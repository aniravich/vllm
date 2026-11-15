# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 bf16 experts on FusedMoE: the routing, how the stacked checkpoint
tensors are loaded, and (on a GPU) the layer against an fp64 reference."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.inkling2.bf16_moe as bf16_moe
from vllm.models.inkling2.bf16_moe import Inkling2BF16MoE, expert_activation, route
from vllm.platforms import current_platform

NUM_EXPERTS, HIDDEN, INTER, TOPK = 16, 64, 32, 4
CONFIG = SimpleNamespace(
    n_shared_experts=0,
    use_gate_bias=True,
    n_routed_experts=NUM_EXPERTS,
    hidden_size=HIDDEN,
    intermediate_size=INTER,
    num_experts_per_tok=TOPK,
    route_scale=8.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
)


def _reference_route(logits, bias, topk, route_scale):
    """One token at a time: the `topk` experts with the largest fp32
    `sigmoid(logit) + bias`, weighted in fp64 by their `sigmoid(logit)`
    normalized to sum to `route_scale`. Returns weights indexed by expert, 0
    if unselected."""
    weights = torch.zeros(logits.shape, dtype=torch.float64)
    keys = torch.sigmoid(logits.float()) + bias.float()
    for token, row in enumerate(logits.double()):
        chosen = torch.argsort(keys[token], descending=True, stable=True)[:topk]
        score = torch.sigmoid(row[chosen])
        weights[token, chosen] = score / score.sum() * route_scale
    return weights


def _by_expert(weights, ids, num_experts):
    return torch.zeros(ids.shape[0], num_experts, dtype=torch.float64).scatter(
        1, ids.long(), weights.double()
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_route_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(200, 384) * 3
    bias = torch.randn(384) * 0.05

    weights, ids = route(logits, bias, 8, 8.0)

    assert weights.shape == ids.shape == (200, 8)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(
        _by_expert(weights, ids, 384),
        _reference_route(logits, bias, 8, 8.0),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_route_selects_with_the_bias_and_weights_without_it():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0]])
    # The bias lifts the two weakest experts over the others.
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])

    weights, ids = route(logits, bias, 2, 8.0)

    assert sorted(ids[0].tolist()) == [4, 5]
    selected = torch.sigmoid(torch.tensor([-2.0, -3.0]))
    expected = selected / selected.sum() * 8.0
    order = ids[0].argsort()
    torch.testing.assert_close(weights[0, order], expected)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_route_stays_finite_when_the_selected_sigmoids_underflow():
    """sigmoid(-150) is 0 in fp32, so normalizing the sigmoids themselves
    would divide 0 by 0."""
    logits = torch.full((3, 12), -150.0)
    bias = torch.zeros(12)
    bias[[1, 5, 7, 10]] = 1.0

    weights, ids = route(logits, bias, 4, 8.0)

    assert all(sorted(row) == [1, 5, 7, 10] for row in ids.tolist())
    torch.testing.assert_close(weights, torch.full((3, 4), 2.0))


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_expert_activation_follows_the_softcaps():
    """A cap of 0 is no cap, as in `ops.activation._softcap`."""
    assert expert_activation(4.0, 25.0) == {
        "activation": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    assert expert_activation(4.0, 0.0) == {
        "activation": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": None,
    }
    assert expert_activation(0.0, 0.0) == {"activation": "silu"}
    with pytest.raises(ValueError, match="mlp_gate_softcap"):
        expert_activation(0.0, 25.0)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_route_prefers_the_lower_index_among_equal_experts():
    """Saturated sigmoids tie exactly in fp32; the reference kernel then
    takes the lower expert ids."""
    logits = torch.full((2, 12), 40.0)
    logits[1, 3] = -1.0
    bias = torch.zeros(12)

    _, ids = route(logits, bias, 4, 8.0)

    assert ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 4]]


class _FakeRoutedExperts(nn.Module):
    """Stand-in for FusedMoE's `RoutedExperts` on one expert-parallel rank:
    it owns the experts in `local_ids`, stores w13 as `[gate; up]`, and its
    loader skips the experts of other ranks (reporting it)."""

    def __init__(self, local_ids):
        super().__init__()
        self.slots = {expert_id: slot for slot, expert_id in enumerate(local_ids)}
        num_local = len(local_ids)
        expert_map = None
        if num_local < NUM_EXPERTS:
            expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
            expert_map[local_ids] = torch.arange(num_local, dtype=torch.int32)
        self.expert_map_manager = SimpleNamespace(expert_map=expert_map)
        self.loaded: list[tuple[str, str, int]] = []
        self.w13_weight = nn.Parameter(
            torch.zeros(num_local, 2 * INTER, HIDDEN, dtype=torch.bfloat16), False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(num_local, HIDDEN, INTER, dtype=torch.bfloat16), False
        )
        for param in (self.w13_weight, self.w2_weight):
            param.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        assert param is getattr(self, weight_name)
        slot = self.slots.get(expert_id)
        if slot is None:
            return False if return_success else None
        if shard_id == "w1":
            param.data[slot, :INTER].copy_(loaded_weight)
        elif shard_id == "w3":
            param.data[slot, INTER:].copy_(loaded_weight)
        else:
            assert shard_id == "w2"
            param.data[slot].copy_(loaded_weight)
        self.loaded.append((weight_name, shard_id, expert_id))
        return True if return_success else None


def _fake_moe(monkeypatch, local_ids=range(NUM_EXPERTS)):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(routed_experts=_FakeRoutedExperts(list(local_ids)))

    monkeypatch.setattr(bf16_moe, "FusedMoEFactory", factory)
    moe = Inkling2BF16MoE(CONFIG, "model.layers.3.mlp")
    return moe, calls[0]


def _checkpoint_experts():
    torch.manual_seed(0)
    w13 = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN).bfloat16()
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER).bfloat16()
    return w13, w2


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("local_ids", [range(16), range(8, 12)])
def test_load_expert_weight_splits_interleaved_gate_and_up_rows(monkeypatch, local_ids):
    """The checkpoint interleaves a w13's rows as [g0, u0, g1, u1, ...];
    FusedMoE (and its SiTU activation) take [gate; up]."""
    moe, _ = _fake_moe(monkeypatch, local_ids)
    w13, w2 = _checkpoint_experts()

    assert moe.load_expert_weight("w13_weight", w13) == (
        "experts.routed_experts.w13_weight"
    )
    assert moe.load_expert_weight("w2_weight", w2) == "experts.routed_experts.w2_weight"

    local = list(local_ids)
    experts = moe.experts.routed_experts
    assert torch.equal(experts.w13_weight[:, :INTER], w13[local, 0::2])
    assert torch.equal(experts.w13_weight[:, INTER:], w13[local, 1::2])
    assert torch.equal(experts.w2_weight, w2[local])
    # Only this rank's experts reach the loader, each shard once.
    assert sorted(experts.loaded) == sorted(
        [("w13_weight", shard, e) for e in local for shard in ("w1", "w3")]
        + [("w2_weight", "w2", e) for e in local]
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_load_expert_weight_raises_when_the_loader_skips_an_expert(monkeypatch):
    """FusedMoE's loader returns False instead of raising when it loads
    nothing."""
    moe, _ = _fake_moe(monkeypatch)
    w13, _ = _checkpoint_experts()
    del moe.experts.routed_experts.slots[5]

    with pytest.raises(ValueError, match=r"expert 5 \(w1\) not loaded"):
        moe.load_expert_weight("w13_weight", w13)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_load_expert_weight_is_repeatable_in_place(monkeypatch):
    """A weight update loads the same tensors again into the same storage."""
    moe, _ = _fake_moe(monkeypatch)
    w13, w2 = _checkpoint_experts()
    moe.load_expert_weight("w13_weight", w13)
    experts = moe.experts.routed_experts
    address = experts.w13_weight.data_ptr()

    moe.load_expert_weight("w13_weight", -w13)

    assert experts.w13_weight.data_ptr() == address
    assert torch.equal(experts.w13_weight[:, :INTER], -w13[:, 0::2])
    assert torch.equal(experts.w13_weight[:, INTER:], -w13[:, 1::2])


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_load_expert_weight_rejects_quantized_checkpoints(monkeypatch):
    moe, _ = _fake_moe(monkeypatch)
    w13, w2 = _checkpoint_experts()

    with pytest.raises(ValueError, match="w13_weight_scale"):
        moe.load_expert_weight("w13_weight_scale", w13)
    with pytest.raises(ValueError, match="torch.uint8"):
        moe.load_expert_weight("w13_weight", w13.to(torch.uint8))
    with pytest.raises(ValueError, match="w2_weight"):
        moe.load_expert_weight("w2_weight", w2[:, :, : INTER // 2])


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_fused_moe_arguments(monkeypatch):
    moe, kwargs = _fake_moe(monkeypatch)

    assert kwargs["prefix"] == "model.layers.3.mlp.experts"
    assert (kwargs["num_experts"], kwargs["top_k"]) == (NUM_EXPERTS, TOPK)
    assert (kwargs["hidden_size"], kwargs["intermediate_size"]) == (HIDDEN, INTER)
    assert kwargs["quant_config"] is None and kwargs["renormalize"] is False
    assert kwargs["params_dtype"] == torch.bfloat16
    assert kwargs["router_logits_dtype"] == torch.float32
    assert kwargs["activation"] == "situ"
    assert kwargs["activation_situ_beta"] == CONFIG.mlp_gate_softcap
    assert kwargs["activation_situ_linear_beta"] == CONFIG.mlp_up_softcap

    torch.manual_seed(0)
    moe.gate.bias.data.normal_()
    logits = torch.randn(5, NUM_EXPERTS)
    routing = kwargs["custom_routing_function"]
    weights, ids = routing(
        hidden_states=None, gating_output=logits, topk=TOPK, renormalize=False
    )
    ref_weights, ref_ids = route(logits, moe.gate.bias, TOPK, CONFIG.route_scale)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_padding_rows_are_routed_to_no_expert(monkeypatch):
    """FusedMoE skips rows whose expert ids are -1 (CUDA graph and
    data-parallel padding); its own routers mark them the same way."""
    moe, kwargs = _fake_moe(monkeypatch)
    torch.manual_seed(0)
    moe.gate.bias.data.normal_()
    logits = torch.randn(5, NUM_EXPERTS)
    is_padding = torch.tensor([False, False, True, False, True])
    monkeypatch.setattr(
        bf16_moe, "get_moe_padding_mask", lambda num_tokens: is_padding[:num_tokens]
    )

    weights, ids = kwargs["custom_routing_function"](
        hidden_states=None, gating_output=logits, topk=TOPK, renormalize=False
    )

    ref_weights, ref_ids = route(logits, moe.gate.bias, TOPK, CONFIG.route_scale)
    assert torch.equal(ids[~is_padding], ref_ids[~is_padding])
    assert (ids[is_padding] == -1).all()
    assert torch.equal(weights, ref_weights)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_bf16_moe_matches_reference(num_tokens, dist_init, workspace_init):
    """The layer on FusedMoE's kernels, loaded from checkpoint-layout
    tensors, against one expert at a time in fp64."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    torch.manual_seed(0)
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    w13, w2 = _checkpoint_experts()
    w13, w2 = (w13 * 0.2).cuda(), (w2 * 0.2).cuda()
    x = (torch.randn(num_tokens, HIDDEN) * 0.5).bfloat16().cuda()

    with set_current_vllm_config(vllm_config):
        moe = Inkling2BF16MoE(CONFIG, "model.layers.0.mlp").cuda()
        moe.gate.weight.data.normal_(std=0.3)
        moe.gate.bias.data.normal_(std=0.05)
        moe.load_expert_weight("w13_weight", w13)
        moe.load_expert_weight("w2_weight", w2)
        experts = moe.experts.routed_experts
        experts.quant_method.process_weights_after_loading(experts)
        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
            out = moe(x)

    assert out.shape == x.shape and out.dtype == torch.bfloat16
    xd = x.double()
    logits = xd @ moe.gate.weight.double().t()
    weights = _reference_route(logits.cpu(), moe.gate.bias.cpu(), TOPK, 8.0).cuda()
    ref = torch.zeros_like(xd)
    for expert_id in range(NUM_EXPERTS):
        gate, up = (
            (w13[expert_id].double() @ xd.t()).t().unflatten(1, (-1, 2)).unbind(2)
        )
        hidden = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
        hidden = hidden * 25 * torch.tanh(up / 25)
        ref += weights[:, expert_id, None] * (hidden @ w2[expert_id].double().t())
    torch.testing.assert_close(out.double(), ref, atol=2e-2, rtol=2e-2)
