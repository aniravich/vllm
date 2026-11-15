# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 checkpoint loading and in-place weight updates, on CPU.

A tiny model with bf16 experts is built for one tensor-parallel rank and
loaded from a synthetic checkpoint in the release's names and shapes. Every
parameter is compared with the slice of the checkpoint its rank owns; then
the checkpoint is delivered again one tensor at a time, as a weight update
does, and has to land in the same storage.
"""

import random
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_test

HIDDEN, LAYERS, LOCAL_LAYERS = 64, 3, (0, 1)
HEADS, KV_HEADS, HEAD_DIM, D_REL, EXTENT, KERNEL = 16, 8, 16, 16, 128, 4
EXPERTS, INTER, VOCAB = 16, 32, 96
TEXT_CONFIG = dict(
    torch_dtype="bfloat16",
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    local_layer_ids=list(LOCAL_LAYERS),
    sliding_window_size=EXTENT,
    rel_extent=EXTENT,
    vocab_size=VOCAB,
    unpadded_vocab_size=90,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    d_rel=D_REL,
    sconv_kernel_size=KERNEL,
    n_layers_per_attn_res_block=2,
    n_routed_experts=EXPERTS,
    num_experts_per_tok=4,
    intermediate_size=INTER,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
)
BF16_EXPERTS = {
    "model": {
        "parameter_groups": [
            {
                "compute_dtype": "bfloat16",
                "tensors": [
                    f"model.llm.layers.{i}.mlp.experts.{name}"
                    for i in range(LAYERS)
                    for name in ("w13_weight", "w2_weight")
                ],
            }
        ]
    }
}


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Random tensors under the release's names, shapes and dtypes (bf16
    experts: `w13_weight` rows interleave gate and up)."""
    gen = torch.Generator().manual_seed(seed)

    def bf16(*shape):
        return torch.randn(shape, generator=gen).bfloat16()

    def fp32(*shape):
        return torch.randn(shape, generator=gen)

    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    ckpt = {
        "embed.weight": bf16(VOCAB, HIDDEN),
        "unembed.weight": bf16(VOCAB, HIDDEN),
        "embed_norm.weight": bf16(HIDDEN),
        "norm.weight": bf16(HIDDEN),
        "attn_res.output_query": fp32(HIDDEN),
    }
    for i in range(LAYERS):
        layer = f"layers.{i}"
        local = i in LOCAL_LAYERS
        if i > 0:
            shape = (HIDDEN,) if local else (3, HIDDEN)
            ckpt[f"attn_res.attn_queries.{i - 1}"] = fp32(*shape)
        ckpt[f"attn_res.mlp_queries.{i}"] = fp32(HIDDEN)
        ckpt |= {
            f"{layer}.attn.wq_du.weight": bf16(q_size, HIDDEN),
            f"{layer}.attn.wr_du.weight": bf16(HEADS * D_REL, HIDDEN),
            f"{layer}.attn.gate_attn.weight": bf16(q_size, HIDDEN),
            f"{layer}.attn.attn_sink_proj.weight": bf16(HEADS, HIDDEN),
            f"{layer}.attn.wk_dv.weight": bf16(kv_size, HIDDEN),
            f"{layer}.attn.wv_dv.weight": bf16(kv_size, HIDDEN),
            f"{layer}.attn.wo_ud.weight": bf16(HIDDEN, q_size),
            f"{layer}.attn.q_l2_clip.scale": bf16(HEADS, HEAD_DIM),
            f"{layer}.attn.rel_l2_clip.scale": bf16(HEADS, D_REL),
            f"{layer}.attn.rel_logits_proj.proj": bf16(D_REL, EXTENT),
            f"{layer}.attn_sconv.weight": bf16(HIDDEN, 1, KERNEL),
            f"{layer}.mlp_sconv.weight": bf16(HIDDEN, 1, KERNEL),
            f"{layer}.mlp_norm.weight": bf16(HIDDEN),
            f"{layer}.mlp.gate.weight": bf16(EXPERTS, HIDDEN),
            f"{layer}.mlp.gate.bias": fp32(EXPERTS),
            f"{layer}.mlp.experts.w13_weight": bf16(EXPERTS, 2 * INTER, HIDDEN),
            f"{layer}.mlp.experts.w2_weight": bf16(EXPERTS, HIDDEN, INTER),
        }
        for conv in ("k_sconv", "v_sconv"):
            if local:
                ckpt[f"{layer}.attn.{conv}.weight"] = bf16(kv_size, 1, KERNEL)
            else:
                ckpt[f"{layer}.attn.{conv}.static_w"] = bf16(KERNEL, kv_size)
                ckpt[f"{layer}.attn.{conv}.weight_proj"] = bf16(
                    KV_HEADS * KERNEL, HIDDEN
                )
        if local:
            ckpt[f"{layer}.attn_norm.weight"] = bf16(HIDDEN)
        else:
            for norm in ("q_input_norm", "k_input_norm", "v_input_norm"):
                ckpt[f"{layer}.{norm}.weight"] = bf16(HIDDEN)
    return {f"model.llm.{name}": tensor for name, tensor in ckpt.items()}


def _expected(
    ckpt: dict[str, torch.Tensor], tp_size: int, tp_rank: int
) -> dict[str, torch.Tensor]:
    """The parameters of one tensor-parallel rank (experts sharded along the
    intermediate dimension), derived from the checkpoint layout."""

    def get(name):
        return ckpt[f"model.llm.{name}"]

    def shard(tensor, dim):
        size = tensor.shape[dim] // tp_size
        return tensor.narrow(dim, tp_rank * size, size)

    expected = {
        "model.embed_norm.weight": get("embed_norm.weight")[None],
        "model.norm.weight": get("norm.weight")[None],
        "model.attn_res.output_query": get("attn_res.output_query")[None]
        * HIDDEN**-0.5,
    }
    for i in range(LAYERS):
        ck, layer = f"layers.{i}", f"model.layers.{i}"
        local = i in LOCAL_LAYERS
        if i > 0:
            query = get(f"attn_res.attn_queries.{i - 1}").view(-1, HIDDEN)
            expected[f"model.attn_res.attn_queries.{i - 1}"] = query * HIDDEN**-0.5
        expected[f"model.attn_res.mlp_queries.{i}"] = (
            get(f"attn_res.mlp_queries.{i}")[None] * HIDDEN**-0.5
        )
        # Query-side projections first, then K and V, then zero rows that
        # align the fused weight to 8.
        fused = [
            shard(get(f"{ck}.attn.{name}.weight"), 0)
            for name in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj")
        ] + [shard(get(f"{ck}.attn.{name}.weight"), 0) for name in ("wk_dv", "wv_dv")]
        rows = sum(part.shape[0] for part in fused)
        fused.append(torch.zeros(-rows % 8, HIDDEN, dtype=torch.bfloat16))
        expected |= {
            f"{layer}.attn.qkvr.weight": torch.cat(fused),
            f"{layer}.attn.wo_ud.weight": shard(get(f"{ck}.attn.wo_ud.weight"), 1),
            f"{layer}.attn.q_scale": shard(get(f"{ck}.attn.q_l2_clip.scale"), 0),
            f"{layer}.attn.r_scale": shard(get(f"{ck}.attn.rel_l2_clip.scale"), 0),
            f"{layer}.attn.rel_logits_proj": get(f"{ck}.attn.rel_logits_proj.proj"),
            f"{layer}.attn_sconv.weight": get(f"{ck}.attn_sconv.weight"),
            f"{layer}.mlp_sconv.weight": get(f"{ck}.mlp_sconv.weight"),
            f"{layer}.mlp_norm.weight": get(f"{ck}.mlp_norm.weight")[None],
            f"{layer}.mlp.gate.weight": get(f"{ck}.mlp.gate.weight"),
            f"{layer}.mlp.gate.bias": get(f"{ck}.mlp.gate.bias"),
        }
        for conv in ("k_sconv", "v_sconv"):
            if local:
                expected[f"{layer}.attn.{conv}.weight"] = shard(
                    get(f"{ck}.attn.{conv}.weight"), 0
                )
            else:
                expected[f"{layer}.attn.{conv}.static_w"] = shard(
                    get(f"{ck}.attn.{conv}.static_w"), 1
                )
                expected[f"{layer}.attn.{conv}.weight_proj"] = shard(
                    get(f"{ck}.attn.{conv}.weight_proj"), 0
                )
        if local:
            expected[f"{layer}.attn_norm.weight"] = get(f"{ck}.attn_norm.weight")[None]
        else:
            expected[f"{layer}.attn_norm.weight"] = torch.stack(
                [
                    get(f"{ck}.{norm}.weight")
                    for norm in ("q_input_norm", "k_input_norm", "v_input_norm")
                ]
            )
        # Checkpoint w13 rows interleave [g0, u0, g1, u1, ...]; the parameter
        # is [gate; up].
        w13, w2 = (
            get(f"{ck}.mlp.experts.{name}") for name in ("w13_weight", "w2_weight")
        )
        expected[f"{layer}.mlp.experts.routed_experts.w13_weight"] = torch.cat(
            [shard(w13[:, 0::2], 1), shard(w13[:, 1::2], 1)], dim=1
        )
        expected[f"{layer}.mlp.experts.routed_experts.w2_weight"] = shard(w2, 2)
    return expected


@pytest.fixture(params=[(1, 0), (8, 0), (8, 5)], ids=["tp1", "tp8-rank0", "tp8-rank5"])
def tp(request):
    return request.param


@pytest.fixture
def model(dist_init, monkeypatch, tp):
    """A tiny Inkling2 with bf16 experts, built as rank `tp[1]` of `tp[0]`."""
    import vllm.distributed.parallel_state as parallel_state
    import vllm.models.inkling2.bf16_moe as bf16_moe
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.inkling2.configs import Inkling2MMConfig
    from vllm.models.inkling2.model import Inkling2ForConditionalGeneration
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    tp_size, tp_rank = tp
    if tp_size > 1:
        monkeypatch.setattr(
            parallel_state,
            "_TP",
            SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank, cpu_group=None),
        )
    # FusedMoE's CPU experts have no SiTU; loading does not depend on the
    # activation.
    monkeypatch.setattr(
        bf16_moe, "expert_activation", lambda gate_cap, up_cap: {"activation": "silu"}
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=Inkling2MMConfig(text_config=TEXT_CONFIG),
        dtype=torch.bfloat16,
        head_dtype=torch.float32,
        enable_prompt_embeds=False,
        multimodal_config=None,
        is_moe=True,
    )
    vllm_config.quant_config = Inkling2QuantConfig(BF16_EXPERTS)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            return Inkling2ForConditionalGeneration(vllm_config=vllm_config)
    finally:
        torch.set_default_dtype(default_dtype)


def _assert_parameters(model, expected, tp_size):
    params = dict(model.named_parameters())
    # The vocabulary shards of embed / unembed are VocabParallelEmbedding's.
    vocab = {"model.embed_tokens.weight", "lm_head.weight"}
    assert params.keys() - vocab == expected.keys()
    for name, tensor in expected.items():
        assert params[name].dtype == tensor.dtype, name
        assert torch.equal(params[name], tensor), name
    if tp_size == 1:
        for name, ckpt_name in zip(sorted(vocab), ("unembed", "embed")):
            weight = model.ckpt[f"model.llm.{ckpt_name}.weight"]
            assert torch.equal(params[name][:VOCAB], weight), name
            assert params[name][VOCAB:].count_nonzero() == 0, name


def test_load_weights_places_every_checkpoint_tensor(model, tp):
    tp_size, tp_rank = tp
    model.ckpt = _checkpoint(0)

    loaded = model.load_weights(model.ckpt.items())

    assert loaded == {name for name, _ in model.named_parameters()}
    _assert_parameters(model, _expected(model.ckpt, tp_size, tp_rank), tp_size)


def test_weight_update_rewrites_the_same_storage(model, tp, monkeypatch):
    """An update delivers the checkpoint over several `load_weights` calls
    (here one tensor each, in any order); loaders write in place, and the
    table derived from `rel_logits_proj` is refilled in place."""
    import vllm.model_executor.model_loader.reload.lifecycle as lifecycle
    from vllm.models.inkling2.ops.sconv import shift_rel_proj

    tp_size, tp_rank = tp
    model.load_weights(_checkpoint(0).items())
    attentions = [layer.attn for layer in model.model.layers]
    for attn in attentions:
        attn.process_weights_after_loading()
    addresses = {name: p.data_ptr() for name, p in model.named_parameters()}
    shifted = [attn.rel_proj_shifted for attn in attentions]

    model.ckpt = _checkpoint(1)
    update = list(model.ckpt.items())
    random.Random(0).shuffle(update)
    monkeypatch.setattr(lifecycle, "_weight_update_active", True)
    loaded: set[str] = set()
    for item in update:
        loaded |= model.load_weights([item])
    monkeypatch.setattr(lifecycle, "_weight_update_active", False)
    for attn in attentions:
        attn.process_weights_after_loading()

    assert loaded == addresses.keys()
    _assert_parameters(model, _expected(model.ckpt, tp_size, tp_rank), tp_size)
    assert addresses == {name: p.data_ptr() for name, p in model.named_parameters()}
    for attn, table in zip(attentions, shifted):
        assert attn.rel_proj_shifted is table
        assert torch.equal(table, shift_rel_proj(attn.rel_logits_proj, attn.is_local))


def test_partial_checkpoint_is_rejected_outside_an_update(model, tp):
    ckpt = _checkpoint(0)
    del ckpt["model.llm.layers.1.attn.wk_dv.weight"]

    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        model.load_weights(ckpt.items())


def test_quantized_expert_tensors_are_rejected_by_bf16_experts(model, tp):
    ckpt = _checkpoint(0)
    name = "model.llm.layers.0.mlp.experts.w13_weight"

    with pytest.raises(ValueError, match="torch.uint8"):
        model.load_weights([(name, ckpt[name].to(torch.uint8))])
    with pytest.raises(ValueError, match="w13_weight_scale"):
        model.load_weights([(f"{name}_scale", ckpt[name])])


@pytest.mark.skip_global_cleanup
def test_expert_format_follows_the_checkpoint_metadata():
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    def group(compute_dtype, layers):
        return {
            "compute_dtype": compute_dtype,
            "tensors": [
                f"model.llm.layers.{i}.mlp.experts.{name}"
                for i in layers
                for name in ("w13_weight", "w2_weight")
            ],
        }

    config = Inkling2QuantConfig(
        {
            "model": {
                "parameter_groups": [
                    group("nvfp4", [0]),
                    group("mxfp8", [1]),
                    group("bfloat16", [2]),
                    {"compute_dtype": "float32", "tensors": ["model.llm.norm.weight"]},
                ]
            }
        }
    )

    assert [config.expert_format(i) for i in range(4)] == [
        "nvfp4",
        "mxfp8",
        "bf16",
        "nvfp4",
    ]
    # bf16 experts need no Blackwell kernel; the quantized ones check for
    # Blackwell when they are built.
    assert config.get_min_capability() == 90


@pytest.mark.skip_global_cleanup
def test_quantized_experts_need_blackwell():
    from vllm.models.inkling2.moe import Inkling2Experts
    from vllm.models.inkling2.ops.device import is_blackwell

    if is_blackwell():
        pytest.skip("the quantized experts run on Blackwell")
    with pytest.raises(ValueError, match="Blackwell"):
        Inkling2Experts(SimpleNamespace(), "nvfp4")
