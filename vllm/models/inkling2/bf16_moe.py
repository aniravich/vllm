# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts with unquantized (bf16) routed experts, on
vLLM's FusedMoE.

For checkpoints whose routed experts are bf16: such experts run on any GPU,
shard over the expert-parallel group (TP x DP) and take in-place weight
updates. The gate, the selection and the routing weights are the ones of
`Inkling2MoE`; the experts run without activation quantization, and the
routing weight multiplies each expert's output instead of its SwiGLU.
"""

from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.base_router import (
    get_moe_padding_mask,
)

# Stacked checkpoint tensors of the routed experts, under `mlp.experts`.
_W13, _W2 = "w13_weight", "w2_weight"


def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 `x @ weight.T` of bf16 operands, accumulated in fp32."""
    if x.is_cuda:
        return torch.mm(x, weight.t(), out_dtype=torch.float32)
    return torch.mm(x.float(), weight.float().t())


def route(
    logits: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing weights and expert ids `(T, topk)` of fp32 gate `logits`
    `(T, E)`.

    Experts are selected by `sigmoid(logits) + bias`, the lower index first
    among equals; a selected expert's weight is its `sigmoid(logits)`,
    normalized over the selection (as a softmax of log-sigmoids, which stays
    finite when they all underflow) and scaled by `route_scale`.
    """
    logits = logits.float()
    keys = torch.sigmoid(logits) + bias
    # A stable sort, for the tie rule (`torch.topk` leaves it open).
    ids = keys.sort(dim=-1, descending=True, stable=True).indices[..., :topk]
    selected = nn.functional.logsigmoid(logits.gather(-1, ids))
    return torch.softmax(selected, dim=-1) * route_scale, ids


def expert_activation(gate_cap: float, up_cap: float) -> dict[str, Any]:
    """FusedMoE's arguments for Inkling2's expert activation `softcap(gate) *
    sigmoid(gate) * softcap(up)`, where `softcap(x) = cap * tanh(x / cap)`
    and a cap of 0 is no cap."""
    if gate_cap > 0:
        # SiTU is beta * tanh(g / beta) * sigmoid(g) times
        # linear_beta * tanh(u / linear_beta), or times u without a linear_beta.
        return {
            "activation": "situ",
            "activation_situ_beta": gate_cap,
            "activation_situ_linear_beta": up_cap if up_cap > 0 else None,
        }
    if up_cap > 0:
        raise ValueError(
            "Inkling2 bf16 experts need mlp_gate_softcap > 0 to cap the up "
            f"projection (mlp_up_softcap = {up_cap})"
        )
    return {"activation": "silu"}


class Inkling2BF16Gate(nn.Module):
    """The gate's parameters, under the checkpoint's names. `Inkling2Gate`
    also holds the counters of its routing kernels: a buffer that no
    checkpoint tensor fills would keep a layerwise weight reload from ever
    seeing this module complete."""

    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )


class Inkling2BF16MoE(nn.Module):
    def __init__(self, config, prefix: str) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2BF16Gate(config)
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=torch.bfloat16,
            # `route` normalizes the weights itself.
            renormalize=False,
            quant_config=None,
            prefix=f"{prefix}.experts",
            custom_routing_function=self._route,
            router_logits_dtype=torch.float32,
            **expert_activation(config.mlp_gate_softcap, config.mlp_up_softcap),
        )

    def _route(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FusedMoE `custom_routing_function`."""
        del hidden_states, renormalize
        assert topk == self.topk
        weights, ids = route(gating_output, self.gate.bias, topk, self.route_scale)
        # CUDA graph and data-parallel padding rows go to no expert (and come
        # out as 0), as with FusedMoE's own routers.
        is_padding = get_moe_padding_mask(ids.shape[0])
        if is_padding is not None and is_padding.shape[0] == ids.shape[0]:
            ids = ids.masked_fill(is_padding[:, None], -1)
        return weights, ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel
        group."""
        logits = router_logits(x, self.gate.weight)
        return self.experts(hidden_states=x, router_logits=logits)

    def load_expert_weight(self, name: str, weight: torch.Tensor) -> str:
        """Load a stacked checkpoint tensor of the routed experts: bf16
        `w13_weight` `(E, 2 * I, H)` with interleaved rows `[g0, u0, g1,
        u1, ...]`, or `w2_weight` `(E, H, I)`. Returns the name of the
        parameter it fills, relative to this module.

        Each of this rank's experts (all of them without expert parallelism)
        goes through FusedMoE's loader, which takes the rank's slice of the
        intermediate dimension (tensor parallel) and stores w13 as
        `[gate; up]`.
        """
        experts = self.experts.routed_experts
        hidden, inter = self.hidden_size, self.intermediate_size
        shapes = {
            _W13: (self.num_experts, 2 * inter, hidden),
            _W2: (self.num_experts, hidden, inter),
        }
        if name not in shapes:
            raise ValueError(
                f"Unexpected tensor for bf16 Inkling2 experts: experts.{name} "
                "(a quantized checkpoint?)"
            )
        param = getattr(experts, name)
        if weight.dtype != param.dtype or weight.shape != shapes[name]:
            raise ValueError(
                f"experts.{name}: checkpoint {weight.dtype} {tuple(weight.shape)} "
                f"!= {param.dtype} {shapes[name]}"
            )
        loader = param.weight_loader
        expert_map = experts.expert_map_manager.expert_map
        local_ids = (
            range(self.num_experts)
            if expert_map is None
            else [e for e, slot in enumerate(expert_map.tolist()) if slot >= 0]
        )
        for expert_id in local_ids:
            expert = weight[expert_id]
            if name == _W13:
                shards = (("w1", expert[0::2]), ("w3", expert[1::2]))
            else:
                shards = (("w2", expert),)
            for shard_id, shard in shards:
                # The loader reports a skipped tensor instead of raising.
                if not loader(
                    param, shard, name, shard_id, expert_id, return_success=True
                ):
                    raise ValueError(
                        f"experts.{name}: expert {expert_id} ({shard_id}) not loaded"
                    )
        return f"experts.routed_experts.{name}"
