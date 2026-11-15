# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor precision metadata of Inkling2 checkpoints (hf_quant_config.json).

Only the routed experts are quantized (NVFP4 or MXFP8); they are handled by
`Inkling2MoE`. Every other layer runs unquantized, and so do the routed
experts a checkpoint lists as bfloat16 (`Inkling2BF16MoE`).
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

_COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4")

ExpertFormat = Literal["nvfp4", "mxfp8", "bf16"]


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.mxfp8_weights: set[str] = set()
        self.bf16_weights: set[str] = set()
        for group in config["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                self.mxfp8_weights.update(group["tensors"])
            elif compute == "bfloat16":
                self.bf16_weights.update(group["tensors"])

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Standard FA4 takes the relative-bias score-mod from SM90 on. The
        # NVFP4 / MXFP8 experts need SM100 and check for it themselves.
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None

    def expert_format(self, layer_id: int) -> ExpertFormat:
        """Storage of a layer's routed experts."""
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        if name in self.mxfp8_weights:
            return "mxfp8"
        if name in self.bf16_weights:
            return "bf16"
        return "nvfp4"
