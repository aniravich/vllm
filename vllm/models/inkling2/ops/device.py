# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

from vllm.platforms import current_platform


@cache
def is_blackwell() -> bool:
    """Whether the GPU runs Inkling2's SM100 kernels: tml_fa4 with the
    sheared relative bias, the Gluon decode attention, FlashInfer's low-M
    GEMM and its TRT-LLM experts. Hopper (SM90) takes standard FA4, torch
    GEMMs and bf16 experts on FusedMoE instead."""
    return current_platform.is_device_capability_family(100)
