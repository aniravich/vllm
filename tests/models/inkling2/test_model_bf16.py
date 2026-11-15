# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test of a tiny random-weight Inkling2 with bf16 experts (TP=1),
which also runs on GPUs without the Blackwell kernels: attention on standard
FA4 and the experts on FusedMoE."""

import json
from typing import NamedTuple

import pytest
import torch
from safetensors.torch import save_file

from tests.models.inkling2.checkpoint_utils import (
    bf16_experts_metadata,
    random_checkpoint,
)
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
# Standard FA4 takes a paged KV cache with a score-mod on these families only.
if _cap is None or _cap.major not in (9, 10, 11):
    pytest.skip(
        "Inkling2 on FA4 requires SM90, SM100 or SM110", allow_module_level=True
    )

NUM_LAYERS = 2
TINY_CONFIG = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": NUM_LAYERS,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
}
# Seed of the checkpoint the numbers below were measured for (they change with
# the tensors `random_checkpoint` draws and with `_conditioned`).
SEED = 12
# The short prompts decode while the long one prefills in 256-token chunks.
PROMPTS = ([7, 8, 9], [5, 5, 5], list(range(2, 702)))
# The token each request is made to generate, so that the two runs decode the
# same sequence whatever they would have sampled.
FORCED_TOKENS = (11, 23, 42)
MAX_TOKENS = 8
# The most a log-probability of the graph run may differ from the eager run's.
# The two run the same kernels on batches padded to different sizes, where a
# GEMM can round an element to the other bf16 neighbour; from there on the
# runs round independently. An offline emulation puts that at 0.08 when up to
# a tenth of the GEMM outputs differ and at 0.14 when all of them do; two
# runs of the same kernels should stay far below either. Wrong results
# (relative logits one column off, an attention output or its log-sum-exp
# lost, decode rows without their conv state) move a compared value by 0.25
# or more, and by more than 0.7 when they are confined to the decode steps;
# a local window one token short moves one by 0.14 and is not caught. To be
# calibrated on the first GPU run, which prints what the two runs differ by.
ATOL = 0.15
# The eager log-probabilities have to be spread at least this wide (measured:
# 5.96 over the prompts, 2.96 over the forced tokens): dummy weights put all
# of them within 1e-3 of -ln(1000), where they would agree whatever the
# graphs compute.
MIN_SPREAD = 1.5


class _Run(NamedTuple):
    """What one request returned, as log-probabilities of the model's
    unrestricted distribution."""

    # Of each prompt token after the first.
    prompt_logprobs: torch.Tensor
    # Per generated position: of the forced token, and of the most likely one.
    forced_logprobs: torch.Tensor
    top_logprobs: torch.Tensor


def _conditioned(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """The generator draws every tensor from N(0, 1). Give the model sane
    magnitudes: norm weights and clip scales near 1, projections whose
    outputs have unit variance, logits of order 1."""
    if "norm.weight" in name or name.endswith(".scale"):
        return 1 + 0.1 * tensor
    if ".attn_res." in name:
        return tensor
    if name.endswith("gate.bias"):
        # Spaced wider than the range of the sigmoid scores, so every token
        # takes the same experts: a top-k choice flipped by a rounding
        # difference between two runs would move a log-probability by O(1).
        return 2.0 * torch.arange(tensor.shape[0], dtype=tensor.dtype)
    if name.endswith("rel_logits_proj.proj"):
        return 0.5 * tensor
    # The attention output as large as the experts', and sinks that reach the
    # log-sum-exp they are compared with.
    gain = 8.0 if name.endswith(("wo_ud.weight", "attn_sink_proj.weight")) else 1.0
    return gain * tensor.shape[-1] ** -0.5 * tensor


def _write_checkpoint(path) -> str:
    tensors = {
        name: _conditioned(name, tensor)
        for name, tensor in random_checkpoint(TINY_CONFIG["text_config"], SEED).items()
    }
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # The checkpoint's precision metadata lists the routed experts as bf16.
    (path / "hf_quant_config.json").write_text(
        json.dumps(bf16_experts_metadata(NUM_LAYERS))
    )
    return str(path)


def _chunked_mixed_run(model: str, enforce_eager: bool) -> list[_Run]:
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=256,
        enforce_eager=enforce_eager,
        # The 256-token piecewise graph is captured on 16 verify-sized requests.
        compilation_config={"cudagraph_capture_sizes": [8, 16, 256]},
        limit_mm_per_prompt={"image": 0},
    )
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in PROMPTS],
        [
            SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=0,
                ignore_eos=True,
                detokenize=False,
                prompt_logprobs=0,
                # The most likely token's and the generated one's, both taken
                # before the restriction to `allowed_token_ids`.
                logprobs=1,
                allowed_token_ids=[token],
            )
            for token in FORCED_TOKENS
        ],
    )
    del llm
    runs = []
    for o, token in zip(outputs, FORCED_TOKENS):
        completion = o.outputs[0]
        assert list(completion.token_ids) == [token] * MAX_TOKENS
        assert o.prompt_logprobs is not None and completion.logprobs is not None
        runs.append(
            _Run(
                torch.tensor(
                    [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]]
                ),
                torch.tensor([d[token].logprob for d in completion.logprobs]),
                torch.tensor(
                    [max(lp.logprob for lp in d.values()) for d in completion.logprobs]
                ),
            )
        )
    return runs


def test_bf16_experts_graphs_match_eager(tmp_path):
    """One local and one full layer with bf16 experts load a checkpoint,
    prefill in chunks next to decodes and decode with the paged KV and
    conv-state caches. Full and piecewise CUDA graphs pad their batches and
    capture or replay the attention, so they have to match eager: in the
    prompt log-probabilities, which the prefill steps produce, and in those
    of the generated positions. Once the long prompt has prefilled, every
    step only decodes and replays the full graph of 8 requests, most of them
    padding."""
    model = _write_checkpoint(tmp_path)
    graph = _chunked_mixed_run(model, enforce_eager=False)
    eager = _chunked_mixed_run(model, enforce_eager=True)

    for field in ("prompt_logprobs", "forced_logprobs"):
        values = torch.cat([getattr(run, field) for run in eager])
        assert values.isfinite().all()
        assert values.max() - values.min() > MIN_SPREAD, field
    for index, (g, e) in enumerate(zip(graph, eager)):
        for field in _Run._fields:
            ours, theirs = getattr(g, field), getattr(e, field)
            # Shown with `pytest -s`: what `ATOL` is to be calibrated with.
            worst = (ours - theirs).abs().max().item()
            print(f"request {index} {field}: max |graph - eager| = {worst:.4f}")
            torch.testing.assert_close(
                ours,
                theirs,
                rtol=0,
                atol=ATOL,
                msg=lambda message, index=index, field=field: (
                    f"request {index} {field}: {message}"
                ),
            )
