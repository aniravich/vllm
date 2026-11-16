# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests of a tiny random-weight Inkling2 with bf16 experts on
FusedMoE, through `LLM`.

The attention path depends on the GPU: Hopper (9.x) runs standard FA4 with
the relative-bias score-mod, Blackwell (10.x) the Blackwell kernels. A pass
on one family is no coverage of the other's attention."""

import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import pytest
import torch
from safetensors.torch import save_file

from tests.models.inkling2.checkpoint_utils import (
    bf16_experts_metadata,
    random_checkpoint,
)
from tests.utils import multi_gpu_marks
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
# The model refuses to build on any other family.
if _cap is None or _cap.major not in (9, 10):
    pytest.skip("Inkling2 requires SM90 or SM100", allow_module_level=True)
ATTENTION_PATH = (
    "the Blackwell kernels"
    if current_platform.is_device_capability_family(100)
    else "standard FA4 with the score-mod"
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
# Six requests that decode together, for `WIDE_GRAPHS`.
SHORT_PROMPTS = tuple([first, first + 1, first + 2] for first in range(20, 80, 10))
SHORT_FORCED_TOKENS = (11, 23, 42, 57, 64, 99)
MAX_TOKENS = 8
# The 256-token piecewise graph is captured on 16 verify-sized requests.
CHUNKED_GRAPHS: dict[str, Any] = {
    "max_num_seqs": 16,
    "max_num_batched_tokens": 256,
    "compilation_config": {"cudagraph_capture_sizes": [8, 16, 256]},
}
# Every batch is padded to 512 rows: prefills replay the piecewise graph of
# 512 tokens, decodes the full graph of 512 requests (a decode graph holds no
# more than `max_num_seqs`, so 16 would keep the decodes at 16 rows).
WIDE_GRAPHS: dict[str, Any] = {
    "max_num_seqs": 512,
    "max_num_batched_tokens": 512,
    "compilation_config": {"cudagraph_capture_sizes": [512]},
}
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
# The reference log-probabilities have to be spread at least this wide (an
# offline emulation puts the narrowest set at 2.1): dummy weights put all of
# them within 1e-3 of -ln(1000), where they would agree whatever is computed.
MIN_SPREAD = 1.5


class _Run(NamedTuple):
    """What one request returned, as log-probabilities of the model's
    unrestricted distribution."""

    # Of each prompt token after the first.
    prompt_logprobs: torch.Tensor
    # Per generated position: of the forced token, and of the most likely one.
    forced_logprobs: torch.Tensor
    top_logprobs: torch.Tensor


def _conditioned(name: str, tensor: torch.Tensor, expert_shift: int) -> torch.Tensor:
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
        bias = 2.0 * torch.arange(tensor.shape[0], dtype=tensor.dtype)
        return bias.roll(expert_shift)
    if name.endswith("rel_logits_proj.proj"):
        return 0.5 * tensor
    # The attention output as large as the experts', and sinks that reach the
    # log-sum-exp they are compared with.
    gain = 8.0 if name.endswith(("wo_ud.weight", "attn_sink_proj.weight")) else 1.0
    return gain * tensor.shape[-1] ** -0.5 * tensor


def _write_checkpoint(path, seed: int = SEED, expert_shift: int = 0) -> str:
    """Every token takes the last 8 of the 16 experts, or with `expert_shift`
    the 8 that many places further on, wrapping around."""
    tensors = {
        name: _conditioned(name, tensor, expert_shift)
        for name, tensor in random_checkpoint(TINY_CONFIG["text_config"], seed).items()
    }
    path.mkdir(exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    # The checkpoint's precision metadata lists the routed experts as bf16.
    (path / "hf_quant_config.json").write_text(
        json.dumps(bf16_experts_metadata(NUM_LAYERS))
    )
    return str(path)


@contextmanager
def _engine(model: str, **kwargs: Any) -> Iterator[LLM]:
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0},
        **(CHUNKED_GRAPHS | kwargs),
    )
    try:
        yield llm
    finally:
        # Returns once the engine's processes are gone: the next engine finds
        # the GPUs free.
        llm.llm_engine.engine_core.shutdown()


def _generate(
    llm: LLM,
    prompts: Sequence[list[int]] = PROMPTS,
    forced_tokens: Sequence[int] = FORCED_TOKENS,
) -> list[_Run]:
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
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
            for token in forced_tokens
        ],
    )
    runs = []
    for o, prompt, token in zip(outputs, prompts, forced_tokens, strict=True):
        completion = o.outputs[0]
        assert list(completion.token_ids) == [token] * MAX_TOKENS
        assert o.prompt_logprobs is not None and completion.logprobs is not None
        prompt_logprobs = [
            next(iter(d.values())).logprob for d in o.prompt_logprobs[1:] if d
        ]
        assert len(prompt_logprobs) == len(prompt) - 1
        runs.append(
            _Run(
                torch.tensor(prompt_logprobs),
                torch.tensor([d[token].logprob for d in completion.logprobs]),
                torch.tensor(
                    [max(lp.logprob for lp in d.values()) for d in completion.logprobs]
                ),
            )
        )
    return runs


def _run(
    model: str,
    prompts: Sequence[list[int]] = PROMPTS,
    forced_tokens: Sequence[int] = FORCED_TOKENS,
    **kwargs: Any,
) -> list[_Run]:
    with _engine(model, **kwargs) as llm:
        return _generate(llm, prompts, forced_tokens)


def _assert_close(ours: list[_Run], reference: list[_Run], what: str) -> None:
    spreads: dict[str, float] = {}
    for field in ("prompt_logprobs", "forced_logprobs"):
        values = torch.cat([getattr(run, field) for run in reference])
        spreads[field] = (values.max() - values.min()).item()
    pairs = [
        (f"request {index} {field}", getattr(run, field), getattr(ref, field))
        for index, (run, ref) in enumerate(zip(ours, reference, strict=True))
        for field in _Run._fields
    ]
    # Shown with `pytest -s`, all of it before a threshold fails: what `ATOL`
    # and `MIN_SPREAD` are to be calibrated with.
    print(f"attention on {ATTENTION_PATH}")
    for field, spread in spreads.items():
        print(f"reference {field}: spread = {spread:.4f}")
    for name, actual, expected in pairs:
        worst = (actual - expected).abs().max().item()
        print(f"{name}: max |{what}| = {worst:.4f}")
    for field, spread in spreads.items():
        # Not finite if the reference holds a NaN or an infinity.
        assert math.isfinite(spread) and spread > MIN_SPREAD, field
    for name, actual, expected in pairs:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=ATOL,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    ("prompts", "forced_tokens", "graphs", "decode_sizes"),
    [
        pytest.param(
            PROMPTS, FORCED_TOKENS, CHUNKED_GRAPHS, [8, 16], id="chunked-prefill"
        ),
        pytest.param(
            SHORT_PROMPTS, SHORT_FORCED_TOKENS, WIDE_GRAPHS, [512], id="512-row-graphs"
        ),
    ],
)
def test_bf16_experts_graphs_match_eager(
    tmp_path, prompts, forced_tokens, graphs, decode_sizes
):
    """One local and one full layer with bf16 experts load a checkpoint,
    prefill in chunks next to decodes and decode with the paged KV and
    conv-state caches. Full and piecewise CUDA graphs pad their batches and
    capture or replay the attention, so they have to match eager: in the
    prompt log-probabilities, which the prefill steps produce, and in those
    of the generated positions. Once the long prompt has prefilled, every
    step only decodes and replays the full graph of 8 requests, most of them
    padding.

    With 512-row graphs, six requests decode next to 506 padding rows. Past
    256 rows the short conv runs in 4-row tiles that skip the conv-state
    cache where no row can need it; the tile of the fifth and sixth request
    ends in padding rows and has to read it, as eager's 6-row batches do."""
    model = _write_checkpoint(tmp_path)
    with _engine(model, **graphs) as llm:
        # The request counts of the full graphs: a decode step is padded to
        # the smallest that holds it.
        compilation = llm.llm_engine.vllm_config.compilation_config
        assert compilation.decode_cudagraph_capture_request_sizes == decode_sizes
        graph = _generate(llm, prompts, forced_tokens)
    eager = _run(model, prompts, forced_tokens, enforce_eager=True, **graphs)
    _assert_close(graph, eager, "graph - eager")


def _experts_per_rank(model) -> int:
    return model.model.layers[0].mlp.experts.routed_experts.w13_weight.shape[0]


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize(
    "expert_parallel",
    [
        pytest.param(False, id="tp2", marks=multi_gpu_marks(num_gpus=2)),
        pytest.param(True, id="tp2-ep", marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_two_gpus_match_one(tmp_path, expert_parallel):
    """Two tensor-parallel ranks hold 4 query heads and 1 KV head each (their
    fused q/k/v/r weight then has alignment rows, which a single rank's has
    not) and all-reduce the attention and expert outputs. The experts are
    split along their intermediate dimension or, with expert parallelism,
    whole: 8 per rank. Either way the log-probabilities are one GPU's, up to
    the rounding of the partial sums (0.07 in an offline emulation, graph
    padding included)."""
    # Expert parallelism places experts 0..7 on the first rank: with the last
    # 8 experts taken, that rank would compute nothing. Take 12..15 and 0..3.
    model = _write_checkpoint(tmp_path, expert_shift=4)
    with _engine(
        model, tensor_parallel_size=2, enable_expert_parallel=expert_parallel
    ) as llm:
        experts = 8 if expert_parallel else 16
        assert llm.apply_model(_experts_per_rank) == [experts, experts]
        two_gpus = _generate(llm)
    one_gpu = _run(model, enforce_eager=True)
    _assert_close(two_gpus, one_gpu, "2 GPUs - 1 GPU")


def _device_tensor_ptrs(model) -> dict[str, int]:
    """Where the model's tensors are on the GPU: its parameters and buffers,
    and what is derived from them."""
    ptrs: dict[str, int] = {}
    for name, module in model.named_modules():
        tensors = {**vars(module), **module._parameters, **module._buffers}
        for attr, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                ptrs[f"{name}.{attr}"] = tensor.data_ptr()
    return ptrs


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("reload_mode", ["layerwise", "direct"])
def test_weight_update_reaches_captured_graphs(tmp_path, monkeypatch, reload_mode):
    """A weight update rewrites the weights of a running engine where they
    are and refills what is derived from them, so the CUDA graphs captured
    before it read the new weights at the addresses they recorded. The engine
    then has to return what an engine started on the new checkpoint returns
    (the two can batch the requests differently, hence `ATOL`)."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    old = _write_checkpoint(tmp_path / "old")
    new = _write_checkpoint(tmp_path / "new", seed=SEED + 1)
    with _engine(old) as llm:
        before = _generate(llm)
        (ptrs,) = llm.apply_model(_device_tensor_ptrs)
        llm.collective_rpc("reload_weights", kwargs={"weights_path": new})
        (new_ptrs,) = llm.apply_model(_device_tensor_ptrs)
        # The prefix cache still holds the old weights' KV: requests that ask
        # for prompt log-probabilities do not read it.
        updated = _generate(llm)
    fresh = _run(new)
    _assert_close(updated, fresh, "updated - fresh")
    # Not a no-op: the checkpoints are further apart than `ATOL`.
    for was, now in zip(before, updated, strict=True):
        assert (now.prompt_logprobs - was.prompt_logprobs).abs().max() > ATOL
    # In place: nothing a captured graph reads has moved.
    moved = [name for name in ptrs if new_ptrs.get(name) != ptrs[name]]
    assert ptrs and moved == []
