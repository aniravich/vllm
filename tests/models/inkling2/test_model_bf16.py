# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test of a tiny random-weight Inkling2 with bf16 experts
(TP=1), which also runs on GPUs without the Blackwell kernels: attention on
standard FA4 and the experts on FusedMoE."""

import json

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
if _cap is None or _cap.major < 9:
    pytest.skip("Inkling2 requires SM90 or newer", allow_module_level=True)

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
# The checkpoint's precision metadata lists the routed experts as bf16.
BF16_EXPERTS = {
    "quant_method": "inkling2",
    "model": {
        "parameter_groups": [
            {
                "compute_dtype": "bfloat16",
                "tensors": [
                    f"model.llm.layers.{i}.mlp.experts.{name}"
                    for i in range(NUM_LAYERS)
                    for name in ("w13_weight", "w2_weight")
                ],
            }
        ]
    },
}


def _write_checkpoint_configs(path) -> str:
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    (path / "hf_quant_config.json").write_text(json.dumps(BF16_EXPERTS))
    return str(path)


def _chunked_mixed_run(model: str, enforce_eager: bool) -> list[list[float]]:
    llm = LLM(
        model=model,
        load_format="dummy",
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
    # The short prompts decode while the long one prefills in 256-token chunks.
    prompts = [[7, 8, 9], [5, 5, 5], list(range(2, 702))]
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            prompt_logprobs=0,
        ),
    )
    del llm
    logprobs = []
    for o in outputs:
        assert len(o.outputs[0].token_ids) == 8
        assert all(0 <= t < 1000 for t in o.outputs[0].token_ids)
        assert o.prompt_logprobs is not None
        logprobs.append(
            [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:] if d]
        )
    return logprobs


def test_bf16_experts_graphs_match_eager(tmp_path):
    """One local and one full layer with bf16 experts load (dummy weights),
    prefill in chunks next to decodes and decode with the paged KV and
    conv-state caches. Full and piecewise CUDA graphs pad their batches and
    capture or replay the attention, so they have to match eager."""
    model = _write_checkpoint_configs(tmp_path)
    graph = _chunked_mixed_run(model, enforce_eager=False)
    eager = _chunked_mixed_run(model, enforce_eager=True)
    for g, e in zip(graph, eager):
        assert torch.isfinite(torch.tensor(e)).all()
        torch.testing.assert_close(torch.tensor(g), torch.tensor(e), rtol=0, atol=1e-2)
