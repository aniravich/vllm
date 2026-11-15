# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic Inkling2 checkpoints with bf16 experts, for tests."""

import torch


def bf16_experts_metadata(num_layers: int) -> dict:
    """What a checkpoint's `hf_quant_config.json` holds when every layer's
    routed experts are bf16."""
    tensors = [
        f"model.llm.layers.{i}.mlp.experts.{name}"
        for i in range(num_layers)
        for name in ("w13_weight", "w2_weight")
    ]
    groups = [{"compute_dtype": "bfloat16", "tensors": tensors}]
    return {"quant_method": "inkling2", "model": {"parameter_groups": groups}}


def random_checkpoint(text_config: dict, seed: int) -> dict[str, torch.Tensor]:
    """Random tensors under the release's names, shapes and dtypes for the
    model `text_config` describes (bf16 experts: `w13_weight` rows interleave
    gate and up). The same tensors for the same arguments."""
    from vllm.models.inkling2.configs import Inkling2ModelConfig

    config = Inkling2ModelConfig(**text_config)
    gen = torch.Generator().manual_seed(seed)

    def bf16(*shape):
        return torch.randn(shape, generator=gen).bfloat16()

    def fp32(*shape):
        return torch.randn(shape, generator=gen)

    hidden, vocab = config.hidden_size, config.padded_vocab_size
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    q_size, kv_size = heads * config.head_dim, kv_heads * config.head_dim
    kernel, d_rel = config.sconv_kernel_size, config.d_rel
    experts, inter = config.n_routed_experts, config.intermediate_size
    ckpt = {
        "embed.weight": bf16(vocab, hidden),
        "unembed.weight": bf16(vocab, hidden),
        "embed_norm.weight": bf16(hidden),
        "norm.weight": bf16(hidden),
        "attn_res.output_query": fp32(hidden),
    }
    for i in range(config.num_hidden_layers):
        layer = f"layers.{i}"
        local = i in config.local_layer_ids
        extent = config.sliding_window_size if local else config.rel_extent
        if i > 0:
            shape = (hidden,) if local else (3, hidden)
            ckpt[f"attn_res.attn_queries.{i - 1}"] = fp32(*shape)
        ckpt[f"attn_res.mlp_queries.{i}"] = fp32(hidden)
        ckpt |= {
            f"{layer}.attn.wq_du.weight": bf16(q_size, hidden),
            f"{layer}.attn.wr_du.weight": bf16(heads * d_rel, hidden),
            f"{layer}.attn.gate_attn.weight": bf16(q_size, hidden),
            f"{layer}.attn.attn_sink_proj.weight": bf16(heads, hidden),
            f"{layer}.attn.wk_dv.weight": bf16(kv_size, hidden),
            f"{layer}.attn.wv_dv.weight": bf16(kv_size, hidden),
            f"{layer}.attn.wo_ud.weight": bf16(hidden, q_size),
            f"{layer}.attn.q_l2_clip.scale": bf16(heads, config.head_dim),
            f"{layer}.attn.rel_l2_clip.scale": bf16(heads, d_rel),
            f"{layer}.attn.rel_logits_proj.proj": bf16(d_rel, extent),
            f"{layer}.attn_sconv.weight": bf16(hidden, 1, kernel),
            f"{layer}.mlp_sconv.weight": bf16(hidden, 1, kernel),
            f"{layer}.mlp_norm.weight": bf16(hidden),
            f"{layer}.mlp.gate.weight": bf16(experts, hidden),
            f"{layer}.mlp.gate.bias": fp32(experts),
            f"{layer}.mlp.experts.w13_weight": bf16(experts, 2 * inter, hidden),
            f"{layer}.mlp.experts.w2_weight": bf16(experts, hidden, inter),
        }
        for conv in ("k_sconv", "v_sconv"):
            if local:
                ckpt[f"{layer}.attn.{conv}.weight"] = bf16(kv_size, 1, kernel)
            else:
                ckpt[f"{layer}.attn.{conv}.static_w"] = bf16(kernel, kv_size)
                ckpt[f"{layer}.attn.{conv}.weight_proj"] = bf16(
                    kv_heads * kernel, hidden
                )
        if local:
            ckpt[f"{layer}.attn_norm.weight"] = bf16(hidden)
        else:
            for norm in ("q_input_norm", "k_input_norm", "v_input_norm"):
                ckpt[f"{layer}.{norm}.weight"] = bf16(hidden)
    return {f"model.llm.{name}": tensor for name, tensor in ckpt.items()}
