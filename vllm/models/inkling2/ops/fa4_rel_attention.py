# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative attention on standard FA4, for GPUs without the Blackwell
kernels.

tml_fa4 reads the relative bias sheared and only on SM100. Elsewhere standard
FA4 adds it through Inkling's score-mod, which reads the logits in their
natural layout `(T, H, extent)`: `unshear_rel_logits` recovers that layout
from what `attention_prologue` wrote.
"""

import torch

from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    _get_score_mod,
    bucket_max_seqlen_q,
)

# Query tile of the sheared layout (`ops.norm.qr_clip_rel_tokens`).
_TILE = 128


def diagonal_offsets(
    seq_idx: torch.Tensor, cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor
) -> torch.Tensor:
    """Per token, the number of keys that follow its own in the key blocks of
    its query tile (`[T]` int64 in `[0, 254]`). It places the token's
    sheared relative logits (`ops.norm.qr_clip_rel_tokens`): of their
    `extent + 256` columns, column `extent + 255 - offset - d` holds
    distance `d`.

    `seq_idx` `[T]` maps tokens to requests; `cu_seqlens_q` `[R + 1]` and
    `seq_lens` `[R]` are the requests' query offsets and KV lengths. Reads
    no value on the host, so it can run in a captured CUDA graph.
    """
    seq = seq_idx.long()
    token = torch.arange(seq.shape[0], device=seq.device)
    q_start = cu_seqlens_q.long()[seq]
    q_len = cu_seqlens_q.long()[seq + 1] - q_start
    q_idx = token - q_start
    past = seq_lens.long()[seq] - q_len
    tile_end = torch.minimum((q_idx // _TILE + 1) * _TILE, q_len)
    key_blocks = (past + tile_end + _TILE - 1) // _TILE
    # CUDA graph padding rows (request 0, past its queries) land outside the
    # range; the clamp keeps their reads in bounds, as the kernel's does.
    return (key_blocks * _TILE - past - q_idx - 1).clamp(0, 2 * _TILE - 2)


def unshear_rel_logits(sheared: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Relative logits `(T, H, extent)` by distance, from the sheared
    `(>= T, H, extent + 256)` logits of `attention_prologue` and the
    tokens' `diagonal_offsets` `[T]`.

    The sheared padding (`-inf` right of the diagonal and past a window, 0
    past the extent of a full layer) is dropped: the attention masks and the
    score-mod's extent check stand in for it.
    """
    num_tokens = offsets.shape[0]
    extent = sheared.shape[2] - 2 * _TILE
    # With the columns reversed, distances 0..extent-1 of a token are the
    # `extent` consecutive columns from its offset.
    windows = sheared[:num_tokens].flip(2).unfold(2, extent, 1)
    token = torch.arange(num_tokens, device=sheared.device)
    return windows[token, :, offsets]


def fa4_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window_size: tuple[int, int],
    out: torch.Tensor,
) -> torch.Tensor:
    """Paged varlen causal attention with the relative bias
    `rel_logits[i, h, i - j]` (0 beyond the extent), over all tokens of the
    batch (prefill and decode alike).

    Writes bf16 `out` `(T, H, D)` and returns the fp32 natural-log LSE
    `(H, T)` of the biased, scaled scores; rows past `cu_seqlens_q[-1]` are
    left unwritten in both. `q` is bf16 `(T, H, D)`, `rel_logits` bf16
    `(T, H, extent)`; the caches are `(num_blocks, block_size, H_kv, D)`.
    """
    from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

    _, lse = flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=bucket_max_seqlen_q(max_query_len),
        page_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
        # cute uses (None, None) to mean "no window".
        window_size=(None, None) if window_size == (-1, -1) else window_size,
        num_splits=1,
        pack_gqa=False,
        score_mod=_get_score_mod(rel_logits.shape[2]),
        aux_tensors=[rel_logits.contiguous()],
        return_lse=True,
        out=out,
    )
    return lse
