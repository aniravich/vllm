# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative attention on standard FA4 (GPUs without the Blackwell
kernels): the un-shear of the prologue's relative logits, the handling of
CUDA graph padding rows, and the attention against the fp32 reference."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.inkling2.test_attention import (
    HEAD_DIM,
    REL_EXTENT,
    SCENARIOS,
    _ref_attention,
    shear_rel_logits,
)
from vllm.models.inkling2.ops.fa4_rel_attention import (
    diagonal_offsets,
    fa4_rel_attention,
    unshear_rel_logits,
)
from vllm.platforms import current_platform
from vllm.triton_utils import gl

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
# Standard FA4 takes a score-mod with paged KV on SM90, SM100 and SM110 only.
requires_sm90 = pytest.mark.skipif(
    _cap is None or _cap.major not in (9, 10, 11),
    reason="FA4 score-mod with paged KV requires SM90, SM100 or SM110",
)
# `vllm.models.inkling2.attention` builds its Gluon kernels at import, which
# vLLM's Triton placeholders (a host without an active GPU driver) cannot.
requires_gluon = pytest.mark.skipif(
    not hasattr(gl, "nvidia"), reason="Inkling2 attention needs Triton's Gluon"
)

UNSHEAR_SCENARIOS = {
    **SCENARIOS,
    "mixed": [(1, 4000), (1, 1), (129, 129), (128, 128), (127, 4127), (256, 1280)],
    # Query tiles and key blocks are 128 wide. The first row of (128, 129) has
    # the largest offset of a real token, 254.
    "boundaries": [
        (1, 128),
        (1, 129),
        (1, 127),
        (128, 129),
        (128, 256),
        (129, 257),
        (255, 255),
    ],
}
FA4_SCENARIOS = {
    **SCENARIOS,
    # FA4 schedules batches of more than 256 requests with another kernel variant.
    "many_decodes": [(1, 1 + 37 * i % 2000) for i in range(300)],
}


def _request_metadata(seqs, device="cpu", num_padding_tokens=0, num_padding_reqs=0):
    """`(seq_idx, cu_seqlens_q, seq_lens)` of a batch. CUDA graph padding
    adds tokens that map to request 0 and empty requests of KV length 0."""
    q_lens = torch.tensor([q_len for q_len, _ in seqs])
    num_tokens = int(q_lens.sum())
    cu_seqlens_q = torch.tensor(
        [0, *q_lens.cumsum(0).tolist(), *[num_tokens] * num_padding_reqs],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(
        [*[kv_len for _, kv_len in seqs], *[0] * num_padding_reqs], dtype=torch.int32
    )
    seq_idx = torch.cat(
        [
            torch.repeat_interleave(torch.arange(len(seqs)), q_lens),
            torch.zeros(num_padding_tokens, dtype=torch.long),
        ]
    ).int()
    return seq_idx.to(device), cu_seqlens_q.to(device), seq_lens.to(device)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("scenario", list(UNSHEAR_SCENARIOS))
@pytest.mark.parametrize("is_local", [False, True])
# A local layer's extent is its window, which a config can shorten.
@pytest.mark.parametrize("extent", [REL_EXTENT, 128])
def test_unshear_rel_logits(scenario, is_local, extent):
    """Un-shearing the reference of what `attention_prologue` writes gives
    back the logits by distance, bit for bit."""
    torch.manual_seed(0)
    seqs = UNSHEAR_SCENARIOS[scenario]
    num_tokens = sum(q_len for q_len, _ in seqs)
    rel_logits = torch.randn(num_tokens, 3, extent, dtype=torch.bfloat16)
    sheared = shear_rel_logits(rel_logits, seqs, 128, is_local)
    offsets = diagonal_offsets(*_request_metadata(seqs))

    out = unshear_rel_logits(sheared, offsets)

    assert out.dtype == rel_logits.dtype
    assert torch.equal(out, rel_logits)
    # Distance 0 of a real token sits in columns [extent + 1, extent + 255].
    assert int(offsets.min()) >= 0
    assert int(offsets.max()) <= 254


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_unshear_reads_padding_rows_in_bounds():
    """A full CUDA graph un-shears its padding rows too: they map to request
    0 past its queries, where the prologue clamps the diagonal. Without the
    same clamp their offsets go negative, and in a large graph out of the
    un-shear's range."""
    torch.manual_seed(0)
    # 3 decodes replayed in a 512-request graph: enough padding rows for an
    # unclamped offset to leave the un-shear's range. Request 0 fills its last
    # key block, so the diagonal of every padding row is past the last column.
    seqs = [(1, 128), (1, 3000), (1, 1)]
    num_tokens, num_padding = len(seqs), 509
    seq_idx, cu_seqlens_q, seq_lens = _request_metadata(
        seqs, num_padding_tokens=num_padding, num_padding_reqs=num_padding
    )
    rel_logits = torch.randn(num_tokens, 2, REL_EXTENT, dtype=torch.bfloat16)
    # The rows of the padding tokens hold whatever the prologue wrote.
    sheared = torch.full(
        (num_tokens + num_padding + 128, 2, REL_EXTENT + 256), 7.0, dtype=torch.bfloat16
    )
    sheared[:num_tokens] = shear_rel_logits(rel_logits, seqs, 128, True)[:num_tokens]

    offsets = diagonal_offsets(seq_idx, cu_seqlens_q, seq_lens)
    out = unshear_rel_logits(sheared, offsets)

    assert offsets.shape == (num_tokens + num_padding,)
    assert int(offsets.min()) >= 0
    assert int(offsets.max()) <= 254
    # The prologue clamps the diagonal of these rows to its last column.
    assert offsets[num_tokens:].count_nonzero() == 0
    assert out.shape == (num_tokens + num_padding, 2, REL_EXTENT)
    assert torch.equal(out[:num_tokens], rel_logits)
    assert torch.equal(out[num_tokens:], torch.full_like(out[num_tokens:], 7.0))


def _dense_fa4(
    q,
    key_cache,
    value_cache,
    rel_logits,
    *,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    max_query_len,
    softmax_scale,
    window_size,
    out,
    pack_gqa=False,
):
    """CPU stand-in for `fa4_rel_attention`: the fp32 reference on the
    logits by distance. Like FA4 it leaves rows past `cu_seqlens_q[-1]`
    unwritten in `out` and uninitialized (NaN here) in the LSE."""
    q_lens = cu_seqlens_q.diff().tolist()
    seqs = [(q_len, int(kv_len)) for q_len, kv_len in zip(q_lens, cache_seqlens)]
    seqs = [seq for seq in seqs if seq[0] > 0]
    assert max(q_len for q_len, _ in seqs) <= max_query_len
    total = int(cu_seqlens_q[-1])
    ref_out, ref_lse = _ref_attention(
        q[:total],
        key_cache,
        value_cache,
        rel_logits[:total],
        seqs,
        block_table,
        softmax_scale,
        None if window_size == (-1, -1) else window_size[0],
    )
    out[:total] = ref_out.to(out.dtype)
    lse = torch.full((q.shape[1], q.shape[0]), float("nan"))
    lse[:, :total] = ref_lse.t()
    return lse


@requires_gluon
@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("graph", ["none", "piecewise", "full"])
@pytest.mark.parametrize("window_left", [None, REL_EXTENT - 1])
def test_score_mod_attention_zeroes_padding_rows(monkeypatch, graph, window_left):
    """`Inkling2Attention._score_mod_attention` on a batch with CUDA graph
    padding: piecewise graphs pad the tokens after `num_actual_tokens`; a
    full graph counts its padding in `num_actual_tokens` and pads the
    requests."""
    import vllm.models.inkling2.attention as attention_module
    from vllm.models.inkling2.attention import Inkling2Attention

    torch.manual_seed(0)
    monkeypatch.setattr(attention_module, "fa4_rel_attention", _dense_fa4)
    num_heads, num_kv_heads, block_size = 6, 1, 16
    # Request 0 fills its last key block, so the padding rows of a full graph
    # (request 0, past its query) need the clamp of `diagonal_offsets`.
    seqs = [(1, 128), (1, 3000), (1, 1), (1, 1025)]
    num_tokens = len(seqs)
    padding = 0 if graph == "none" else 4
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    key_cache = torch.nn.functional.normalize(
        torch.randn(num_blocks, block_size, num_kv_heads, HEAD_DIM), dim=-1
    )
    key_cache = (key_cache * 2).bfloat16()
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, HEAD_DIM).bfloat16()
    block_table = (torch.randperm(num_blocks - 1) + 1)[: len(seqs) * max_blocks]
    block_table = block_table.view(len(seqs), max_blocks).int()
    q = torch.randn(num_tokens + padding, num_heads, HEAD_DIM)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(num_tokens, num_heads, REL_EXTENT, dtype=torch.bfloat16)
    sheared = torch.zeros(
        num_tokens + padding + 128, num_heads, REL_EXTENT + 256, dtype=torch.bfloat16
    )
    sheared[:num_tokens] = shear_rel_logits(
        rel_logits, seqs, 128, window_left is not None
    )[:num_tokens]
    padding_reqs = padding if graph == "full" else 0
    seq_idx, cu_seqlens_q, seq_lens = _request_metadata(
        seqs, num_padding_tokens=padding, num_padding_reqs=padding_reqs
    )
    block_table = torch.cat(
        [block_table, torch.zeros(padding_reqs, max_blocks, dtype=torch.int32)]
    )
    md = attention_module.Inkling2AttentionMetadata(
        num_actual_tokens=num_tokens + (padding if graph == "full" else 0),
        max_query_len=1,
        query_start_loc=cu_seqlens_q,
        max_seq_len=max(kv_len for _, kv_len in seqs),
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.empty(0, dtype=torch.long),
    )
    layer = SimpleNamespace(
        num_heads=num_heads,
        scaling=HEAD_DIM**-0.5,
        window_size=(-1, -1) if window_left is None else (window_left, 0),
        _split_kv_cache=lambda: (key_cache, value_cache),
    )

    out, lse = Inkling2Attention._score_mod_attention(
        layer, md, SimpleNamespace(seq_idx=seq_idx), q, sheared
    )

    ref_out, ref_lse = _ref_attention(
        q[:num_tokens],
        key_cache,
        value_cache,
        rel_logits,
        seqs,
        block_table,
        HEAD_DIM**-0.5,
        window_left,
    )
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert lse.shape == (num_heads, q.shape[0]) and lse.dtype == torch.float32
    torch.testing.assert_close(out[:num_tokens].float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse.t()[:num_tokens], ref_lse)
    # Padding rows come out as zeros (the gate/sink then writes zeros).
    assert out[num_tokens:].count_nonzero() == 0
    assert lse[:, num_tokens:].count_nonzero() == 0
    # The offsets are kept for the other layers that share the metadata.
    assert md.rel_offsets is not None
    assert md.rel_offsets.shape == (md.num_actual_tokens,)
    # A full graph's padding rows take the prologue's clamp: a negative
    # offset would wrap in the un-shear without a visible effect here.
    assert md.rel_offsets[num_tokens:].count_nonzero() == 0
    kept = md.rel_offsets
    Inkling2Attention._score_mod_attention(
        layer, md, SimpleNamespace(seq_idx=None), q, sheared
    )
    assert md.rel_offsets is kept


@requires_sm90
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left,scenario",
    [
        # At TP 8 a rank has 6 query heads over 1 KV head, and the conv-state
        # page makes the attention blocks 256 tokens.
        (6, 1, 256, REL_EXTENT - 1, "prefill"),
        (6, 1, 256, None, "prefill"),
        (6, 1, 256, REL_EXTENT - 1, "chunked_prefill"),
        (6, 1, 256, None, "chunked_prefill"),
        (6, 1, 256, REL_EXTENT - 1, "decode"),
        (6, 1, 256, None, "decode"),
        (6, 1, 256, REL_EXTENT - 1, "many_decodes"),
        (6, 1, 256, None, "many_decodes"),
        (12, 2, 128, REL_EXTENT - 1, "prefill"),
        (12, 2, 64, None, "decode"),
        (48, 8, 16, None, "prefill"),
        (48, 8, 16, REL_EXTENT - 1, "chunked_prefill"),
        (48, 8, 128, REL_EXTENT - 1, "decode"),
    ],
)
@torch.inference_mode()
def test_fa4_rel_attention(
    num_heads, num_kv_heads, block_size, window_left, scenario, pack_gqa
):
    """Prologue-layout logits -> un-shear -> standard FA4 with the score-mod
    matches the fp32 reference, output and LSE."""
    torch.manual_seed(0)
    device = "cuda"
    seqs = FA4_SCENARIOS[scenario]
    q_lens = [q_len for q_len, _ in seqs]
    num_tokens = sum(q_lens)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1

    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    kv = kv.bfloat16().transpose(1, 2)
    key_cache, value_cache = kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    seq_idx, cu_seqlens_q, seq_lens = _request_metadata(seqs, device)
    scale = HEAD_DIM**-0.5
    sheared = shear_rel_logits(rel_logits, seqs, 128, window_left is not None)

    out = torch.empty_like(q)
    lse = fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        unshear_rel_logits(sheared, diagonal_offsets(seq_idx, cu_seqlens_q, seq_lens)),
        block_table=block_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max(q_lens),
        softmax_scale=scale,
        window_size=(-1, -1) if window_left is None else (window_left, 0),
        out=out,
        pack_gqa=pack_gqa,
    )

    ref_out, ref_lse = _ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    assert lse.shape == (num_heads, num_tokens) and lse.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.t(), ref_lse, atol=1e-3, rtol=1e-4)


@requires_sm90
@pytest.mark.parametrize("block_size", [16, 64, 128, 256])
@torch.inference_mode()
def test_fa4_rel_attention_window_over_freed_blocks(block_size):
    """A window layer's blocks before the window are freed: their page-table
    entries point at the null block, whose rows the kernel may load but the
    window masks."""
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, window_left = 6, 1, REL_EXTENT - 1
    seqs = [(1, 3000), (1, 1025), (1, 5001)]
    num_tokens = len(seqs)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    # The null block is zero-filled, as vLLM keeps it.
    kv[0] = 0
    key_cache, value_cache = kv.bfloat16().split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    freed_table = block_table.clone()
    for i, (_, kv_len) in enumerate(seqs):
        # Blocks wholly before the first key of the window.
        freed_table[i, : max(kv_len - 1 - window_left, 0) // block_size] = 0
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    seq_idx, cu_seqlens_q, seq_lens = _request_metadata(seqs, device)
    scale = HEAD_DIM**-0.5

    out = torch.empty_like(q)
    lse = fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        rel_logits,
        block_table=freed_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=1,
        softmax_scale=scale,
        window_size=(window_left, 0),
        out=out,
    )

    ref_out, ref_lse = _ref_attention(
        q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
    )
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.t(), ref_lse, atol=1e-3, rtol=1e-4)


@requires_sm90
@pytest.mark.parametrize("window_left", [None, REL_EXTENT - 1])
# FA4 finds a tile's request another way in batches of more than 256.
@pytest.mark.parametrize("scenario,graph_reqs", [("decode", 8), ("many_decodes", 512)])
@torch.inference_mode()
def test_fa4_rel_attention_with_full_graph_padding(scenario, graph_reqs, window_left):
    """What `_score_mod_attention` hands FA4 in a full CUDA graph: token rows
    after the batch's, and empty requests whose page-table rows are the null
    block. FA4 has to compute the real rows and leave the others of `out` as
    they are, which is where the zeros of the padding rows come from."""
    torch.manual_seed(0)
    device = "cuda"
    num_heads, num_kv_heads, block_size = 6, 1, 256
    seqs = FA4_SCENARIOS[scenario]
    num_tokens = len(seqs)
    num_padding = graph_reqs - num_tokens
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    key_cache, value_cache = kv.bfloat16().split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    padded_table = torch.cat(
        [block_table, block_table.new_zeros(num_padding, max_blocks)]
    )
    q = torch.randn(num_tokens + num_padding, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    sheared = rel_logits.new_zeros(
        num_tokens + num_padding + 128, num_heads, REL_EXTENT + 256
    )
    sheared[:num_tokens] = shear_rel_logits(
        rel_logits, seqs, 128, window_left is not None
    )[:num_tokens]
    seq_idx, cu_seqlens_q, seq_lens = _request_metadata(
        seqs, device, num_padding_tokens=num_padding, num_padding_reqs=num_padding
    )
    scale = HEAD_DIM**-0.5

    # A sentinel shows a write to the rows after the batch's tokens.
    out = torch.full_like(q, 7.0)
    lse = fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        unshear_rel_logits(sheared, diagonal_offsets(seq_idx, cu_seqlens_q, seq_lens)),
        block_table=padded_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=1,
        softmax_scale=scale,
        window_size=(-1, -1) if window_left is None else (window_left, 0),
        out=out,
    )

    ref_out, ref_lse = _ref_attention(
        q[:num_tokens],
        key_cache,
        value_cache,
        rel_logits,
        seqs,
        block_table,
        scale,
        window_left,
    )
    assert lse.shape == (num_heads, num_tokens + num_padding)
    torch.testing.assert_close(out[:num_tokens].float(), ref_out, atol=2e-2, rtol=2e-2)
    # FA4 leaves the LSE of the rows after the batch's tokens unwritten.
    torch.testing.assert_close(lse.t()[:num_tokens], ref_lse, atol=1e-3, rtol=1e-4)
    assert torch.equal(out[num_tokens:], torch.full_like(out[num_tokens:], 7.0))
