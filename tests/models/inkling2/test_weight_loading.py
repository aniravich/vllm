# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 checkpoint loading and in-place weight updates, on CPU.

A tiny model with bf16 experts is built as one rank of a parallel layout and
loaded from a synthetic checkpoint in the release's names and shapes: every
parameter is compared with the slice of the checkpoint its rank owns. Weight
updates then go through vLLM's reload functions, in both reload modes: they
have to land in the same storage, survive a retry after an interrupted
update, and leave untouched what they are not sent.
"""

import random
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from tests.models.inkling2.checkpoint_utils import (
    bf16_experts_metadata,
    random_checkpoint,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    finish_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.platforms import current_platform

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
# The model is built on the CPU wherever the tests run.
CPU = torch.device("cpu")


BF16_EXPERTS = bf16_experts_metadata(LAYERS)


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    return random_checkpoint(TEXT_CONFIG, seed)


def _expected(
    ckpt: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
    dp_size: int = 1,
    dp_rank: int = 0,
    expert_parallel: bool = False,
) -> dict[str, torch.Tensor]:
    """The parameters of one rank, derived from the checkpoint layout. A
    tensor-parallel rank holds every expert, sharded along the intermediate
    dimension. With expert parallelism the experts are spread over the
    `tp_size * dp_size` ranks instead: rank `dp_rank * tp_size + tp_rank`
    holds its consecutive share of them, whole."""
    # Without expert parallelism FusedMoE shards the experts over the ranks
    # of every replica (`dp_size * tp_size`), which is not modelled here.
    assert expert_parallel or dp_size == 1

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
        gate, up = w13[:, 0::2], w13[:, 1::2]
        if expert_parallel:
            share = EXPERTS // (tp_size * dp_size)
            first = (dp_rank * tp_size + tp_rank) * share
            gate, up, w2 = (t[first : first + share] for t in (gate, up, w2))
        else:
            gate, up, w2 = shard(gate, 1), shard(up, 1), shard(w2, 2)
        experts = f"{layer}.mlp.experts.routed_experts"
        expected[f"{experts}.w13_weight"] = torch.cat([gate, up], dim=1)
        expected[f"{experts}.w2_weight"] = w2
    return expected


class Layout(NamedTuple):
    """The rank a model is built as: `tp_rank` of a tensor-parallel group in
    data-parallel replica `dp_rank`, with or without expert parallelism."""

    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    expert_parallel: bool = False


LAYOUTS = {
    "tp1": Layout(),
    "tp8-rank0": Layout(8, 0),
    "tp8-rank5": Layout(8, 5),
    "tp8-ep-rank5": Layout(8, 5, expert_parallel=True),
    "dp2-tp8-ep-rank13": Layout(8, 5, 2, 1, expert_parallel=True),
}


@pytest.fixture(params=list(LAYOUTS.values()), ids=list(LAYOUTS))
def layout(request):
    return request.param


@pytest.fixture
def vllm_config():
    """The configuration of a tiny text-only Inkling2 with bf16 experts."""
    from vllm.config import VllmConfig
    from vllm.models.inkling2.configs import Inkling2MMConfig
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    config = VllmConfig()
    # Stands in for a `ModelConfig`, which needs a checkpoint directory and
    # imports the model class in a subprocess: the fields that building,
    # loading and updating the model read.
    config.model_config = SimpleNamespace(
        hf_config=Inkling2MMConfig(text_config=TEXT_CONFIG),
        dtype=torch.bfloat16,
        head_dtype=torch.float32,
        enable_prompt_embeds=False,
        multimodal_config=None,
        quantization="inkling2",
        word_embeddings_untied_by_checkpoint=False,
    )
    config.quant_config = Inkling2QuantConfig(BF16_EXPERTS)
    return config


@pytest.fixture
def gluon():
    """Skips where the model cannot be imported: its attention builds Gluon
    kernels at import, and vLLM's stand-ins for a missing or disabled Triton
    have no Gluon."""
    from vllm.triton_utils import gl

    if not hasattr(gl, "nvidia"):
        pytest.skip("Inkling2 attention needs Triton's Gluon")


@pytest.fixture
def build_model(gluon, dist_init, monkeypatch, vllm_config):
    """Returns a function that builds the model `vllm_config` describes as
    one rank of a parallel layout, the way vLLM's loader builds it."""
    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.model_loader.reload.lifecycle as lifecycle
    import vllm.models.inkling2.bf16_moe as bf16_moe
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.models.inkling2.model import Inkling2ForConditionalGeneration
    from vllm.utils.torch_utils import set_default_torch_dtype

    # The model is refused by the host's GPU family, whatever it is built on.
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major not in (9, 10):
        pytest.skip("Inkling2 is refused on this GPU family")
    # FusedMoE's CPU experts have no SiTU and repack their weights after
    # loading. Loading does not depend on the activation, and the Triton
    # experts Inkling2 runs on keep the checkpoint's layout.
    monkeypatch.setattr(
        bf16_moe, "expert_activation", lambda gate_cap, up_cap: {"activation": "silu"}
    )
    monkeypatch.setattr(
        UnquantizedFusedMoEMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )
    # A test that fails in the middle of an update must not leave the
    # process-wide flag set for the tests after it.
    monkeypatch.setattr(lifecycle, "_weight_update_active", False)

    def build(layout: Layout = LAYOUTS["tp1"]) -> Inkling2ForConditionalGeneration:
        # One process cannot be a rank of a larger group: the layers read
        # their rank and the group's size from these.
        groups = {
            "_TP": (layout.tp_size, layout.tp_rank),
            "_DP": (layout.dp_size, layout.dp_rank),
        }
        for group, (size, rank) in groups.items():
            if size > 1:
                stub = SimpleNamespace(
                    world_size=size, rank_in_group=rank, cpu_group=None
                )
                monkeypatch.setattr(parallel_state, group, stub)
        vllm_config.parallel_config.enable_expert_parallel = layout.expert_parallel
        with (
            set_default_torch_dtype(torch.bfloat16),
            set_current_vllm_config(vllm_config),
        ):
            model = Inkling2ForConditionalGeneration(vllm_config=vllm_config)  # type: ignore[abstract]
            # As vLLM's loader does for every model it builds.
            record_metadata_for_reloading(model)
        return model

    return build


@pytest.fixture
def model(build_model, layout):
    """A tiny Inkling2 with bf16 experts, built as the rank `layout`."""
    return build_model(layout)


def _assert_parameters(model, ckpt, layout):
    """Every parameter of `model` holds its rank's share of `ckpt`."""
    expected = _expected(ckpt, *layout)
    params = dict(model.named_parameters())
    # The vocabulary shards of embed / unembed are VocabParallelEmbedding's.
    vocab = {"model.embed_tokens.weight": "embed", "lm_head.weight": "unembed"}
    assert params.keys() - vocab.keys() == expected.keys()
    for name, tensor in expected.items():
        assert params[name].dtype == tensor.dtype, name
        assert torch.equal(params[name], tensor), name
    if layout.tp_size == 1:
        for name, ckpt_name in vocab.items():
            weight = ckpt[f"model.llm.{ckpt_name}.weight"]
            assert torch.equal(params[name][:VOCAB], weight), name
            assert params[name][VOCAB:].count_nonzero() == 0, name


def test_load_weights_places_every_checkpoint_tensor(model, layout):
    ckpt = _checkpoint(0)

    loaded = model.load_weights(ckpt.items())

    assert loaded == {name for name, _ in model.named_parameters()}
    _assert_parameters(model, ckpt, layout)


def _cold_start(model, model_config):
    """Load checkpoint 0 and process it, as vLLM's loader does. Returns what
    the kernels of captured CUDA graphs go on reading after an update: the
    storage of every parameter, and each attention layer's table derived
    from `rel_logits_proj`."""
    model.load_weights(_checkpoint(0).items())
    process_weights_after_loading(model, model_config, CPU)
    addresses = {name: p.data_ptr() for name, p in model.named_parameters()}
    tables = [layer.attn.rel_proj_shifted for layer in model.model.layers]
    return addresses, tables


def _assert_updated(model, ckpt, layout, kept):
    """`model` holds `ckpt` where `_cold_start` found its weights."""
    from vllm.models.inkling2.ops.sconv import shift_rel_proj

    addresses, tables = kept
    _assert_parameters(model, ckpt, layout)
    assert addresses == {name: p.data_ptr() for name, p in model.named_parameters()}
    for layer, table in zip(model.model.layers, tables):
        attn = layer.attn
        assert attn.rel_proj_shifted is table, attn.prefix
        fresh = shift_rel_proj(attn.rel_logits_proj, attn.is_local)
        assert torch.equal(table, fresh), attn.prefix


def _update_per_tensor(model, model_config, weights) -> set[str]:
    """A weight update as vLLM's weight-transfer engines deliver it: one
    `load_weights` call per tensor between `start_reload` and
    `finish_reload`. Returns the parameters the calls reported."""
    loaded: set[str] = set()
    start_reload(model)
    for item in weights:
        loaded |= model.load_weights([item])
    finish_reload(model, model_config, CPU)
    return loaded


def _update_one_call(model, model_config, weights) -> set[str]:
    """A weight update through the low-level layerwise API
    (`docs/training/layerwise.md`): the whole stream to one `load_weights`
    call, with `weight_update_active()` false. Returns the parameters the
    call reported."""
    initialize_layerwise_reload(model)
    loaded = model.load_weights(weights)
    finalize_layerwise_reload(model, model_config)
    return loaded


@pytest.mark.parametrize(
    ("mode", "update"),
    [
        ("layerwise", _update_per_tensor),
        ("direct", _update_per_tensor),
        ("layerwise", _update_one_call),
    ],
    ids=["layerwise", "direct", "layerwise-one-call"],
)
def test_weight_update_rewrites_the_same_storage(
    model, vllm_config, layout, monkeypatch, mode, update
):
    """An update delivers the checkpoint's tensors in any order. A layerwise
    reload buffers them on meta parameters and replays them into each module
    once it is complete; a direct reload lets the loaders write in place.
    Either way the weights end in the storage they had, and the table derived
    from `rel_logits_proj` is refilled in place."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    kept = _cold_start(model, vllm_config.model_config)
    ckpt = _checkpoint(1)
    weights = list(ckpt.items())
    random.Random(0).shuffle(weights)

    loaded = update(model, vllm_config.model_config, weights)

    assert loaded == {name for name, _ in model.named_parameters()}
    _assert_updated(model, ckpt, layout, kept)


def _interrupted(ckpt, last):
    """The tensors of `ckpt` up to `last`, then the trainer is gone."""
    for name, tensor in ckpt.items():
        yield name, tensor
        if name == last:
            raise ConnectionError("the trainer went away")


@pytest.mark.parametrize(
    ("update", "last"),
    [
        (_update_per_tensor, "model.llm.layers.1.mlp.experts.w13_weight"),
        (_update_one_call, "model.llm.layers.1.mlp.experts.w13_weight"),
        (_update_one_call, "model.llm.layers.2.attn.gate_attn.weight"),
    ],
    ids=["per-tensor-after-w13", "one-call-after-w13", "one-call-inside-qkvr"],
)
def test_complete_update_repairs_an_interrupted_one(
    model, vllm_config, layout, monkeypatch, update, last
):
    """Nothing ends the layerwise reload of an update that dies, so the next
    update finds the dead one's tensors buffered. A module that had received
    part of its tensors is then complete, and gets its parameters back, in
    the middle of the retry: after some of the experts of a stacked tensor,
    or some of the shards of a fused weight."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    kept = _cold_start(model, vllm_config.model_config)
    with pytest.raises(ConnectionError):
        update(model, vllm_config.model_config, _interrupted(_checkpoint(1), last))
    ckpt = _checkpoint(2)

    update(model, vllm_config.model_config, ckpt.items())

    _assert_updated(model, ckpt, layout, kept)


# Tensors an update may come without: one of several of their module (a
# pseudo-query, a shard of a fused attention weight, one of the two expert
# tensors), or every tensor of their module (a gate, its experts, a fused
# attention weight).
PART_OF_MODULES = [
    "model.llm.attn_res.mlp_queries.1",
    "model.llm.layers.1.attn.wk_dv.weight",
    "model.llm.layers.1.mlp.experts.w2_weight",
]
WHOLE_MODULES = [
    "model.llm.layers.1.mlp.gate.weight",
    "model.llm.layers.1.mlp.gate.bias",
    "model.llm.layers.1.mlp.experts.w13_weight",
    "model.llm.layers.1.mlp.experts.w2_weight",
    *(
        f"model.llm.layers.2.attn.{name}.weight"
        for name in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")
    ),
]


def _without(ckpt, names):
    """`ckpt` without the tensors `names`, all of which it holds."""
    assert set(names) <= ckpt.keys()
    return {name: tensor for name, tensor in ckpt.items() if name not in names}


@pytest.mark.parametrize(
    ("mode", "omitted"),
    [
        ("layerwise", WHOLE_MODULES),
        ("direct", WHOLE_MODULES),
        ("direct", PART_OF_MODULES),
    ],
    ids=["layerwise-whole-modules", "direct-whole-modules", "direct-part-of-modules"],
)
def test_update_keeps_the_tensors_it_is_not_sent(
    model, vllm_config, layout, monkeypatch, mode, omitted
):
    """Modules an update leaves out keep their weights in both modes. Under
    a direct reload so do the tensors missing from a module the update does
    touch: its loaders write in place. (A layerwise reload rebuilds such a
    module from what it was sent.)"""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    kept = _cold_start(model, vllm_config.model_config)
    weights = _without(_checkpoint(1), omitted)

    _update_per_tensor(model, vllm_config.model_config, weights.items())

    _assert_updated(model, _checkpoint(0) | weights, layout, kept)


def test_partial_checkpoint_is_rejected_outside_an_update(build_model):
    model = build_model()
    ckpt = _checkpoint(0)
    del ckpt["model.llm.layers.1.attn.wk_dv.weight"]

    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        model.load_weights(ckpt.items())


def test_quantized_expert_tensors_are_rejected_by_bf16_experts(build_model):
    model = build_model()
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


@pytest.fixture
def off_blackwell():
    """Skips on the GPUs that run Inkling2's SM100 kernels."""
    from vllm.models.inkling2.ops.device import is_blackwell

    if is_blackwell():
        pytest.skip("Blackwell runs the SM100 kernels")


@pytest.mark.skip_global_cleanup
def test_quantized_experts_need_blackwell(off_blackwell):
    from vllm.models.inkling2.moe import Inkling2Experts

    with pytest.raises(ValueError, match="Blackwell"):
        Inkling2Experts(SimpleNamespace(), "nvfp4")


# A tiny vision tower, in the fields of the release's `vision_config`.
VISION_CONFIG = dict(
    temporal_patch_size=2,
    native_patch_size=20,
    adapter_merge_size=2,
    projection_hidden_dim=256,
    projector_bias=False,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    mlp_ratio=4.0,
    posemb_grid_size=32,
    layer_norm_eps=1e-5,
)


def test_vision_tower_needs_blackwell(off_blackwell, build_model, vllm_config):
    """The tower's attention only has an SM100 kernel, and the tower is built
    unless images are limited to 0."""
    from vllm.config import MultiModalConfig

    model_config = vllm_config.model_config
    model_config.hf_config.vision_config = VISION_CONFIG
    model_config.multimodal_config = MultiModalConfig()

    with pytest.raises(ValueError, match="vision tower needs a Blackwell GPU"):
        build_model()


def test_text_only_model_is_built_without_the_vision_tower(build_model, vllm_config):
    from vllm.config import MultiModalConfig

    model_config = vllm_config.model_config
    model_config.hf_config.vision_config = VISION_CONFIG
    model_config.multimodal_config = MultiModalConfig(limit_per_prompt={"image": 0})

    assert build_model().visual is None


def test_layerwise_update_refuses_vision_tensors(build_model, vllm_config, monkeypatch):
    """The tower's loader writes to its parameters, which a layerwise reload
    has replaced by meta ones, all but the biases: the update is refused
    instead of leaving new biases on old weights."""
    import vllm.models.inkling2.model as model_module
    from vllm.config import MultiModalConfig

    # The tower is only built on Blackwell.
    monkeypatch.setattr(model_module, "is_blackwell", lambda: True)
    model_config = vllm_config.model_config
    model_config.hf_config.vision_config = VISION_CONFIG
    model_config.multimodal_config = MultiModalConfig()
    model = build_model()
    initialize_layerwise_reload(model)

    with pytest.raises(ValueError, match="cannot be updated by a layerwise reload"):
        model.load_weights([("model.visual.pooler.weight", torch.zeros(1))])


def test_dspark_draft_needs_blackwell(off_blackwell, build_model, vllm_config):
    # Stands in for a `SpeculativeConfig`, which resolves the DSpark draft
    # from a `ModelConfig`: the fields the target model reads.
    vllm_config.speculative_config = SimpleNamespace(
        method="dspark", num_speculative_tokens=7, max_num_new_slots_for_drafting=0
    )

    with pytest.raises(ValueError, match="DSpark draft needs a Blackwell GPU"):
        build_model()


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
def test_other_gpu_families_are_refused(
    off_blackwell, build_model, monkeypatch, capability
):
    """Off Blackwell only Hopper is served: the other families have no
    attention path."""
    import vllm.models.inkling2.model as model_module
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(
        model_module.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(*capability),
    )

    with pytest.raises(ValueError, match="this GPU has compute capability"):
        build_model()
