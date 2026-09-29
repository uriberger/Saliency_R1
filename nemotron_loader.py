#!/usr/bin/env python
"""Loading NVIDIA's remote-code Nemotron VLMs, in the one place both sides can reach.

Two callers need the SAME five repairs and they must not drift apart:

  * the measuring side -- `sink_location_probe.load_model`, and everything downstream of
    it (the scan, the arms, `omni_train_step_bench.py`);
  * the TRAINING side -- `trl/grpo_trainer_qwen3.py`, which builds the policy itself with
    `getattr(transformers, config.architectures[0])`. That resolution is exactly right for
    a natively supported model and raises AttributeError for a `trust_remote_code` one,
    which is what every Nemotron VLM here is.

A second copy of these would be a second thing to fix when transformers drifts again, and
the failures they cover are the kind that surface 25 GB of weights into the load or
several frames away from the cause. So this module is the single definition and
`sink_location_probe` re-exports it; nothing here changed on the way across.

WHAT THE FIVE ARE, in the order a load meets them:

    _vendor_mamba_rmsnorm        the decoder raises at IMPORT without `rmsnorm_fn`
    _shim_tied_weights_keys      `all_tied_weights_keys` that 5.13 reads and the wrapper
                                 never sets -- NVIDIA's own fix, applied to the older repo
    _repair_radio_summary_idxs   an index buffer the checkpoints omit and the ViT indexes
    _shim_masking_api            `create_causal_mask`'s renamed / dropped arguments
    _shim_single_process_group   a world of one, so the 12B's forward can ask its rank
    _shim_cache_params_alias     `past_key_values` as an alias of `cache_params`
    _shim_cache_position         the `cache_position` 5.13 stopped passing to `generate`

(Seven functions, five checkpoints' worth of gaps; `docs/omni-training-blockers.md` counts
the ones a training run meets.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

#: Where `vendor/mamba_ssm_min` lives. `__file__` finds it when this module is imported
#: from the repo root, which is how the probes import it -- but `patch_trl_nemotron.sh`
#: also COPIES this file into `trl_repo_nemotron/trl/trainer/`, where walking up from
#: `__file__` lands in the TRL clone and finds nothing. So the search is: an explicit
#: SR1_REPO, then every parent of this file, then the repo path the launchers hardcode.
_REPO_ENV = "SR1_REPO"


def _repo_candidates():
    import os

    seen = []
    env = os.environ.get(_REPO_ENV)
    if env:
        seen.append(Path(env))
    here = Path(__file__).resolve()
    seen.extend(here.parents)
    seen.append(Path("/home/uberger/scratch/research/saliency_r1"))
    return seen


def is_remote_code(path):
    """Does this checkpoint carry its own modelling code? -> (bool, config).

    The config is handed back because loading it is the expensive half and every caller
    needs it immediately afterwards.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return bool(getattr(cfg, "auto_map", None)), cfg


def load_model(path, device, attn_impl="eager", quant=None, dtype=torch.bfloat16,
               place=True):
    """A remote-code Nemotron VLM, repaired. -> (processor, model).

    `attn_impl` is accepted and pinned to eager for the reason below; it is a parameter so
    a caller that has a working sdpa path one day does not have to reach inside.

    `place=False` leaves the model on the meta/CPU side and skips `.eval()`, for the
    trainer -- accelerate moves it, and a model that has been `.eval()`d before
    `get_peft_model` is one more thing to remember to undo.
    """
    from transformers import AutoModel, AutoProcessor

    remote, cfg = is_remote_code(path)
    if not remote:
        raise SystemExit(f"{path} is not a remote-code checkpoint; load it the normal way")
    _vendor_mamba_rmsnorm()
    # The attention implementation has to be set on EVERY SUB-CONFIG before construction.
    # The wrapper builds its language model as `NemotronHForCausalLM(config.llm_config)`,
    # passing the sub-config straight through, so an `attn_implementation=` argument to
    # the outer class never reaches it -- and `llm_config` ships with flash_attention_2
    # baked in, which is not installed here.
    #
    # Eager rather than sdpa: the wrapper declares no SDPA support and refuses it.
    impl = attn_impl
    for sub in ("llm_config", "text_config", "vision_config", "sound_config"):
        c = getattr(cfg, sub, None)
        if c is not None:
            c._attn_implementation = impl
    cfg._attn_implementation = impl
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    _shim_tied_weights_keys(cfg, path)
    kw = {}
    if quant is not None:
        # bitsandbytes swaps the Linear modules out DURING `from_pretrained` and places
        # them itself, so `device_map` replaces the `.to(device)` below -- calling both
        # raises. The vision tower is left alone by `llm_int8_skip_modules` at the call
        # site: it is ~600M of the 33B and it is the thing whose output geometry every
        # patch statistic is defined on, so shrinking it would confound the very
        # comparison this argument is for.
        kw = {"quantization_config": quant, "device_map": device}
    model = AutoModel.from_pretrained(path, config=cfg, dtype=dtype,
                                      trust_remote_code=True, **kw)
    apply_runtime_shims(model, cfg)
    if quant is not None or not place:
        return processor, (model.eval() if place else model)
    return processor, model.to(device).eval()


def load_any(path, adapter, device, attn_impl, native):
    """A checkpoint of either kind. -> (processor, model).

    THE DISPATCH, in one place. `overlap_probe.load_model` resolves the architecture as
    `getattr(transformers, config.architectures[0])`, which is exactly right for a
    natively supported model and raises AttributeError for a `trust_remote_code` one --
    and every Nemotron VLM here is the latter. So every probe that wants to run on both
    needs this four-line branch, and `sink_location_probe.load_model` was the first copy
    of it. A second and a third would be three things to fix the next time either side
    moves.

    `native` is passed in rather than imported because the native loader lives in
    `overlap_probe.py`, which pulls in the reward stack; this module is imported BY the
    trainer and must not acquire that dependency.
    """
    remote, _cfg = is_remote_code(path)
    if not remote:
        return native(path, adapter, device, attn_impl)
    if adapter:
        raise SystemExit(f"--adapter is not supported on the remote-code checkpoint {path}")
    # Eager, always: the wrapper declares no SDPA support and refuses it. It is also what
    # makes `NemotronHAttention.forward` hand back real softmax weights, which is the whole
    # of `Family.attention_weights_are_returned`.
    return load_model(path, device, attn_impl="eager")


def apply_runtime_shims(model, cfg=None):
    """The four repairs that only make sense on a CONSTRUCTED model.

    Split out because a caller that builds the model itself -- the trainer, through
    `from_pretrained` on the resolved architecture class -- still needs every one of them,
    and calling them in the wrong order is a crash several frames from the cause.
    """
    if cfg is None:
        cfg = model.config
    _repair_radio_summary_idxs(model, cfg)
    _shim_masking_api()
    _shim_single_process_group()
    _shim_cache_params_alias(model)
    _shim_cache_position(model)
    _shim_logits_to_keep(model)
    return model


def _shim_cache_position(model):
    """Hand `prepare_inputs_for_generation` the `cache_position` 5.13 stopped passing.

    The same API drift `_shim_masking_api` covers, one function further on. The Base
    repo's `prepare_inputs_for_generation` slices the new tokens out of `input_ids` with

        input_ids = input_ids[:, cache_position]

    and transformers 5.13 no longer supplies the argument -- it derives positions from
    `past_key_values` and `position_ids` instead -- so it arrives None and `generate`
    dies on the first step. `forward` is unaffected, which is why the measured pass got
    through and only the selftest's greedy decode did not.

    Derived, not guessed, and it reproduces the original contract exactly: the cache
    knows how many tokens it has already seen, `input_ids` carries everything so far, and
    the difference is what is new.

        prefill  past=0,  L new       -> arange(0, L)     -> the slice is a no-op
        decode   past=L,  1 new       -> arange(L, L+1)   -> selects the last token

    `get_seq_length()` reads the attention layers, which is the only part of a
    HybridMambaAttentionDynamicCache that has a sequence dimension at all.
    """
    import functools

    lm = getattr(model, "language_model", None)
    fn = getattr(lm, "prepare_inputs_for_generation", None)
    if fn is None or getattr(fn, "_sr1_cache_pos_shim", False):
        return

    # `wraps` is load-bearing, not tidiness. `generate` decides whether a model can take
    # `inputs_embeds` by INSPECTING THE SIGNATURE of this very method, and the wrapper is
    # `(*args, **kwargs)` -- which reads as "no such parameter" and makes generate refuse
    # the wrapper's own `inputs_embeds` call with "doesn't have its forwarding
    # implemented". `wraps` sets `__wrapped__`, which `inspect.signature` follows back to
    # the real parameter list.
    @functools.wraps(fn)
    def prepare_inputs_for_generation(*args, **kwargs):
        if kwargs.get("cache_position") is None:
            ids = kwargs.get("input_ids", args[0] if args else None)
            cache = kwargs.get("past_key_values")
            if ids is not None:
                past = cache.get_seq_length() if cache is not None else 0
                n_new = int(ids.shape[1]) - int(past)
                if n_new > 0:
                    kwargs["cache_position"] = torch.arange(
                        past, past + n_new, device=ids.device)
        return fn(*args, **kwargs)

    prepare_inputs_for_generation._sr1_cache_pos_shim = True
    lm.prepare_inputs_for_generation = prepare_inputs_for_generation


def _shim_cache_params_alias(model):
    """`past_key_values` as a read-only alias of `cache_params`, for the 12B's forward.

    `NVIDIA-Nemotron-Nano-12B-v2-VL`'s wrapper ends `forward` with

        return CausalLMOutputWithPast(..., past_key_values=outputs.past_key_values, ...)

    but its language model returns `NemotronHCausalLMOutput`, whose cache field is called
    `cache_params` -- a Mamba hybrid carries convolution and SSM state, not a KV cache.
    There is no such attribute, so every `forward()` on this checkpoint raises
    AttributeError after the whole model has run. Only `generate()` works as shipped,
    which is the one path the model card demonstrates; this module measures the forward.

    An alias, not a value: it renames the field the wrapper is reaching for and touches
    no arithmetic. `cache_params` keeps working and stays the only real dict key, so
    anything reading the output as a mapping sees exactly what it saw before.

    Patched on the class in the loaded module rather than in the file on disk, because
    the modules cache is re-downloaded whenever the repo changes.
    """
    lm = getattr(model, "language_model", None)
    mod = sys.modules.get(type(lm).__module__) if lm is not None else None
    out_cls = getattr(mod, "NemotronHCausalLMOutput", None) if mod is not None else None
    if out_cls is None or "past_key_values" in vars(out_cls):
        return
    if "cache_params" not in getattr(out_cls, "__dataclass_fields__", {}):
        return
    out_cls.past_key_values = property(lambda self: self.cache_params)


def _shim_tied_weights_keys(cfg, path):
    """The tied-weights bookkeeping the 12B's wrapper predates -- NVIDIA's own fix.

    transformers 5.13 finishes `from_pretrained` in `mark_tied_weights_as_initialized`,
    which reads `self.all_tied_weights_keys`. `PreTrainedModel` fills that in during
    `post_init()`, and this wrapper never calls it: it assembles a vision tower, an
    `mlp1` projector and a language model and ties nothing. So loading dies with
    AttributeError AFTER all 25 GB of weights are on the device.

    `{}` is not a guess. The Omni's copy of the same wrapper sets exactly
    `self.all_tied_weights_keys = {}` in its own `__init__` -- NVIDIA already fixed this
    in the newer of the two releases, and this applies their fix to the older one.

    Wrapping `__init__` rather than setting a class attribute, because the loader
    UPDATES and POPS this mapping: a class-level dict would be shared by every instance
    built in the process. `hasattr` first, so a repo that grows its own copy keeps it.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ref = (getattr(cfg, "auto_map", None) or {}).get("AutoModel")
    if not ref:
        return
    cls = get_class_from_dynamic_module(ref, path)
    if getattr(cls, "_sr1_tied_shim", False):
        return
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    cls.__init__ = __init__
    cls._sr1_tied_shim = True


def _vendor_mamba_rmsnorm():
    """Put `vendor/mamba_ssm_min` on the path, so the 12B can be imported at all.

    `NVIDIA-Nemotron-Nano-12B-v2-VL`'s decoder raises at IMPORT time without
    `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`, and every Mamba layer's
    `MambaRMSNormGated.forward` is a call to it -- so this is not a fast path that
    degrades. `vendor/mamba_ssm_min/README.md` is why it is one vendored upstream file
    rather than a `pip install` into a SHARED env, and why having no dist-info is the
    point: `is_mamba_2_ssm_available()` keeps reading False, the fused SSM kernels stay
    off, and the 12B runs the same torch-native Mamba path the Omni row was measured on.

    A real installation wins: this appends, so an installed `mamba_ssm` is found first
    and nothing here shadows it.
    """
    import importlib.util

    if importlib.util.find_spec("mamba_ssm") is not None:
        return
    for root in _repo_candidates():
        here = root / "vendor" / "mamba_ssm_min"
        if here.is_dir():
            if str(here) not in sys.path:
                sys.path.append(str(here))
            return
    raise SystemExit(
        "vendor/mamba_ssm_min is not beside this file and no SR1_REPO points at it; "
        "the Nemotron decoder cannot be imported without `rmsnorm_fn`")


def _shim_single_process_group():
    """A world of one, so the 12B's forward can ask which rank it is.

    `NVIDIA-Nemotron-Nano-12B-v2-VL`'s forward logs its ViT batch size under a bare

        if torch.distributed.get_rank() == 0:

    which raises "Default process group has not been initialized" on a single process.
    Its Omni sibling guards the same line with `is_initialized()`; this checkpoint does
    not, so every picture would fail at the first forward.

    Initialising a real one-rank group is the smaller lie than stubbing `get_rank`: it
    is the state the model's own code is written against, it leaves the forward's
    arithmetic untouched, and `HashStore` keeps it in-process so the probe's per-GPU
    shards cannot collide on a rendezvous port. A group that already exists -- the
    trainer's, under accelerate -- is left exactly as it is.
    """
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo", store=dist.HashStore(),
                                rank=0, world_size=1)


def _shim_masking_api():
    """Bridge the one transformers API change NVIDIA's decoder code predates.

    `modeling_nemotron_h.py` targets transformers 4.55.4 and calls

        create_causal_mask(config=..., input_embeds=..., cache_position=..., ...)

    where 5.13 spells the first `inputs_embeds` and has dropped `cache_position`
    entirely -- it derives the positions from `past_key_values` and `position_ids`. Those
    are the only two differences, so this renames one argument and drops the other in the
    remote module's own namespace.

    Patched HERE rather than in the file on disk because the modules cache is
    re-downloaded whenever the repo changes, so an edit there is silently lost; and
    rather than by pinning transformers 4.55 in a second environment, which is the
    heavier alternative and stays available if more drift turns up. A shim that changed
    the MASK would change the model's output, so the selftest's greedy decode -- which
    compares the scan against stock attention token for token, and would produce nonsense
    under a broken mask -- is what says this is inert.
    """
    import sys

    n = 0
    for name, mod in list(sys.modules.items()):
        if "transformers_modules" not in name:
            continue
        fn = getattr(mod, "create_causal_mask", None)
        if fn is None or getattr(fn, "_sl_shimmed", False):
            continue

        def wrapped(*args, _orig=fn, **kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return _orig(*args, **kw)

        wrapped._sl_shimmed = True
        mod.create_causal_mask = wrapped
        n += 1
    if n:
        print(f"[load] bridged create_causal_mask in {n} remote module(s): "
              "input_embeds -> inputs_embeds, cache_position dropped", flush=True)


def _repair_radio_summary_idxs(model, cfg):
    """Restore a buffer NVIDIA's VLM checkpoints omit and their vision code then indexes.

    `RADIOModel.summary_idxs` is a registered buffer of INDICES -- the forward does
    `all_summary[:, self.summary_idxs]`. The Nemotron checkpoints do not ship it, so
    transformers reports it MISSING, newly-initialises it with random floats, and the
    first vision forward dies in a CUDA device-side assert several frames away from the
    cause (it surfaced inside an RMSNorm in the projector).

    The value is not invented here: it is read from NVIDIA's own standalone release of
    the same encoder, named by the vision config's `auto_map`. If that repo is not
    cached, this refuses rather than guessing -- a wrong index set would silently select
    the wrong summary tokens instead of crashing.
    """
    import glob

    radio = getattr(getattr(model, "vision_model", None), "radio_model", None)
    got = getattr(radio, "summary_idxs", None) if radio is not None else None
    if got is None:
        return                       # the buffer was never registered; the path is unused
    # Restored WHENEVER it is recoverable, not only when it looks wrong. transformers
    # preserves the registered dtype, so a missing int64 buffer comes back as int64 full
    # of uninitialised memory -- which `is_floating_point` reports as False, and an
    # earlier version of this guard therefore skipped exactly the case it existed for.
    # These are indices into a fixed teacher list, not learned weights, so overwriting
    # with the upstream value is a no-op when the checkpoint did ship them.
    ref = (getattr(cfg, "vision_config", None) or object())
    amap = getattr(ref, "auto_map", None) or {}
    repo = str(amap.get("AutoModel", "")).split("--")[0]
    if not repo:
        raise SystemExit("RADIO's summary_idxs was newly initialised and no upstream "
                         "encoder repo is named in vision_config.auto_map to recover it")
    from safetensors.torch import load_file
    snaps = sorted(glob.glob("/home/uberger/scratch/cache/hf_cache/hub/models--"
                             + repo.replace("/", "--") + "/snapshots/*"))
    for f in (sorted(glob.glob(snaps[-1] + "/*.safetensors")) if snaps else []):
        for k, v in load_file(f).items():
            if k.endswith("summary_idxs"):
                was = got.tolist()[:4]
                radio.summary_idxs = v.to(radio.summary_idxs.device)
                print(f"[load] RADIO summary_idxs {was} -> {v.tolist()} (from {repo})",
                      flush=True)
                return
    raise SystemExit(f"RADIO's summary_idxs was newly initialised and {repo} is not "
                     "cached, so the real value cannot be recovered. Fetch it first.")


def _shim_logits_to_keep(model):
    """Let the VLM wrapper pass `logits_to_keep` through to its language model.

    `NemotronHForCausalLM.forward` takes `logits_to_keep: int | torch.Tensor = 0` and the
    VLM wrapper around it does not, so it never reaches the decoder: the lm_head runs over
    the WHOLE sequence, prompt included, and the backward runs over all of it too. On a
    131,072-token vocabulary at ~1,400 positions that is a 369 MB logits tensor plus its
    graph, against 269 MB for the completion alone -- and the trainer already knows how to
    ask for less. It checks `"logits_to_keep" in inspect.signature(model.forward)` and
    quietly skips the argument when it is absent, so the cost is invisible.

    The wrapper's forward calls `self.language_model(...)` with a fixed kwarg list, which
    cannot be edited from outside. So this is two wrappers and a box: the outer one accepts
    the argument and puts it in the box, the inner one takes it out on the way past.

    `functools.wraps` is deliberately NOT used on the outer forward. It would set
    `__wrapped__`, `inspect.signature` would follow it back to the original parameter list,
    and the trainer would go on believing the argument is unsupported -- which is the one
    thing this exists to change.
    """
    import inspect

    cls = type(model)
    if getattr(cls, "_sr1_logits_to_keep_shim", False):
        return
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    try:
        if "logits_to_keep" not in inspect.signature(lm.forward).parameters:
            return                      # the decoder cannot take it either; nothing to do
        if "logits_to_keep" in inspect.signature(cls.forward).parameters:
            return                      # a release that already forwards it
    except (TypeError, ValueError):
        return

    pending = {"n": 0}
    outer = cls.forward
    inner = type(lm).forward

    def forward(self, *args, logits_to_keep=0, **kwargs):
        pending["n"] = logits_to_keep
        try:
            return outer(self, *args, **kwargs)
        finally:
            pending["n"] = 0

    def lm_forward(self, *args, **kwargs):
        if pending["n"] and "logits_to_keep" not in kwargs:
            kwargs["logits_to_keep"] = pending["n"]
        return inner(self, *args, **kwargs)

    cls.forward = forward
    type(lm).forward = lm_forward
    cls._sr1_logits_to_keep_shim = True
    print("[load] the wrapper now forwards logits_to_keep to the language model",
          flush=True)
