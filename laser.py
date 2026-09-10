#!/usr/bin/env python
"""LASER's two attention rewards, and the one forward that feeds them.

`docs/laser-go-no-go.md` is the design; `laser_probe.py` is the harness. This module is
the measurement, and it is a TRANSCRIPTION: every constant and every line of arithmetic
below is read off `laser_repo/verl/workers/actor/dp_actor.py` and
`laser_repo/verl/utils/reward_score/openr1_verl.py`, not off the paper. The two disagree,
and the released code is what produced the reported numbers.

WHERE THE CODE DIFFERS FROM THE WIKI PAGE (../vlm_reasoning/wiki/laser-implementation.md)

  window          10 with stride 5, not 20 with stride 10. The function signature says 20;
                  both call sites pass `window_size = 10`, and the call site wins.
  sink set        NOT massive-activation dimensions in the final layer's hidden states.
                  The released code never asks for hidden states. `S` is the set of visual
                  tokens receiving more than mean + 2*sd of the attention paid by the LAST
                  PROMPT TOKEN, layer- and head-averaged.
  alpha_t         a MEAN over non-sink visual tokens, not a sum.
  R_supp          per generated token, against tau = 0.9 -- so a uniform map (ratio 1.0)
                  is already penalised -- then averaged over tokens.
  omega           0.05 for R_vis and 0.1 for R_supp, on top of the per-reward `scale`.

THE ATTENTION IS THE POINT, SO IT IS COMPUTED THE WAY THE REWARD COMPUTES IT

    A[t, j] = (1/L) sum_l (1/H) sum_h softmax(q_t k^T scaling + mask)[j]

Layer-mean, head-mean, RAW softmax probabilities -- no value weighting, no layer sum. That
is `compute_slice_for_sample` in `laser_repo/verl/workers/actor/attention_capture.py`, and
it is a different quantity from the one `vlm/saliency.py` and the overlap reward read.

`LaserScan` lifts upstream's Q/K trick, which the wiki page recommends taking whichever
path we end up on: the real forward keeps SDPA, and only the (rows we need, kv) slice of
the logits is ever materialised. Upstream needs 266 lines of mirrored module internals to
do it. We need none, because a registered attention implementation is handed `query` and
`key` ALREADY post-RoPE and post-QK-norm -- which is exactly what the mirror existed to
reproduce.

    import laser
    scan = laser.install(model)
    scan.arm(prompt_len=P)                        # teacher-forced prompt ++ completion
    model(**case, use_cache=False)
    got = scan.result()                           # A[t, j], a_bos[j], per-cell columns
    scan.uninstall()
"""

from __future__ import annotations

import math

import numpy as np
import torch

import sink_shift as SS

IMAGE_TOKEN_ID = SS.IMAGE_TOKEN_ID       # <|image_pad|>
IMPL_NAME = "laser_scan"

# ---------------------------------------------------------------------------
# the constants, from laser_repo. Sourced individually because two of them are
# overridden at the call site and reading only the signatures gets them wrong.
# ---------------------------------------------------------------------------
#: dp_actor.py sets this at BOTH call sites, overriding the signature default of 20.
WINDOW_SIZE = 10
SENSITIVITY = 5.0                   # `stability_reward_sensitivity`
STABILITY_SCALE = 0.1               # `stability_reward_scale`
STABILITY_PENALTY = 0.015           # `stability_reward_penalty`, the aggregate path's
DECAY_RATE = 1.5                    # early-weighted variant
WEIGHT_MODE = "exp"
TAU = 0.9                           # `compute_sink_suppression_reward`
SUPP_BETA = 0.5
SUPP_SCALE = 1.0
#: openr1_verl.py -- the weights the two rewards enter the total with.
FORMAT_WEIGHT = 0.3
OMEGA_VIS = 0.05
OMEGA_SUPP = 0.1
#: train.sh -- plain GRPO runs for this many steps before the attention terms switch on.
ATTENTION_START_STEP = 20


# ---------------------------------------------------------------------------
# the rewards -- pure numpy, no torch, no model, CPU-testable
# ---------------------------------------------------------------------------
def sink_mask(a_bos):
    """`a_bos > mean + 2*sd` -> [m] bool. Upstream's whole sink identification.

    The standard deviation is torch's, which is UNBIASED (ddof=1). numpy's default is
    ddof=0, and taking it would move the threshold by a factor of sqrt(m/(m-1)) -- about
    0.3% at m=160, which is small until it changes which patches are in the set at the
    margin. `test_laser_cpu.py` pins it against torch.
    """
    a = np.asarray(a_bos, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return np.zeros(a.size, dtype=bool)
    return a > a.mean() + 2.0 * a.std(ddof=1)


def alpha_per_step(A, sinks=None):
    """LASER's `avg_attentions_per_step`. -> [T].

    The MEAN over visual tokens, taken over V \\ S when the sink set is supplied. A mean
    rather than a sum is what makes `alpha` comparable across pictures with different
    patch counts -- and it also means dropping a sink RAISES alpha, because the sink's
    mass leaves the numerator but its patch leaves the denominator too.
    """
    A = np.asarray(A, dtype=np.float64)
    if sinks is None:
        return A.mean(axis=1)
    keep = ~np.asarray(sinks, dtype=bool)
    if not keep.any():
        return np.zeros(A.shape[0], dtype=np.float64)
    return A[:, keep].mean(axis=1)


def windows_of(alpha, window_size=WINDOW_SIZE):
    """`alpha.unfold(0, window_size, window_size // 2).mean(-1)` -> [n_windows] or None.

    None -- not an empty array -- when `t <= window_size`, because upstream short-circuits
    to a reward of exactly 0.0 there and the two cases have to stay distinguishable: a
    rollout that scored zero for being short is not a rollout that scored zero on merit.
    """
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    t, step = a.size, max(1, int(window_size) // 2)
    if t <= int(window_size):
        return None
    n = (t - int(window_size)) // step + 1
    idx = np.arange(int(window_size))[None, :] + step * np.arange(n)[:, None]
    return a[idx].mean(axis=1)


def early_weights(n_windows, decay_rate=DECAY_RATE, weight_mode=WEIGHT_MODE):
    """Finding 1's weighting: earlier windows count for more. -> [n_windows], mean 1.

    Renormalised to sum to `n_windows`, which is why upstream can share one `scale`
    between the weighted and unweighted variants.
    """
    n = int(n_windows)
    idx = np.arange(n, dtype=np.float64)
    denom = float(max(n - 1, 1))
    if weight_mode == "exp":
        raw = np.exp(-decay_rate * (idx / denom))
    elif weight_mode == "linear":
        raw = np.clip(1.0 - decay_rate * (idx / denom), 1e-6, None)
    elif weight_mode == "power":
        raw = (idx + 1.0) ** (-decay_rate)
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    return raw * (n / (raw.sum() + 1e-12))


def visual_grounding_reward(alpha, window_size=WINDOW_SIZE, sensitivity=SENSITIVITY,
                            scale=STABILITY_SCALE, penalty=STABILITY_PENALTY,
                            early_weighted=False, decay_rate=DECAY_RATE,
                            weight_mode=WEIGHT_MODE):
    """`R_vis`. -> float.

    Every window is scored against THE TRAJECTORY'S OWN PEAK WINDOW, which is detached, so
    this is a measure of flatness and not of level: a rollout whose visual attention is
    uniformly low earns the same as one that is uniformly high. That is the reward as
    released, and §1 of the design document says so before any number is collected -- the
    probe reports r(R_vis, mean alpha) precisely so the claim is checked rather than
    repeated.
    """
    wm = windows_of(alpha, window_size)
    if wm is None:
        return 0.0
    ratios = wm / (wm.max() + 1e-12)
    per = np.exp(-sensitivity * (1.0 - ratios)) - penalty
    if early_weighted:
        per = per * early_weights(wm.size, decay_rate, weight_mode)
    return float(per.sum() * scale)


def sink_suppression_reward(A, sinks, tau=TAU, beta=SUPP_BETA, scale=SUPP_SCALE):
    """`R_supp`. -> float. 1.0 only if every step keeps the sinks at or below `tau`.

    `ratio` is the sinks' mean per-patch attention over ALL visual tokens' mean per-patch
    attention -- an enrichment, in the same units as `E_ring` in
    docs/sink-location-by-image-type.md. With `tau = 0.9` the ceiling is unreachable by a
    map that merely gives the sinks their fair share.
    """
    A = np.asarray(A, dtype=np.float64)
    s = np.asarray(sinks, dtype=bool)
    if A.size == 0 or s.sum() == 0:
        return 0.0
    ratio = A[:, s].mean(axis=1) / (A.mean(axis=1) + 1e-10)
    return float(np.exp(-beta * np.clip(ratio - tau, 0.0, None)).mean() * scale)


def sink_ratio_per_step(A, sinks):
    """The `r_t` inside `R_supp`, exposed so the report can show what it is reacting to."""
    A = np.asarray(A, dtype=np.float64)
    s = np.asarray(sinks, dtype=bool)
    if A.size == 0 or s.sum() == 0:
        return np.zeros(A.shape[0], dtype=np.float64)
    return A[:, s].mean(axis=1) / (A.mean(axis=1) + 1e-10)


def total_reward(acc, fmt, r_vis, r_supp, omega_vis=OMEGA_VIS, omega_supp=OMEGA_SUPP,
                 format_weight=FORMAT_WEIGHT):
    """`openr1_verl.compute_score_vanilla`, without the repetition penalty.

    The gate is MULTIPLICATIVE on both `acc` and `fmt`, so an incorrect rollout carries no
    attention term at all -- which is what makes the within-group spread among CORRECT
    rollouts, and not the spread over all eight, the quantity that decides whether GRPO
    can see this reward.

    `repetition_reward` is dropped: it is orthogonal to the attention terms, it is the one
    component whose upstream form we would not adopt unchanged, and including it would put
    a constant in every column of the report that reacts to nothing being measured here.
    """
    gate = float(acc) * float(fmt)
    return (float(acc) + format_weight * float(fmt)
            + omega_vis * float(r_vis) * gate + omega_supp * float(r_supp) * gate)


def response_query_slice(n_valid):
    """Upstream's `desc_start:desc_end`, as a slice into the response. -> slice.

    `desc_start = prompt_len + 1` and `desc_end = desc_start + n_valid - 1`, both absolute,
    so in response coordinates the queries are `1 .. n_valid - 2`: the FIRST response token
    is skipped and the LAST is dropped. Neither is documented upstream and neither looks
    deliberate, but both change which tokens `R_vis` sees at the ends of a short rollout,
    and the point of a replication is to inherit the arithmetic rather than improve it.
    """
    n = int(n_valid)
    return slice(1, max(1, n - 1))


# ---------------------------------------------------------------------------
# the collector
# ---------------------------------------------------------------------------
class LaserScan:
    """One teacher-forced forward -> `A[t, j]`, `a_bos[j]`, and the per-cell columns.

    Registered as an attention implementation, like `sink_shift.SinkShift` and
    `sink_location.SinkScan`. Unlike either, it EDITS NOTHING and does not take the softmax
    on the forward's own path: the model's output comes from stock SDPA, and the reward's
    slice is computed beside it from the same post-RoPE `query` and `key`. Only the rows
    the reward reads are ever materialised, so the peak is one layer's
    `[1, H, 1 + n_response, kv]` in fp32 rather than the full `[1, H, T, T]`.

    Memory at a 1,500-token sequence with 512 response tokens and 36 layers: ~100 MB
    transient, and the accumulators are `[n_response, m]` plus `[L, H, m]` -- under 2 MB.
    """

    def __init__(self, model, want_cells=True):
        self.model = model
        self.want_cells = bool(want_cells)
        # A teacher-forced forward carries prompt ++ completion, so nothing in the input
        # says where one ends and the other begins. The caller has to say.
        self.prompt_len = None
        self.paused = False
        self.img_cols = None
        self.grid = None
        self.n_layers_seen = 0
        self.kv_len = 0
        self.scaling = 1.0
        self._resp = None
        self._bos = None
        self._cells = {}
        self._handles = []
        self._prev_impl = None
        self._text_cfg = None

    # -- arming -----------------------------------------------------------
    def arm(self, prompt_len):
        """Declare the prompt/response boundary and clear the accumulators.

        `img_cols` is cleared too, so a forward whose pre-hook never located a picture
        produces a `None` result rather than silently reusing the PREVIOUS prompt's
        columns -- which, on a picture with a different grid, would put every column
        statistic on the wrong patch and leave nothing downstream able to tell.
        """
        self.prompt_len = int(prompt_len)
        self.img_cols, self.grid = None, None
        self.reset()
        return self

    def reset(self):
        self._resp = None
        self._bos = None
        self._cells = {}
        self.n_layers_seen = 0
        return self

    # -- per-prompt layout ------------------------------------------------
    def _locate(self, input_ids, image_grid_thw):
        """The image token run and its patch grid, exactly as `sink_location` finds them.

        A third implementation of the same lookup would be a third thing to drift; this
        calls `sink_location.locate_image_runs` and `test_laser_cpu.py` checks the columns
        it produces against `sink_shift`'s independent locator on the same input.
        """
        import sink_location as SL

        runs, grids = SL.locate_image_runs(input_ids, image_grid_thw)
        if not runs:
            self.img_cols, self.grid = None, None
            return False
        if len(runs) != 1:
            raise RuntimeError(
                f"{len(runs)} pictures in the prompt: LASER's sink set is defined over one "
                "picture's visual tokens and this collector will not guess a split")
        t, gh, gw = grids[0]
        self.img_cols = runs[0].to(input_ids.device)
        self.grid = (gh, gw)
        return True

    def _pre_hook(self, module, args, kwargs):
        # `paused` has to be honoured HERE and not only in the attention function.
        # Sampling G rollouts uses `num_return_sequences=G`, which expands the batch to G
        # before the first forward, and a batch guard that fires while the collector is
        # deliberately asleep would refuse the generation it exists to let run at speed.
        if self.paused:
            return None
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None or ids.dim() != 2:
            return None
        if ids.shape[0] != 1:
            raise RuntimeError(
                f"batch of {ids.shape[0]}: the sink set is per picture and this collector "
                "is only correct at batch size 1. Set `paused` around `generate`.")
        if bool((ids == IMAGE_TOKEN_ID).any()):
            self._locate(ids, kwargs.get("image_grid_thw"))
        return None

    # -- results ----------------------------------------------------------
    def result(self):
        """-> dict, or None if this forward carried no picture or no response.

        `A` is the layer-mean of the per-layer head-means, which is the order upstream
        takes them in and, since every layer contributes exactly one head-mean over the
        same rows, the same thing as the joint mean. The division happens here rather than
        per layer so a missing layer is visible as a wrong count instead of a quiet bias.
        """
        if self._resp is None or self.n_layers_seen == 0:
            return None
        if self.n_layers_seen != self._n_text_layers():
            raise RuntimeError(
                f"{self.n_layers_seen} layers contributed but the text stack has "
                f"{self._n_text_layers()}: the attention implementation was not installed "
                "everywhere, and a layer-mean over a subset is not LASER's quantity")
        n = float(self.n_layers_seen)
        out = {
            "A": (self._resp / n).cpu().numpy().astype(np.float32),
            "a_bos": None if self._bos is None else (self._bos / n).cpu().numpy().astype(np.float32),
            "grid": list(self.grid),
            "kv_len": int(self.kv_len),
            "prompt_len": int(self.prompt_len),
            "n_layers": int(self.n_layers_seen),
        }
        if self._cells:
            layers = sorted(self._cells)
            if layers != list(range(len(layers))):
                raise RuntimeError(f"layers {layers} are not a contiguous block from 0")
            out["cells"] = torch.stack([self._cells[l] for l in layers]).cpu().numpy()
        return out

    def _n_text_layers(self):
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        return int(getattr(cfg, "num_hidden_layers", 0))

    # -- install ----------------------------------------------------------
    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, _make_laser_attention(self))
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self._text_cfg = cfg
        self._prev_impl = cfg._attn_implementation
        cfg._attn_implementation = IMPL_NAME
        for m in self.model.modules():
            if type(m).__name__ == "Qwen3VLTextAttention" and hasattr(m, "layer_idx"):
                m.config._attn_implementation = IMPL_NAME
        self._handles.append(self.model.register_forward_pre_hook(self._pre_hook,
                                                                  with_kwargs=True))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._text_cfg is not None and self._prev_impl is not None:
            self._text_cfg._attn_implementation = self._prev_impl
            for m in self.model.modules():
                if type(m).__name__ == "Qwen3VLTextAttention":
                    m.config._attn_implementation = self._prev_impl
        self._prev_impl = None


def install(model, **kwargs):
    return LaserScan(model, **kwargs).install()


def _make_laser_attention(state: LaserScan):
    """Stock SDPA for the model, a second small softmax for the reward. Edits nothing."""

    def laser_attention_forward(module, query, key, value, attention_mask,
                                dropout=0.0, scaling=None, is_causal=None, **kwargs):
        out = SS._sdpa(module, query, key, value, attention_mask, dropout, scaling,
                       is_causal, **kwargs)
        if state.img_cols is None or state.paused or state.prompt_len is None:
            return out
        if query.shape[2] <= 1:
            return out              # a decode step: the reward reads a full forward only
        _record(state, module, query, key, attention_mask, scaling, is_causal)
        return out

    return laser_attention_forward


def _needed_rows(state: LaserScan, q_start, q_len, device):
    """(bos row, response rows) as offsets into this forward's query block.

    `bos` is upstream's name for query position `prompt_len - 1` -- the last token of the
    prompt, which is not a BOS token in any Qwen3-VL prompt. The name is kept because it
    is the name in the code the sink set comes from, and renaming it here would make the
    two impossible to read side by side.
    """
    lo, hi = q_start, q_start + q_len
    bos_abs = state.prompt_len - 1
    bos = (torch.tensor([bos_abs - q_start], device=device)
           if lo <= bos_abs < hi else torch.zeros(0, dtype=torch.long, device=device))
    r0 = max(state.prompt_len, lo)
    resp = torch.arange(r0 - q_start, hi - q_start, device=device)
    return bos, resp


def _record(state: LaserScan, module, query, key, attention_mask, scaling, is_causal):
    """One layer's contribution: softmax over the needed rows only, then head-mean."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    k = SS._repeat_kv(key, n_rep)
    q_len, kv_len = query.shape[2], k.shape[2]
    q_start = kv_len - q_len
    if scaling is None:
        scaling = module.head_dim ** -0.5
    state.scaling = float(scaling)
    state.kv_len = int(kv_len)

    bos, resp = _needed_rows(state, q_start, q_len, query.device)
    rows = torch.cat([bos, resp])
    if rows.numel() == 0:
        return
    q_sel = query[:, :, rows, :]                                   # [1, H, R, d]
    logits = torch.matmul(q_sel.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[..., rows, :kv_len].float()
    else:
        # No mask means the fast path relied on `is_causal`, so the causal structure has
        # to be rebuilt here or the softmax would normalise over the future as well and
        # every probability would be too small by a different amount per row.
        qpos = (rows + q_start)[:, None]
        kpos = torch.arange(kv_len, device=query.device)[None, :]
        logits = logits.masked_fill(kpos > qpos, float("-inf"))
    a = torch.softmax(logits, dim=-1)                              # [1, H, R, kv]
    w = a[0][:, :, state.img_cols]                                 # [H, R, m]
    head_mean = w.mean(0)                                          # [R, m]

    n_bos = int(bos.numel())
    if n_bos:
        b = head_mean[0].float()
        state._bos = b if state._bos is None else state._bos + b
    body = head_mean[n_bos:].float()
    if body.shape[0]:
        state._resp = body if state._resp is None else state._resp + body
        if state.want_cells:
            layer_idx = int(getattr(module, "layer_idx", -1))
            state._cells[layer_idx] = w[:, n_bos:, :].sum(1).float()
    state.n_layers_seen += 1


# ---------------------------------------------------------------------------
# reading a collected rollout
# ---------------------------------------------------------------------------
def score_rollout(A, a_bos, n_valid, apply_rectification=True, early_weighted=True,
                  **kw):
    """Everything the report needs about one rollout. -> dict.

    `A` is the FULL response block; the `desc_start:desc_end` slice is applied here rather
    than at collection time so the off-by-one in `response_query_slice` stays visible and
    reversible. `n_query` is what upstream's `t <= window_size` short-circuit sees.
    """
    A = np.asarray(A, dtype=np.float64)
    sel = response_query_slice(n_valid)
    Aq = A[sel]
    sinks = sink_mask(a_bos) if (apply_rectification and a_bos is not None) \
        else np.zeros(A.shape[1], dtype=bool)
    alpha = alpha_per_step(Aq, sinks if apply_rectification else None)
    wm = windows_of(alpha)
    r_vis = visual_grounding_reward(alpha, early_weighted=early_weighted, **kw)
    r_supp = sink_suppression_reward(Aq, sinks) if sinks.any() else 0.0
    ratio = sink_ratio_per_step(Aq, sinks)
    return {
        "n_query": int(Aq.shape[0]),
        "n_sinks": int(sinks.sum()),
        "sink_frac": float(sinks.mean()) if sinks.size else float("nan"),
        "sinks": sinks,
        "alpha": alpha,
        "windows": wm,
        "short": wm is None,
        "arg_window": None if wm is None else int(np.argmax(wm)),
        "n_windows": 0 if wm is None else int(wm.size),
        "r_vis": r_vis,
        "r_vis_flat": visual_grounding_reward(alpha, early_weighted=False, **kw),
        "r_supp": r_supp,
        "sink_ratio": ratio,
        "mean_alpha": float(alpha.mean()) if alpha.size else float("nan"),
        "image_mass": float(Aq.sum(axis=1).mean()) if Aq.size else float("nan"),
    }


def decay_spearman(alpha):
    """Rank correlation of alpha with the step index. -> float, or NaN when degenerate."""
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if a.size < 4 or not np.isfinite(a).all() or a.min() == a.max():
        return float("nan")
    r = np.argsort(np.argsort(a)).astype(np.float64)
    t = np.arange(a.size, dtype=np.float64)
    return float(np.corrcoef(r, t)[0, 1])


def ring_agreement(sinks, gh, gw):
    """`S` against the one-patch border. -> dict.

    The cross-check `docs/laser-go-no-go.md` §3.3 exists for. If LASER's sink set and the
    border ring are the same tokens, then everything
    docs/sink-location-by-image-type.md established about the ring -- that it is a
    vision-encoder positional stamp, and that its across-query CV rules it out as a sink --
    is a statement about `S`, and `R_supp` is suppressing the encoder.
    """
    import sink_location as SL

    ring = SL.ring_set(gh, gw).reshape(-1)
    s = np.asarray(sinks, dtype=bool).reshape(-1)
    if s.size != ring.size:
        raise ValueError(f"{s.size} sinks against a {gh}x{gw} grid")
    inter = float((s & ring).sum())
    union = float((s | ring).sum())
    return {
        "jaccard": inter / union if union else float("nan"),
        "sink_on_ring": inter / float(s.sum()) if s.sum() else float("nan"),
        "ring_area": float(ring.mean()),
        # how much more of `S` lands on the ring than the ring's area share would give
        "enrichment": ((inter / float(s.sum())) / ring.mean()
                       if s.sum() and ring.mean() > 0 else float("nan")),
    }


def column_cv(A, cols):
    """Across-QUERY coefficient of variation of the named columns. -> float.

    The sink literature's second leg, and the one
    docs/sink-location-by-image-type.md pre-registered at 0.5: a token that is merely
    large is a peak, and a token that is large AND does not move with the query is a sink.
    Computed per column over the generated tokens and then averaged, so a set with one
    stable column and one volatile one does not average into a false pass.
    """
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(cols, dtype=bool)
    if A.shape[0] < 2 or c.sum() == 0:
        return float("nan")
    sub = A[:, c]
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mu > 0, sd / mu, np.nan)
    return float(np.nanmean(cv)) if np.isfinite(cv).any() else float("nan")


def within_group_sd(values, groups):
    """Mean over groups of the sd within a group. -> (mean sd, n groups used).

    GRPO's advantage is the group-centred reward over the group's own spread, so this --
    not the spread over the whole run -- is what decides whether a reward term can move
    the policy. docs/overlap-reward-hack-set-a.md is what happens when it is small and the
    weight is not.
    """
    by = {}
    for v, g in zip(values, groups):
        if v is not None and np.isfinite(v):
            by.setdefault(g, []).append(float(v))
    sds = [float(np.std(v, ddof=1)) for v in by.values() if len(v) >= 2]
    return (float(np.mean(sds)) if sds else float("nan")), len(sds)
