#!/usr/bin/env python
"""E0 -- the RoPE phase lock-in test: does the image-attention profile march with position?

WHAT IS BEING MEASURED
----------------------
In Qwen3-VL a text token at index p carries the M-RoPE position (p, p, p), and a
patch at row r, column c of an image anchored at s carries (s, s+r, s+c).  The
three per-axis offsets that reach the attention logit are therefore

    dt = d,        dh = d - r,        dw = d - c,        with  d = p - s

so every (r, c) dependence enters ONLY through d - r and d - c.  Adding 1 to d is
identical to subtracting 1 from r and from c: the whole positional overlay that
RoPE lays on the image slides one patch row down and one patch column right per
generated token, in every frequency channel at once.  The t-axis offset does not
depend on the patch at all, so it shifts image-vs-text mass but cannot change the
shape of the profile over patches.

The overlay is faint next to content.  But it depends on exactly one variable, d,
and d is arbitrary with respect to what is in the picture.  So: bucket a few
hundred thousand (token, head) attention rows by the phase of one RoPE channel,
average inside each bucket -- content mushes out, the overlay adds up coherently
-- subtract the grand mean, and ask whether consecutive buckets are the same
picture shifted by the predicted number of patches.

THE PREDICTION HAS NO FREE PARAMETERS
-------------------------------------
Bucketing by the phase of a channel with angular frequency theta into B buckets
makes consecutive buckets differ in d by 2*pi/(theta*B), and the overlay shifts by
that many patches.  For Qwen3-VL-8B (rope_theta 5e6, interleaved M-RoPE) the
fastest H channel is theta = 0.7858 rad/position, so 8 buckets => +1.00 row per
bucket; the fastest W channel is theta = 0.6175, so 10 buckets => +1.02 columns
per bucket.  Halving the bucket count doubles the predicted shift -- a second
prediction from the same law, free, on the same data.

What survives the bucket average is only the channel you bucketed on: consecutive
h8 buckets differ in d by 8.0, over which the W channel turns 4.9 rad, so the
column half of the diagonal averages away inside the bucket.  Expect a march
along the binning's own axis and none on the other -- see predicted_pair_2d.

WHY LOG-ATTENTION, MEAN-CENTRED OVER THE PATCHES
------------------------------------------------
log softmax_i = z_i - logZ and logZ does not depend on i, so mean-centring the log
attention across the image patches recovers the raw q.k logit up to a constant --
the quantity the algebra above is about -- without ever needing the softmax
denominator.  It also discards the image-vs-text mass effect ("visual fading"),
which is a different and already-documented phenomenon.  This probe is about the
SHAPE of the profile, never its total.

CONTROLS
--------
  perm8  the h8 buckets with their LABELS permuted independently per case.  Counts
         per bucket per case are preserved exactly, so is every scrap of content,
         and the only thing destroyed is the alignment of the overlay ACROSS cases
         -- which is precisely what the hypothesis claims exists.
         An iid random bucket draw is NOT a matched floor and must not be used:
         a case's d values are consecutive integers, so they fill the phase
         buckets almost perfectly evenly and that case's content cancels in the
         residual far better than it does under random assignment.  Measured, the
         iid floor sits ~5x above the phase binning on pure noise, which would
         rig the power comparison in the hypothesis's favour.  See
         test_rope_phase_cpu.py::test_no_march_when_nothing_is_planted.
  h4     same channel as h8 with four buckets, so the predicted shift doubles.
  decoy  a frequency that is not a RoPE channel.  The law still applies to it
         (any bucketing by phase separates d), so it is a second rate check
         rather than a pure null -- read it as such.

Content cannot know d mod 8, so a signal that lands on the predicted shift for
h8, doubles for h4, appears on the W axis at the W rate, and vanishes under perm8
is not a content artefact.

STAGES
------
  scan    (GPU) generate one completion per case, teacher-force it, replay every
          layer in eager to recover the softmax weights, accumulate bucket sums.
          Writes scan/shardNN.npz; re-running skips finished shards.
  report  (CPU, fine on the login node) merge the shards, form residuals, run the
          shift test, print the table.

  python rope_phase_probe.py --stage scan   --out-dir DIR --shard 0 --num-shards 8
  python rope_phase_probe.py --stage report --out-dir DIR

Cost: one generation plus one teacher-forced forward with a 36-layer eager replay
per case: ~5 s/case on an H100, so ~21 min for 256 cases on one GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IMAGE_TOKEN_ID = 151655          # Qwen2-VL/2.5-VL/3-VL all share it; config wins when present

# The text tower's attention class, per model family.  All three take layer_idx,
# resolve their kernel through config._attn_implementation, and return
# (output, attn_weights) -- which is what the eager replay in EagerAttentionTap
# needs.  The vision towers' attention classes are deliberately not here.
TEXT_ATTENTION_CLASSES = (
    "Qwen3VLTextAttention",      # Qwen3-VL: interleaved M-RoPE, H and W on fast channels
    "Qwen2_5_VLAttention",       # Qwen2.5-VL: chunked M-RoPE, W on channels that barely turn
    "Qwen2VLAttention",
)

# Kept verbatim from the host project's trainer.  It only has to put the model on
# the distribution it was tuned for; nothing here parses the completion.
SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> tags, "
    "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
)

# Both are external to this directory on purpose: point them wherever the model and
# an image+question dataset happen to live.  The dataset needs `image` and `problem`
# columns, which is all this probe reads.
DEFAULT_DATASET = os.environ.get("ROPE_PHASE_DATASET", "")
DEFAULT_MODEL = os.environ.get("ROPE_PHASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


def load_samples(dataset_path: str, n: int, seed: int):
    """n rows drawn without replacement from the whole dataset, by seed.

    Deliberately no train/holdout carve: that is host-project trainer semantics and
    this probe does not care which rows the model was tuned on -- the overlay it
    measures is a property of positions, not of content.
    """
    from datasets import load_dataset, load_from_disk

    if os.path.isfile(os.path.join(dataset_path, "state.json")):
        ds = load_from_disk(dataset_path)
        if hasattr(ds, "keys"):
            ds = ds["train"]
    elif os.path.isdir(dataset_path):
        ds = load_dataset(dataset_path, split="train")
    else:
        raise SystemExit(f"dataset not found: {dataset_path} (pass --dataset)")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(np.arange(len(ds)), size=min(n, len(ds)), replace=False).tolist())
    # `problem` and `image` are all E0/E1 need, but a probe that scores against
    # annotations needs the annotations too, so any of these that the dataset
    # happens to carry come along.  Additive: nothing that used the old three keys
    # sees a difference.
    extra = [c for c in ("bbox", "solution", "dataset", "split", "question_id")
             if c in ds.column_names]
    out = []
    for i in idx:
        row = ds[int(i)]
        rec = {"row_index": int(i), "question": row["problem"], "image": row["image"]}
        rec.update({c: row[c] for c in extra})
        out.append(rec)
    return out


def load_model(base_path: str, adapter: str | None, device: str, attn_impl: str = "sdpa"):
    import transformers
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_path, padding_side="left")
    config = AutoConfig.from_pretrained(base_path)
    architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    if adapter:
        from peft import PeftModel

        # Merging in-memory keeps the module tree plain, so the attention hook still
        # finds the bare attention class rather than a PEFT wrapper.
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return processor, model.to(device).eval()


def build_prompt(processor, question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(processor, model, image, question, max_new_tokens, temperature, device):
    inputs = processor(
        text=[build_prompt(processor, question)], images=[[image]], return_tensors="pt",
        padding=True, padding_side="left", add_special_tokens=False,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=1.0,
                         top_k=0, max_new_tokens=max_new_tokens,
                         pad_token_id=processor.tokenizer.pad_token_id)
    ids = out[0, prompt_len:].tolist()
    eos = processor.tokenizer.eos_token_id
    if eos in ids:
        ids = ids[: ids.index(eos) + 1]
    return inputs, prompt_len, ids


def teacher_forced_case(prompt_inputs, comp_ids, device):
    """prompt ++ one completion, as a single teacher-forced forward.

    mm_token_type_ids must be extended over the completion with zeros -- text -- or
    M-RoPE position ids come out wrong rather than missing.
    """
    ids = torch.cat([prompt_inputs["input_ids"],
                     torch.tensor([comp_ids], device=device)], dim=1)
    case = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for k in ("pixel_values", "image_grid_thw"):
        if k in prompt_inputs:
            case[k] = prompt_inputs[k]
    if prompt_inputs.get("mm_token_type_ids") is not None:
        zeros = torch.zeros(1, len(comp_ids), dtype=torch.long, device=device)
        case["mm_token_type_ids"] = torch.cat([prompt_inputs["mm_token_type_ids"], zeros], dim=1)
    return case


TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# which RoPE channels carry which axis, and how fast they turn
# ---------------------------------------------------------------------------
def rope_params(text_config):
    """(head_dim, base, mrope_section, interleaved), tolerating either config spelling."""
    rs = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
    if not isinstance(rs, dict):
        rs = dict(rs)
    head_dim = int(getattr(text_config, "head_dim", None)
                   or text_config.hidden_size // text_config.num_attention_heads)
    base = float(rs.get("rope_theta", None) or getattr(text_config, "rope_theta", 10000.0))
    n = head_dim // 2
    section = list(rs.get("mrope_section") or [n, 0, 0])
    return head_dim, base, section, bool(rs.get("mrope_interleaved", False))


def axis_channels(text_config):
    """{'t'|'h'|'w': (freq_indices, angular frequencies)} for the text tower's M-RoPE.

    Chunked (Qwen2.5-VL): contiguous blocks in section order.  Interleaved
    (Qwen3-VL): THWTHW... over the first 3*section[1] entries with the tail all T,
    mirroring transformers' apply_interleaved_mrope.
    """
    head_dim, base, section, interleaved = rope_params(text_config)
    n = head_dim // 2
    idx = {"t": [], "h": [], "w": []}
    if interleaved:
        hset = set(range(1, section[1] * 3, 3))
        wset = set(range(2, section[2] * 3, 3))
        for j in range(n):
            idx["h" if j in hset else "w" if j in wset else "t"].append(j)
    else:
        edges = [0, section[0], section[0] + section[1], n]
        idx["t"] = list(range(edges[0], edges[1]))
        idx["h"] = list(range(edges[1], edges[2]))
        idx["w"] = list(range(edges[2], edges[3]))
    return {k: (v, [base ** (-2.0 * j / head_dim) for j in v]) for k, v in idx.items()}


def fastest_theta(text_config, axis: str) -> float:
    _, freqs = axis_channels(text_config)[axis]
    if not freqs:
        raise SystemExit(f"axis {axis!r} has no M-RoPE channels in this config")
    return max(freqs)


# ---------------------------------------------------------------------------
# bucketings
# ---------------------------------------------------------------------------
@dataclass
class Binning:
    name: str
    kind: str          # "phase" | "shuffle" (same buckets, labels permuted per case)
    theta: float       # angular frequency, rad per position unit
    nbins: int
    axis: str          # "row" | "col" -- which image axis the shift test slides

    @property
    def predicted_shift(self) -> float:
        """Patches of drift between consecutive buckets, from the law alone."""
        if self.kind != "phase":
            return 0.0
        return TWO_PI / (self.theta * self.nbins)


def default_binnings(text_config, decoy: float) -> list[Binning]:
    th, tw = fastest_theta(text_config, "h"), fastest_theta(text_config, "w")
    return [
        Binning("h8", "phase", th, 8, "row"),
        Binning("h4", "phase", th, 4, "row"),
        Binning("w10", "phase", tw, 10, "col"),
        Binning("decoy8", "phase", decoy, 8, "row"),
        Binning("perm8", "shuffle", th, 8, "row"),
    ]


def phase_bins(d, theta: float, nbins: int):
    """Bucket index from the phase of one channel.  Works on a torch tensor or ndarray.

    Bucketing on the phase rather than on `d % period` matters: the period is
    7.996 positions, not 8, and over a 400-token completion the difference slips
    by half a patch.
    """
    if torch.is_tensor(d):
        ph = torch.remainder(d.double() * theta, TWO_PI)
        return torch.clamp((ph / (TWO_PI / nbins)).long(), 0, nbins - 1)
    ph = np.remainder(np.asarray(d, dtype=np.float64) * theta, TWO_PI)
    return np.clip((ph / (TWO_PI / nbins)).astype(np.int64), 0, nbins - 1)


def make_bins(d, binnings, rng):
    """Bucket index per token, one vector per binning.  `rng` is drawn once per case."""
    out = {}
    for b in binnings:
        base = phase_bins(d, b.theta, b.nbins)
        if b.kind == "phase":
            out[b.name] = base
            continue
        # "shuffle": relabel this case's buckets by a random permutation.  Counts are
        # preserved exactly; only the cross-case alignment of the overlay is lost.
        perm = rng.permutation(b.nbins)
        if torch.is_tensor(base):
            lut = torch.as_tensor(perm, dtype=torch.long, device=base.device)
            out[b.name] = lut[base]
        else:
            out[b.name] = perm[base]
    return out


class BinAccumulator:
    """sum[name][layer, head, bin, patch] and the per-token variance, on device."""

    def __init__(self, binnings, n_layers, n_heads, n_patch, device):
        self.binnings = list(binnings)
        self.sum = {
            b.name: torch.zeros(n_layers, n_heads, b.nbins, n_patch,
                                dtype=torch.float32, device=device)
            for b in self.binnings
        }
        self.count = {
            b.name: torch.zeros(b.nbins, dtype=torch.float64, device=device)
            for b in self.binnings
        }
        self.sumsq = torch.zeros(n_layers, n_heads, n_patch, dtype=torch.float32, device=device)
        self.ntok = 0
        self.ncase = 0
        self.dmin = self.dmax = None
        # Kept on the device: an .item() here would sync 36 times per case for one
        # integer, which is what RolloutFlow's comment warns about.
        self.clamped = torch.zeros((), dtype=torch.float64, device=device)

    @property
    def nclamped(self) -> int:
        return int(self.clamped.item())

    def add_layer(self, row: int, x: torch.Tensor, bins: dict):
        """x: [heads, tokens, patches] float32, already mean-centred over patches."""
        for b in self.binnings:
            self.sum[b.name][row].index_add_(1, bins[b.name], x)
        self.sumsq[row] += x.pow(2).sum(1)

    def add_case(self, bins: dict, d_keep):
        n_tok = int(len(d_keep))
        for b in self.binnings:
            ones = torch.ones(n_tok, dtype=torch.float64, device=self.count[b.name].device)
            self.count[b.name].index_add_(0, bins[b.name], ones)
        self.ntok += n_tok
        self.ncase += 1
        lo, hi = float(d_keep.min()), float(d_keep.max())
        self.dmin = lo if self.dmin is None else min(self.dmin, lo)
        self.dmax = hi if self.dmax is None else max(self.dmax, hi)

    def to_npz(self, meta: dict) -> dict:
        out = {"__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
        for b in self.binnings:
            out[f"sum::{b.name}"] = self.sum[b.name].detach().cpu().numpy()
            out[f"count::{b.name}"] = self.count[b.name].detach().cpu().numpy()
        out["sumsq"] = self.sumsq.detach().cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# the shift test (pure numpy -- this is what test_rope_phase_cpu.py drives)
# ---------------------------------------------------------------------------
def bucket_residuals(sums: np.ndarray, counts: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """[L, H, B, P] sums + [B] counts -> [L, H, B, gh, gw] residual maps.

    Residual = bucket mean minus the grand mean over buckets, i.e. exactly the part
    of the profile that knows which bucket it is in.  Content that is common to
    every bucket -- which is nearly all of it -- cancels here.
    """
    if np.any(counts <= 0):
        raise SystemExit(f"empty bucket(s): counts={counts.tolist()}")
    means = sums / counts[None, None, :, None]
    resid = means - means.mean(axis=2, keepdims=True)
    return resid.reshape(*resid.shape[:3], gh, gw)


def _crop_pair(a: np.ndarray, b: np.ndarray, s: int, axis: int):
    """Views implementing the comparison  a(x)  vs  b(x - s)  on the valid overlap.

    Deliberately not a circular roll: the column period (10.2) does not divide the
    grid, so wrapping would fold the pattern onto itself and manufacture agreement.
    """
    n = a.shape[axis]
    if abs(s) >= n:
        raise ValueError(f"shift {s} exceeds axis length {n}")
    sl_a = slice(s, n) if s >= 0 else slice(0, n + s)
    sl_b = slice(0, n - s) if s >= 0 else slice(-s, n)
    ia = [slice(None)] * a.ndim
    ib = [slice(None)] * b.ndim
    ia[axis], ib[axis] = sl_a, sl_b
    return a[tuple(ia)], b[tuple(ib)]


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r over the last two axes, broadcast over the leading ones."""
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    num = (a * b).sum(axis=(-2, -1))
    den = np.sqrt((a * a).sum(axis=(-2, -1)) * (b * b).sum(axis=(-2, -1)))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def shift_scores(resid: np.ndarray, axis: str, shifts) -> np.ndarray:
    """[L, H, B, gh, gw] -> [L, H, len(shifts)].

    score(s) = mean over consecutive bucket pairs of corr(R_{k+1}, R_k shifted by s).
    Buckets are compared in order and the k = B-1 -> 0 wrap is skipped, so nothing
    depends on the pattern period dividing the grid.
    """
    ax = 3 if axis == "row" else 4
    out = np.zeros(resid.shape[:2] + (len(shifts),), dtype=np.float64)
    nxt, cur = resid[:, :, 1:], resid[:, :, :-1]
    for i, s in enumerate(shifts):
        a, b = _crop_pair(nxt, cur, int(s), ax)
        out[:, :, i] = _corr(a, b).mean(axis=2)
    return out


def shift_scores_2d(resid: np.ndarray, shifts) -> np.ndarray:
    """[L, H, B, gh, gw] -> [L, H, len(shifts), len(shifts)] over (row, col) shifts jointly.

    This is the actual prediction.  Bucketing by ANY channel's phase separates
    consecutive buckets by the same step in d, and d enters as both d-r and d-c,
    so the overlay translates DIAGONALLY by (delta, delta) -- not along one axis.
    Sliding rows alone leaves the column component unmatched, which drags the
    best row-only shift back toward zero and splits heads between 0 and +1.
    """
    nxt, cur = resid[:, :, 1:], resid[:, :, :-1]
    out = np.zeros(resid.shape[:2] + (len(shifts), len(shifts)), dtype=np.float64)
    for i, sr in enumerate(shifts):
        a1, b1 = _crop_pair(nxt, cur, int(sr), 3)
        for j, sc in enumerate(shifts):
            a2, b2 = _crop_pair(a1, b1, int(sc), 4)
            out[:, :, i, j] = _corr(a2, b2).mean(axis=2)
    return out


def bracket_ints(p: float) -> list[int]:
    """Whole-patch shifts a predicted rate is allowed to land on.

    2*pi/(theta*B) is 0.9995 for h8, not 1: bracketing that as {0, 1} would accept
    a head that found no march at all.  Snap when the prediction is within 0.15 of
    an integer, bracket only when it genuinely falls between two.
    """
    return [int(round(p))] if abs(p - round(p)) < 0.15 else [int(math.floor(p)), int(math.ceil(p))]


def predicted_pair_2d(predicted: float, axis: str):
    """(row shifts, col shifts) expected under a binning on `axis`.

    NOT (delta, delta).  A true step in d does translate the overlay diagonally,
    but bucketing by one channel's phase only makes THAT channel coherent inside a
    bucket: consecutive h8 buckets differ in d by 8.0, over which the W channel
    turns 4.9 rad, so the column component averages away within the bucket and
    only the rows march.  Bucketing on W is the mirror image.  So each binning
    should show a march along its own axis and nothing along the other, which is a
    sharper prediction than the diagonal and distinguishes the two channels.
    """
    b = bracket_ints(predicted)
    return (b, [0]) if axis == "row" else ([0], b)


def summarize_2d(scores2d, shifts, predicted: float, axis: str):
    shifts = np.asarray(shifts)
    L, H = scores2d.shape[:2]
    flat = scores2d.reshape(L * H, -1).argmax(axis=1)
    sr = shifts[flat // len(shifts)]
    sc = shifts[flat % len(shifts)]
    pr, pc = predicted_pair_2d(predicted, axis)
    ok = np.isin(sr, pr) & np.isin(sc, pc)
    return {"row": sr, "col": sc, "frac_at_predicted": float(ok.mean()),
            "predicted_pair": (pr, pc),
            "mode": (int(np.bincount(sr - shifts.min()).argmax() + shifts.min()),
                     int(np.bincount(sc - shifts.min()).argmax() + shifts.min()))}


def summarize(resid, scores, shifts, predicted: float):
    """Per-head argmax, the fraction landing on the predicted shift, residual power."""
    shifts = np.asarray(shifts)
    arg = shifts[np.argmax(scores, axis=-1)]
    flat = arg.reshape(-1)
    hist = {int(s): int((flat == s).sum()) for s in shifts}
    # The shift test only resolves whole patches, but the predicted rate need not be
    # a whole number -- decoy8 predicts +2.53.  Score against the bracketing pair in
    # that case, and widen the chance rate to match, rather than rounding and then
    # calling a correct answer a miss.
    p = float(predicted)
    pred_set = bracket_ints(p)
    return {
        "predicted_shift": p,
        "predicted_ints": pred_set,
        "argmax": arg,
        "hist": hist,
        "frac_at_predicted": float(np.isin(flat, pred_set).mean()),
        "frac_expected_by_chance": len(pred_set) / len(shifts),
        "median_argmax": float(np.median(flat)),
        "mean_score_curve": scores.reshape(-1, len(shifts)).mean(axis=0),
        "resid_power": float((resid ** 2).mean()),
    }


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
def square_image(image, side: int):
    """Force one grid for every case so bucket means can be pooled across images.

    Aspect ratio is not preserved.  That is deliberate: resampling maps onto a
    common frame afterwards would rescale the very spatial period being measured
    and cancel it across differently-shaped images.  Distorting the input instead
    keeps the patch grid exact and only changes what the model is looking at.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > side:                       # cap the long side first, aspect kept,
        f = side / max(w, h)                   # so the square resize is not an upsample
        image = image.resize((max(1, round(w * f)), max(1, round(h * f))), 2)
    return image.resize((side, side), 2)  # BICUBIC


class EagerAttentionTap:
    """Forward hook on every Qwen3VLTextAttention that replays its own module in eager.

    sdpa throws the softmax weights away, so each layer is re-run in eager,
    reduced into the accumulator, and dropped.  Peak memory is one layer's
    [H, P, P] rather than all of them at once.
    """

    def __init__(self, model, acc: BinAccumulator, img_cols, prompt_len: int):
        self.acc, self.img_cols, self.prompt_len = acc, img_cols, prompt_len
        self.armed = False
        self.mask = None
        self.bins = None
        self.keep = None
        self.eps = 1e-20
        self._reentry = False
        self.handles, layers = [], []
        mods = [m for m in model.modules()
                if type(m).__name__ in TEXT_ATTENTION_CLASSES and getattr(m, "layer_idx", None) is not None]
        if not mods:
            raise RuntimeError(
                f"no text-tower attention modules found; looked for {TEXT_ATTENTION_CLASSES}. "
                "Add this model's class there if it has the same forward contract.")
        self.layers = sorted(int(m.layer_idx) for m in mods)
        self.row_of = {l: i for i, l in enumerate(self.layers)}
        for m in mods:
            self.handles.append(
                m.register_forward_hook(self._make(int(m.layer_idx)), with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def arm(self, mask, bins, keep):
        self.mask, self.bins, self.keep, self.armed = mask, bins, keep, True

    def disarm(self):
        self.armed = False
        self.mask = self.bins = self.keep = None

    def _make(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            if self._reentry or not self.armed:
                return None
            self._reentry = True
            kw = dict(kwargs)
            kw["attention_mask"] = self.mask     # sdpa may have passed None; eager needs it
            kw["past_key_values"] = None         # never double-update the KV cache
            kw["use_cache"] = False
            prev = module.config._attn_implementation
            module.config._attn_implementation = "eager"
            try:
                attn = module(*args, **kw)[1]
            finally:
                module.config._attn_implementation = prev
                self._reentry = False
            a = attn[0][:, self.prompt_len:, :][:, :, self.img_cols]   # [H, T, P]
            a = a[:, self.keep, :].float()
            self.acc.clamped += (a < self.eps).sum()
            x = a.clamp_min(self.eps).log()
            x -= x.mean(-1, keepdim=True)        # log softmax minus logZ == the raw logit
            self.acc.add_layer(self.row_of[layer_idx], x, self.bins)
            del a, x, attn
            return None
        return hook


def token_distances(model, case, prompt_len: int, image_token_id: int):
    """d = p - s for every completion token, from the model's own position ids.

    s is the image anchor (the t/h/w value of the first patch, where r = c = 0);
    text tokens carry t = h = w so axis 0 is the whole story for them.
    """
    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "get_rope_index"):
        raise SystemExit("model has no get_rope_index; is this a Qwen*-VL checkpoint?")
    if case.get("mm_token_type_ids") is None:
        raise SystemExit("no mm_token_type_ids: M-RoPE position ids cannot be rebuilt, "
                         "and d would be wrong rather than missing")
    pos, _ = inner.get_rope_index(
        case["input_ids"],
        case["mm_token_type_ids"],
        image_grid_thw=case.get("image_grid_thw"),
        video_grid_thw=None,
        attention_mask=case.get("attention_mask"),
    )
    img_cols = (case["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if img_cols.numel() == 0:
        raise SystemExit("no image tokens in the teacher-forced case")
    s = pos[0, 0, img_cols[0]]
    return (pos[0, 0, prompt_len:] - s).long(), img_cols


@torch.no_grad()
def scan(args, device):
    out_dir = Path(args.out_dir)
    shard_path = out_dir / "scan" / f"shard{args.shard:02d}.npz"
    if shard_path.exists() and not args.overwrite:
        print(f"[skip] {shard_path} exists")
        return
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dataset:
        raise SystemExit("--dataset is required (or set ROPE_PHASE_DATASET)")
    rows = load_samples(args.dataset, args.n_samples, args.seed)
    rows = rows[args.shard::args.num_shards]
    if args.max_cases:
        rows = rows[: args.max_cases]
    print(f"[scan] shard {args.shard}/{args.num_shards}: {len(rows)} cases", flush=True)

    processor, model = load_model(args.base_model, args.adapter or None, device)
    tcfg = model.config.text_config
    binnings = default_binnings(tcfg, args.decoy_theta)
    for b in binnings:
        print(f"       {b.name:7s} kind={b.kind:6s} theta={b.theta:.4f} "
              f"nbins={b.nbins:2d} axis={b.axis} predicted shift={b.predicted_shift:+.3f}")

    image_token_id = int(getattr(model.config, "image_token_id", None) or IMAGE_TOKEN_ID)
    torch.manual_seed(args.seed + args.shard)
    rng = np.random.default_rng(args.seed + 1000 * args.shard)

    acc = tap = None
    gh = gw = None
    n_heads = tcfg.num_attention_heads
    kept_cases = 0

    for ci, row in enumerate(rows):
        image = square_image(row["image"], args.image_side)
        try:
            inputs, prompt_len, comp_ids = generate(
                processor, model, image, row["question"],
                args.max_new_tokens, args.temperature, device)
        except Exception as exc:                      # OOM or a bad sample: skip, keep going
            print(f"[warn] case {ci} generate failed: {exc}", flush=True)
            continue
        if len(comp_ids) < args.min_tokens:
            continue

        grid = inputs["image_grid_thw"][0].tolist()
        merge = int(model.config.vision_config.spatial_merge_size)
        g_h, g_w = grid[1] // merge, grid[2] // merge
        if gh is None:
            gh, gw = g_h, g_w
            # The tap discovers the layers, so build it first and size the
            # accumulator from what it actually found rather than from the config.
            tap = EagerAttentionTap(model, None, None, 0)
            acc = BinAccumulator(binnings, len(tap.layers), n_heads, gh * gw, device)
            tap.acc = acc
        elif (g_h, g_w) != (gh, gw):
            print(f"[warn] case {ci} grid {(g_h, g_w)} != {(gh, gw)}; skipped", flush=True)
            continue

        case = teacher_forced_case(inputs, comp_ids, device)
        d, img_cols = token_distances(model, case, prompt_len, image_token_id)
        if img_cols.numel() != gh * gw:
            print(f"[warn] case {ci} has {img_cols.numel()} image tokens, "
                  f"grid says {gh * gw}; skipped", flush=True)
            continue

        keep = (d >= args.d_min) & (d <= args.d_max)
        if int(keep.sum()) < args.min_tokens:
            continue
        d_keep = d[keep]
        bins = make_bins(d_keep, binnings, rng)

        seq = case["input_ids"].shape[-1]
        mdtype = next(model.parameters()).dtype
        add = torch.zeros(seq, seq, dtype=mdtype, device=device)
        add.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device),
                                    diagonal=1), torch.finfo(mdtype).min)

        tap.img_cols, tap.prompt_len = img_cols, prompt_len
        tap.arm(add[None, None], bins, keep)
        try:
            model(**case)
        finally:
            tap.disarm()
            del add
        acc.add_case(bins, d_keep)
        kept_cases += 1
        if (ci + 1) % args.log_every == 0:
            print(f"[scan] {ci + 1}/{len(rows)} cases, {acc.ntok} tokens kept", flush=True)

    if tap is not None:
        tap.close()
    if acc is None or acc.ntok == 0:
        raise SystemExit("no usable cases in this shard")

    meta = {
        "gh": gh, "gw": gw, "n_heads": n_heads, "layers": tap.layers,
        "ntok": acc.ntok, "ncase": acc.ncase, "nclamped": acc.nclamped,
        "shard": args.shard, "num_shards": args.num_shards,
        "d_observed": [acc.dmin, acc.dmax],
        "n_samples": args.n_samples, "seed": args.seed,
        "image_side": args.image_side, "d_min": args.d_min, "d_max": args.d_max,
        "base_model": args.base_model, "adapter": args.adapter or "",
        "dataset": args.dataset,
        "binnings": [b.__dict__ for b in binnings],
    }
    np.savez(shard_path, **acc.to_npz(meta))
    print(f"[scan] wrote {shard_path}  cases={kept_cases} tokens={acc.ntok} "
          f"clamped={acc.nclamped}", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def load_scan(out_dir: Path):
    shards = sorted((out_dir / "scan").glob("shard*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}/scan -- run --stage scan first")
    sums, counts, sumsq, meta = {}, {}, None, None
    ntok = ncase = nclamped = 0
    seen_shards = {}
    for p in shards:
        z = np.load(p, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        if meta is None:
            meta = m
        elif (m["gh"], m["gw"]) != (meta["gh"], meta["gw"]):
            raise SystemExit(f"{p.name} has grid {(m['gh'], m['gw'])}, expected "
                             f"{(meta['gh'], meta['gw'])} -- do not merge these")
        # A --gpus 1 run writes shard00 = every case; a later --gpus 8 run skips
        # shard00 as already done and adds seven shards that each re-cover part of
        # it.  Merging those double-counts most of the corpus and the report would
        # not look wrong.  Refuse instead.
        for key in ("num_shards", "n_samples", "seed", "dataset", "base_model", "image_side"):
            if key in m and key in meta and m[key] != meta[key]:
                raise SystemExit(
                    f"{p.name} was scanned with {key}={m[key]!r} but an earlier shard "
                    f"used {key}={meta[key]!r}. These cover overlapping cases; merging "
                    f"them would double-count. Delete {p.parent} and rescan with one "
                    f"consistent setting.")
        sid = m.get("shard")
        if sid is not None and sid in seen_shards:
            raise SystemExit(f"{p.name} and {seen_shards[sid]} are both shard {sid}")
        seen_shards[sid] = p.name
        ntok += m["ntok"]
        ncase += m["ncase"]
        nclamped += m.get("nclamped", 0)
        for k in z.files:
            if k.startswith("sum::"):
                name = k.split("::", 1)[1]
                sums[name] = sums.get(name, 0) + z[k]
            elif k.startswith("count::"):
                name = k.split("::", 1)[1]
                counts[name] = counts.get(name, 0) + z[k]
        sumsq = z["sumsq"] if sumsq is None else sumsq + z["sumsq"]
    expected = meta.get("num_shards", len(shards))
    meta.update(ntok=ntok, ncase=ncase, nclamped=nclamped,
                n_shards=len(shards), shards_expected=expected)
    if len(shards) != expected:
        print(f"[report] WARNING: {len(shards)} of {expected} shards present -- this is "
              f"a partial scan, not the full corpus", file=sys.stderr)
    return sums, counts, sumsq, meta


def ascii_profile(resid_head: np.ndarray, axis: str, width: int = 60) -> list[str]:
    """One row per bucket: the residual collapsed onto the shift axis, as a ramp.

    Reading it: the bright band should step one column to the right per line.
    """
    prof = resid_head.mean(axis=2 if axis == "row" else 1)      # [B, n]
    lo, hi = prof.min(), prof.max()
    if hi <= lo:
        return ["(flat)"]
    chars = " .:-=+*#%@"
    out, track = [], None
    for k, row in enumerate(prof):
        idx = np.clip(((row - lo) / (hi - lo) * (len(chars) - 1)).astype(int), 0, len(chars) - 1)
        s = "".join(chars[i] for i in idx)
        # A repeating stripe puts several equal crests on the grid, so a plain argmax
        # hops between them and hides the very march it is meant to show.  Follow one
        # crest instead: nearest local peak to where the last bucket left it.
        if track is None:
            track = int(np.argmax(row))
        else:
            w = 3
            lo_i, hi_i = max(0, track - w), min(len(row), track + w + 1)
            track = lo_i + int(np.argmax(row[lo_i:hi_i]))
        out.append(f"  bucket {k}: |{s}|  crest@{track:2d}")
    return out


def report(args):
    out_dir = Path(args.out_dir)
    sums, counts, sumsq, meta = load_scan(out_dir)
    gh, gw = meta["gh"], meta["gw"]
    binnings = [Binning(**b) for b in meta["binnings"]]
    var_tok = float((sumsq / max(meta["ntok"], 1)).mean())

    lines = []
    P = lines.append
    P("=" * 78)
    P("E0 -- RoPE phase lock-in test")
    P("=" * 78)
    P(f"model    : {meta['base_model']}" + (f" + {meta['adapter']}" if meta["adapter"] else ""))
    P(f"data     : {meta['dataset']}")
    P(f"scan     : {meta['n_shards']} shards, {meta['ncase']} cases, {meta['ntok']} tokens, "
      f"{len(meta['layers'])} layers x {meta['n_heads']} heads = "
      f"{len(meta['layers']) * meta['n_heads']} heads total")
    P(f"grid     : {gh} x {gw} patches (image_side={meta['image_side']}), "
      f"d in [{meta['d_min']}, {meta['d_max']}]")
    P(f"per-token variance of the mean-centred log-attention: {var_tok:.4f}")
    if meta["nclamped"]:
        P(f"NOTE {meta['nclamped']} attention entries hit the 1e-20 floor "
          f"({100 * meta['nclamped'] / max(meta['ntok'] * gh * gw * len(meta['layers']) * meta['n_heads'], 1):.2g}% "
          f"of entries) -- bf16 underflow, see the docstring")
    P("")

    shifts = list(range(-args.max_shift, args.max_shift + 1))
    dlo, dhi = (meta.get("d_observed") or [None, None])
    dspan = (dhi - dlo) if (dlo is not None and dhi is not None) else None
    results, degenerate = {}, []
    for b in binnings:
        empty = int((counts[b.name] <= 0).sum())
        if empty:
            turn = b.theta * dspan if dspan else float("nan")
            degenerate.append(
                f"{b.name}: {empty}/{b.nbins} buckets are empty. theta={b.theta:.3g} turns "
                f"{turn:.3g} rad over the observed d range ({dlo:.0f}..{dhi:.0f}), i.e. "
                f"{turn / TWO_PI:.3g} of a cycle, so every token falls in the same phase "
                f"bucket. This channel is too slow to separate the data -- which is itself "
                f"the measurement, not a failure.")
            continue
        resid = bucket_residuals(sums[b.name], counts[b.name], gh, gw)
        sc = shift_scores(resid, b.axis, shifts)
        results[b.name] = (b, resid, summarize(resid, sc, shifts, b.predicted_shift))
    if degenerate:
        P("--- degenerate binnings (skipped)")
        for line in degenerate:
            P("    " + line)
        P("")
    if not results:
        raise SystemExit("every binning was degenerate; nothing to report")

    floor = results["perm8"][2]["resid_power"] if "perm8" in results else float("nan")
    if "perm8" not in results:
        P("NOTE perm8 is absent, so /floor and null% are undefined for this run")

    # 1/#shifts is a poor reference: the null's argmax is edge-heavy, because bucket
    # residuals sum to zero across buckets and so are mildly anti-correlated at zero
    # shift, and because the crop leaves fewer, noisier elements at large shifts.
    # Quote what perm8 actually does at each binning's predicted shift instead.
    null_arg = results["perm8"][2]["argmax"].reshape(-1) if "perm8" in results else None

    def null_rate(pred_ints):
        if null_arg is None:
            return float("nan")
        return float(np.isin(null_arg, pred_ints).mean())

    P(f"{'binning':8s} {'axis':4s} {'theta':>7s} {'B':>3s} {'pred':>6s} {'median':>7s} "
      f"{'%@pred':>7s} {'null%':>7s} {'power':>9s} {'/floor':>7s}")
    P("-" * 78)
    for name, (b, resid, s) in results.items():
        P(f"{name:8s} {b.axis:4s} {b.theta:7.4f} {b.nbins:3d} "
          f"{b.predicted_shift:+6.2f} {s['median_argmax']:+7.1f} "
          f"{100 * s['frac_at_predicted']:6.1f}% {100 * null_rate(s['predicted_ints']):6.1f}% "
          f"{s['resid_power']:9.2e} {s['resid_power'] / floor:7.2f}")
    P("")

    # The diagonal test.  d enters as both d-r and d-c, so the overlay translates in
    # BOTH axes at once; sliding rows alone leaves the column component unmatched and
    # drags the best row-only shift toward zero.
    shifts2 = list(range(-3, 4))
    res2 = {n: summarize_2d(shift_scores_2d(r, shifts2), shifts2, b_.predicted_shift, b_.axis)
            for n, (b_, r, _s) in results.items()}

    def null_rate_2d(pred, axis):
        if "perm8" not in res2:
            return float("nan")
        pr, pc = predicted_pair_2d(pred, axis)
        n = res2["perm8"]
        return float((np.isin(n["row"], pr) & np.isin(n["col"], pc)).mean())

    P("--- 2-D shift test: rows and columns slid jointly")
    P("    A bucketing on one channel only makes THAT channel coherent inside a bucket,")
    P("    so the expected signature is a march along its own axis and none on the other.")
    P(f"    {'binning':8s} {'predicted':>12s} {'mode(row,col)':>14s} {'%@pred':>8s} {'null%':>7s}")
    for name, (b, _r, _s) in results.items():
        r2 = res2[name]
        pr, pc = r2["predicted_pair"]
        lbl = f"({'/'.join(map(str, pr))},{'/'.join(map(str, pc))})"
        P(f"    {name:8s} {lbl:>12s} "
          f"{str(r2['mode']):>14s} {100 * r2['frac_at_predicted']:7.1f}% "
          f"{100 * null_rate_2d(b.predicted_shift, b.axis):6.1f}%")
    P("")
    P("pred   = shift in patches per bucket predicted by 2*pi/(theta*B), no fitting")
    P("null%  = what perm8, the matched control, puts at that same shift -- the")
    P("         empirical false-positive rate, which is what %@pred must beat")
    P("median = median over heads of the shift that best aligns bucket k+1 with bucket k")
    P("%@pred = fraction of heads whose best shift is the predicted one; when the")
    P("         predicted rate is not a whole number of patches the two bracketing")
    P("         integers both count, and the chance column widens to match")
    P("/floor = residual power relative to perm8, the matched noise floor")
    P("")

    for name, (b, resid, s) in results.items():
        P(f"--- {name}: score curve averaged over heads "
          f"(predicted peak at {b.predicted_shift:+.2f})")
        P("      shift : " + " ".join(f"{v:+5d}" for v in shifts))
        P("      score : " + " ".join(f"{v:+5.2f}" for v in s["mean_score_curve"]))
        P("      heads : " + " ".join(f"{s['hist'][v]:5d}" for v in shifts))
        P("")

    # the strongest single head under the primary binning, drawn
    if args.show not in results:
        raise SystemExit(f"--show {args.show!r} is not one of {sorted(results)}")
    b, resid, s = results[args.show]
    power = (resid ** 2).mean(axis=(2, 3, 4))
    li, hi_ = np.unravel_index(int(np.argmax(power)), power.shape)
    P(f"--- {args.show}: strongest head = layer {meta['layers'][li]} head {hi_} "
      f"(residual power {power[li, hi_]:.2e}, best shift {s['argmax'][li, hi_]:+d})")
    P(f"    residual collapsed onto the {b.axis} axis, one line per bucket:")
    lines.extend(ascii_profile(resid[li, hi_], b.axis))
    P("")

    per_layer = {}
    for name, (b, resid, s) in results.items():
        per_layer[name] = np.isin(s["argmax"], s["predicted_ints"]).mean(axis=1)
    P("--- fraction of heads at the predicted shift, by layer")
    P("      layer : " + " ".join(f"{l:4d}" for l in meta["layers"]))
    for name in results:
        P(f"      {name:6s}: " + " ".join(f"{v:4.2f}" for v in per_layer[name]))
    P("")

    h8 = results.get("h8")
    if h8 is not None:
        share = (h8[2]["resid_power"] - floor) / var_tok if var_tok > 0 else float("nan")
        P(f"positional share of the profile: (power(h8) - power(perm8)) / var_token "
          f"= {share:.4f}")
        P("  i.e. the part of a token's image-attention profile that is explained by "
          "where the token")
        P("  sits relative to the image, rather than by content.")
    P("")
    P("READING IT: the hypothesis predicts %@pred well above chance for h8, w10 and h4,")
    P("with h4's peak at twice h8's, and perm8 flat at chance.  If every binning sits at")
    P("chance with a tight noise floor, the language model is not using M-RoPE's spatial")
    P("channels to select image patches -- which is the other horn, and also a result.")

    text = "\n".join(lines)
    print(text)
    (out_dir / "report.txt").write_text(text + "\n")
    np.savez_compressed(out_dir / "residuals.npz",
                        **{f"resid::{n}": r for n, (_, r, _) in results.items()},
                        layers=np.asarray(meta["layers"]))
    print(f"\n[report] wrote {out_dir / 'report.txt'} and {out_dir / 'residuals.npz'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["scan", "report"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--max-cases", type=int, default=0, help="cap per shard, 0 = no cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    # stimulus / capture
    ap.add_argument("--image-side", type=int, default=768,
                    help="square side in pixels; 768 -> a 24x24 patch grid, three "
                         "vertical periods of the fastest H channel")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--d-min", type=int, default=0)
    ap.add_argument("--d-max", type=int, default=10 ** 9)
    ap.add_argument("--decoy-theta", type=float, default=0.31,
                    help="a frequency that is not an M-RoPE channel; the same law "
                         "predicts its (different) rate")
    ap.add_argument("--log-every", type=int, default=10)
    # report
    ap.add_argument("--max-shift", type=int, default=4)
    ap.add_argument("--show", default="h8", help="binning to draw the ASCII profile for")
    args = ap.parse_args()

    if args.stage == "scan":
        scan(args, args.device)
    else:
        report(args)


if __name__ == "__main__":
    main()
