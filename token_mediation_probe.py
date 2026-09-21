#!/usr/bin/env python
"""Does the token the LLM leans on carry its own pixels, or a stamp from elsewhere?

`docs/sink-location-by-image-type.md` §16 established that the border mark rides inside
the patch vector: permuting which embedding sits in which grid slot moves the attention,
and the peak follows the embedding. That is a claim about WHERE the mark lives. It says
nothing about whether the marked token's CONTENT came from the pixels underneath it.

This probe asks that, by corrupting one grid cell at two different stages and comparing:

    after   the encoder runs on the clean picture; one LLM-facing row (and its three
            DeepStack features) is then replaced by the row a DONOR picture produced for
            the same slot. Position ids, token count, grid and prompt are untouched, and
            the token is still there to be attended to -- only its content changed.

    before  the same cell's 32x32 pixels are destroyed and the encoder re-run. Every row
            it emits may move, because the tower has full attention.

Both are the same corruption at two stages, so the pair is a mediation decomposition:

    E_after / E_before  ~ 1   the cell's information really does live in its own token
                        ~ 0   it does not -- the content went somewhere else
                        > 1   the token carries MORE than its pixels: a register

Why not the `-inf` attention knockout of arXiv:2411.17491 as the primary instrument: at a
cell that is absorbing attention mass, masking it renormalises the softmax over every
other token, and a large effect then cannot be told apart from "information was removed".
Replacing the content holds the attention target in place. The knockout is kept as a
third arm (`--with-ko`) because it is the published construction and the comparison
between it and `swap` is itself informative -- but it is not what the ratio is built on.

WHICH TOKEN. The peak is read per picture off the clean pass's own all-head patch map,
which this probe has to compute anyway, and cell (0,0) is a second named arm.

That was built expecting the two to differ. They do not. `docs/peak-location-results.md`
puts the peak in a literal corner only ~27% of the time, but that was 30 natural
photographs at one layer and two heads; on the all-head map over the 12-type corpus the
peak is cell (0,0) on 88.0% of pictures and on the ring on 97.3%. So `swap_peak` and
`swap_first` are the same cell most of the time and agree to the fourth decimal. Both are
kept: the agreement is the measurement, and a corpus where they came apart would need
them separate.

THE METRIC IS NOT "DID THE ANSWER CHANGE". Every arm is scored by teacher-forcing the
CLEAN chain, so the trajectory is held fixed and the null is exactly zero -- no
self-consistency floor to subtract, no chaotic divergence to swamp the signal. What comes
back per arm is the per-token KL along the chain, the answer span's log-probability drop,
and the first position where the greedy argmax leaves the clean chain.

    python token_mediation_probe.py run --model M --out DIR [--limit N]
    python token_mediation_probe.py report --out DIR
    python token_mediation_probe.py selftest --model M
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_tm_overlap_probe", "overlap_probe.py")
SLP = _load_module("_tm_sink_location_probe", "sink_location_probe.py")
sys.path.insert(0, str(REPO))
import sink_location as SL  # noqa: E402
import vlm_family as VF  # noqa: E402

#: The corpus this project's border result was measured on. 1,800 pictures, 150 in each
#: of twelve image types, with a dev/test flag already fixed at 457/1,343. Reusing it is
#: what makes this causal arm comparable to the correlational one that motivated it.
DEFAULT_CORPUS = "outputs/sink_location/coldstart"

#: Arms. `kind` picks the machinery, `where` picks the cell.
ARMS = (
    ("swap_peak", "swap", "peak"),
    ("swap_first", "swap", "first"),
    ("swap_rand", "swap", "rand"),
    ("pix_peak", "pix", "peak"),
    ("pix_first", "pix", "first"),
    ("pix_rand", "pix", "rand"),
    # The norm-matched control. Every cell is pushed the SAME distance at every injection
    # point, so `swapn_peak - swapn_rand` cannot be a magnitude effect. If the plain
    # swap's gap survives here it is about direction -- what the vector says -- and if it
    # collapses, the peak was only ever a bigger vector.
    ("swapn_peak", "swapn", "peak"),
    ("swapn_first", "swapn", "first"),
    ("swapn_rand", "swapn", "rand"),
)
KO_ARMS = (
    ("ko_peak", "ko", "peak"),
    ("ko_rand", "ko", "rand"),
)


# ---------------------------------------------------------------------------
# the intervention on the encoder's output rows
# ---------------------------------------------------------------------------
class RowCapture:
    """Keep the LLM-facing rows a picture produced, DeepStack features included.

    `sink_location.VisionTap` keeps their norms; this keeps the tensors, because a swap
    needs the vectors themselves. The three DeepStack features are injected into the
    language model's early layers under the same row indexing as `pooler_output`, so a
    swap that moved only the pooled row would leave three quarters of the token's content
    in place and read as "replacing it did nothing".
    """

    def __init__(self, model, family=None):
        self.model = model
        self.family = family or VF.family_for(model)
        self.rows = None
        self.deepstack = None
        self._handles = []

    def _hook(self, module, args, out):
        pool = getattr(out, "pooler_output", None)
        if pool is None:
            raise RuntimeError(
                "this family's row module returns no pooler_output, so there is no row "
                "to capture; RowCapture is written against Qwen3-VL's tower")
        self.rows = pool.detach().clone()
        feats = getattr(out, "deepstack_features", None)
        self.deepstack = [f.detach().clone() for f in feats] if feats else []
        return out

    def install(self):
        self._handles.append(
            self.family.row_module(self.model).register_forward_hook(self._hook))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False


class RowSwap:
    """Overwrite specific LLM-facing rows with rows a donor picture produced.

    The counterpart of `sink_location.PatchPermute`: that one moves every row and leaves
    the multiset alone, this one changes one row's content and leaves every position
    alone. Both go through the same module output for the same reason -- the tower has
    already run, so nothing is recomputed and nothing but the vector is different.

    `magnitudes` turns this into the norm-matched control. Without it the substitution is
    literal, and the perturbation it applies is whatever `donor - target` happens to be
    at that cell -- so a cell whose vector is an outlier gets a bigger push, and "the
    peak token matters more" cannot be told from "the peak token is a bigger vector".
    With it, the row moves along the same direction but by a PRESCRIBED distance at every
    injection point, so two cells given the same magnitude differ only in direction.
    """

    def __init__(self, model, idx, donor_rows, donor_deepstack, family=None,
                 magnitudes=None):
        self.model = model
        self.family = family or VF.family_for(model)
        self.idx = list(idx)
        self.donor_rows = donor_rows
        self.donor_deepstack = donor_deepstack
        #: None, or one magnitude per injection point: [pool, ds0, ds1, ds2].
        self.magnitudes = magnitudes
        self.applied = 0
        self._handles = []

    @staticmethod
    def _move(target, donor, m):
        """target + m * unit(donor - target), or the literal donor row when m is None.

        In float32, then cast back. These rows are bfloat16, which carries about three
        decimal digits: computing `||donor - target||` in it makes the m/n ratio wrong by
        ~1% even when m IS that norm, so the matched arm did not reproduce the literal
        swap and the selftest's reconstruction check failed by a fifth of the signal.
        """
        if m is None:
            return donor.to(target.dtype)
        t = target.float()
        delta = donor.float() - t
        n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (t + delta * (float(m) / n)).to(target.dtype)

    def _hook(self, module, args, out):
        pool = out.pooler_output
        if max(self.idx) >= int(pool.shape[0]):
            raise RuntimeError(
                f"swap index {max(self.idx)} past this picture's {int(pool.shape[0])} "
                "rows: the donor and the target disagree about the grid")
        import torch

        idx = torch.as_tensor(self.idx, device=pool.device, dtype=torch.long)
        mags = self.magnitudes
        pool[idx] = self._move(pool[idx], self.donor_rows[idx],
                               None if mags is None else mags[0])
        feats = getattr(out, "deepstack_features", None)
        if feats:
            if len(feats) != len(self.donor_deepstack):
                raise RuntimeError(
                    f"{len(feats)} DeepStack features on the target and "
                    f"{len(self.donor_deepstack)} on the donor")
            for k, (f, d) in enumerate(zip(feats, self.donor_deepstack)):
                f[idx] = self._move(f[idx], d[idx],
                                    None if mags is None else mags[k + 1])
        self.applied += 1
        return out

    def install(self):
        self._handles.append(
            self.family.row_module(self.model).register_forward_hook(self._hook))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False


# ---------------------------------------------------------------------------
# the intervention in pixel space
# ---------------------------------------------------------------------------
def snap_to_grid(image, patch_px=32):
    """Resize so the picture is an exact whole number of patches on both axes.

    Without this the processor's own resize sits between the cell index and the pixels,
    and a cell's 32x32 block lands a pixel or two off on some picture sizes. `Family.
    encoder_px` exists for the same reason and A10 cuts its blocks at exactly this size.
    Snapping first makes the processor's resize the identity, so a cell IS a block.
    """
    w, h = image.size
    nw = max(patch_px, int(round(w / patch_px)) * patch_px)
    nh = max(patch_px, int(round(h / patch_px)) * patch_px)
    if (nw, nh) == (w, h):
        return image
    from PIL import Image

    return image.resize((nw, nh), Image.BICUBIC)


def cell_rect(idx, gh, gw, patch_px=32):
    """Grid cell -> its pixel box on a snapped picture. Row-major, which is token order."""
    if not 0 <= idx < gh * gw:
        raise ValueError(f"cell {idx} outside a {gh}x{gw} grid")
    row, col = divmod(int(idx), gw)
    return (col * patch_px, row * patch_px, (col + 1) * patch_px, (row + 1) * patch_px)


def corrupt_cell(image, idx, gh, gw, mode="mean", patch_px=32, seed=0):
    """Destroy one cell's pixels. -> a new picture, same size.

    `mean` fills with the picture's own mean colour. It is the default because uniform
    noise is a salient out-of-distribution stimulus in its own right -- it can attract
    attention and be described -- which would inflate the before-arm for a reason that
    has nothing to do with removing information. `noise` and `shuffle` are kept so the
    conclusion can be checked against a corruption that fails differently.
    """
    from PIL import Image

    out = image.copy()
    box = cell_rect(idx, gh, gw, patch_px)
    arr = np.asarray(image, dtype=np.float32)
    if mode == "mean":
        fill = np.tile(arr.reshape(-1, arr.shape[-1]).mean(0),
                       (patch_px, patch_px, 1))
    elif mode == "noise":
        rng = np.random.default_rng(seed)
        fill = rng.integers(0, 256, size=(patch_px, patch_px, arr.shape[-1]))
    elif mode == "shuffle":
        rng = np.random.default_rng(seed)
        block = arr[box[1]:box[3], box[0]:box[2]].reshape(-1, arr.shape[-1])
        fill = block[rng.permutation(block.shape[0])].reshape(patch_px, patch_px, -1)
    else:
        raise ValueError(f"unknown corruption mode {mode!r}")
    out.paste(Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8)), box[:2])
    return out


# ---------------------------------------------------------------------------
# scoring -- the clean chain, held fixed
# ---------------------------------------------------------------------------
def answer_span(processor, comp):
    """The answer's offsets inside the completion. -> (lo, hi) half-open.

    After `</think>` when the chain has one, and the boxed span inside that when it has
    one of those too. A plain instruct model with no tags answers from the first token,
    and the whole completion is then the answer -- recorded as `span` so the two are
    never pooled.
    """
    text = processor.tokenizer.decode(comp, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
    marker = text.find("</think>")
    if marker < 0:
        return 0, len(comp), "whole_completion"
    out = processor.tokenizer([text])
    tok = out.char_to_token(0, marker)
    if tok is None:
        return 0, len(comp), "whole_completion"
    # `out` re-tokenises the decoded text, so its indices are the reward's own skew --
    # the same approximation `sink_location_probe.observe_spans` documents. Clamped to
    # the ids this forward actually carries.
    lo = min(max(int(tok), 0), len(comp))
    return lo, len(comp), "post_think"


def clean_logprobs(model, case, prompt_len, n_comp):
    """log p over the vocabulary at every position that predicts a completion token.

    -> ([n_comp, V] float32, [n_comp] argmax) on the model's device. Position P+j-1
    predicts comp[j], so the slice starts one before the completion and is the same
    length as it.

    The argmax comes back because divergence has to be measured against THIS pass, not
    against the ids `generate` produced. Greedy decoding runs through the fused cached
    kernel and this forward recomputes the whole sequence, so the two disagree on a few
    positions by numerics alone -- and an identity intervention would then be scored as
    having diverged. Against the clean teacher-forced argmax the identity is exactly zero
    divergences, which is the property the whole design rests on.
    """
    import torch

    with torch.no_grad():
        out = model(**case, use_cache=False)
    lg = out.logits[0, prompt_len - 1: prompt_len - 1 + n_comp].float()
    lp = torch.log_softmax(lg, dim=-1)
    return lp, lp.argmax(-1)


def score_against(model, case, prompt_len, comp, clean_lp, clean_argmax):
    """One intervened forward, scored against the clean chain. -> metrics dict.

    Every quantity here is zero when the intervention is the identity, which is what
    makes the null exact and the floor unnecessary.
    """
    import torch

    n = len(comp)
    with torch.no_grad():
        out = model(**case, use_cache=False)
    lg = out.logits[0, prompt_len - 1: prompt_len - 1 + n].float()
    lp = torch.log_softmax(lg, dim=-1)
    p_clean = clean_lp.exp()
    kl = (p_clean * (clean_lp - lp)).sum(-1)                    # [n]
    ids = torch.as_tensor(comp, device=lp.device, dtype=torch.long)
    d_lp = (clean_lp.gather(-1, ids[:, None]) - lp.gather(-1, ids[:, None]))[:, 0]
    diverge = (lp.argmax(-1) != clean_argmax).nonzero()
    return {
        "kl": kl.cpu().numpy(),
        "dlogp": d_lp.cpu().numpy(),
        "first_diverge": int(diverge[0, 0]) if diverge.numel() else -1,
        "n_comp": n,
    }


def summarise(m, lo, hi):
    """Per-arm scalars. Log-space for the pooled ones, because KL is heavy-tailed.

    A cell's mean raw KL is set by whichever sample happened to spike; the mean of
    log(KL) is not, and it is what every comparison in the report is taken on.
    """
    kl = np.asarray(m["kl"], dtype=np.float64)
    # KL is non-negative in exact arithmetic and a hair negative in fp32 whenever the
    # intervention did nothing at that position. Unclamped, a single such position makes
    # log() NaN and drops the whole sample from every pooled statistic -- and the samples
    # it drops are exactly the ones where the arm had no effect, which would bias every
    # mean upward. `kl_neg` keeps the clamp visible instead of silent.
    neg = int((kl < 0).sum())
    kl = np.maximum(kl, 0.0)
    ans = kl[lo:hi] if hi > lo else kl[:0]
    eps = 1e-12
    return {
        "kl_neg": neg,
        "kl_mean": float(kl.mean()),
        "kl_logmean": float(np.log(kl + eps).mean()),
        "kl_median": float(np.median(kl)),
        "kl_max": float(kl.max()),
        # Cheap insurance: a pooled statistic that turns out to be the wrong one can be
        # recomputed from these instead of from a second run of the whole grid.
        "kl_q": [float(q) for q in np.percentile(kl, [10, 25, 50, 75, 90, 99])],
        "kl_answer": float(ans.mean()) if ans.size else float("nan"),
        "kl_answer_logmean": (float(np.log(ans + eps).mean()) if ans.size
                              else float("nan")),
        "dlogp_answer": (float(np.asarray(m["dlogp"])[lo:hi].sum()) if hi > lo
                         else float("nan")),
        "first_diverge": m["first_diverge"],
        "n_comp": m["n_comp"],
    }


# ---------------------------------------------------------------------------
# the peak cell
# ---------------------------------------------------------------------------
def peak_cell(res, gh, gw, min_mass=0.002):
    """The all-head patch map's argmax. -> (idx, map) or (None, None).

    The same pooled map the published tables and heatmaps are built on, so "the token
    the language model leans on" means here exactly what it means there.
    """
    q = None
    for name in ("generated", "image", "text", "all"):
        if res.get(name):
            q = res[name]
            break
    if q is None:
        for name, val in res.items():
            if isinstance(val, dict) and "col_sum" in val:
                q = val
                break
    if q is None:
        return None, None
    m = SL.pooled_patch_map(q["col_sum"], q["row_total"], q["n_rows"],
                            min_mass=min_mass, col_null=q.get("col_null"))
    m = np.asarray(m, dtype=np.float64)[: gh * gw]
    if m.size != gh * gw or not np.isfinite(m).any():
        return None, None
    return int(np.nanargmax(m)), m


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def pick_cells(peak, gh, gw, rng):
    """peak / first / a uniformly random OTHER cell. -> dict name -> idx.

    The random cell is the null the whole experiment rests on: without it, "ablating the
    peak hurts" cannot be told from "ablating anything hurts". It excludes the peak and
    cell 0 so the three arms never silently measure the same token.
    """
    n = gh * gw
    banned = {int(peak), 0}
    choices = [i for i in range(n) if i not in banned]
    return {"peak": int(peak), "first": 0,
            "rand": int(rng.choice(choices)) if choices else int(peak)}


def donor_for(row, by_grid, grid, rng):
    """A picture with the SAME grid, so a cell index means the same slot in both."""
    pool = [k for k in by_grid.get(grid, []) if k["key"] != row["key"]]
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


def _corpus(args):
    """-> (rows, image root, donor pool). Shared by `run` and `map`.

    The donor pool is the WHOLE corpus, not the sampled subset. A donor has to have the
    target's grid so a cell index means the same slot in both, and this corpus has 99
    distinct grids -- drawing donors from a 150-picture sample alone left 28 of them with
    no partner and no swap arm at all.
    """
    corpus = Path(args.corpus)
    rows = SLP.read_manifest(corpus.parent if corpus.name == "corpus" else corpus,
                             args.types)
    if args.split in ("dev", "test"):
        want = args.split == "dev"
        rows = [r for r in rows if bool(r["dev"]) is want]
    img_root = corpus / "corpus" / "images"
    if not img_root.exists():
        img_root = corpus / "images"
    return rows, img_root, list(rows)


def _census(fam, processor, donor_pool, img_root, cache=None):
    """Grid per picture, and the pictures grouped by grid. -> (by_grid, census).

    Before the model runs: the grid is a property of the picture's size alone. Cached to
    disk because every shard needs the WHOLE corpus (a donor can come from outside the
    shard) and four shards each opening and resizing 1,800 pictures is four times the
    CPU for one answer.
    """
    from PIL import Image

    by_grid, census = {}, {}
    if cache is not None and Path(cache).exists():
        census = {k: tuple(v) for k, v in json.loads(Path(cache).read_text()).items()}
    for r in donor_pool:
        g = census.get(r["key"])
        if g is None:
            im = snap_to_grid(PROBE.prepare_image(
                Image.open(img_root / Path(r["image"]).name).convert("RGB")))
            g = fam.grid_of(processor, im)
            census[r["key"]] = g
        by_grid.setdefault(tuple(g), []).append(r)
    if cache is not None and not Path(cache).exists():
        tmp = Path(f"{cache}.{os.getpid()}")
        tmp.write_text(json.dumps({k: list(v) for k, v in census.items()}))
        tmp.replace(cache)                       # atomic: shards race to write it
    print(f"[census] {len(by_grid)} distinct grids over {len(donor_pool)} pictures, "
          f"modal {max(by_grid, key=lambda k: len(by_grid[k]))}", flush=True)
    return by_grid, census


def run(args):
    import torch
    from PIL import Image

    rows, img_root, donor_pool = _corpus(args)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = out_dir / f"rows_shard{args.shard}.jsonl"
    done = set()
    if sink.exists() and not args.rebuild:
        for line in sink.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:                                    # noqa: BLE001
                pass
    todo = [r for i, r in enumerate(rows)
            if i % args.shards == args.shard and r["key"] not in done]
    print(f"[run] {len(todo)} pictures on shard {args.shard}/{args.shards}"
          f" ({len(done)} already done)", flush=True)
    if not todo:
        return

    # (processor, model), in that order -- `overlap_probe.load_model`'s own signature.
    processor, model = SLP.load_model(args.model, args.adapter, args.device,
                                      args.attn_impl)
    fam = SLP.load_family(model, processor, args.system_prompt)
    scan = SL.install(model, family=fam, want_key_stats=False)
    arms = list(ARMS) + (list(KO_ARMS) if args.with_ko else [])
    by_grid, census = _census(fam, processor, donor_pool, img_root,
                              cache=Path(args.corpus) / 'grid_census.json')

    n_ok = 0
    with open(sink, "a") as fh:
        for r in todo:
            try:
                rec = one_picture(model, processor, fam, scan, r, img_root, census,
                                  by_grid, arms, args)
            except Exception as exc:                             # noqa: BLE001
                print(f"[run] {r['key']}: FAILED {type(exc).__name__}: {str(exc)[:200]}",
                      flush=True)
                continue
            if rec is None:
                print(f"[run] {r['key']}: skipped", flush=True)
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"[run] {n_ok}/{len(todo)}", flush=True)
    scan.uninstall()
    print(f"[run] wrote {n_ok} rows to {sink}", flush=True)


def one_picture(model, processor, fam, scan, r, img_root, census, by_grid, arms, args):
    import torch
    from PIL import Image

    rng = random.Random(f"{args.seed}:{r['key']}")
    image = snap_to_grid(PROBE.prepare_image(
        Image.open(img_root / Path(r["image"]).name).convert("RGB")))
    question = r["question"]
    device = args.device

    # -- the clean pass: one generation, then one measured teacher-forced forward -----
    gen = SLP.generate_then_teacher_force(model, processor, [image], question, device,
                                          scan, args.max_new_tokens)
    if gen is None:
        return None
    inputs, prompt_len, comp = gen
    case = fam.teacher_forced_case(inputs, comp, device)
    scan.reset()
    scan.prompt_len_override = prompt_len
    # The target's own rows come out of this same forward -- the tower runs in it anyway,
    # so the norm-matched arm costs no extra encode.
    tcap = RowCapture(model, family=fam).install()
    try:
        with torch.no_grad():
            model(**case, use_cache=False)
    finally:
        scan.prompt_len_override = None
        tcap.uninstall()
    target_rows, target_deep = tcap.rows, tcap.deepstack
    res = scan.result()
    if res is None or not res["grids"]:
        return None
    _t, gh, gw = res["grids"][0]
    if (gh, gw) != census[r["key"]]:
        raise RuntimeError(f"grid census said {census[r['key']]} but the scan saw "
                           f"{(gh, gw)}: the snap is not the processor's resize")
    peak, pmap = peak_cell(res, gh, gw, args.min_mass)
    if peak is None:
        return None

    clean_lp, clean_am = clean_logprobs(model, case, prompt_len, len(comp))
    lo, hi, span_kind = answer_span(processor, comp)
    text = processor.tokenizer.decode(comp, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)

    cells = pick_cells(peak, gh, gw, rng)
    donor = donor_for(r, by_grid, (gh, gw), rng)
    rec = {
        "key": r["key"], "type": r["type"], "dev": bool(r["dev"]),
        "question": question, "grid": [gh, gw], "n_image_tokens": gh * gw,
        "peak": peak, "peak_row": peak // gw, "peak_col": peak % gw,
        "peak_mass": float(pmap[peak]), "cells": cells,
        "span": span_kind, "answer_lo": lo, "answer_hi": hi,
        "n_comp": len(comp), "chain": text,
        "donor": donor["key"] if donor else None,
        "arms": {},
    }

    # -- the donor's rows, captured once and reused by every swap arm ----------------
    donor_rows = donor_deep = None
    if donor is not None:
        dim = snap_to_grid(PROBE.prepare_image(
            Image.open(img_root / Path(donor["image"]).name).convert("RGB")))
        cap = RowCapture(model, family=fam).install()
        scan.paused = True
        try:
            d_inputs = SLP.build_inputs(fam, processor, [dim], question, device)
            with torch.no_grad():
                model(**d_inputs, use_cache=False)
            donor_rows, donor_deep = cap.rows, cap.deepstack
        finally:
            scan.paused = False
            cap.uninstall()
        if donor_rows is None or int(donor_rows.shape[0]) != gh * gw:
            donor_rows = donor_deep = None
            rec["donor"] = None

    # -- the matched magnitude, and the norms that motivate the control ---------------
    # One magnitude per injection point, the SMALLEST of the three cells' own
    # perturbations: scaling every cell down to a common distance never asks a row to
    # move further than the literal swap would have moved it.
    mags, rec["norms"] = matched_magnitudes(target_rows, target_deep, donor_rows,
                                            donor_deep, cells)

    scan.paused = True
    try:
        for name, kind, where in arms:
            idx = cells[where]
            try:
                m = run_arm(model, processor, fam, kind, idx, image, question, device,
                            gh, gw, comp, prompt_len, clean_lp, clean_am, donor_rows,
                            donor_deep, mags,
                            args, rng)
            except Exception as exc:                             # noqa: BLE001
                rec["arms"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                continue
            rec["arms"][name] = None if m is None else summarise(m, lo, hi)
            if m is not None:
                rec["arms"][name]["cell"] = idx
    finally:
        scan.paused = False
    return rec


def matched_magnitudes(target_rows, target_deep, donor_rows, donor_deep, cells):
    """One perturbation distance per injection point, common to every cell. -> (mags, diag)

    `mags[k]` is the smallest of the three cells' own `||donor - target||` at injection
    point k, so the matched arm never pushes a row further than the literal swap would.
    `diag` carries the per-cell row norms and delta norms, which is what says whether the
    peak was a magnitude outlier in the first place -- `sink_encoder_probe` reports the
    mark as an alignment effect rather than a magnitude one, and this is the check of
    that on the rows the language model actually consumes.
    """
    if target_rows is None or donor_rows is None:
        return None, None
    tgt = [target_rows] + list(target_deep or [])
    don = [donor_rows] + list(donor_deep or [])
    want = sorted({int(v) for v in cells.values()})
    mags, diag = [], {"row": {}, "delta": {}}
    for k, (t, d) in enumerate(zip(tgt, don)):
        per = {}
        for c in want:
            per[c] = float((d[c].float() - t[c].float()).norm())
        mags.append(min(per.values()))
        for name, c in cells.items():
            diag["row"].setdefault(name, []).append(float(t[int(c)].float().norm()))
            diag["delta"].setdefault(name, []).append(per[int(c)])
    return mags, diag


def run_arm(model, processor, fam, kind, idx, image, question, device, gh, gw, comp,
            prompt_len, clean_lp, clean_am, donor_rows, donor_deep, mags, args, rng,
            clean_case=None):
    """One arm, scored against the clean chain. -> metrics or None if not applicable.

    `clean_case` is the prompt++completion the CLEAN picture builds. A swap arm changes
    nothing upstream of the vision tower's output -- same picture, same prompt, same ids
    -- so rebuilding it per arm only re-runs the image processor. At three cells per
    picture that was invisible; over every cell of the grid it is most of the run, and
    with four shards sharing one node's CPUs it dominated: 280 s per picture against the
    30 s a single process needed.
    """
    import torch

    if kind in ("swap", "swapn"):
        if donor_rows is None or (kind == "swapn" and mags is None):
            return None
        if clean_case is not None:
            case = clean_case
        else:
            inputs = SLP.build_inputs(fam, processor, [image], question, device)
            case = fam.teacher_forced_case(inputs, comp, device)
        sw = RowSwap(model, [idx], donor_rows, donor_deep, family=fam,
                     magnitudes=mags if kind == "swapn" else None).install()
        try:
            m = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
        finally:
            sw.uninstall()
        if sw.applied == 0:
            raise RuntimeError("the swap hook never fired: the tower did not re-run, so "
                               "this arm measured the clean picture")
        return m

    if kind == "pix":
        dirty = corrupt_cell(image, idx, gh, gw, mode=args.corrupt,
                             seed=rng.randrange(1 << 30))
        inputs = SLP.build_inputs(fam, processor, [dirty], question, device)
        case = fam.teacher_forced_case(inputs, comp, device)
        return score_against(model, case, prompt_len, comp, clean_lp, clean_am)

    if kind == "ko":
        inputs = SLP.build_inputs(fam, processor, [image], question, device)
        case = fam.teacher_forced_case(inputs, comp, device)
        mask = ko_mask(fam, case, idx, device)
        if mask is None:
            return None
        case = dict(case)
        case["attention_mask"] = mask
        return score_against(model, case, prompt_len, comp, clean_lp, clean_am)

    raise ValueError(f"unknown arm kind {kind!r}")


def ko_mask(fam, case, idx, device):
    """Causal mask with one image column closed to every later position. -> [1,1,L,L].

    The published construction (arXiv:2411.17491 Eq. 3) blocks image->text only, which on
    a causal decoder leaves image->image->text open: with 36 layers and DeepStack feeding
    the same positions at layers 5, 11 and 17, the token's content reaches the text
    anyway. Blocking every later position is the ablation that construction is usually
    read as being.
    """
    import torch

    ids = case["input_ids"]
    L = int(ids.shape[1])
    run = (ids[0] == fam.image_token_id).nonzero().flatten()
    if run.numel() == 0 or idx >= int(run.numel()):
        return None
    col = int(run[idx])
    m = torch.full((L, L), torch.finfo(torch.float32).min, device=device)
    m = torch.triu(m, diagonal=1)
    m[col + 1:, col] = torch.finfo(torch.float32).min
    return m[None, None]


# ---------------------------------------------------------------------------
# the full map -- every cell, not three
# ---------------------------------------------------------------------------
def canonical_weights(gh, gw, g):
    """Area overlap between a gh x gw patch grid and a canonical g x g one. -> [gh*gw, g*g]

    This corpus has 99 distinct grids and the modal one covers 13.6% of pictures, so
    there is no common cell index to average over and a heatmap has to be accumulated in
    NORMALISED coordinates. Exact rectangle overlap, not nearest-neighbour: a 70-token
    picture and a 256-token one otherwise contribute at different effective resolutions
    and the map quietly becomes a map of picture size.

    Each row sums to that cell's area share of the picture, so summing a picture's whole
    map over the canonical grid conserves its total.
    """
    ry = np.linspace(0.0, 1.0, gh + 1)
    rx = np.linspace(0.0, 1.0, gw + 1)
    cy = np.linspace(0.0, 1.0, g + 1)
    cx = np.linspace(0.0, 1.0, g + 1)
    # overlap[i, j] on each axis independently, then the outer product per cell pair
    oy = np.clip(np.minimum(ry[1:, None], cy[None, 1:]) -
                 np.maximum(ry[:-1, None], cy[None, :-1]), 0, None)     # [gh, g]
    ox = np.clip(np.minimum(rx[1:, None], cx[None, 1:]) -
                 np.maximum(rx[:-1, None], cx[None, :-1]), 0, None)     # [gw, g]
    w = np.einsum("ia,jb->ijab", oy, ox).reshape(gh * gw, g * g)
    return w


def one_map(model, processor, fam, scan, r, img_root, census, by_grid, args):
    """Every cell of one picture, in three variants. -> record or None.

    `swap` is the literal substitution, `swapn` holds the perturbation distance fixed
    across every cell of the picture, and `pix` destroys the cell's pixels and re-runs
    the tower. The pair of swap maps is the point: ||row|| runs 28.9 at the peak against
    12.7 at a random cell, so the literal map is largely a picture of the norms and the
    matched one is what is left when that is taken out.
    """
    import torch
    from PIL import Image

    rng = random.Random(f"{args.seed}:map:{r['key']}")
    image = snap_to_grid(PROBE.prepare_image(
        Image.open(img_root / Path(r["image"]).name).convert("RGB")))
    question, device = r["question"], args.device

    gen = SLP.generate_then_teacher_force(model, processor, [image], question, device,
                                          scan, args.max_new_tokens)
    if gen is None:
        return None
    inputs, prompt_len, comp = gen
    case = fam.teacher_forced_case(inputs, comp, device)
    scan.reset()
    scan.prompt_len_override = prompt_len
    tcap = RowCapture(model, family=fam).install()
    try:
        with torch.no_grad():
            model(**case, use_cache=False)
    finally:
        scan.prompt_len_override = None
        tcap.uninstall()
    res = scan.result()
    if res is None or not res["grids"]:
        return None
    _t, gh, gw = res["grids"][0]
    n_cells = gh * gw
    peak, pmap = peak_cell(res, gh, gw, args.min_mass)
    clean_lp, clean_am = clean_logprobs(model, case, prompt_len, len(comp))
    lo, hi, span_kind = answer_span(processor, comp)

    donor = donor_for(r, by_grid, (gh, gw), rng)
    if donor is None:
        return None
    dim = snap_to_grid(PROBE.prepare_image(
        Image.open(img_root / Path(donor["image"]).name).convert("RGB")))
    cap = RowCapture(model, family=fam).install()
    scan.paused = True
    try:
        with torch.no_grad():
            model(**SLP.build_inputs(fam, processor, [dim], question, device),
                  use_cache=False)
        donor_rows, donor_deep = cap.rows, cap.deepstack
    finally:
        scan.paused = False
        cap.uninstall()
    if donor_rows is None or int(donor_rows.shape[0]) != n_cells:
        return None

    # One magnitude per injection point for the WHOLE picture -- the median cell's own
    # perturbation. The three-cell arms used the min of three, which is not defined here
    # and would anyway be set by whichever cell happened to be smallest.
    tgt = [tcap.rows] + list(tcap.deepstack or [])
    don = [donor_rows] + list(donor_deep or [])
    deltas = [(d[:n_cells].float() - t[:n_cells].float()).norm(dim=-1).cpu().numpy()
              for t, d in zip(tgt, don)]
    mags = [float(np.median(dn)) for dn in deltas]
    rownorm = tgt[0][:n_cells].float().norm(dim=-1).cpu().numpy()

    out = {v: [None] * n_cells for v in ("swap", "swapn", "pix")}
    scan.paused = True
    try:
        for c in range(n_cells):
            for variant in args.variants:
                try:
                    m = run_arm(model, processor, fam, variant,
                                c, image, question, device, gh, gw, comp, prompt_len,
                                clean_lp, clean_am, donor_rows, donor_deep, mags, args,
                                rng, clean_case=case)
                except Exception:                                # noqa: BLE001
                    m = None
                if m is not None:
                    out[variant][c] = round(summarise(m, lo, hi)["kl_logmean"], 5)
    finally:
        scan.paused = False

    return {
        "key": r["key"], "type": r["type"], "dev": bool(r["dev"]),
        "grid": [gh, gw], "n_cells": n_cells, "peak": peak,
        "peak_mass": None if peak is None else float(pmap[peak]),
        "span": span_kind, "n_comp": len(comp), "donor": donor["key"],
        "mags": mags, "row_norm": [round(float(x), 3) for x in rownorm],
        "delta_norm": [round(float(x), 3) for x in deltas[0]],
        "map": {v: out[v] for v in args.variants},
    }


def run_map(args):
    from PIL import Image

    rows, img_root, donor_pool = _corpus(args)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = out_dir / f"map_shard{args.shard}.jsonl"
    done = set()
    if sink.exists() and not args.rebuild:
        for line in sink.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:                                    # noqa: BLE001
                pass
    todo = [r for i, r in enumerate(rows)
            if i % args.shards == args.shard and r["key"] not in done]
    print(f"[map] {len(todo)} pictures on shard {args.shard}/{args.shards}, "
          f"variants {args.variants}", flush=True)
    if not todo:
        return

    processor, model = SLP.load_model(args.model, args.adapter, args.device,
                                      args.attn_impl)
    fam = SLP.load_family(model, processor, args.system_prompt)
    scan = SL.install(model, family=fam, want_key_stats=False)
    by_grid, census = _census(fam, processor, donor_pool, img_root,
                              cache=Path(args.corpus) / 'grid_census.json')

    n_ok = 0
    with open(sink, "a") as fh:
        for r in todo:
            try:
                rec = one_map(model, processor, fam, scan, r, img_root, census, by_grid,
                              args)
            except Exception as exc:                             # noqa: BLE001
                print(f"[map] {r['key']}: FAILED {type(exc).__name__}: {str(exc)[:200]}",
                      flush=True)
                continue
            if rec is None:
                print(f"[map] {r['key']}: skipped", flush=True)
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok += 1
            print(f"[map] {n_ok}/{len(todo)}  {rec['key']}  {rec['n_cells']} cells",
                  flush=True)
    scan.uninstall()
    print(f"[map] wrote {n_ok} rows to {sink}", flush=True)


def map_report(args):
    """Pool the per-picture maps onto one canonical grid and draw it."""
    recs = []
    for p in sorted(Path(args.out).glob("map_shard*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:                                    # noqa: BLE001
                pass
    if not recs:
        raise SystemExit(f"no map rows under {args.out}")
    g = args.canon
    variants = [v for v in ("swap", "swapn", "pix") if v in recs[0]["map"]]
    print(f"\n{len(recs)} pictures, {sum(r['n_cells'] for r in recs):,} cells, "
          f"canonical {g}x{g}\n")

    for variant in variants + ["row_norm"]:
        acc = np.zeros(g * g)
        wsum = np.zeros(g * g)
        for r in recs:
            gh, gw = r["grid"]
            vals = (r["map"][variant] if variant in r["map"] else r.get(variant))
            v = np.array([np.nan if x is None else x for x in vals], dtype=np.float64)
            if v.size != gh * gw or not np.isfinite(v).any():
                continue
            # Within-picture standardisation: without it a handful of pictures with big
            # overall effects set the map, and it becomes a map of which pictures are
            # fragile rather than of which positions matter.
            mu, sd = np.nanmean(v), np.nanstd(v)
            z = (v - mu) / (sd if sd > 1e-9 else 1.0)
            w = canonical_weights(gh, gw, g)
            ok = np.isfinite(z)
            acc += (z[ok, None] * w[ok]).sum(0)
            wsum += w[ok].sum(0)
        m = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan).reshape(g, g)
        print(f"--- {variant}  (within-picture z, + = this position matters more) ---")
        for row in m:
            print("  " + " ".join(f"{x:+5.2f}" if np.isfinite(x) else "    ."
                                  for x in row))
        ring = np.zeros((g, g), bool)
        ring[0], ring[-1], ring[:, 0], ring[:, -1] = True, True, True, True
        print(f"  ring {np.nanmean(m[ring]):+.3f}   interior {np.nanmean(m[~ring]):+.3f}"
              f"   corner(0,0) {m[0, 0]:+.3f}   max {np.nanmax(m):+.3f} at "
              f"{np.unravel_index(np.nanargmax(m), m.shape)}\n")
        np.save(Path(args.out) / f"map_{variant}_{g}x{g}.npy", m)
    print(f"maps written to {args.out}/map_*_{g}x{g}.npy")


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def read_rows(out_dir):
    rows = []
    for p in sorted(Path(out_dir).glob("rows_shard*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:                                    # noqa: BLE001
                pass
    return rows


def _paired(rows, a, b, field):
    """Values of `field` for two arms on the pictures where BOTH were measured."""
    xs, ys = [], []
    for r in rows:
        ra, rb = r["arms"].get(a), r["arms"].get(b)
        if not ra or not rb or "error" in ra or "error" in rb:
            continue
        va, vb = ra.get(field), rb.get(field)
        if va is None or vb is None or not (math.isfinite(va) and math.isfinite(vb)):
            continue
        xs.append(va)
        ys.append(vb)
    return np.asarray(xs), np.asarray(ys)


def _wilcoxon(x, y):
    """Paired sign statistic plus a bootstrap CI on the median difference.

    Not scipy: this repo's probes do not depend on it, and a rank test on a few hundred
    pairs is three lines. The reported quantity is the median paired difference and how
    often the first arm exceeds the second, which is what a rank test is testing anyway.
    """
    d = x - y
    if d.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(0)
    boot = np.array([np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(2000)])
    return {
        "n": int(d.size),
        "median_diff": float(np.median(d)),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "frac_gt": float((d > 0).mean()),
    }


def report(args):
    rows = read_rows(args.out)
    if not rows:
        raise SystemExit(f"no rows under {args.out}")
    field = args.field
    print(f"\n{len(rows)} pictures, metric = {field}\n")

    names = []
    for r in rows:
        for k in r["arms"]:
            if k not in names:
                names.append(k)
    print(f"{'arm':<12} {'n':>5} {'median':>10} {'mean':>10} {'p90':>10} {'err':>5}")
    for name in names:
        vals = [r["arms"][name][field] for r in rows
                if r["arms"].get(name) and "error" not in r["arms"][name]
                and r["arms"][name].get(field) is not None
                and math.isfinite(r["arms"][name][field])]
        errs = sum(1 for r in rows
                   if r["arms"].get(name) and "error" in r["arms"][name])
        if not vals:
            print(f"{name:<12} {0:>5} {'-':>10} {'-':>10} {'-':>10} {errs:>5}")
            continue
        v = np.asarray(vals)
        print(f"{name:<12} {v.size:>5} {np.median(v):>10.4f} {v.mean():>10.4f} "
              f"{np.percentile(v, 90):>10.4f} {errs:>5}")

    print("\nPaired contrasts (positive = first arm moves the model more)")
    for a, b in (("swap_peak", "swap_rand"), ("swap_first", "swap_rand"),
                 ("swap_peak", "swap_first"), ("pix_peak", "pix_rand"),
                 ("swap_peak", "pix_peak"), ("ko_peak", "swap_peak"),
                 # The control. If `swapn_peak - swapn_rand` holds up next to
                 # `swap_peak - swap_rand`, the gap is about what the vector says and not
                 # about how big it is.
                 ("swapn_peak", "swapn_rand"), ("swapn_first", "swapn_rand"),
                 ("swap_peak", "swapn_peak")):
        x, y = _paired(rows, a, b, field)
        s = _wilcoxon(x, y)
        if not s.get("n"):
            continue
        print(f"  {a:<11} - {b:<11} n={s['n']:<5} median {s['median_diff']:+.4f} "
              f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]  first bigger on "
              f"{100 * s['frac_gt']:.0f}%")

    # NOT on `field`. A through-origin slope is only a ratio on a ratio scale, and
    # `kl_logmean` is a log: every arm sits near -10.5, so regressing one on the other
    # returns ~1.0 by arithmetic whatever the arms did. The decomposition is taken on raw
    # mean KL, where "twice the effect" is twice the number.
    print("\nMediated fraction  E_after / E_before, through-origin slope on kl_mean")
    for where in ("peak", "first", "rand"):
        x, y = _paired(rows, f"swap_{where}", f"pix_{where}", "kl_mean")
        if x.size < 8:
            continue
        # Through-origin slope of swap on pix. Ratios per picture are unstable when the
        # denominator is near zero; the slope is not, and it is the quantity a mediation
        # decomposition actually names.
        slope = float((x * y).sum() / max((y * y).sum(), 1e-12))
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            i = rng.integers(0, x.size, x.size)
            boot.append((x[i] * y[i]).sum() / max((y[i] * y[i]).sum(), 1e-12))
        b = np.percentile(boot, [2.5, 97.5])
        print(f"  {where:<6} n={x.size:<5} slope {slope:6.3f} "
              f"[{b[0]:.3f}, {b[1]:.3f}]")

    # Is the peak a magnitude outlier at all? If its row norm and its delta norm sit on
    # top of the random cell's, the norm-matched arm was never going to change anything
    # and the control is confirming rather than rescuing the result.
    have = [r for r in rows if r.get("norms")]
    if have:
        print("\nRow and perturbation norms at the pooled injection point "
              f"(n={len(have)}, median)")
        print(f"  {'cell':<8}{'||row||':>10}{'||donor-row||':>16}")
        for name in ("peak", "first", "rand"):
            rw = [r["norms"]["row"][name][0] for r in have if name in r["norms"]["row"]]
            dl = [r["norms"]["delta"][name][0] for r in have
                  if name in r["norms"]["delta"]]
            if rw:
                print(f"  {name:<8}{np.median(rw):>10.2f}{np.median(dl):>16.2f}")

    print("\nWhere the peak sits")
    gh = np.array([r["grid"][0] for r in rows])
    gw = np.array([r["grid"][1] for r in rows])
    pr = np.array([r["peak_row"] for r in rows])
    pc = np.array([r["peak_col"] for r in rows])
    border = ((pr == 0) | (pc == 0) | (pr == gh - 1) | (pc == gw - 1))
    corner = (((pr == 0) | (pr == gh - 1)) & ((pc == 0) | (pc == gw - 1)))
    first = (pr == 0) & (pc == 0)
    print(f"  on the ring {100 * border.mean():.1f}%   in a corner "
          f"{100 * corner.mean():.1f}%   cell (0,0) {100 * first.mean():.1f}%")
    print(f"  peak mass   median {np.median([r['peak_mass'] for r in rows]):.4f}"
          f"   vs uniform {np.median([1 / r['n_image_tokens'] for r in rows]):.4f}")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest(args):
    """Every claim this probe rests on, checked on one real picture.

    The identity checks are the ones that matter: an arm that silently does nothing
    produces a small effect, and a small effect is exactly what a null result looks
    like. Each of these fails loudly instead.
    """
    import torch
    from PIL import Image

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("geometry, without a model")
    check("cell_rect row-major", cell_rect(5, 4, 4) == (32, 32, 64, 64))
    check("cell_rect last cell", cell_rect(15, 4, 4) == (96, 96, 128, 128))
    im = Image.new("RGB", (100, 70), (10, 20, 30))
    sn = snap_to_grid(im)
    check("snap is a whole number of patches", sn.size == (96, 64), str(sn.size))
    # A corruption that lands anywhere but the named cell would attribute one cell's
    # effect to another, so the changed pixels are checked against the rect, not counted.
    gh_t, gw_t = sn.size[1] // 32, sn.size[0] // 32
    for cell in (0, gw_t - 1, gh_t * gw_t - 1):
        d = corrupt_cell(sn, cell, gh_t, gw_t, mode="noise", seed=1)
        moved = np.abs(np.asarray(sn, np.int16) - np.asarray(d, np.int16)).sum(-1) > 0
        want = np.zeros_like(moved)
        x0, y0, x1, y1 = cell_rect(cell, gh_t, gw_t)
        want[y0:y1, x0:x1] = True
        check(f"corruption of cell {cell} lands inside its rect",
              bool((moved & ~want).sum() == 0) and bool(moved.any()),
              f"{int(moved.sum())} px moved, {int((moved & ~want).sum())} outside")
        check(f"corruption of cell {cell} keeps the size", d.size == sn.size)

    # The canonical rebinning. A heatmap pooled over 99 grids is only as trustworthy as
    # this, and every failure mode here is silent: weights that do not conserve area turn
    # the map into a map of picture size.
    print("\ncanonical rebinning")
    for (gh, gw) in ((16, 16), (8, 16), (12, 16), (7, 13), (1, 1)):
        w = canonical_weights(gh, gw, 12)
        check(f"{gh}x{gw} rows sum to their area share",
              np.allclose(w.sum(1), 1.0 / (gh * gw)), f"{w.sum(1).min():.6f}")
        check(f"{gh}x{gw} columns tile the canonical grid",
              np.allclose(w.sum(0), 1.0 / 144), f"{w.sum(0).min():.6f}")
        check(f"{gh}x{gw} conserves total", abs(w.sum() - 1.0) < 1e-9)
    # A constant map must come back constant, whatever the source grid.
    for (gh, gw) in ((16, 16), (8, 16), (7, 13)):
        w = canonical_weights(gh, gw, 12)
        got = (np.full(gh * gw, 3.0)[:, None] * w).sum(0) / w.sum(0)
        check(f"{gh}x{gw} constant map stays constant", np.allclose(got, 3.0),
              f"spread {got.max() - got.min():.2e}")
    # A left-half/right-half step must land on the left/right half of the canonical grid.
    gh, gw = 8, 16
    v = np.array([1.0 if (i % gw) < gw // 2 else -1.0 for i in range(gh * gw)])
    w = canonical_weights(gh, gw, 12)
    m = ((v[:, None] * w).sum(0) / w.sum(0)).reshape(12, 12)
    check("a left/right step maps to left/right",
          m[:, :6].mean() > 0.99 and m[:, 6:].mean() < -0.99,
          f"left {m[:, :6].mean():+.3f} right {m[:, 6:].mean():+.3f}")

    if not args.model:
        print("\n(no --model: stopping before the model checks)")
        return 0 if ok else 1

    print("\nwith the model")
    # (processor, model), in that order -- `overlap_probe.load_model`'s own signature.
    processor, model = SLP.load_model(args.model, args.adapter, args.device,
                                      args.attn_impl)
    fam = SLP.load_family(model, processor, args.system_prompt)
    scan = SL.install(model, family=fam, want_key_stats=False)
    corpus = Path(args.corpus)
    rows = SLP.read_manifest(corpus.parent if corpus.name == "corpus" else corpus, None)
    img_root = corpus / "corpus" / "images"
    if not img_root.exists():
        img_root = corpus / "images"
    r = rows[0]
    image = snap_to_grid(PROBE.prepare_image(
        Image.open(img_root / Path(r["image"]).name).convert("RGB")))
    gen = SLP.generate_then_teacher_force(model, processor, [image], r["question"],
                                          args.device, scan, 64)
    check("the model generated something", gen is not None)
    if gen is None:
        return 1
    inputs, prompt_len, comp = gen
    case = fam.teacher_forced_case(inputs, comp, args.device)
    scan.reset()
    scan.prompt_len_override = prompt_len
    with torch.no_grad():
        model(**case, use_cache=False)
    scan.prompt_len_override = None
    res = scan.result()
    _t, gh, gw = res["grids"][0]
    check("the snapped picture keeps its grid",
          fam.grid_of(processor, image) == (gh, gw), f"{(gh, gw)}")
    clean_lp, clean_am = clean_logprobs(model, case, prompt_len, len(comp))

    m = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
    check("the identity forward has zero KL", float(np.max(m["kl"])) < 1e-4,
          f"max {float(np.max(m['kl'])):.2e}")
    check("the identity forward never diverges", m["first_diverge"] == -1)

    cap = RowCapture(model, family=fam).install()
    with torch.no_grad():
        model(**SLP.build_inputs(fam, processor, [image], r["question"], args.device),
              use_cache=False)
    cap.uninstall()
    check("the capture saw one row per image token",
          cap.rows is not None and int(cap.rows.shape[0]) == gh * gw,
          f"{None if cap.rows is None else int(cap.rows.shape[0])} vs {gh * gw}")
    check("the capture saw the three DeepStack features", len(cap.deepstack) == 3,
          f"{len(cap.deepstack)}")

    # A swap with the picture's OWN rows must be the identity: if it is not, the hook is
    # writing into the wrong place and every swap number is meaningless.
    sw = RowSwap(model, [0], cap.rows, cap.deepstack, family=fam).install()
    m0 = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
    sw.uninstall()
    check("a self-swap is the identity", float(np.max(m0["kl"])) < 1e-3,
          f"max {float(np.max(m0['kl'])):.2e}")
    check("the swap hook fired", sw.applied > 0, f"{sw.applied} forwards")

    # The norm-matched path, against the two magnitudes whose answers are known: zero
    # must be the identity, and the cell's own ||donor - target|| must reproduce the
    # literal swap. A rescale that silently does nothing would pass neither.
    rolled = [cap.rows.roll(1, 0)] , [f.roll(1, 0) for f in cap.deepstack]
    donor_r, donor_d = rolled[0][0], rolled[1]
    zero = RowSwap(model, [0], donor_r, donor_d, family=fam,
                   magnitudes=[0.0] * (1 + len(cap.deepstack))).install()
    mz = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
    zero.uninstall()
    check("a zero-magnitude matched swap is the identity",
          float(np.max(mz["kl"])) < 1e-3, f"max {float(np.max(mz['kl'])):.2e}")

    own = [float((d[0].float() - t[0].float()).norm())
           for d, t in zip([donor_r] + donor_d, [cap.rows] + cap.deepstack)]
    lit = RowSwap(model, [0], donor_r, donor_d, family=fam).install()
    ml = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
    lit.uninstall()
    mt = RowSwap(model, [0], donor_r, donor_d, family=fam, magnitudes=own).install()
    mm = score_against(model, case, prompt_len, comp, clean_lp, clean_am)
    mt.uninstall()
    gap = float(np.max(np.abs(np.asarray(ml["kl"]) - np.asarray(mm["kl"]))))
    check("matching to the cell's own norm reproduces the literal swap",
          gap < 1e-3, f"max |diff| {gap:.2e}")
    check("the literal swap of a different row is NOT the identity",
          float(np.max(ml["kl"])) > 1e-6, f"max {float(np.max(ml['kl'])):.2e}")

    # Reusing the clean case for swap arms is a 9x speedup and it must be a bit-for-bit
    # no-op. If it were not, the map and the three-cell arms would be different
    # experiments wearing the same name.
    a_cached = run_arm(model, processor, fam, "swap", 0, image, r["question"],
                       args.device, gh, gw, comp, prompt_len, clean_lp, clean_am,
                       donor_r, donor_d, None, args, random.Random(0), clean_case=case)
    a_fresh = run_arm(model, processor, fam, "swap", 0, image, r["question"],
                      args.device, gh, gw, comp, prompt_len, clean_lp, clean_am,
                      donor_r, donor_d, None, args, random.Random(0), clean_case=None)
    d = float(np.max(np.abs(np.asarray(a_cached["kl"]) - np.asarray(a_fresh["kl"]))))
    check("reusing the clean case changes nothing", d == 0.0, f"max |diff| {d:.2e}")

    peak, pmap = peak_cell(res, gh, gw)
    check("a peak cell was found", peak is not None, f"cell {peak} of {gh * gw}")
    if peak is not None:
        check("the peak carries more than uniform mass",
              pmap[peak] > 1.0 / (gh * gw),
              f"{pmap[peak]:.4f} vs {1 / (gh * gw):.4f}")

    mask = ko_mask(fam, case, 0, args.device)
    if mask is not None:
        case_id = dict(case)
        L = int(case["input_ids"].shape[1])
        neg = torch.finfo(torch.float32).min
        case_id["attention_mask"] = torch.triu(
            torch.full((L, L), neg, device=args.device), diagonal=1)[None, None]
        # NOT a gate. The knockout is the published construction and an optional third
        # arm; the mediation pair this probe is built on needs no custom mask at all. So
        # a family whose mask plumbing refuses a 4D tensor disables `--with-ko` and the
        # run proceeds -- failing the whole selftest here would block the primary
        # experiment on an arm it does not use.
        try:
            m1 = score_against(model, case_id, prompt_len, comp, clean_lp, clean_am)
            good = float(np.max(m1["kl"])) < 1e-3
            check("a plain causal 4D mask reproduces the unmasked forward", good,
                  f"max {float(np.max(m1['kl'])):.2e}")
        except Exception as exc:                                 # noqa: BLE001
            print(f"  [skip] this model rejects a 4D attention mask "
                  f"({type(exc).__name__}: {str(exc)[:70]}) -- --with-ko unavailable, "
                  "the swap/pixel pair is unaffected")

    scan.uninstall()
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default=None)
        p.add_argument("--adapter", default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--attn-impl", default="sdpa")
        p.add_argument("--system-prompt", default="auto")
        p.add_argument("--corpus", default=DEFAULT_CORPUS)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--split", default="all", choices=("all", "dev", "test"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--corrupt", default="mean", choices=("mean", "noise", "shuffle"))
    p.add_argument("--min-mass", type=float, default=0.002)
    p.add_argument("--with-ko", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("map")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--split", default="all", choices=("all", "dev", "test"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--corrupt", default="mean", choices=("mean", "noise", "shuffle"))
    p.add_argument("--min-mass", type=float, default=0.002)
    p.add_argument("--variants", nargs="+", default=["swap", "swapn", "pix"],
                   choices=("swap", "swapn", "pix"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("mapreport")
    p.add_argument("--out", required=True)
    p.add_argument("--canon", type=int, default=12)

    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--field", default="kl_logmean")

    p = sub.add_parser("selftest")
    common(p)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args) or 0
    if args.cmd == "map":
        return run_map(args) or 0
    if args.cmd == "mapreport":
        return map_report(args) or 0
    if args.cmd == "report":
        return report(args) or 0
    return selftest(args)


if __name__ == "__main__":
    sys.exit(main())
