#!/usr/bin/env python
"""Per-head correlation scan: which of the 1152 heads' step-level overlap predicts
whether the completion was right?

The intervention probe asked a different question at the wrong granularity. Forcing
all 32 heads of a layer at once is NOT an upper bound on what one head does -- heads
can carry opposing contributions that cancel, and o_proj mixes them -- so its
layer-level null says nothing about individual heads. This scans every head directly,
on the property the project cares about: does this head's attention to the objects a
step names predict getting the answer right?

For every observe step of every case, and for EVERY (layer, head), the step's map is
that head's attention over image patches, mean-reduced over the step's tokens (the
trainer's token_reduction=mean), scored against the step's own per-step DINO union:

    mean_in_v2 = mean inside the union / mean over the whole map      (chance = 1.0)
    auroc      = P(in-box patch outranks out-box patch)               (chance = 0.5)

Two correlation set-ups:

  step        one observation per observe step; the label is its COMPLETION's
              correctness, repeated across that completion's steps.
  completion  the completion's steps are averaged into one overlap value; one
              observation per completion.

Steps within a completion share a label and are not independent, so `step`-level
significance is anti-conservative. CIs are therefore bootstrapped over COMPLETIONS in
both set-ups, which is the fix.

Correctness is the trainer's own `accuracy_reward` on the model's own greedy answer,
recovered by decoding the continuation of its own chain -- not a first-token match,
which capitalisation biases to 0.38 against a true 0.55.

Selecting a winner from 1152 candidates is where a ranking becomes an artefact, so
`report` splits cases by row_index parity: heads are ranked on the odd half and
re-scored on the even. A head that survives that is a candidate; one that does not is
selection noise.

1152 IS A QWEN3-VL NUMBER. The scan asks `vlm_family` which layers have an attention
matrix, and on a Mamba-Transformer hybrid most of them do not: the Omni's
`hybrid_override_pattern` puts attention at 6 of its 52 decoder layers, so the search is
6 x 32 = 192 cells and the parity split has far less multiplicity to survive. "The scan
saw fewer layers than the model has" is the CORRECT outcome there and a bug anywhere
else, so the count is printed against the decoder's own depth rather than assumed. It
also dissolves the layer question: covering every attention layer means the LAYER is
selected here too, not chosen by relative depth and defended afterwards.

THE UNION IS UNCAPPED, here and in the `prepare` that built the cases -- only the
per-BOX cap (0.5) ran, and N boxes each under it can cover the image between them. The
median step's union covers 54% of the patch grid and the top decile 89%, and every map
measured so far reads lower the larger it gets (r(union, auroc) = -0.55 for the mean
over all 1152 heads). `report` therefore prints the level by union decile before
anything else, and `--max-union` restricts every number after it to a subset. Fix that
threshold before looking at a confirmation set; chosen afterwards it is a researcher
degree of freedom, and the confirmation draws are single use.

The scan also writes the box-free columns of `saliency_sharpness.py` -- how CONCENTRATED
each head's map is, with no boxes involved -- plus the head's total image mass and the
covariates (patch count, step token count, dataset) those need to be controlled for.
They cost one sort over the patch axis and are scored by `sharpness_report.py`, not by
`--stage report` here, because the question they answer is a comparison ACROSS map
families rather than a ranking within this one.

    bash launch_head_correlation.sh --gpus 8 --out-dir DIR --cases-dir <probe out-dir>
    python head_correlation_probe.py --stage report --out-dir DIR [--max-union 0.5]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_hc_overlap_probe", "overlap_probe.py")
IV = _load_module("_hc_intervene", "intervene_probe.py")
SHARP = _load_module("_hc_sharpness", "saliency_sharpness.py")
NL = _load_module("_hc_nemotron_loader", "nemotron_loader.py")
sys.path.insert(0, str(REPO))
import vlm_family as VF  # noqa: E402

IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID


def load_family(model, processor):
    """The family adapter, with the prompt the REWARD was computed under.

    `sink_location_probe.load_family` lets each family choose its own prompt, because the
    question there is where a model looks when run the way it is normally run. The
    question here is which head's reading of the reward's own map predicts correctness,
    and every GRPO run in this repo -- Qwen3-VL's and the Omni's alike -- generates under
    `grpo_vlm_qwen3.SYSTEM_PROMPT`. So the project prompt goes on unconditionally, and
    `intervene_probe --stage prepare` does the same thing for the same reason: the chains
    this reads were written under it.
    """
    fam = VF.family_for(model, processor)
    fam.system_prompt = PROBE.SYSTEM_PROMPT
    return fam


# ---------------------------------------------------------------------------
# all-layer, all-head attention capture
# ---------------------------------------------------------------------------
class AllHeadCapture:
    """Hooks every attention layer; keeps only [heads, step_rows, image_patches].

    TWO WAYS TO GET THE SOFTMAX WEIGHTS, and the family says which one this model needs.

      * Qwen3-VL runs under sdpa, which never materialises them, so each hook re-runs its
        own module in eager -- the trainer's single-layer trick, installed on all 36. The
        transient [1, H, S, S] is sliced immediately and dropped, so peak memory is one
        layer's worth rather than 36.
      * The Omni is loaded eager throughout (`nemotron_loader` pins it, because the
        wrapper declares no SDPA support) and `NemotronHAttention.forward` already returns
        `(attn_output, attn_weights)`. Re-running it there would cost a second attention
        per layer to recompute a tensor the module just handed over, and would have to
        rebuild the causal mask to do it. `Family.attention_weights_are_returned` is that
        fact, and this reads the module's own output instead.

    Which layers exist is also the family's answer. On a hybrid most decoder layers have
    no attention matrix at all, so the hook count is REPORTED against the decoder's depth
    rather than assumed to equal it -- a scan that silently installed on 6 of 52 layers
    and a scan that silently installed on nothing look the same from the outside.
    """

    def __init__(self, model, family):
        self.family = family
        self.returns_weights = bool(family.attention_weights_are_returned)
        self.mask = None
        self.rows = None
        self.cols = None
        self.out = {}
        self._reentry = False
        self.handles = []
        self.layers = []
        for m in model.modules():
            if type(m).__name__ in family.attn_classes and hasattr(m, "layer_idx"):
                li = int(m.layer_idx)
                self.layers.append(li)
                self.handles.append(
                    m.register_forward_hook(self._make(li), with_kwargs=True))
        self.layers.sort()
        if not self.layers:
            raise RuntimeError(
                f"no {'/'.join(family.attn_classes)} modules on this {family.name} model; "
                "the scan would install on nothing and report empty cells as a result")
        self.n_decoder_layers = None
        try:
            self.n_decoder_layers = len(family.decoder(model).layers)
        except Exception:                 # a family whose decoder is shaped differently
            pass

    def describe(self):
        n = len(self.layers)
        of = ("" if self.n_decoder_layers is None
              else f" of {self.n_decoder_layers} decoder layers")
        return f"{n} attention layer(s){of}: {self.layers}"

    def close(self):
        for h in self.handles:
            h.remove()

    def _keep(self, layer_idx, attn):
        sl = attn[0][:, self.rows][:, :, self.cols]        # [H, n_rows, n_patches]
        self.out[layer_idx] = torch.relu(sl).float().cpu().numpy()

    def _make(self, layer_idx):
        def hook(module, args, kwargs, output):
            if self._reentry or self.rows is None:
                return None
            if self.returns_weights:
                attn = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if attn is None:
                    raise RuntimeError(
                        f"{self.family.name} declares attention_weights_are_returned but "
                        f"layer {layer_idx} handed back {type(output).__name__} with no "
                        "weights -- the scan would have nothing to measure")
                self._keep(layer_idx, attn)
                return None
            self._reentry = True
            kw = dict(kwargs)
            kw["attention_mask"] = self.mask
            kw["past_key_values"] = None          # never double-update the KV cache
            kw["use_cache"] = False
            prev = module.config._attn_implementation
            module.config._attn_implementation = "eager"
            try:
                _o, attn = module(*args, **kw)
            finally:
                module.config._attn_implementation = prev
                self._reentry = False
            self._keep(layer_idx, attn)
            del attn
            return None
        return hook


@torch.no_grad()
def scan_case(model, processor, fam, cap, case, image, device, answer_max_tokens):
    """-> (maps [L,H,n_steps,n_patches], model's own answer, kept step indices)."""
    inputs = fam.build_inputs(processor, [image], case["question"], device)
    prompt_len = inputs["input_ids"].shape[1]
    chain = case["chain_ids"]
    gh, gw = case["grid"]
    cols = (inputs["input_ids"][0] == fam.image_token_id).nonzero(as_tuple=True)[0]
    if cols.numel() != gh * gw:
        return None                       # grid and image tokens disagree: skip, not guess

    spans, kept = [], []
    for si, st in enumerate(case["steps"]):
        a, b = prompt_len + st["tok_a"], prompt_len + st["tok_b"]
        if b > prompt_len + len(chain) or b <= a:
            continue
        spans.append((a, b))
        kept.append(si)
    if not spans:
        return None
    rows = torch.cat([torch.arange(a, b, device=device) for a, b in spans])

    # prompt ++ chain, in whatever multimodal bookkeeping this family carries. Qwen3-VL's
    # override extends `mm_token_type_ids` over the completion with zeros, which is what
    # the two explicit `if` blocks here used to do; the Omni carries `image_flags`.
    fwd = fam.teacher_forced_case(inputs, chain, device)
    fwd.update(fam.forward_defaults)
    seq = int(fwd["input_ids"].shape[1])
    # Only the re-run path needs a mask handed to it; a family that returns its own
    # weights already ran under the mask the model built.
    cap.mask = (None if cap.returns_weights
                else IV.causal_mask(seq, next(model.parameters()).dtype, device))
    cap.rows, cap.cols, cap.out = rows, cols, {}

    # THE MODEL'S OWN ANSWER, two ways.
    #
    # `prepare` writes it into the case when it knows it -- it generated the whole
    # completion in one pass, so the tokens after `</think>` ARE the answer this chain
    # got, with no second forward and no second decoding rule. Cases prepared before that
    # existed do not carry it, and for them this recovers it the way it always did: a
    # greedy continuation of the chain off the same forward's KV cache.
    #
    # The fallback needs a cache, which is exactly the thing `forward_defaults` turns off
    # on a hybrid (`use_cache=False`, so no Mamba+KV cache is built and thrown away on
    # every pass) -- so it is overridden here, for that path only.
    answer = case.get("answer_text")
    if answer is None:
        fwd["use_cache"] = True
        fwd.pop("logits_to_keep", None)
    out = model(**fwd)
    cap.rows = None                                # disarm before the decode

    if answer is None:
        past, nxt = out.past_key_values, out.logits[0, -1].argmax().view(1, 1)
        got = [int(nxt)]
        eos = processor.tokenizer.eos_token_id
        for _ in range(answer_max_tokens - 1):
            if got[-1] == eos:
                break
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, o.logits[0, -1].argmax().view(1, 1)
            got.append(int(nxt))
        answer = processor.tokenizer.decode(got, skip_special_tokens=True)

    lens = [b - a for a, b in spans]
    H = cap.out[cap.layers[0]].shape[0]
    maps = np.zeros((len(cap.layers), H, len(spans), gh * gw), dtype=np.float32)
    for li, L in enumerate(cap.layers):
        arr = cap.out[L]
        o = 0
        for si, n in enumerate(lens):
            maps[li, :, si] = arr[:, o:o + n].mean(axis=1)   # token_reduction=mean
            o += n
    cap.out = {}
    return maps, answer, kept


# ---------------------------------------------------------------------------
# metrics, vectorised over (layer, head, step)
# ---------------------------------------------------------------------------
def metrics(maps, masks):
    """maps [L,H,S,P], masks [S,P] bool -> (mean_in_v2, auroc), each [L,H,S].

    Average ranks for ties: attention maps have many near-identical near-zero
    patches, and argsort would break those arbitrarily and bias auroc.
    """
    from scipy.stats import rankdata

    Lc, Hc, Sc, P = maps.shape
    flat = maps.reshape(Lc * Hc * Sc, P).astype(np.float64)
    ranks = rankdata(flat, axis=-1)
    v2 = np.full(Lc * Hc * Sc, np.nan)
    au = np.full(Lc * Hc * Sc, np.nan)
    for si in range(Sc):
        m = masks[si]
        k = int(m.sum())
        if k == 0 or k == P:
            continue                   # degenerate union: no in/out contrast to score
        pos = np.arange(Lc * Hc) * Sc + si
        sub, rk = flat[pos], ranks[pos]
        mean_all = sub.mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            v2[pos] = np.where(mean_all > 0, sub[:, m].mean(axis=1) / mean_all, np.nan)
        au[pos] = (rk[:, m].sum(axis=1) - k * (k + 1) / 2.0) / (k * (P - k))
    return v2.reshape(Lc, Hc, Sc), au.reshape(Lc, Hc, Sc)


# ---------------------------------------------------------------------------
# stage: scan
# ---------------------------------------------------------------------------
def scan(args, device):
    out = Path(args.out_dir)
    dest = out / "scan" / f"shard{args.shard:02d}.npz"
    if dest.exists() and not args.overwrite:
        print(f"[scan] {dest} exists -- nothing to do (--overwrite to redo)")
        return
    cases, cfg, _fp = IV.load_cases(Path(args.cases_dir), args.shard, args.num_shards,
                                    args.max_cases)
    imgs = IV.load_case_images(cfg, f"_hc{args.shard}")
    missing = [c["row_index"] for c in cases if c["row_index"] not in imgs]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(cases)} cases have no image "
                         f"(e.g. row {missing[0]}); cases were prepared with {cfg}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    processor, model = NL.load_any(args.base_model, args.adapter or None, device,
                                   args.attn_impl, PROBE.load_model)
    fam = load_family(model, processor)
    cap = AllHeadCapture(model, fam)
    print(f"[scan] shard {args.shard}: {len(cases)} cases x {cap.describe()}   "
          f"family {fam.name}, weights "
          f"{'returned' if cap.returns_weights else 're-run in eager'}", flush=True)
    prog = IV.Progress(out / "progress" / f"scan{args.shard:02d}.json", len(cases),
                       f"scan{args.shard}", args.log_every)

    V2, AU, SH, NEG, MASS, ROW, STEP, COR, UNI, NPAT, NTOK, DSET = (
        [], [], [], [], [], [], [], [], [], [], [], [])
    dropped = 0
    try:
        for case in cases:
            try:
                r = scan_case(model, processor, fam, cap, case,
                              imgs[case["row_index"]]["image"], device,
                              args.answer_max_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[scan] OOM on row {case['row_index']}; skipped", flush=True)
                r = None
            if r is None:
                dropped += 1
                prog.tick()
                continue
            maps, answer, kept = r
            gh, gw = case["grid"]
            steps = [case["steps"][i] for i in kept]
            masks = np.stack([IV.unb64u8(st["mask_q"], (gh, gw)).astype(bool).reshape(-1)
                              for st in steps])
            v2, au = metrics(maps, masks)
            # Box-free concentration, on the same maps in the same pass. This costs a
            # sort over the patch axis and nothing else, so it rides along free rather
            # than justifying a second 8-GPU scan of the same 1,157 cases.
            sh, neg = SHARP.sharpness(maps, (gh, gw))        # [L,H,S,M], [L,H,S]
            # This head's total attention to the image, the magnitude that goes with
            # the shape. It is the covariate the concentration columns have to be
            # held against: image mass is the strongest single correlate of
            # correctness measured on this corpus, and "sharper" must not be it in
            # disguise -- which the sharpness columns cannot be, since they are
            # computed on the L1-normalised map, but the control should still run.
            mass = maps.sum(-1)                              # [L,H,S]
            grade = PROBE.accuracy_reward(
                [[{"role": "assistant", "content": f"</think> {answer}"}]],
                [case["gold"]])[0]
            if grade is None:                    # ungradable answer: not "wrong"
                dropped += 1
                prog.tick()
                continue
            for si, st in enumerate(steps):
                V2.append(v2[:, :, si])
                AU.append(au[:, :, si])
                SH.append(sh[:, :, si])
                NEG.append(neg[:, :, si])
                MASS.append(mass[:, :, si])
                ROW.append(case["row_index"])
                STEP.append(si)
                COR.append(float(grade))
                UNI.append(st["union_frac"])
                NPAT.append(gh * gw)
                NTOK.append(st["tok_b"] - st["tok_a"])
                DSET.append(case.get("dataset", ""))
            prog.tick()
    finally:
        cap.close()
        prog.close()
    if not V2:
        raise SystemExit("no steps scored")
    np.savez_compressed(
        dest,
        v2=np.stack(V2).astype(np.float32), auroc=np.stack(AU).astype(np.float32),
        sharp=np.stack(SH).astype(np.float32),
        neg_frac=np.stack(NEG).astype(np.float32),
        mass=np.stack(MASS).astype(np.float32),
        sharp_names=np.array(SHARP.SHARP_NAMES),
        row=np.array(ROW), step=np.array(STEP),
        correct=np.array(COR, dtype=np.float32), union=np.array(UNI, dtype=np.float32),
        npatch=np.array(NPAT), ntok=np.array(NTOK), dataset=np.array(DSET),
        layers=np.array(cap.layers))
    print(f"[scan] shard {args.shard}: {len(V2)} steps from "
          f"{len(set(ROW))} completions, {dropped} cases dropped -> {dest}")
    print(SHARP.describe(np.stack(SH)))


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def col_corr(X, y):
    """Pearson r of every column of X [N, L, H] against y [N], NaN-aware. -> [L, H]."""
    ok = np.isfinite(X) & np.isfinite(y)[:, None, None]
    n = ok.sum(0).astype(np.float64)
    Xs = np.where(ok, X, 0.0).astype(np.float64)
    ys = np.where(ok, y[:, None, None], 0.0).astype(np.float64)
    sx, sy = Xs.sum(0), ys.sum(0)
    sxx, syy, sxy = (Xs * Xs).sum(0), (ys * ys).sum(0), (Xs * ys).sum(0)
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0)) * np.sqrt(np.maximum(n * syy - sy ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((den > 0) & (n >= 8), num / den, np.nan)
    return r


def union_decile_table(uni, cols, null=0.5, nbins=10):
    """Print each column's mean level within deciles of the step's DINO union size.

    `cols` maps a display name to that column's per-step values [N]; `uni` is the
    per-step union fraction, `null` the metric's chance level.

    None of the probes caps the union, and it is large: on set_a the median step's
    union covers 54% of the patch grid and the top decile covers 89%. Every map
    measured so far falls monotonically as it grows, so a single pooled level mixes
    two different questions. Midrank AUROC has chance exactly 0.5 for a mask of any
    size at a random location -- averaged over toroidal shifts each pair contributes
    symmetrically, because the mask's autocorrelation is symmetric -- so the curve is
    real map/mask structure and not an artefact of the statistic. What it is not is
    interchangeable across bins: above ~0.5 coverage the union has stopped localising
    the thing the step names (those steps read "The image shows a group of people
    outdoors"), and the level there answers a different question from the level on a
    small union. Read the curve before reading any pooled number, and see --max-union
    to restrict everything else to a subset of it.

    mean_in_v2 is deliberately not tabulated here: its ceiling is 1/union, so it falls
    with union size for a mechanical reason this table could not distinguish from the
    structural one.
    """
    uni = np.asarray(uni, dtype=np.float64)
    edges = np.percentile(uni, np.linspace(0, 100, nbins + 1))
    keeps = [(uni >= edges[i]) & ((uni <= edges[i + 1]) if i == nbins - 1
                                  else (uni < edges[i + 1])) for i in range(nbins)]
    w = 7
    print(f"\n=== level by union-size decile (chance {null:.1f}) ===")
    print(f"   {'bin from':>12}" + "".join(f"{edges[i]:>{w}.2f}" for i in range(nbins))
          + f"{'r(union)':>10}")
    print(f"   {'mean union':>12}"
          + "".join(f"{uni[k].mean():>{w}.2f}" for k in keeps))
    print(f"   {'n steps':>12}" + "".join(f"{int(k.sum()):>{w}d}" for k in keeps))
    for name, v in cols.items():
        v = np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v)
        r = (np.corrcoef(uni[ok], v[ok])[0, 1] if ok.sum() >= 8 and v[ok].std() > 0
             else np.nan)
        with np.errstate(invalid="ignore"):
            cells = [np.nanmean(v[k]) for k in keeps]
        print(f"   {name:>12}" + "".join(f"{c:>{w}.3f}" for c in cells)
              + f"{r:>+10.3f}")
    print(f"   r(union) is over steps, not bins. A column flat in union size is one "
          f"whose reading does not depend on how much of the image the step's boxes "
          f"cover.")


def apply_union_cap(max_union, uni, arrays):
    """Drop steps whose DINO union exceeds `max_union`. -> (kept arrays, mask).

    A no-op at max_union <= 0. The cap is applied here, at report time, rather than in
    intervene_probe's `prepare`: dropping steps at case-construction time would change
    the case set under all four probes at once and break comparability with every
    number already published against these cases.
    """
    if not max_union or float(max_union) <= 0:
        return arrays, np.ones(len(uni), dtype=bool)
    keep = np.asarray(uni) <= float(max_union)
    if keep.sum() < 12:
        raise SystemExit(f"--max-union {max_union} keeps only {int(keep.sum())} steps")
    return tuple(None if a is None else a[keep] for a in arrays), keep


def label_audit(cases_dir, row, cor):
    """How much of "wrong" is the grader rather than the model. -> (soft label, note).

    THE LABEL THIS PROBE RANKS ON is `accuracy_reward` on the model's own answer, and that
    reward parses the gold with math_verify and falls back to an EXACT STRING MATCH when
    math_verify yields nothing -- which it does for `C`, for `horses`, for `Yes`. So a
    model answering `A` scores and the same model answering `(C) water supply`,
    `\\boxed{A}` or `The cup stands on a shelf.` scores zero for being verbose.

    On a cold-started Qwen3-VL that is nearly free: it was trained into the terse format.
    On a base checkpoint it is not. The Nemotron-Omni reads 0.167 under the strict rule and
    ~0.44 under `answer_grading` on the same completions, and it also echoes the SYSTEM
    PROMPT's own illustration -- it writes the literal string "This is my answer." before
    answering. Roughly half the CORRECT completions therefore carry a "wrong" label, which
    attenuates every correlation the ranking is built from.

    This never changes a number by itself. It prints the two accuracies and how many
    completions disagree, so the size of the problem is on the record; `--regrade soft` is
    what actually swaps the label, and the report says so when it does.
    """
    cases = {}
    for f in sorted((Path(cases_dir) / "cases").glob("shard*.json")):
        for c in json.loads(f.read_text())["cases"]:
            if c.get("answer_text") is not None:
                cases[int(c["row_index"])] = c
    if not cases:
        return None, (f"\n[label] no case under {cases_dir} carries `answer_text`, so the "
                      "strict grade cannot be audited. Cases prepared before that field "
                      "existed; re-run prepare to get the audit.")
    AG = _load_module("_hc_answer_grading", "answer_grading.py")
    soft = np.array(cor, dtype=np.float32)
    n_missing = 0
    per_row = {}
    for i, r in enumerate(row):
        c = cases.get(int(r))
        if c is None:
            n_missing += 1
            continue
        if int(r) not in per_row:
            g = AG.grade_completion(c["answer_text"], c["gold"])
            per_row[int(r)] = float(bool(g["soft"]))
        soft[i] = per_row[int(r)]
    uniq, first = np.unique(row, return_index=True)
    strict_acc = float(np.asarray(cor)[first].mean())
    soft_acc = float(soft[first].mean())
    flipped = int((np.asarray(cor)[first] != soft[first]).sum())
    note = (f"\n[label] {len(uniq)} completions   strict (the trainer's accuracy_reward) "
            f"{strict_acc:.3f}   soft (answer_grading) {soft_acc:.3f}   "
            f"{flipped} disagree"
            + (f"   [{n_missing} steps had no case]" if n_missing else "")
            + "\n        strict is what the REWARD optimises and stays the primary "
              "number. soft is what a verbose-but-correct answer can pass; a large gap "
              "means the ranking below is built on a label that is partly grading noise."
              "\n        --regrade soft re-runs everything on the soft label.")
    return soft, note


def report(args):
    out = Path(args.out_dir)
    files = sorted((out / "scan").glob("shard*.npz"))
    if not files:
        raise SystemExit(f"no scan output under {out / 'scan'}")
    d = [np.load(f) for f in files]
    v2 = np.concatenate([x["v2"] for x in d])
    au = np.concatenate([x["auroc"] for x in d])
    row = np.concatenate([x["row"] for x in d])
    cor = np.concatenate([x["correct"] for x in d])
    uni = np.concatenate([x["union"] for x in d])
    layers = d[0]["layers"]
    N, Lc, Hc = v2.shape

    # Before anything is ranked: how much of "wrong" is the grader rather than the model.
    if args.cases_dir and Path(args.cases_dir, "cases").is_dir():
        soft, note = label_audit(args.cases_dir, row, cor)
        print(note)
        if args.regrade == "soft":
            if soft is None:
                raise SystemExit("--regrade soft, but no case carries `answer_text`")
            cor = soft
            print("        *** EVERY NUMBER BELOW IS ON THE SOFT LABEL ***")
    elif args.regrade == "soft":
        raise SystemExit("--regrade soft needs --cases-dir pointing at the prepare "
                         "out-dir whose cases carry the model's own answers")

    # THE INCUMBENT is whatever the run being compared against rewarded, and on a model
    # that is not Qwen3-VL-8B it may not exist: layer 22 is a Mamba layer on the Omni, so
    # `np.where(layers == 22)[0][0]` raises. It defaults to L22 h28,31 so the published
    # Qwen3-VL report is unchanged, and the rows are simply omitted when the layer has no
    # attention matrix on this model -- rather than falling back to index 0, which would
    # label some other layer's heads with the incumbent's name.
    inc_layer = int(args.incumbent_layer)
    inc_heads = [int(h) for h in str(args.incumbent_heads).split(",") if h.strip()]
    has_inc = bool((layers == inc_layer).any())
    li_inc = int(np.where(layers == inc_layer)[0][0]) if has_inc else None

    # The union curve is reported on everything, before any cap -- it is the thing the
    # cap is chosen from, so restricting it first would hide the tail being cut.
    bl, bh = divmod(int(np.nanargmax(np.nanmean(au.reshape(N, -1), axis=0))), Hc)
    curves = {f"mean {Lc * Hc}": np.nanmean(au.reshape(N, -1), axis=1),
              f"L{int(layers[bl])}H{bh}": au[:, bl, bh]}
    if has_inc:                          # the rewarded heads, when the run has them
        for h in inc_heads:
            if h < Hc:
                curves[f"L{inc_layer}H{h}"] = au[:, li_inc, h]
    union_decile_table(uni, curves, null=0.5)
    print(f"   the rows are auroc: the mean over all {Lc * Hc} heads, the single "
          f"head with the highest pooled level"
          + (", and the rewarded heads." if has_inc else
             f", and NO incumbent row -- layer {inc_layer} has no attention matrix on "
             f"this model, whose attention layers are {[int(x) for x in layers]}."))

    (v2, au, row, cor, uni), keep = apply_union_cap(args.max_union, uni,
                                                    (v2, au, row, cor, uni))
    if not keep.all():
        print(f"\n--max-union {args.max_union}: {int(keep.sum())}/{N} steps and "
              f"{len(np.unique(row))} completions kept. Everything below is that "
              f"subset; the table above is not.")
        N = len(row)

    uniq = np.unique(row)
    print(f"steps {N}   completions {len(uniq)}   layers {Lc}   heads {Hc}   "
          f"accuracy {cor[np.unique(row, return_index=True)[1]].mean():.3f}")

    # completion-level: mean of the completion's steps, one observation per completion
    idx = np.searchsorted(uniq, row)
    ccor = np.zeros(len(uniq))
    np.maximum.at(ccor, idx, cor)          # label is constant within a completion
    cagg = {}
    for name, arr in (("mean_in_v2", v2), ("auroc", au)):
        s = np.zeros((len(uniq), Lc, Hc))
        n = np.zeros((len(uniq), Lc, Hc))
        np.add.at(s, idx, np.nan_to_num(arr, nan=0.0))
        np.add.at(n, idx, np.isfinite(arr).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cagg[name] = np.where(n > 0, s / n, np.nan)

    sel_c = (uniq % 2 == 1)                # select on odd rows, confirm on even
    sel_s = (row % 2 == 1)
    for name, sarr, carr in (("mean_in_v2", v2, cagg["mean_in_v2"]),
                             ("auroc", au, cagg["auroc"])):
        for setup, X, y, sel in (("step", sarr, cor, sel_s),
                                 ("completion", carr, ccor, sel_c)):
            r_all = col_corr(X, y)
            r_sel = col_corr(X[sel], y[sel])
            r_out = col_corr(X[~sel], y[~sel])
            print(f"\n=== {name} / {setup}-level "
                  f"(n={len(y)}) ===")
            print("  per-LAYER (max |r| over its 32 heads, all data) -- pick layers here:")
            order = np.argsort(-np.nan_to_num(np.nanmax(np.abs(r_all), axis=1)))
            print(f"   {'rank':>4} {'layer':>5} {'max|r|':>8} {'head':>5} {'mean|r|':>8}")
            for k, li in enumerate(order[: args.top_layers]):
                h = int(np.nanargmax(np.abs(r_all[li])))
                print(f"   {k + 1:>4} {int(layers[li]):>5} "
                      f"{np.nanmax(np.abs(r_all[li])):>8.4f} {h:>5} "
                      f"{np.nanmean(np.abs(r_all[li])):>8.4f}")
            print("  TOP HEADS ranked on ODD rows, re-scored on EVEN (held out):")
            flat = np.abs(np.nan_to_num(r_sel, nan=0.0)).ravel()
            print(f"   {'layer':>5} {'head':>5} {'r(select)':>10} {'r(HELD OUT)':>12} "
                  f"{'r(all)':>8}")
            for t in np.argsort(-flat)[: args.top_heads]:
                l, h = divmod(int(t), Hc)
                print(f"   {int(layers[l]):>5} {h:>5} {r_sel[l, h]:>+10.4f} "
                      f"{r_out[l, h]:>+12.4f} {r_all[l, h]:>+8.4f}")
            for hh in (inc_heads if has_inc else []):
                if hh >= Hc:
                    continue
                rank = int((np.abs(np.nan_to_num(r_all))
                            > abs(r_all[li_inc, hh])).sum()) + 1
                print(f"   incumbent L{inc_layer}H{hh}: r(all) "
                      f"{r_all[li_inc, hh]:>+.4f}  rank {rank} of {Lc * Hc}")
            # The label is part of the identity of these numbers, so a soft run writes
            # beside the strict one rather than over it.
            tag = "" if args.regrade == "off" else f"_{args.regrade}"
            np.savez_compressed(out / f"corr_{name}_{setup}{tag}.npz",
                                r_all=r_all, r_sel=r_sel, r_out=r_out, layers=layers,
                                max_union=np.array(args.max_union),
                                label=np.array(args.regrade))
    print(f"\n-> {out}/corr_*{'' if args.regrade == 'off' else '_' + args.regrade}.npz")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="scan", choices=["scan", "report"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases-dir", default="",
                   help="an intervene_probe out-dir whose cases/ holds the chains and "
                        "per-step DINO unions; defaults to --out-dir")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--answer-max-tokens", type=int, default=16)
    p.add_argument("--top-layers", type=int, default=10)
    p.add_argument("--top-heads", type=int, default=15)
    p.add_argument("--incumbent-layer", type=int, default=22,
                   help="report stage: the (layer, heads) the run being compared against "
                        "already rewards, printed with their rank among all cells. The "
                        "default is the Qwen3-VL pair; on a model whose attention layers "
                        "do not include it the rows are omitted rather than relabelled")
    p.add_argument("--incumbent-heads", default="28,31")
    p.add_argument("--regrade", default="off", choices=["off", "soft"],
                   help="report stage: which correctness label to rank on. `off` is the "
                        "trainer's own accuracy_reward, what the scan stored and what the "
                        "REWARD optimises -- the primary number. `soft` re-derives it "
                        "from the cases' stored answers with answer_grading, which a "
                        "verbose-but-correct answer can pass; it needs --cases-dir. The "
                        "audit line prints both either way, because the gap between them "
                        "is how much of the ranking is grading noise")
    p.add_argument("--max-union", type=float, default=0.0,
                   help="report stage: drop steps whose DINO union covers more than "
                        "this fraction of the patch grid (0 = off, the default and "
                        "what every published number used). The union is the metric's "
                        "reference region and it is uncapped everywhere else, "
                        "including intervene_probe's `prepare`; see the decile table "
                        "the report prints before applying this.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not args.cases_dir:
        args.cases_dir = args.out_dir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        return report(args)
    return scan(args, args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
