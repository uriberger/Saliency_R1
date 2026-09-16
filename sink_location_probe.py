#!/usr/bin/env python
"""Is the sink in the outer ring, or is the outer ring just the background?

`docs/sink-location-by-image-type.md` is the design; `sink_location.py` is the
measurement. This is the harness.

    python sink_location_probe.py --stage corpus   --out-dir DIR
    python sink_location_probe.py --stage selftest --out-dir DIR --model M
    python sink_location_probe.py --stage scan     --out-dir DIR --model M --shard i --num-shards n
    python sink_location_probe.py --stage arms     --out-dir DIR --model M --shard i --num-shards n
    python sink_location_probe.py --stage report   --out-dir DIR
    python sink_location_probe.py --stage monitor  --out-dir DIR

STAGES

  corpus    Twelve image types to disk, as prepared pictures plus a manifest. CPU only,
            no model. Everything downstream reads this directory and never touches
            `datasets` again, so the GPU stages are deterministic and shard trivially.
            The manifest also carries the dev/test flag: heads are SELECTED on dev and
            REPORTED on test, because picking the sink heads on the same images that
            report their effect is the easiest way to manufacture a result here.

  selftest  Gates every run, and must pass. Five checks:
              - the scan's attention implementation reproduces stock SDPA, both in the
                logits and token for token under greedy decoding. It edits nothing, and
                that has to be a measurement rather than a claim.
              - the ring's measured area equals (2gh+2gw-4)/(gh*gw) on every real grid,
                and the two negative-control sets come back at 1.00 on a shuffled map.
              - THE COORDINATE FRAME. A picture with one bright patch at a known place,
                put through every transform, must land where `patch_correspondence` says.
                This is where the experiment is most likely to go quietly wrong: an
                off-by-one in the rotation decodes the wrong frame and answers the
                content-versus-position question confidently and backwards.
              - the patch permutation is a permutation: the multiset of patch embeddings
                is unchanged and `mode=identity` reproduces the baseline exactly.
              - the causal column correction. In the image->image query set the first
                patch is visible to every query and the last to one, which is a top-row
                gradient of exactly the shape under test. It must be divided out.

  scan      One prefill per picture, batch 1, every layer and every head, plus the
            hidden-state norms and the vision tower's own patch norms. Append-only
            JSONL for resume, with the bulk arrays in sidecar npz parts.

  arms      The same scan on transformed pictures, paired with each picture's own
            baseline. This is where the causal weight is: `rot180` against the sky
            confound, `zoom60` against the ring-is-background reading, `donut` and
            `canvas` against both, and `permute` -- which decouples content from grid
            slot without touching a pixel -- against every objection that a pixel-space
            transform changed something unmodelled.

  report    The budget first, then per type, then the arms. Nothing is printed as a raw
            percentage: the ring is 23% of a 16x16 grid and 50% of a 6x8 one, so every
            number is a lift or an enrichment over the geometry it came from.

WHY PREFILL AND NOT GENERATION. The claim being tested was measured on generated tokens.
A prefill readout is ~10x cheaper and lets the whole design fit in an hour of one node.
`--stage selftest --tie-back` is what licenses the substitution: it recomputes the
generated-token readout on the same pictures and reports the agreement. Below the
pre-registered threshold, the honest move is to pay for generation, not to reword the
conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_sl_overlap_probe", "overlap_probe.py")
IV = _load_module("_sl_intervene", "intervene_probe.py")
sys.path.insert(0, str(REPO))
import sink_location as SL  # noqa: E402
import sink_shift as SS  # noqa: E402
import vlm_family as VF  # noqa: E402

# The pair the reward trained, kept as a named cell so every table can carry the
# "and what does it say at the cell the whole project is built on" column. It is a fact
# about Qwen3-VL-8B and about this project's reward; on any other family the same two
# indices name two arbitrary heads, so the report prints that block only for `qwen3_vl`.
TRAINED_LAYER, TRAINED_HEADS = 22, (28, 31)
TRAINED_FAMILY = "qwen3_vl"

DEV_FRAC = 0.25          # share of each type reserved for choosing heads


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------
#: type -> how to get it. `valset` reads the arrow sets this project already built and
#: filters on their `dataset` column; `hf` and `parquet` read the benchmark caches
#: eval_mini/benchmarks.py already uses. Every one of these is on disk and offline.
#:
#: The order of the types is the order of the argument in the write-up: photographs
#: first, then the types where BLANK AND BORDER COME APART, which is what makes the
#: observational half of this experiment discriminative on its own.
CORPUS = {
    "photo_vqa": dict(
        kind="valset", sets=("val_natural", "set_a"),
        datasets=("gqa", "aokvqa", "visual7w", "openimages", "vsr"),
        note="photographs, object and relation VQA -- the corpus the claim was made on"),
    "photo_aerial": dict(
        kind="valset", sets=("val_natural", "set_a"), datasets=("visdrone",),
        note="aerial: no sky, no centred subject, so photographer bias cannot explain it"),
    "chart": dict(
        kind="valset", sets=("val_nonnatural", "set_b"), datasets=("virl_charts",),
        note="charts and tables: the largest blank region is INTERIOR"),
    "document": dict(
        kind="valset", sets=("val_nonnatural", "set_b"), datasets=("docvqa",),
        note="scanned documents: ink runs to the margins, so the border is not blank"),
    "infographic": dict(
        kind="valset", sets=("val_nonnatural", "set_b"), datasets=("infographicsvqa",),
        note="dense, coloured, edge to edge"),
    "math_figure": dict(
        kind="valset", sets=("val_nonnatural", "set_b"), datasets=("virl_math_geo",),
        note="line art on white: most of the picture is background, and not at the edge"),
    "science_diagram": dict(
        kind="valset", sets=("val_nonnatural", "set_b"), datasets=("virl_science",),
        note="labelled diagrams"),
    "puzzle_abstract": dict(
        kind="jsonl_images", repo="VisuLogic/VisuLogic", file="data.jsonl",
        # the pictures live inside images.zip, never unpacked; lmms-eval reads them the
        # same way (eval_mini/benchmarks.py's `local_data`)
        zip="images.zip", image_key="image_path", question_key="question",
        note="Raven-style grids: content is uniformly tiled, including the border"),
    "puzzle_board": dict(
        kind="hf", repo="declare-lab/AlgoPuzzleVQA", split="data",
        question_key="question", image_key="image",
        note="boards fill the frame, so the border is a board edge and is informative"),
    "synthetic_popout": dict(
        kind="parquet", repo="salbench-vlm/salbench", glob="P3/shard_*.parquet",
        question_key="question", image_key="image",
        note="homogeneous distractor field: 'background' is not a place, so H2 has "
             "nowhere to point"),
    "exam_page": dict(
        kind="hf", repo="MMMU/MMMU_Pro", config="standard (10 options)", split="test",
        question_key="question", image_key="image_1",
        note="mixed text and figure layout"),
    "illusion": dict(
        kind="hf", repo="csebuetnlp/illusionVQA-Soft-Localization", split="test",
        question_key="question", image_key="image",
        note="low-texture fields"),
}

DEFAULT_QUESTION = "What is shown in this image?"


def _hf_root():
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _snapshot(repo):
    """The one cached snapshot of a repo, or a clear failure. No network."""
    base = _hf_root() / ("datasets--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(base.glob("*")) if base.is_dir() else []
    if not snaps:
        raise SystemExit(f"{repo} is not in the cache under {base}; this stage is offline "
                         "by design -- fetch it first or drop the type with --types")
    return snaps[-1]


def _load_valset(spec, want, seed, exclude_train):
    """Rows of the named `dataset` sources, val sets first, training sets only to top up.

    The training sets are 50,000 rows and the validation sets are 256, so several types
    cannot reach a usable n from validation alone. That is fine for the base model and
    the cold start, which never saw set_a or set_b -- and it is NOT fine for a
    GRPO-trained checkpoint, which did. `--val-only` is the flag for that case, and the
    manifest records which set every row came from so a mistake is visible afterwards.
    """
    from datasets import load_from_disk

    rng = np.random.default_rng(seed)
    want_ds = set(spec["datasets"])
    got = []
    for name in spec["sets"]:
        if len(got) >= want:
            break
        if exclude_train and not name.startswith("val"):
            continue
        path = REPO / "cold_data" / "grpo_sets" / name
        if not path.is_dir():
            path = Path("/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/"
                        "uberger/research/saliency_r1/cold_data/grpo_sets") / name
        if not path.is_dir():
            continue
        ds = load_from_disk(str(path))
        if hasattr(ds, "keys"):
            ds = ds["train"]
        idx = np.flatnonzero(np.isin(np.asarray(ds["dataset"]), sorted(want_ds)))
        rng.shuffle(idx)
        for i in idx[: want - len(got)]:
            r = ds[int(i)]
            got.append(dict(image=r["image"], question=r["problem"],
                            source=f"{name}:{r['dataset']}", ref=str(r.get("question_id"))))
    return got


def _load_generic(spec, want, seed):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    kind = spec["kind"]
    if kind == "parquet":
        files = sorted(str(p) for p in _snapshot(spec["repo"]).glob(spec["glob"]))
        ds = load_dataset("parquet", data_files=files, split="train")
    elif kind == "jsonl_images":
        import zipfile
        from PIL import Image
        snap = _snapshot(spec["repo"])
        rows = [json.loads(l) for l in (snap / spec["file"]).read_text().splitlines() if l]
        rng.shuffle(rows)
        zf = zipfile.ZipFile(snap / spec["zip"]) if spec.get("zip") else None
        names = set(zf.namelist()) if zf else set()
        out = []
        for r in rows:
            if len(out) >= want:
                break
            rel = r[spec["image_key"]]
            if zf is not None:
                if rel not in names:
                    continue
                im = Image.open(io.BytesIO(zf.read(rel)))
            else:
                p = snap / rel
                if not p.exists():
                    continue
                im = Image.open(p)
            out.append(dict(image=im, question=r.get(spec["question_key"])
                            or DEFAULT_QUESTION, source=spec["repo"],
                            ref=str(r.get("id"))))
        return out
    else:
        ds = load_dataset(spec["repo"], spec.get("config"), split=spec.get("split", "test"))
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= want:
            break
        r = ds[int(i)]
        im = r.get(spec["image_key"])
        if im is None:
            continue
        q = r.get(spec["question_key"]) or DEFAULT_QUESTION
        out.append(dict(image=im, question=str(q), source=spec["repo"], ref=str(i)))
    return out


def stage_corpus(args):
    """Materialise every type to disk, prepared exactly as the trainer prepares images."""
    out = Path(args.out_dir) / "corpus"
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists() and not args.rebuild:
        for line in manifest.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except json.JSONDecodeError:
                continue
    mode = "a" if done else "w"
    n_by_type = {}
    with open(manifest, mode) as fh:
        for tname in args.types:
            spec = CORPUS[tname]
            try:
                rows = (_load_valset(spec, args.per_type, args.seed, args.val_only)
                        if spec["kind"] == "valset"
                        else _load_generic(spec, args.per_type, args.seed))
            except Exception as exc:                       # one dead source is not fatal
                print(f"[corpus] {tname}: SKIPPED -- {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            kept = 0
            for j, r in enumerate(rows):
                key = f"{tname}-{j:04d}"
                if key in done:
                    kept += 1
                    continue
                try:
                    im = PROBE.prepare_image(r["image"])
                except Exception as exc:
                    print(f"[corpus] {key}: unreadable ({exc})", flush=True)
                    continue
                path = out / "images" / f"{key}.png"
                im.save(path)
                # dev/test by a hash of the key, so the split is stable across rebuilds
                # and independent of the draw order.
                h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                fh.write(json.dumps({
                    "key": key, "type": tname, "source": r["source"], "ref": r["ref"],
                    "question": r["question"], "image": f"images/{key}.png",
                    "size": list(im.size), "dev": h < DEV_FRAC}) + "\n")
                kept += 1
            n_by_type[tname] = kept
            print(f"[corpus] {tname:<18} {kept:>4}   {spec['note']}", flush=True)
    print(f"\n{sum(n_by_type.values())} pictures under {out}")
    print("Types with fewer than 60 pictures are under-powered for a per-type CI; the "
          "report says so\nin its own header rather than leaving it to be noticed.")
    return 0


def read_manifest(out_dir, types=None):
    path = Path(out_dir) / "corpus" / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"no corpus at {path}; run --stage corpus first")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if types and r["type"] not in types:
            continue
        r["path"] = str(path.parent / r["image"])
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# one measured picture
# ---------------------------------------------------------------------------
def load_family(model, processor, system_prompt="auto"):
    """The adapter for this model, with the system prompt the run asked for.

    `auto` means the family's own preference: Qwen3-VL gets the project's trainer prompt,
    so `docs/sink-location-by-image-type.md` §16-17 reproduce, and the other two get
    none, because putting a `<think>` system prompt in front of LLaVA-1.5 would measure
    an off-distribution model. `none` puts every family on the same footing, which is
    what the cross-model tables are run under.
    """
    fam = VF.family_for(model, processor)
    if system_prompt == "project" or (system_prompt == "auto" and fam.uses_project_prompt):
        fam.system_prompt = PROBE.SYSTEM_PROMPT
    elif system_prompt not in ("auto", "none"):
        fam.system_prompt = system_prompt
    return fam


def build_inputs(fam, processor, images, question, device, **kw):
    """The prompt, at batch size 1, with one or two pictures, in the family's template."""
    return fam.build_inputs(processor, images, question, device, **kw)


def generate_then_teacher_force(model, processor, images, question, device, scan,
                                max_new_tokens):
    """Write an answer at full speed, then measure ONE forward over prompt ++ answer.

    Not 256 decode steps with the explicit softmax on every layer: the scan is paused for
    the generation, so it runs through the fused kernel, and the whole completion is then
    measured in a single teacher-forced pass. That pass is `overlap_probe.teacher_forced_case`
    -- the same construction `grpo_trainer_qwen3.py` uses to compute the reward -- so the
    `generated` query set here is the reward's own view and not an approximation of it.

    Returns (inputs, prompt_len, completion ids) or None if nothing was generated.
    """
    import torch

    inputs = build_inputs(scan.family, processor, images, question, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    scan.paused = True
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=processor.tokenizer.pad_token_id)
    finally:
        scan.paused = False
    comp = out[0][prompt_len:].tolist()
    eos = processor.tokenizer.eos_token_id
    if eos in comp:
        comp = comp[: comp.index(eos) + 1]
    return (inputs, prompt_len, comp) if comp else None


def measure(model, processor, images, question, device, scan, tap=None,
            want_hidden=True, max_new_tokens=0, tile=0, min_mass=0.002, **proc_kwargs):
    """One prefill. -> the reduced cells, the maps, the norms, and the geometry.

    Everything this experiment reads comes out of this single forward: the column view at
    every layer and head, the span budget, the key statistics, the LLM's hidden-state
    norms and the vision tower's own patch norms. Nothing is generated.

    `tile` picks which of the picture's grids to score. It is 0 everywhere except in
    InternVL's tiled arm, where one picture becomes several 448px tiles plus a thumbnail
    and each is its own 16x16 grid: scoring them as one grid would call a tile boundary
    an interior edge of the picture and a border of the tile at the same time.
    """
    import torch

    gen = None
    if max_new_tokens > 0:
        gen = generate_then_teacher_force(model, processor, images, question, device,
                                          scan, max_new_tokens)
    if gen is None:
        inputs = build_inputs(scan.family, processor, images, question, device,
                              **proc_kwargs)
        case, scan.prompt_len_override = inputs, None
    else:
        inputs, prompt_len, comp = gen
        case = scan.family.teacher_forced_case(inputs, comp, device)
        scan.prompt_len_override = prompt_len
    scan.reset()
    try:
        with torch.no_grad():
            out = model(**case, output_hidden_states=bool(want_hidden), use_cache=False)
    finally:
        scan.prompt_len_override = None
    res = scan.result()
    if res is None or not res["grids"]:
        return None
    if tile >= len(res["grids"]):
        return None
    # Every grid shares one column axis -- two pictures, or one picture's tiles, are laid
    # out end to end -- so the block being scored is an offset and a length, not a slice
    # from zero. `lo` is 0 for the single-grid case, which is every family's default.
    sizes = [t * gh * gw for t, gh, gw in res["grids"]]
    lo = int(sum(sizes[:tile]))
    t, gh, gw = res["grids"][tile]
    n_first = gh * gw
    cut = lambda a: None if a is None else a[..., lo:lo + n_first]   # noqa: E731

    got = {"grid": [gh, gw], "kv_len": res["kv_len"], "n_grids": len(res["grids"]),
           "tile": int(tile), "n_image_tokens": res["n_image_tokens"]}
    prim = res[SL.PRIMARY_Q]
    stats, peak = SL.reduce_cells(
        cut(prim["col_sum"]), cut(prim["col_sq"]), prim["n_rows"], prim["row_total"],
        gh, gw, res["kv_len"],
        knorm=cut(res["knorm"]), logit_sum=cut(res["logit_sum"]),
        scaling=res["scaling"], seed=0,
        col_null=None if prim["col_null"] is None
        else prim["col_null"][lo:lo + n_first])
    got["stats"], got["peak"] = stats, peak

    sec = res.get("image")
    if sec is not None and len(res["grids"]) == 1:
        s2, p2 = SL.reduce_cells(sec["col_sum"], None, sec["n_rows"], sec["row_total"],
                                 gh, gw, res["kv_len"], col_null=sec["col_null"])
        got["stats_img_q"], got["peak_img_q"] = s2, p2

    # The tokens the model WROTE, and the union of those with the question's tokens.
    # Column sums are additive and the row totals are sums over the same rows, so `all` is
    # the exact union rather than an average of two averages -- which would silently
    # re-weight a 12-token question against a 250-token answer.
    gq = res.get("generated")
    if gq is not None and gq["n_rows"] > 0:
        got["n_generated"] = int(gq["n_rows"])
        sg, pg = SL.reduce_cells(cut(gq["col_sum"]), None, gq["n_rows"],
                                 gq["row_total"], gh, gw, res["kv_len"])
        got["stats_gen"], got["peak_gen"] = sg, pg
        sa, pa = SL.reduce_cells(
            cut(prim["col_sum"]) + cut(gq["col_sum"]), None,
            prim["n_rows"] + gq["n_rows"], prim["row_total"] + gq["row_total"],
            gh, gw, res["kv_len"])
        got["stats_all"], got["peak_all"] = sa, pa

    # the layer-mean map, which is what the per-patch regression and the radial profiles
    # are fitted on. Head-mean per layer: 36 x N floats, not 36 x 32 x N.
    mean = cut(prim["col_sum"]).mean(1)
    got["maps"] = mean / np.maximum(mean.sum(-1, keepdims=True), 1e-30)

    # ... and the ALL-HEAD pooled map, once per query set. This is the per-patch
    # decomposition of the cross-model tables -- summing it over any patch set
    # reproduces that table's numerator -- so a heatmap drawn from it cannot disagree
    # with the numbers printed beside it. [N] floats each.
    got["map_q"] = SL.pooled_patch_map(cut(prim["col_sum"]), prim["row_total"],
                                       prim["n_rows"], min_mass)
    if gq is not None and gq["n_rows"] > 0:
        got["map_gen"] = SL.pooled_patch_map(cut(gq["col_sum"]), gq["row_total"],
                                             gq["n_rows"], min_mass)
        got["map_all"] = SL.pooled_patch_map(
            cut(prim["col_sum"]) + cut(gq["col_sum"]),
            prim["row_total"] + gq["row_total"],
            prim["n_rows"] + gq["n_rows"], min_mass)

    if res["spans"] is not None:
        denom = np.maximum(prim["row_total"], 1e-30)
        got["spans"] = np.stack([res["spans"][s] / denom for s in SL.SPANS], axis=-1)

    if want_hidden and getattr(out, "hidden_states", None) is not None:
        cols = scan.img_cols[lo:lo + n_first]
        ring = SL.ring_set(gh, gw).reshape(-1)
        rows = []
        for h in out.hidden_states:
            nrm = h[0, cols].float().norm(dim=-1).cpu().numpy()
            rows.append([nrm[ring].mean(), nrm[~ring].mean(), nrm.max(),
                         float(np.argmax(nrm))])
        got["hnorm"] = np.asarray(rows, dtype=np.float32)
    if tap is not None and tap.norms is not None:
        nrm = tap.norms[lo:lo + n_first]
        ring = SL.ring_set(gh, gw).reshape(-1)
        got["vnorm"] = np.asarray([nrm[ring].mean(), nrm[~ring].mean(), nrm.max(),
                                   float(np.argmax(nrm))], dtype=np.float32)
    return got


# ---------------------------------------------------------------------------
# storage -- JSONL for resume, npz parts for the bulk
# ---------------------------------------------------------------------------
class Sink:
    """Append-only results: one JSONL line per unit, bulk arrays in sidecar npz parts.

    Resume is by KEY out of the JSONL, exactly as `sink_shift_probe.done_keys` does it,
    and the arrays are flushed in parts so a killed shard loses at most one part rather
    than a whole shard's worth of GPU time.
    """

    def __init__(self, out_dir, stage, shard, flush_every=200):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage, self.shard = stage, shard
        self.jsonl = self.dir / f"{stage}_shard{shard}.jsonl"
        self.flush_every = int(flush_every)
        self.buf = []
        self.part = 0
        while (self.dir / f"{stage}_shard{shard}_part{self.part}.npz").exists():
            self.part += 1
        self.fh = open(self.jsonl, "a")

    def done(self):
        if not self.jsonl.exists():
            return set()
        keys = set()
        for line in self.jsonl.read_text().splitlines():
            try:
                keys.add(json.loads(line)["unit"])
            except (json.JSONDecodeError, KeyError):
                continue          # a torn last line from a killed job is not a result
        return keys

    def write(self, unit, meta, arrays):
        self.buf.append((unit, arrays))
        self.fh.write(json.dumps(dict(meta, unit=unit)) + "\n")
        self.fh.flush()
        if len(self.buf) >= self.flush_every:
            self.flush()

    #: stored as integers, because they are LABELS on a grid. float16 is exact to 2048
    #: and the grids here are far smaller, but a patch index that rounds is a patch index
    #: that points at the wrong patch, and nothing downstream could tell.
    INT_FIELDS = ("peak", "peak_img_q", "peak_gen", "peak_all", "perm")

    def flush(self):
        """One encoding for every field: flat values, per-unit shapes, per-unit indices.

        Fields are both ragged AND optional -- `maps` has a different N per grid, and
        `perm` exists only for the permutation arms. An earlier version stacked
        fixed-shape fields and skipped any field some unit lacked, which silently dropped
        `perm` from every part it shared with a non-permutation arm, i.e. all of them.
        Carrying the unit indices costs a few bytes and removes the whole class of bug.
        """
        if not self.buf:
            return
        packed = {"units": np.asarray([u for u, _ in self.buf])}
        for name in sorted({k for _u, a in self.buf for k in a}):
            have = [(i, a[name]) for i, (_u, a) in enumerate(self.buf) if a.get(name) is not None]
            if not have:
                continue
            dtype = np.int32 if name in self.INT_FIELDS else np.float16
            packed[name] = np.concatenate([v.reshape(-1) for _i, v in have]).astype(dtype)
            packed[name + "__shapes"] = np.asarray([v.shape for _i, v in have],
                                                   dtype=np.int64)
            packed[name + "__idx"] = np.asarray([i for i, _v in have], dtype=np.int64)
        np.savez_compressed(self.dir / f"{self.stage}_shard{self.shard}_part{self.part}.npz",
                            **packed)
        self.part += 1
        self.buf = []

    def close(self):
        self.flush()
        self.fh.close()


def arrays_of(got):
    """The bulk fields of one measurement, as the npz stores them.

    `perm` is here because without it the permutation arm cannot be read: "did the peak
    follow the patch embedding it was sitting on" needs to know where each embedding went,
    and that is the one question no pixel-space transform can answer.
    """
    keep = ("stats", "peak", "maps", "spans", "hnorm", "vnorm", "stats_img_q", "perm",
            "stats_gen", "peak_gen", "stats_all", "peak_all",
            "map_q", "map_gen", "map_all")
    return {k: np.asarray(got[k]) for k in keep if got.get(k) is not None}


def meta_of(got, row, extra=None, fam=None):
    """The JSONL line: everything the report needs before it opens an npz.

    `family` and `view` are here because the same corpus is now measured by three models
    with three geometries. Without them a mixed output directory would pool a 24x24
    centre crop with a 16x16 squash and a per-picture native grid, and every enrichment
    would still look like a number.
    """
    gh, gw = got["grid"]
    st = got["stats"]
    trained = st[TRAINED_LAYER, list(TRAINED_HEADS)].mean(0) if st.shape[0] > TRAINED_LAYER \
        else np.full(len(SL.STAT_NAMES), np.nan)
    m = {
        "key": row["key"], "type": row["type"], "source": row.get("source"),
        "dev": bool(row.get("dev")), "grid": [gh, gw], "size": row.get("size"),
        "ring_area_frac": SL.ring_area_frac(gh, gw),
        "n_image_tokens": got["n_image_tokens"], "kv_len": got["kv_len"],
        "family": None if fam is None else fam.name,
        "view": list(got.get("view", SL.FULL_VIEW)),
        "tile": got.get("tile", 0), "n_grids": got.get("n_grids", 1),
        "trained_cell": {n: _f(trained[i]) for i, n in enumerate(SL.STAT_NAMES)},
    }
    return dict(m, **(extra or {}))


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else x


# ---------------------------------------------------------------------------
# stage: scan
# ---------------------------------------------------------------------------
def stage_scan(args):
    import torch

    rows = read_manifest(args.out_dir, args.types)
    if args.scan_rows_per_type:
        by_type = {}
        for r in rows:
            by_type.setdefault(r["type"], []).append(r)
        rows = [r for t in sorted(by_type)
                for r in by_type[t][: args.scan_rows_per_type]]
    mine = rows[args.shard::args.num_shards]
    sink = Sink(args.out_dir, "scan", args.shard, args.flush_every)
    seen = sink.done()
    todo = [r for r in mine if r["key"] not in seen]
    prog = IV.Progress(Path(args.out_dir) / "progress" / f"scan{args.shard}.json",
                       len(mine), f"scan/{args.shard}", already_done=len(mine) - len(todo))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    fam = load_family(model, processor, args.system_prompt)
    print(f"[scan] family={fam.name}  system prompt="
          f"{'(none)' if not fam.system_prompt else fam.system_prompt[:40] + '...'}",
          flush=True)
    scan = SL.install(model, family=fam, want_key_stats=not args.no_key_stats)
    tap = SL.VisionTap(model, family=fam).install()
    try:
        from PIL import Image
        for r in todo:
            im = Image.open(r["path"]).convert("RGB")
            got = measure(model, processor, [im], r["question"], device, scan, tap,
                          want_hidden=not args.no_hidden,
                          max_new_tokens=args.max_new_tokens, min_mass=args.min_mass)
            if got is None:
                print(f"[scan] {r['key']}: no picture located, skipped", flush=True)
                prog.tick()
                continue
            gh, gw = got["grid"]
            # The covariates are read on the region the ENCODER saw. Handing them the
            # whole picture where the processor centre-cropped it would line every
            # covariate up against the wrong patch, silently, in the one table that puts
            # "border" and "background" in the same units.
            got["view"] = fam.view_box(im)
            cs = SL.content_stats(VF.view_crop(im, got["view"]), gh, gw)
            got["content"] = np.stack([cs[k] for k in CONTENT_KEYS])
            sink.write(r["key"],
                       meta_of(got, r, {"n_generated": got.get("n_generated")}, fam),
                       dict(arrays_of(got), content=got["content"]))
            prog.tick()
    finally:
        sink.close()
        tap.uninstall()
        scan.uninstall()
    prog.close()
    return 0


CONTENT_KEYS = ("pix_var", "edge", "sat", "blank")


# ---------------------------------------------------------------------------
# stage: arms
# ---------------------------------------------------------------------------
def stage_arms(args):
    import torch
    from PIL import Image

    rows = [r for r in read_manifest(args.out_dir, args.types) if not r["dev"]]
    rng = np.random.default_rng(args.seed)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    chosen = []
    for t, rs in sorted(by_type.items()):
        idx = rng.permutation(len(rs))[: args.arm_rows_per_type]
        chosen += [rs[i] for i in sorted(idx)]
    mine = chosen[args.shard::args.num_shards]

    sink = Sink(args.out_dir, "arms", args.shard, args.flush_every)
    seen = sink.done()
    units = [(r, a) for r in mine for a in args.arms
             if f"{r['key']}|{a}" not in seen]
    prog = IV.Progress(Path(args.out_dir) / "progress" / f"arms{args.shard}.json",
                       len(mine) * len(args.arms), f"arms/{args.shard}",
                       already_done=len(mine) * len(args.arms) - len(units))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    fam = load_family(model, processor, args.system_prompt)
    print(f"[arms] family={fam.name}", flush=True)
    scan = SL.install(model, family=fam, want_key_stats=False)
    partners = _partner_images(chosen)
    try:
        for r, arm in units:
            im = Image.open(r["path"]).convert("RGB")
            out = _run_arm(model, processor, im, r, arm, device, scan, partners, args,
                           fam)
            for got, extra, suffix in out:
                if got is None:
                    print(f"[arms] {r['key']}|{arm}: "
                          f"{extra.get('skipped', 'no picture')}", flush=True)
                    continue
                got.setdefault("view", fam.view_box(im))
                sink.write(f"{r['key']}|{arm}{suffix}",
                           meta_of(got, r, dict(extra, arm=arm), fam), arrays_of(got))
            prog.tick()
    finally:
        sink.close()
        scan.uninstall()
    prog.close()
    return 0


PROMPT_SWAPS = ("Describe the image.", "What is in this picture?",
                "Answer the question about this image.")


def _partner_images(rows):
    """For `two_images`: each picture's partner, drawn from a DIFFERENT type.

    A partner of the same type would confound "is the second picture also a sink" with
    "are these two pictures alike". Deterministic in the manifest order so the arm is
    reproducible without carrying a seed into the results.
    """
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    types = sorted(by_type)
    out = {}
    for i, t in enumerate(types):
        other = by_type[types[(i + 1) % len(types)]]
        for j, r in enumerate(by_type[t]):
            out[r["key"]] = other[j % len(other)]
    return out


def _run_arm(model, processor, im, row, arm, device, scan, partners, args, fam):
    """One (picture, arm) measurement. -> [(got, extra, unit suffix)].

    A list because InternVL's `tiled` arm produces one measurement per tile and every
    other arm produces exactly one. The suffix is what keeps those apart in the unit key,
    and therefore in the resume set.
    """
    from PIL import Image

    if arm in ("permute", "permute_identity"):
        mode = "shuffle" if arm == "permute" else "identity"
        perm = SL.PatchPermute(model, mode=mode, seed=args.seed, family=fam).install()
        try:
            got = measure(model, processor, [im], row["question"], device, scan,
                          want_hidden=False)
        finally:
            perm.uninstall()
        if got is None:
            return [(None, {}, "")]
        got["perm"] = (None if perm.perm is None
                       else perm.perm.detach().cpu().numpy().astype(np.int32))
        return [(got, {"perm_mode": mode}, "")]

    if arm in ("permute_pixels", "permute_pixels_identity"):
        # A10. The encoder's OWN position embeddings are the one thing A9 cannot rule
        # out: A9 shuffles the rows the encoder emitted, so the stamp travels with the
        # token either way. This shuffles the PIXELS of each grid cell before the encoder
        # runs, so if the attractor appears at whatever content now sits at the top-left
        # of the ViT grid, the encoder's position embedding is writing it.
        mode = "shuffle" if arm == "permute_pixels" else "identity"
        grid = fam.grid_of(processor, VF.view_crop(im, fam.view_box(im)))
        src = VF.view_crop(im, fam.view_box(im))
        tim, perm = VF.block_permute(src, grid, fam.encoder_px, seed=args.seed,
                                     mode=mode)
        got = measure(model, processor, [tim], row["question"], device, scan,
                      want_hidden=False)
        if got is None:
            return [(None, {}, "")]
        if tuple(got["grid"]) != tuple(grid):
            # The blocks were cut on a grid the processor then did not choose, so the
            # permutation does not describe what the encoder saw. Refuse rather than
            # report a correspondence for a different picture.
            return [(None, {"skipped": f"grid moved {grid} -> {tuple(got['grid'])}"}, "")]
        got["perm"] = np.asarray(perm, dtype=np.int32)
        got["view"] = SL.FULL_VIEW          # the cell-aligned crop IS the whole picture
        return [(got, {"perm_mode": mode, "block_grid": list(grid)}, "")]

    if arm == "tiled":
        # InternVL only: the same picture with its dynamic tiling switched back on, so a
        # picture becomes up to 12 tiles of 448px plus a thumbnail, each its own 16x16
        # grid. Scored PER TILE, because "the outer ring" of a tile is an interior edge
        # of the picture and conflating the two produces a confident, meaningless number.
        if fam.name != "internvl":
            return [(None, {"skipped": f"tiling is not a thing on {fam.name}"}, "")]
        out = []
        for k in range(args.max_tiles):
            got = measure(model, processor, [im], row["question"], device, scan,
                          want_hidden=False, tile=k, crop_to_patches=True)
            if got is None:
                break
            out.append((got, {"tile_of": got["n_grids"]}, f"#{k}"))
        return out or [(None, {"skipped": "no tile located"}, "")]

    if arm == "two_images":
        partner = partners.get(row["key"])
        if partner is None:
            return [(None, {"skipped": "no partner picture"}, "")]
        other = Image.open(partner["path"]).convert("RGB")
        got = measure(model, processor, [im, other], row["question"], device, scan,
                      want_hidden=False)
        return [(got, {"partner": partner["key"],
                       "partner_type": partner["type"]}, "")]

    if arm == "prompt_swap":
        q = PROMPT_SWAPS[abs(hash(row["key"])) % len(PROMPT_SWAPS)]
        got = measure(model, processor, [im], q, device, scan, want_hidden=False)
        return [(got, {"question_used": q}, "")]

    px = fam.patch_px(im)
    tim, inv, tmeta = SL.transform(arm, im, patch_px=px)
    tmeta = dict(tmeta, patch_px=px)
    if tim is None:
        return [(None, tmeta, "")]
    got = measure(model, processor, [tim], row["question"], device, scan,
                  want_hidden=False)
    if got is None:
        return [(None, tmeta, "")]
    got["view"] = fam.view_box(tim)
    return [(got, dict(tmeta, transform=arm), "")]


# ---------------------------------------------------------------------------
# stage: selftest
# ---------------------------------------------------------------------------
def bright_patch_image(size, grid, where, view=SL.FULL_VIEW):
    """A flat dark field with one bright square, at a KNOWN patch of a known grid.

    `view` is the part of the picture the grid covers. On LLaVA-1.5 the processor
    centre-crops, so a marker painted at cell (r, c) of the PICTURE would land at a
    different cell of the GRID -- which is the off-by-one this check exists to find, and
    would find in the wrong place.
    """
    from PIL import Image

    W, H = size
    gh, gw = grid
    r, c = where
    u0, v0, u1, v1 = view
    x0 = int(W * (u0 + c * (u1 - u0) / gw))
    x1 = max(x0 + 1, int(W * (u0 + (c + 1) * (u1 - u0) / gw)))
    y0 = int(H * (v0 + r * (v1 - v0) / gh))
    y1 = max(y0 + 1, int(H * (v0 + (r + 1) * (v1 - v0) / gh)))
    a = np.full((H, W, 3), 40, dtype=np.uint8)
    a[y0:y1, x0:x1] = 240
    return Image.fromarray(a, "RGB")


def _within(flat_idx, gh, gw, radius):
    """Flat boolean mask of the patches within `radius` Chebyshev of one patch."""
    r, c = divmod(int(flat_idx), gw)
    rr = np.abs(np.arange(gh)[:, None] - r)
    cc = np.abs(np.arange(gw)[None, :] - c)
    return (np.maximum(rr, cc) <= radius).reshape(-1)


def content_response(image, blank, gh, gw, view=SL.FULL_VIEW):
    """Edge energy of the probe minus edge energy of the SAME transform on a flat field.

    Brightness will not do. `pad_white` paints a border brighter than the marker, `canvas`
    lays a mid-grey field over most of the picture, and both would win an argmax over
    grey levels while telling us nothing about where the marker went. Differencing
    against the transform's own blank cancels every edge the transform itself introduced
    and leaves only the marker.

    Read on the region the encoder sees, not on the picture: outside the view there is no
    patch for the energy to land in.
    """
    a, b = VF.view_crop(image, view), VF.view_crop(blank, view)
    return SL.content_stats(a, gh, gw)["edge"] - SL.content_stats(b, gh, gw)["edge"]


def frame_check(grid_fn, size, view_fn=None, patch_px=32, tol=1, min_response=1e-3,
                ratio=3.0):
    """Every transform's pixel->patch mapping, against a picture with one marked patch.

    The check runs BACKWARDS, which is the only way one probe picture can serve every
    arm: pick a patch of the TRANSFORMED grid, ask `patch_correspondence` which baseline
    patch it claims to show, put the marker there, and see whether the transformed
    picture's marker lands on the patch we picked. Forwards -- one fixed marker, every
    arm -- cannot work, and finding out why was worth the failed run: `zoom60` crops the
    border away, `donut` paints the middle out, and `canvas` shrinks the marker until a
    flat grey field outscores it. No single placement survives them all.

    Nor does one target patch per arm. A central target is the right choice for `zoom`
    and the wrong one for `donut`, which erases exactly that. So the candidates are tried
    in order of centrality and the first one whose marker SURVIVES the transform -- a
    response above `min_response` and `ratio` times the rest of the grid -- is the one
    the assertion is made on. An arm where no candidate survives is a failure, not a skip.

    `tol` is Chebyshev patches. Resampling spreads a one-patch marker over its neighbours
    under `zoom`, `pad` and `canvas`, so 1 is the honest tolerance -- and it is still far
    tighter than any rotation error, which lands the marker on the wrong side entirely.
    """
    from PIL import Image

    view_fn = view_fn or (lambda im: SL.FULL_VIEW)
    failures, checked = [], 0
    flat = Image.new("RGB", size, (40, 40, 40))
    gh0, gw0 = grid_fn(flat)
    view0 = view_fn(flat)
    for arm in SL.ARMS:
        blank, inv, _meta = SL.transform(arm, flat, patch_px=patch_px)
        if blank is None:
            continue
        gh, gw = grid_fn(blank)
        view = view_fn(blank)
        corr = SL.patch_correspondence(inv, gh, gw, gh0, gw0, view, view0)
        order = sorted(range(gh * gw),
                       key=lambda i: abs(i // gw - (gh - 1) / 2) + abs(i % gw - (gw - 1) / 2))
        checked += 1
        for target in order:
            if corr[target] < 0:
                continue
            base = int(corr[target])
            probe = bright_patch_image(size, (gh0, gw0), (base // gw0, base % gw0), view0)
            resp = content_response(SL.transform(arm, probe, patch_px=patch_px)[0],
                                    blank, gh, gw, view)
            peak = int(np.argmax(resp))
            # "Localised" has to mean localised to a NEIGHBOURHOOD, not to one patch.
            # `zoom60` magnifies the marker by 1/0.6 and `res384` resamples the grid, so
            # in both the marker legitimately straddles two patches and its immediate
            # neighbour scores nearly as high. Comparing against the best OTHER patch
            # rejected those two arms as "not surviving" when they had survived perfectly
            # well; comparing against everything outside the peak's own neighbourhood is
            # the test that was meant.
            near = _within(peak, gh, gw, tol)
            far = resp[~near]
            if resp[peak] < min_response or (far.size
                                             and resp[peak] < ratio * max(far.max(), 1e-12)):
                continue                       # this patch does not survive the transform
            d = max(abs(peak // gw - target // gw), abs(peak % gw - target % gw))
            if d > tol:
                failures.append(f"{arm}(marker at {peak}, correspondence says {target})")
            break
        else:
            failures.append(f"{arm}(no patch survives the transform at all)")
    return failures, checked


def stage_selftest(args):
    import torch
    from PIL import Image

    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    rows = read_manifest(args.out_dir, args.types)[: args.selftest_rows]
    if not rows:
        raise SystemExit("no corpus; run --stage corpus first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    fam = load_family(model, processor, args.system_prompt)
    print(f"\nselftest  model={args.model}  family={fam.name}  {len(rows)} pictures")
    print(f"          decoder attention {fam.attn_classes}, rows from "
          f"{fam.row_classes}, image token {fam.image_token_id}, delimiters "
          f"{fam.vision_start_ids or '(none)'}/{fam.vision_end_ids or '(none)'}",
          flush=True)

    # 1. the scan edits nothing -----------------------------------------------
    # Not against zero. The scan takes the explicit float32 softmax where the fused kernel
    # works in bfloat16, so it CANNOT be bit-identical, and an absolute threshold on the
    # logits is a guess about how much bf16 drifts over 36 layers. The reference is
    # transformers' OWN eager path, which differs from SDPA for exactly the same reason
    # and is not under test: the scan has to be no further from the fused kernel than
    # stock unfused attention already is.
    im0 = Image.open(rows[0]["path"]).convert("RGB")
    inputs = build_inputs(fam, processor, [im0], rows[0]["question"], device)

    def last_logits():
        with torch.no_grad():
            return model(**inputs, use_cache=False).logits[0, -1].float().cpu()

    def greedy_ids():
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=args.selftest_tokens,
                                  do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id)[0].tolist()

    base_logits, base_ids = last_logits(), greedy_ids()
    with _attn_impl(model, "eager", fam):
        eager_logits, eager_ids = last_logits(), greedy_ids()
    scan = SL.install(model, family=fam)
    try:
        scan_logits, scan_ids = last_logits(), greedy_ids()
        d_scan = float((scan_logits - base_logits).abs().max())
        d_eager = float((eager_logits - base_logits).abs().max())
        check("the scan is no further from the fused kernel than stock eager is",
              d_scan <= 2 * d_eager + 1e-4,
              f"scan {d_scan:.2e} vs eager {d_eager:.2e} (both against sdpa)")
        check("the scan picks the same next token",
              int(scan_logits.argmax()) == int(base_logits.argmax()))
        # Against EAGER, not against SDPA. The reference has to be the path the scan is a
        # copy of: both take an explicit softmax where the fused kernel does not, so both
        # drift from it by the same bf16 rounding, and on a model whose greedy decode has
        # near-ties -- LLaVA-1.5's fp16 weights read in bf16 are one -- that drift flips a
        # token ten steps in. Asserting scan == SDPA there fails a scan that is provably
        # doing nothing, and the number that says so is printed rather than dropped.
        check("the scan reproduces stock unfused attention (greedy tokens)",
              scan_ids == eager_ids,
              f"{sum(a == b for a, b in zip(scan_ids, eager_ids))}/{len(eager_ids)} equal")
        same_sdpa = sum(a == b for a, b in zip(eager_ids, base_ids))
        print(f"        (unfused vs the fused kernel: {same_sdpa}/{len(base_ids)} tokens "
              f"equal -- bf16 kernel drift, and not something the scan did)")

        # 2. geometry and the negative controls -------------------------------
        got = measure(model, processor, [im0], rows[0]["question"], device, scan,
                      want_hidden=False)
        gh, gw = got["grid"]
        ring = SL.ring_set(gh, gw)
        check("ring area is (2gh+2gw-4)/(gh*gw)",
              abs(ring.mean() - SL.ring_area_frac(gh, gw)) < 1e-12,
              f"{gh}x{gw}: {ring.mean():.4f}")
        rng = np.random.default_rng(0)
        flat = rng.random((1, 1, gh * gw))
        st, _pk = SL.reduce_cells(flat, None, 1, flat.sum(-1), gh, gw, 100)
        for name, want in (("ctrl_block_share", SL.named_sets(gh, gw)["ctrl_block"].mean()),
                           ("ring_share", ring.mean())):
            e = st[0, 0, SL.STAT_INDEX[name]] / want
            check(f"a shuffled map gives enrichment 1.0 for {name}", abs(e - 1) < 0.35,
                  f"{e:.3f}")

        # 3. THE COORDINATE FRAME ---------------------------------------------
        # On the real pictures' own sizes, not a convenient one: the processor rounds the
        # grid, and a size where the rounding is benign proves nothing about the corpus.
        # Each family needs its OWN frame check: the patch ordering after InternVL's
        # pixel shuffle, and the centre crop LLaVA's processor takes, are exactly where
        # this would go quietly wrong.
        bad, checked = [], 0
        for size in sorted({tuple(r["size"]) for r in rows}):
            f, n = frame_check(lambda im: fam.grid_of(processor, im), size,
                               view_fn=fam.view_box,
                               patch_px=fam.patch_px(Image.new("RGB", size)))
            bad += [f"{size}:{x}" for x in f]
            checked += n
        check("every transform's pixel->patch mapping decodes where it claims",
              not bad, ", ".join(bad) if bad else f"{checked} (transform, size) pairs")
        probe_im = bright_patch_image(tuple(rows[0]["size"]), (1, 1), (0, 0))
        pg = fam.grid_of(processor, probe_im)
        tg90 = fam.grid_of(processor, SL.transform("rot90", probe_im)[0])
        want90 = (pg[1], pg[0]) if fam.fixed_grid is None else tuple(pg)
        check("rot90 does to the grid what this family's processor does",
              tuple(tg90) == want90,
              f"{pg} -> {tg90}" + ("  (fixed grid: it cannot transpose)"
                                   if fam.fixed_grid else ""))

        # 4. the permutation is a permutation ---------------------------------
        tap = SL.VisionTap(model, family=fam)
        perm = SL.PatchPermute(model, mode="shuffle", seed=7, family=fam).install()
        tap.install()                      # registered second, so it sees the permuted rows
        try:
            measure(model, processor, [im0], rows[0]["question"], device, scan,
                    want_hidden=False)
            shuffled = tap.norms.copy()
            p = perm.perm.detach().cpu().numpy()
        finally:
            tap.uninstall(); perm.uninstall()
        tap2 = SL.VisionTap(model, family=fam).install()
        try:
            measure(model, processor, [im0], rows[0]["question"], device, scan,
                    want_hidden=False)
            plain = tap2.norms.copy()
        finally:
            tap2.uninstall()
        check("the patch permutation moves the same vectors",
              np.allclose(np.sort(plain), np.sort(shuffled)) and
              np.allclose(shuffled, plain[p]),
              f"{len(p)} patches")
        ident = SL.PatchPermute(model, mode="identity", family=fam).install()
        try:
            g2 = measure(model, processor, [im0], rows[0]["question"], device, scan,
                         want_hidden=False)
        finally:
            ident.uninstall()
        check("permutation mode=identity is the identity",
              np.allclose(g2["maps"], got["maps"], atol=1e-6))

        # 4b. A10 -- the PIXEL-BLOCK permutation, before the encoder -----------
        # The arm that separates "the mark is in the embedding" from "the encoder's own
        # position embedding put it there". Its correctness rests on two things: that the
        # blocks really are a permutation of the picture's cells, and that the grid the
        # blocks were cut on is the grid the processor then chooses -- otherwise the
        # correspondence describes a different picture.
        base_view = fam.view_box(im0)
        src = VF.view_crop(im0, base_view)
        grid0 = fam.grid_of(processor, src)
        p_im, p_perm = VF.block_permute(src, grid0, fam.encoder_px, seed=7)
        i_im, i_perm = VF.block_permute(src, grid0, fam.encoder_px, mode="identity")
        check("A10 cuts the picture into whole grid cells and puts them all back",
              sorted(p_perm.tolist()) == list(range(grid0[0] * grid0[1]))
              and np.array_equal(i_perm, np.arange(grid0[0] * grid0[1]))
              and p_im.size == i_im.size,
              f"{grid0[0]}x{grid0[1]} blocks of {fam.encoder_px}px")
        check("A10's pixels are the same pixels, only moved",
              np.array_equal(np.sort(np.asarray(p_im).reshape(-1)),
                             np.sort(np.asarray(i_im).reshape(-1))))
        g10 = measure(model, processor, [p_im], rows[0]["question"], device, scan,
                      want_hidden=False)
        g10i = measure(model, processor, [i_im], rows[0]["question"], device, scan,
                       want_hidden=False)
        check("A10 keeps the grid the blocks were cut on",
              g10 is not None and g10i is not None
              and tuple(g10["grid"]) == tuple(grid0) == tuple(g10i["grid"]),
              f"cut on {tuple(grid0)}, got "
              f"{None if g10 is None else tuple(g10['grid'])}")

        # 5. the causal column correction -------------------------------------
        cn = scan.column_null("image")
        check("the image->image query set is corrected by its own position-blind null",
              cn is not None and cn[0] > cn[-1] > 0,
              f"{cn[0]:.2e} down to {cn[-1]:.2e}" if cn is not None else "missing")
        check("the text query set needs no correction",
              scan.column_null("text") is None)

        if args.tie_back:
            # `sink_shift` is this project's Qwen3-VL edit machinery, and the tie-back
            # borrows its collector to read a generated chain. On another family the
            # honest route is --max-new-tokens, which measures the same thing through
            # this module's own `generated` query set.
            if fam.name != TRAINED_FAMILY:
                print("\n  tie-back: skipped -- it reads through sink_shift, which is "
                      f"{TRAINED_FAMILY} only.\n  Use --max-new-tokens for the "
                      "generated-token readout on this family.")
            else:
                ok &= _tie_back(model, processor, rows, device, scan, args)
    finally:
        scan.uninstall()

    print(f"\n  {'SELFTEST PASS' if ok else 'SELFTEST FAIL'}")
    return 0 if ok else 1


class _attn_impl:
    """Swap the text decoder's attention implementation for the length of a `with`.

    The same two-place switch `SinkScan.install` makes -- the text config AND every
    attention module's own config -- because relying on them being one object is how a
    reference measurement quietly becomes a second measurement of the thing under test.
    """

    def __init__(self, model, impl, family):
        self.model, self.impl, self.prev = model, impl, None
        self.family = family

    def __enter__(self):
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self.cfg, self.prev = cfg, cfg._attn_implementation
        self._set(self.impl)
        return self

    def __exit__(self, *exc):
        self._set(self.prev)
        return False

    def _set(self, impl):
        self.cfg._attn_implementation = impl
        for m in self.model.modules():
            if type(m).__name__ in self.family.attn_classes:
                m.config._attn_implementation = impl


def _tie_back(model, processor, rows, device, scan, args):
    """1a and 1b: reproduce the generated-token readout, and price the prefill proxy.

    The claim under test was measured on tokens the model WROTE. The whole experiment is
    prefill-only. This is the check that licenses the substitution, and its threshold is
    fixed in the design document rather than chosen after seeing the number.
    """
    import torch

    from PIL import Image
    print("\n  tie-back: prefill vs generated tokens, on the same pictures")
    pre, gen = [], []
    ss = None
    for r in rows[: args.tieback_rows]:
        im = Image.open(r["path"]).convert("RGB")
        got = measure(model, processor, [im], r["question"], device, scan,
                      want_hidden=False)
        if got is None:
            continue
        gh, gw = got["grid"]
        ring = SL.ring_set(gh, gw).reshape(-1)
        cell = got["stats"][TRAINED_LAYER, list(TRAINED_HEADS)].mean(0)
        pre.append(cell[SL.STAT_INDEX["ring_share"]] / SL.ring_area_frac(gh, gw))

        scan.uninstall()
        ss = SS.install(model, arm="centre", alpha=0.0, layers=[TRAINED_LAYER],
                        heads=list(TRAINED_HEADS)).collect(TRAINED_LAYER,
                                                           list(TRAINED_HEADS))
        try:
            inputs = build_inputs(scan.family, processor, [im], r["question"], device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=args.tieback_tokens,
                               do_sample=False,
                               pad_token_id=processor.tokenizer.pad_token_id)
            smap = SS.collected_map(ss)
        finally:
            ss.uninstall()
            scan.install()
        if smap is None or smap.sum() <= 0:
            pre.pop()
            continue
        gen.append(float(smap.reshape(-1)[ring].sum() / smap.sum())
                   / SL.ring_area_frac(gh, gw))
    if len(pre) < 4:
        print("    too few paired pictures to say anything")
        return True
    a, b = np.asarray(pre), np.asarray(gen)
    rho = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    print(f"    prefill ring enrichment {a.mean():.3f}, generated {b.mean():.3f}, "
          f"Spearman {rho:.3f} over {len(a)} pictures")
    good = rho >= args.tieback_rho
    print(f"  {'PASS' if good else 'FAIL'}  the prefill readout tracks the generated one "
          f"(pre-registered rho >= {args.tieback_rho})")
    if not good:
        print("        Below threshold the honest move is to pay for generation, not to "
              "reword\n        the conclusion. See the design document, stage 1b.")
    return good


# ---------------------------------------------------------------------------
# reading results back
# ---------------------------------------------------------------------------
def read_stage(out_dir, stage):
    """-> (metadata rows, {unit: {field: array}}). Parts are globbed, order is by unit.

    DEDUPLICATED BY UNIT, keeping the last write. Resume is per shard -- a shard skips
    what is in its OWN jsonl -- so re-running the same directory at a different shard
    count re-measures the units that changed hands, and both copies are on disk. The
    arrays are a dict and collapse on their own; the metadata is a list and would not,
    which would quietly double-weight those pictures in every average below.
    """
    d = Path(out_dir)
    seen = {}
    for p in sorted(d.glob(f"{stage}_shard*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[r.get("unit")] = r
    meta = list(seen.values())
    arrays = {}
    for p in sorted(d.glob(f"{stage}_shard*_part*.npz")):
        with np.load(p, allow_pickle=False) as z:
            units = [str(u) for u in z["units"]]
            for name in z.files:
                if name == "units" or name.endswith(("__shapes", "__idx")):
                    continue
                flat, off = z[name], 0
                for i, sh in zip(z[name + "__idx"], z[name + "__shapes"]):
                    n = int(np.prod(sh))
                    a = flat[off:off + n].reshape(sh)
                    arrays.setdefault(units[int(i)], {})[name] = _pad_stats(name, a)
                    off += n
    return meta, arrays


def _pad_stats(name, a):
    """Widen a stats array written before a statistic was added. -> NaN in the new slots.

    `STAT_NAMES` is append-only by contract, so a run from before an append is a prefix
    of the current layout and padding it is exact. Without this, `STAT_INDEX` would read
    a new name out of an old array and silently return whatever float happened to sit at
    that offset -- or, worse, index past the end only on some units.
    """
    if not name.startswith("stats") or a.ndim != 3 or a.shape[-1] >= len(SL.STAT_NAMES):
        return a
    out = np.full(a.shape[:-1] + (len(SL.STAT_NAMES),), np.nan, dtype=a.dtype)
    out[..., :a.shape[-1]] = a
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_mean(x, n_boot=10000, seed=20260907):
    """Mean with a bootstrap CI over the unit of analysis, which is the PICTURE."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)), int(x.size))


def boot_paired(a, b, n_boot=10000, seed=20260907):
    """Mean of a - b over the pictures BOTH ran, paired by common random numbers."""
    shared = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in shared], dtype=float)
    d = d[np.isfinite(d)]
    return boot_mean(d, n_boot, seed)


def holm(pvals):
    """Holm-Bonferroni, returned in the input order."""
    order = np.argsort(pvals)
    out = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out


def ci_excludes(lo, hi, value):
    return np.isfinite(lo) and np.isfinite(hi) and (lo > value or hi < value)


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def at_cells(stats, stat, cells):
    """The mean of one statistic over a set of (layer, head) cells. NaN-safe."""
    v = np.asarray([stats[l, h, SL.STAT_INDEX[stat]] for l, h in cells], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def choose_cells(meta, arrays, k, min_mass):
    """The k cells with the largest ring enrichment ON THE DEV SPLIT.

    Selection and reporting must not share pictures. A cell picked because it looked
    extreme on a sample will look extreme on that sample again; the test split is what
    makes the number mean something. Cells whose image mass is below `min_mass` are
    excluded outright -- a head that ignores the picture has a ring share, and it is
    noise wearing a statistic's name.
    """
    ring_i, mass_i = SL.STAT_INDEX["ring_share"], SL.STAT_INDEX["image_mass"]
    acc, mass, n = None, None, 0
    for m in meta:
        if not m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        e = np.asarray(a[..., ring_i], dtype=float) / m["ring_area_frac"]
        acc = e if acc is None else acc + np.nan_to_num(e)
        mm = np.asarray(a[..., mass_i], dtype=float)
        mass = mm if mass is None else mass + np.nan_to_num(mm)
        n += 1
    if acc is None:
        return [(TRAINED_LAYER, h) for h in TRAINED_HEADS], 0
    acc, mass = acc / max(1, n), mass / max(1, n)
    acc = np.where(mass >= min_mass, acc, -np.inf)
    flat = np.argsort(acc.reshape(-1))[::-1][:k]
    return [(int(i // acc.shape[1]), int(i % acc.shape[1])) for i in flat], n


def report_budget(meta, arrays):
    """Where the attention row actually is. Everything later is read against this."""
    tot = {s: [] for s in SL.SPANS}
    for m in meta:
        a = arrays.get(m["unit"], {}).get("spans")
        if a is None:
            continue
        for i, s in enumerate(SL.SPANS):
            tot[s].append(float(np.nanmean(a[..., i])))
    if not tot["image"]:
        return
    print("\n" + "=" * 78)
    print("1. THE BUDGET -- what share of an attention row lands where, over all "
          "layers and heads")
    print(f"    {'span':<14} {'share of the row':>18}")
    for s in SL.SPANS:
        m, lo, hi, n = boot_mean(tot[s])
        print(f"    {s:<14} {m:>18.4f}   [{lo:.4f}, {hi:.4f}]  n={n}")
    img = boot_mean(tot["image"])[0]
    print(f"\n  The picture receives {img:.4f} of a row. Read every ring percentage "
          f"below against\n  that: at {img:.3f} the ring's mass is at most "
          f"{img:.4f} of what the model attends to, and\n  the honest claim is 'WITHIN "
          "the picture, the border is favoured'.")


def report_types(meta, arrays, cells, args):
    print("\n" + "=" * 78)
    print(f"2. PER IMAGE TYPE, at the {len(cells)} dev-selected cells "
          f"{cells[:4]}{'...' if len(cells) > 4 else ''}")
    print("   E_ring = the border's share of the picture's attention / the border's share")
    print("   of the patches. 1.0 = no effect. L_ring = P(peak on the border) - that same")
    print("   chance level. Pre-registered: E_ring >= 1.5 is 'the sink concentrates on the")
    print("   ring'; below 1.2 with a CI excluding 1.5 is a FAILURE of the claim for that")
    print("   type. CIs bootstrap over pictures; p is Holm-adjusted across types.")
    print(f"\n    {'type':<18} {'n':>4} {'grid':>8} {'ring area':>10} {'E_ring':>8} "
          f"{'95% CI':>18} {'L_ring':>8} {'95% CI':>18} {'verdict':>10}")
    types = sorted({m["type"] for m in meta})
    rows, pv = [], []
    for t in types:
        sel = lambda m, t=t: m["type"] == t and not m.get("dev")   # noqa: E731
        e = {}
        lift = {}
        grids, area = [], []
        for m in meta:
            if not sel(m):
                continue
            a = arrays.get(m["unit"], {}).get("stats")
            if a is None:
                continue
            e[m["key"]] = at_cells(a, "ring_share", cells) / m["ring_area_frac"]
            lift[m["key"]] = at_cells(a, "peak_in_ring", cells) - m["ring_area_frac"]
            grids.append(tuple(m["grid"]))
            area.append(m["ring_area_frac"])
        em, elo, ehi, n = boot_mean(list(e.values()), args.n_boot)
        lm, llo, lhi, _ = boot_mean(list(lift.values()), args.n_boot)
        modal = max(set(grids), key=grids.count) if grids else (0, 0)
        # a two-sided bootstrap p for "E_ring = 1", read off the CI's tail count
        pv.append(_boot_p(list(e.values()), 1.0, args.n_boot))
        rows.append((t, n, modal, float(np.mean(area)) if area else np.nan,
                     em, elo, ehi, lm, llo, lhi))
    adj = holm(np.asarray(pv)) if pv else []
    for (t, n, modal, area, em, elo, ehi, lm, llo, lhi), p in zip(rows, adj):
        if not np.isfinite(em):
            verdict = "no data"
        elif em >= 1.5 and elo > 1.2 and p < 0.05:
            verdict = "RING"
        elif ehi < 1.5:
            verdict = "FAILS"
        else:
            verdict = "weak"
        print(f"    {t:<18} {n:>4} {f'{modal[0]}x{modal[1]}':>8} {area:>10.3f} "
              f"{em:>8.3f} {f'[{elo:.3f}, {ehi:.3f}]':>18} {lm:>+8.3f} "
              f"{f'[{llo:+.3f}, {lhi:+.3f}]':>18} {verdict:>10}   p={p:.3g}")
    print("\n  RING = the claim holds for this type. FAILS = the CI excludes the "
          "pre-registered\n  effect size, so the border is NOT where this type's "
          "attention concentrates.")


def _boot_p(vals, null, n_boot, seed=20260907):
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if x.size < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    p = 2 * min((m <= null).mean(), (m >= null).mean())
    return float(min(1.0, max(p, 1.0 / n_boot)))


def report_shape(meta, arrays, cells, args):
    """The radial profile and the four edges: where on the border, and is it symmetric."""
    print("\n" + "=" * 78)
    print("3. THE SHAPE OF IT -- radial profile, the four edges, and the two end patches")
    print("   Each column is enrichment: that set's share of the picture's attention over")
    print("   its share of the patches. A 2D BORDER effect has no reason to prefer the top")
    print("   over the bottom, or the first corner over the other three. A SEQUENCE effect")
    print("   has every reason to: the picture is raster-ordered, so `top` and `left` are")
    print("   its early tokens and `first` is the token right after <|vision_start|>.")
    names = [("ring_share", "ring"), ("depth1_share", "depth1"), ("deep_share", "deep"),
             ("top_share", "top"), ("bottom_share", "bottom"), ("left_share", "left"),
             ("right_share", "right"), ("corner_share", "corner"),
             ("first_patch_share", "first"), ("last_patch_share", "last"),
             ("ctrl_block_share", "ctrl_int")]
    pooled = {}
    print(f"\n    {'type':<18} " + " ".join(f"{n:>8}" for _s, n in names))
    for t in sorted({m["type"] for m in meta}):
        vals = {n: [] for _s, n in names}
        for m in meta:
            if m["type"] != t or m.get("dev"):
                continue
            a = arrays.get(m["unit"], {}).get("stats")
            if a is None:
                continue
            gh, gw = m["grid"]
            sets = SL.named_sets(gh, gw)
            one = 1.0 / (gh * gw)
            frac = {k: sets[k].mean() for k in
                    ("ring", "depth1", "deep", "top", "bottom", "left", "right", "corner")}
            frac.update(first=one, last=one, ctrl_int=sets["ctrl_block"].mean())
            for s, n in names:
                if frac[n] > 0:
                    vals[n].append(at_cells(a, s, cells) / frac[n])
        print(f"    {t:<18} " + " ".join(
            f"{np.nanmean(vals[n]) if vals[n] else float('nan'):>8.2f}"
            for _s, n in names))
        for _s, n in names:
            pooled.setdefault(n, []).extend(vals[n])
    print("\n  `first` and `last` are single patches -- the top-left and bottom-right")
    print("  corners -- priced against a flat map's 1/N. `corner` averages all four, so a")
    print("  `first` far above `corner` says the sink is ONE token and not the geometry.")
    print("\n  `ctrl_int` is a contiguous ring-sized block placed at random, which lands in")
    print("  the INTERIOR. On a real map it should read like `depth1`/`deep`, not like 1.00:")
    print("  a depleted interior is the same fact as an enriched ring, stated twice. The")
    print("  metric's own check is the selftest, which runs it on a SHUFFLED map, where the")
    print("  answer must be 1.00 and nothing else.")
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in pooled.items()}


def report_cells(meta, arrays, args):
    """Is this one cell, or is it the model? The layer x head picture."""
    ring_i, mass_i = SL.STAT_INDEX["ring_share"], SL.STAT_INDEX["image_mass"]
    acc, mass, n = None, None, 0
    for m in meta:
        if m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        e = np.asarray(a[..., ring_i], dtype=float) / m["ring_area_frac"]
        acc = np.nan_to_num(e) if acc is None else acc + np.nan_to_num(e)
        mm = np.nan_to_num(np.asarray(a[..., mass_i], dtype=float))
        mass = mm if mass is None else mass + mm
        n += 1
    if acc is None:
        return
    acc, mass = acc / n, mass / n
    live = mass >= args.min_mass
    print("\n" + "=" * 78)
    print("4. IS IT ONE CELL OR IS IT THE MODEL")
    print(f"    {int(live.sum())} of {acc.size} (layer, head) cells clear the "
          f"{args.min_mass} image-mass floor")
    if live.any():
        print(f"    of those, {float((acc[live] > 1).mean()):.3f} have E_ring > 1 and "
              f"{float((acc[live] > 1.5).mean()):.3f} have E_ring > 1.5")
    print(f"\n    {'layer':>5} {'mean E_ring':>12} {'best head':>10} {'that head':>10} "
          f"{'image mass':>11}")
    for l in range(acc.shape[0]):
        row = np.where(live[l], acc[l], np.nan)
        if not np.isfinite(row).any():
            continue
        h = int(np.nanargmax(row))
        print(f"    {l:>5} {np.nanmean(row):>12.3f} {h:>10} {row[h]:>10.3f} "
              f"{mass[l].mean():>11.5f}")
    if acc.shape[0] > TRAINED_LAYER and _family_of(meta) == TRAINED_FAMILY:
        for h in TRAINED_HEADS:
            print(f"    the rewarded cell L{TRAINED_LAYER}h{h}: E_ring "
                  f"{acc[TRAINED_LAYER, h]:.3f}, image mass "
                  f"{mass[TRAINED_LAYER, h]:.5f}")


def report_s3(meta, arrays, cells, args):
    """Is there a SINK, or only a peak? Magnitude and query-invariance, together.

    Reported twice. At the dev-selected cells, which is where every other table is read --
    and at the cell with the LARGEST column anywhere in the model, which is the fair place
    to look for a sink. The selected cells maximise ring enrichment, not magnitude, so
    finding no sink there would say nothing about whether the model has one.
    """
    rows = {"at the ring cells": [[], [], []], "at the strongest cell anywhere": [[], [], []]}
    mass_i, mag_i, cv_i = (SL.STAT_INDEX["image_mass"], SL.STAT_INDEX["peak_uniform_x"],
                           SL.STAT_INDEX["peak_cv"])
    best_cells = []
    for m in meta:
        if m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        sel = rows["at the ring cells"]
        for lst, stat in zip(sel, ("peak_uniform_x", "peak_cv", "entropy_norm")):
            lst.append(at_cells(a, stat, cells))
        mag = np.where(np.asarray(a[..., mass_i], dtype=float) >= args.min_mass,
                       np.asarray(a[..., mag_i], dtype=float), np.nan)
        if not np.isfinite(mag).any():
            continue
        flat = int(np.nanargmax(mag))
        l, h = divmod(flat, a.shape[1])
        best_cells.append((l, h))
        alt = rows["at the strongest cell anywhere"]
        alt[0].append(float(a[l, h, mag_i]))
        alt[1].append(float(a[l, h, cv_i]))
        alt[2].append(float(a[l, h, SL.STAT_INDEX["entropy_norm"]]))
    if not rows["at the ring cells"][0]:
        return
    print("\n" + "=" * 78)
    print("5. IS IT A SINK, OR ONLY A PEAK")
    verdicts = {}
    for where, (mag, cv, ent) in rows.items():
        if not mag:
            continue
        print(f"\n  {where}:")
        for name, vals, note in (
                ("peak / uniform", mag, "times uniform the top image column is"),
                ("peak CV across queries", cv, "small = the same column for every query"),
                ("normalised entropy", ent, "1.0 = the picture's attention is flat")):
            mm, lo, hi, n = boot_mean(vals, args.n_boot)
            print(f"    {name:<24} {mm:>8.3f}  [{lo:.3f}, {hi:.3f}]  n={n}   {note}")
        mm, c = boot_mean(mag, args.n_boot)[0], boot_mean(cv, args.n_boot)[0]
        verdicts[where] = ("a sink by both legs" if mm >= args.sink_x and c <= args.sink_cv
                           else "a peak, not a sink" if mm < args.sink_x
                           else "large but query-DEPENDENT -- a peak that moves")
    if best_cells:
        top = sorted({c: best_cells.count(c) for c in set(best_cells)}.items(),
                     key=lambda kv: -kv[1])[:5]
        print(f"\n    the strongest cell is {top[0][0]} on {top[0][1]}/{len(best_cells)} "
              f"pictures; the top five are {[c for c, _n in top]}")
    print(f"\n  Verdict at the pre-registered thresholds (>= {args.sink_x}x uniform and "
          f"CV <= {args.sink_cv}):")
    for where, v in verdicts.items():
        print(f"    {where:<32} {v}")
    if not any("sink by both" in v for v in verdicts.values()):
        print("\n  Then the word 'sink' should be dropped from the claim even if every ring")
        print("  number above is high: what is on the border is a PEAK, and a peak that")
        print("  moves with the query is not what the sink literature is describing. That")
        print("  is a real result about wording, not a null.")


def report_blank(meta, arrays, cells, args):
    """H2's home turf: when there IS a big blank interior, does the attention go there?"""
    rows = []
    for m in meta:
        if m.get("dev"):
            continue
        d = arrays.get(m["unit"], {})
        maps, content = d.get("maps"), d.get("content")
        if maps is None or content is None:
            continue
        gh, gw = m["grid"]
        blank = np.asarray(content[CONTENT_KEYS.index("blank")], dtype=float) > 0.5
        interior = (SL.depth_map(gh, gw) >= 1).reshape(-1)
        target = blank & interior
        if target.mean() < args.blank_min:
            continue
        p = np.asarray(maps, dtype=float).mean(0)
        p = p / max(p.sum(), 1e-30)
        rows.append((m["type"], float(p[target].sum() / target.mean()),
                     float(p[SL.ring_set(gh, gw).reshape(-1)].sum()
                           / SL.ring_area_frac(gh, gw))))
    if not rows:
        print("\n6. INTERIOR BLANK -- no picture has an interior blank region above "
              f"{args.blank_min:.0%} of the grid")
        return
    print("\n" + "=" * 78)
    print("6. INTERIOR BLANK vs THE BORDER -- the observational test of 'background'")
    print(f"   Restricted to pictures whose blank INTERIOR region covers at least "
          f"{args.blank_min:.0%} of the\n   grid. If sinks seek background, E_blank "
          "should beat E_ring here. If they seek the\n   border, E_blank sits near 1 "
          "while E_ring stays high.")
    print(f"\n    {'type':<18} {'n':>4} {'E_blank_interior':>17} {'E_ring':>9} "
          f"{'blank - ring':>14}")
    for t in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == t]
        b = boot_mean([r[1] for r in sub], args.n_boot)
        e = boot_mean([r[2] for r in sub], args.n_boot)
        d = boot_mean([r[1] - r[2] for r in sub], args.n_boot)
        print(f"    {t:<18} {len(sub):>4} {b[0]:>17.3f} {e[0]:>9.3f} "
              f"{d[0]:>+14.3f}  [{d[1]:+.3f}, {d[2]:+.3f}]")
    all_d = boot_mean([r[1] - r[2] for r in rows], args.n_boot)
    return {"blank_minus_ring": all_d[0], "blank_minus_ring_lo": all_d[1],
            "blank_minus_ring_hi": all_d[2], "n": len(rows)}


def report_content(meta, arrays, args):
    """M4: does `ring` survive once the content covariates are in the model?

    One pooled least-squares fit per type, on standardised per-patch covariates, with the
    patch's share of the picture's attention as the response. The reported quantity is the
    RING coefficient after content, and the share of variance each block explains. It is a
    linear fit on a bounded response and it is not the last word -- it is the number that
    puts "outer ring" and "background" in the same units, which nothing else here does.
    """
    print("\n" + "=" * 78)
    print("7. THE REGRESSION -- does 'ring' survive 'background'?")
    print("   response: the patch's share of the picture's attention x N (1.0 = its "
          "share of\n   a flat map). Covariates standardised within picture; unit of "
          "analysis is the patch.")
    got = []
    print(f"\n    {'type':<18} {'n_pix':>7} {'b_ring':>8} {'b_blank':>9} {'b_edge':>8} "
          f"{'b_pixvar':>9} {'R2 all':>7} {'R2 no ring':>11}")
    for t in sorted({m["type"] for m in meta}):
        X, y = [], []
        for m in meta:
            if m["type"] != t or m.get("dev"):
                continue
            d = arrays.get(m["unit"], {})
            maps, content = d.get("maps"), d.get("content")
            if maps is None or content is None:
                continue
            gh, gw = m["grid"]
            p = np.asarray(maps, dtype=float).mean(0)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                continue
            n = gh * gw
            y.append(p / s * n)
            c = np.asarray(content, dtype=float)
            cols = [SL.ring_set(gh, gw).reshape(-1).astype(float),
                    c[CONTENT_KEYS.index("blank")],
                    _z(c[CONTENT_KEYS.index("edge")]),
                    _z(c[CONTENT_KEYS.index("pix_var")]),
                    _z(SL.depth_map(gh, gw).reshape(-1).astype(float)),
                    np.ones(n)]
            X.append(np.stack(cols, axis=1))
        if not X:
            continue
        X, y = np.concatenate(X), np.concatenate(y)
        b, r2 = _ols(X, y)
        _b2, r2_noring = _ols(X[:, 1:], y)
        print(f"    {t:<18} {len(y):>7} {b[0]:>8.3f} {b[1]:>9.3f} {b[2]:>8.3f} "
              f"{b[3]:>9.3f} {r2:>7.3f} {r2_noring:>11.3f}")
        got.append((b[0], r2 - r2_noring))
    print("\n  b_ring is the ring's partial effect AFTER blankness, edge energy, pixel")
    print("  variance and radial depth. A b_ring that stays large with a small R2 gap is")
    print("  'the border, not the background'; a b_ring that collapses is the opposite.")
    return {"b_ring": float(np.mean([g[0] for g in got])) if got else float("nan"),
            "r2_gap": float(np.mean([g[1] for g in got])) if got else float("nan"),
            "b_ring_positive": (float(np.mean([g[0] > 0 for g in got])) if got
                                else float("nan"))}


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    ss = ((y - y.mean()) ** 2).sum()
    return b, float(1 - (resid ** 2).sum() / ss) if ss > 0 else float("nan")


def report_norms(meta, arrays, args):
    """M1: are the border patches norm outliers before the language model sees them?"""
    v_ring, v_int, h_by_layer = [], [], {}
    for m in meta:
        d = arrays.get(m["unit"], {})
        if d.get("vnorm") is not None:
            v_ring.append(float(d["vnorm"][0])); v_int.append(float(d["vnorm"][1]))
        hn = d.get("hnorm")
        if hn is not None:
            for l in range(hn.shape[0]):
                h_by_layer.setdefault(l, []).append(float(hn[l, 0]) / max(float(hn[l, 1]),
                                                                          1e-9))
    facts = {}
    if not v_ring and not h_by_layer:
        return facts
    print("\n" + "=" * 78)
    print("8. MASSIVE ACTIVATIONS -- ||h|| on the border over ||h|| in the interior")
    if v_ring:
        r = boot_mean(np.asarray(v_ring) / np.maximum(np.asarray(v_int), 1e-9), args.n_boot)
        print(f"    vision tower output (before ANY text token): {r[0]:.3f} "
              f"[{r[1]:.3f}, {r[2]:.3f}]")
        print("    ABOVE 1: the border patches are norm outliers before a single text")
        print("      token exists, so the encoder decided it and the LLM inherited it --")
        print("      the register story (Darcet et al.; Sun et al.).")
        print("    AT OR BELOW 1: the border's pull is NOT a big-feature effect. Whatever")
        print("      makes those columns attractive, it is not that they arrive large, and")
        print("      section 9 is then the place the mechanism has to come from.")
        facts["vision_ring_over_interior"] = r[0]
    if h_by_layer:
        print(f"\n    {'LLM layer':>10} {'ring/interior norm':>20}")
        for l in sorted(h_by_layer)[:: max(1, len(h_by_layer) // 12)]:
            print(f"    {l:>10} {float(np.mean(h_by_layer[l])):>20.3f}")
        facts["llm_ring_over_interior_max"] = max(
            float(np.mean(v)) for v in h_by_layer.values())
    return facts


def report_key_split(meta, arrays, cells, args):
    """M2: is the border's advantage bigger keys, or keys that point where queries do?"""
    kr, ka = [], []
    for m in meta:
        if m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        for l, h in cells:
            k1, k2 = a[l, h, SL.STAT_INDEX["knorm_ring"]], a[l, h, SL.STAT_INDEX["knorm_interior"]]
            a1, a2 = a[l, h, SL.STAT_INDEX["align_ring"]], a[l, h, SL.STAT_INDEX["align_interior"]]
            if np.isfinite(k1) and np.isfinite(k2) and k2 > 0:
                kr.append(float(k1 / k2))
            if np.isfinite(a1) and np.isfinite(a2):
                ka.append(float(a1 - a2))
    if not kr:
        return {}
    print("\n" + "=" * 78)
    print("9. WHERE THE LOGIT COMES FROM  (mean logit = scaling * ||k|| * alignment)")
    k = boot_mean(kr, args.n_boot)
    al = boot_mean(ka, args.n_boot)
    print(f"    ||k|| ring / interior       {k[0]:>8.3f}  [{k[1]:.3f}, {k[2]:.3f}]")
    print(f"    alignment ring - interior   {al[0]:>+8.4f}  [{al[1]:+.4f}, {al[2]:+.4f}]")
    print("    A ratio near 1 with a positive alignment gap is the sink-direction story;")
    print("    a large ratio with no alignment gap is the register story. They have")
    print("    different fixes, which is why they are split rather than summed.")
    return {"knorm_ratio": k[0], "align_gap": al[0], "align_gap_lo": al[1]}


def report_arms(out_dir, cells, args):
    """The causal half. Every arm is paired against the same picture's own baseline."""
    meta, arrays = read_stage(out_dir, "arms")
    if not meta:
        return {}
    by_arm = {}
    for m in meta:
        by_arm.setdefault(m.get("arm"), []).append(m)
    # Most arms are paired against `identity`. A10 is not: it has to resize the picture
    # so the grid divides it exactly before it can move whole cells, and that resize is
    # not free, so its baseline is the SAME resize with the identity permutation.
    bases = {a: {m["key"]: m for m in rows} for a, rows in by_arm.items()}
    if "identity" not in bases:
        print("\n(arms: no `identity` baseline was run, so nothing can be paired)")
        return {}
    print("\n" + "=" * 78)
    print("10. THE ARMS -- same picture, one thing changed, paired against its own "
          "baseline")
    print("    Enrichment deltas at the dev-selected cells. `follow content` is the share")
    print("    of pictures whose peak patch SHOWS the baseline's peak patch; `follow slot`")
    print("    is the share whose peak sits at the same grid position. For a positional")
    print("    sink the second is high and the first is at chance.")
    print("    An arm marked (=) leaves every patch showing what it showed -- `donut`")
    print("    repaints in place, the resolution ladder only rescales -- so for those two")
    print("    columns are the same question asked twice and neither discriminates.")
    print(f"\n    {'arm':<20} {'n':>4} {'dE_ring':>9} {'95% CI':>18} {'dE_top':>8} "
          f"{'dE_bottom':>10} {'follow content':>15} {'follow slot':>12}")
    out = {}
    for arm in sorted(by_arm):
        if arm == "identity":
            continue
        base = bases.get(SL.ARM_BASELINE.get(arm, "identity"), {})
        e_arm, e_base, tops, bots, fc, fs = {}, {}, [], [], [], []
        for m in by_arm[arm]:
            b = base.get(m["key"])
            if b is None:
                continue
            a_arm = arrays.get(m["unit"], {}).get("stats")
            a_base = arrays.get(b["unit"], {}).get("stats")
            if a_arm is None or a_base is None:
                continue
            gh, gw = m["grid"]
            gh0, gw0 = b["grid"]
            f_arm, f_base = SL.named_sets(gh, gw), SL.named_sets(gh0, gw0)
            # Keyed by UNIT, not by picture: the tiled arm produces one unit per tile and
            # they all pair against the same picture's baseline, which is what "paired by
            # common random numbers" means when one side is finer than the other.
            u = m["unit"]
            e_arm[u] = at_cells(a_arm, "ring_share", cells) / f_arm["ring"].mean()
            e_base[u] = at_cells(a_base, "ring_share", cells) / f_base["ring"].mean()
            for lst, name in ((tops, "top"), (bots, "bottom")):
                stat = f"{name}_share"
                lst.append(at_cells(a_arm, stat, cells) / f_arm[name].mean()
                           - at_cells(a_base, stat, cells) / f_base[name].mean())
            pk_a = arrays.get(m["unit"], {}).get("peak")
            pk_b = arrays.get(b["unit"], {}).get("peak")
            if pk_a is None or pk_b is None or arm in SL.NO_FOLLOW:
                continue
            # The MODE over the selected cells, not the median: peak indices are labels on
            # a grid, and the median of two corners is a patch neither head chose.
            pa = _mode([int(pk_a[l, h]) for l, h in cells])
            pb = _mode([int(pk_b[l, h]) for l, h in cells])
            # The permutation is its own correspondence: slot i now holds the embedding
            # (A9) or the pixel block (A10) that was at perm[i]. Nothing else in this
            # table can ask "did the peak follow the VECTOR" without also having moved
            # pixels around.
            corr = (np.asarray(arrays[m["unit"]]["perm"], dtype=np.int64)
                    if arm.startswith("permute") and "perm" in arrays.get(m["unit"], {})
                    else _arm_correspondence(arm, gh, gw, gh0, gw0, b.get("size"),
                                             m.get("view"), b.get("view"),
                                             m.get("patch_px")))
            if corr is not None and 0 <= pa < corr.size:
                fc.append(float(corr[pa] == pb))
            if (gh, gw) == (gh0, gw0):
                fs.append(float(pa == pb))
        d = boot_paired(e_arm, e_base, args.n_boot)
        out[arm] = {"baseline": SL.ARM_BASELINE.get(arm, "identity"),
                    "dE_ring": d[0], "dE_ring_lo": d[1], "dE_ring_hi": d[2],
                    "dE_top": float(np.mean(tops)) if tops else float("nan"),
                    "dE_bottom": float(np.mean(bots)) if bots else float("nan"),
                    "follow_content": float(np.mean(fc)) if fc else float("nan"),
                    "follow_slot": float(np.mean(fs)) if fs else float("nan"),
                    "identity_frame": _identity_frame(arm), "n": d[3]}
        tag = arm + (" (=)" if _identity_frame(arm) else "")
        print(f"    {tag:<20} {d[3]:>4} {d[0]:>+9.3f} "
              f"{f'[{d[1]:+.3f}, {d[2]:+.3f}]':>18} "
              f"{np.mean(tops) if tops else float('nan'):>+8.3f} "
              f"{np.mean(bots) if bots else float('nan'):>+10.3f} "
              f"{np.mean(fc) if fc else float('nan'):>15.3f} "
              f"{np.mean(fs) if fs else float('nan'):>12.3f}")
    print("\n  `permute` is the one that cannot be answered with 'your transform changed")
    print("  the content'. It moves nothing but which slot holds which patch embedding.")
    return out


def _identity_frame(arm):
    """True when the arm's grid correspondence is the identity, so `follow content` and
    `follow slot` are the same measurement and their contrast says nothing."""
    if arm in SL.SPECIAL_ARMS:
        return False
    from PIL import Image
    try:
        _im, inv, _meta = SL.transform(arm, Image.new("RGB", (320, 224)))
    except Exception:
        return False
    if inv is None:
        return False
    corr = SL.patch_correspondence(inv, 7, 10, 7, 10)
    return bool(np.array_equal(corr, np.arange(70)))


def _mode(values):
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _arm_correspondence(arm, gh, gw, gh0, gw0, size=None, view=None, view0=None,
                        patch_px=None):
    """The transform's grid mapping, rebuilt from the BASELINE PICTURE'S OWN pixel size.

    `pad_*` derives its border fraction from the picture's width and height AND from one
    patch's width, so a correspondence built on a guessed size -- or on Qwen3-VL's 32px
    patch when the run was LLaVA's 14px one -- is a correspondence for a different
    transform. The manifest carries the real size, the view and the patch width, and the
    report passes all three through rather than reconstructing any of them.
    """
    if arm in SL.SPECIAL_ARMS:
        return None
    from PIL import Image
    w, h = (size if size else (gw0 * 32, gh0 * 32))
    try:
        _im, inv, _meta = SL.transform(arm, Image.new("RGB", (int(w), int(h)),
                                                      (128, 128, 128)),
                                       patch_px=int(patch_px or 32))
    except Exception:
        return None
    if inv is None:
        return None
    return SL.patch_correspondence(inv, gh, gw, gh0, gw0,
                                   tuple(view or SL.FULL_VIEW),
                                   tuple(view0 or SL.FULL_VIEW))


#: Each entry is (hypothesis, what it would mean, [(prediction, reader) ...]) where the
#: reader takes the collected facts and returns (verdict, evidence string) or None when
#: the arm or table it needs was not run. Written as a table rather than as prose so the
#: conclusion is a function of the numbers and not of whoever is reading them.
HYPOTHESES = (
    ("H1  sequence position",
     "the picture is raster-ordered, so its early tokens are the sequence's early keys;\n"
     "     the first patch also sits immediately after <|vision_start|>",
     (("the FIRST patch beats the average corner",
       lambda f: _cmp(f["shape"].get("first"), f["shape"].get("corner"), ">", 1.5)),
      ("top and left beat bottom and right",
       lambda f: _cmp(_avg(f["shape"], "top", "left"),
                      _avg(f["shape"], "bottom", "right"), ">", 2.0)),
      ("rot180 does NOT move the border's share",
       lambda f: _near(f["arms"].get("rot180", {}).get("dE_ring"), 0.0, 0.1)),
      ("rot180 does NOT move the top row's share",
       lambda f: _near(f["arms"].get("rot180", {}).get("dE_top"), 0.0, 0.5)),
      # The prediction this hypothesis lives or dies by, and the one it was originally
      # not given. If the sink is held by the LLM's own position -- the slot, its
      # M-RoPE coordinates, its distance from <|vision_start|> -- then permuting which
      # embedding sits in which slot must leave the peak exactly where it was. Scoring
      # H1 without this is the selective reading this whole section exists to prevent.
      ("the peak stays in its SLOT when the EMBEDDINGS are permuted",
       lambda f: _cmp(f["arms"].get("permute", {}).get("follow_slot"), 0.8, ">", 0.0)))),

    ("H2  background content",
     "sinks go where there is nothing to look at, and in a photograph that is the edge",
     (("a blank INTERIOR region beats the border",
       lambda f: _cmp(f["blank"].get("blank_minus_ring"), 0.0, ">", 0.0)),
      ("zooming in, so the border becomes foreground, weakens it",
       lambda f: _cmp(f["arms"].get("zoom60", {}).get("dE_ring"), -0.2, "<", 0.0)),
      ("a NOISY pad does not attract what a blank pad does",
       lambda f: _cmp(f["arms"].get("pad_noise_1", {}).get("dE_ring"),
                      f["arms"].get("pad_grey_1", {}).get("dE_ring"), ">", 0.1)),
      ("`ring` does not survive the content covariates",
       lambda f: _cmp(f["content"].get("b_ring"), 0.2, "<", 0.0)))),

    ("H3  2D border geometry",
     "the ViT's position embeddings and the LLM's 2D M-RoPE put the border at extremal\n"
     "     coordinates, and border patches are built from a truncated neighbourhood",
     (("the border is enriched at all",
       lambda f: _cmp(f["shape"].get("ring"), 1.5, ">", 0.0)),
      ("the peak stays in its SLOT when the pixels move",
       lambda f: _cmp(_follow(f["arms"], "follow_slot"), 0.8, ">", 0.0)),
      ("the four edges are roughly symmetric",
       lambda f: _cmp(_avg(f["shape"], "top", "left"),
                      _avg(f["shape"], "bottom", "right"), "<", 2.0)))),

    ("H4  registers / massive activations",
     "some tokens carry outlier-norm states used as scratch space, and they are\n"
     "     allocated to low-information patches",
     (("the border arrives with a bigger norm from the vision tower",
       lambda f: _cmp(f["norms"].get("vision_ring_over_interior"), 1.05, ">", 0.0)),
      ("the border's keys are bigger",
       lambda f: _cmp(f["keys"].get("knorm_ratio"), 1.05, ">", 0.0)))),

    # H5 was not in the design document. It is what H1 and H3 turn into once the
    # permutation arm is read, and it is written here in the same before-the-fact form so
    # a later run can fail it: the mark is IN THE EMBEDDING, put there by the vision
    # tower's own position embedding, and the language model attends to the mark rather
    # than to the slot or to the pixels.
    ("H5  the vision tower's positional signature  (post hoc -- see the caveat)",
     "the ViT stamps its border patches, the stamp travels inside the embedding, and\n"
     "     the LLM attends to the stamp. Neither the pixels nor the slot hold it",
     (("rotating the pixels does NOT move it -- so not content",
       lambda f: _near(f["arms"].get("rot180", {}).get("dE_ring"), 0.0, 0.1)),
      ("permuting the embeddings DOES move it -- so not the slot",
       lambda f: _cmp(f["arms"].get("permute", {}).get("follow_slot"), 0.2, "<", 0.0)),
      ("...and the peak follows the EMBEDDING it was sitting on",
       lambda f: _cmp(f["arms"].get("permute", {}).get("follow_content"), 0.8, ">", 0.0)),
      ("the stamp is not a big norm (that would be H4)",
       lambda f: _cmp(f["norms"].get("vision_ring_over_interior"), 1.05, "<", 0.0)),
      ("it is an alignment with what the queries look for",
       lambda f: _cmp(f["keys"].get("align_gap_lo"), 0.0, ">", 0.0)))),
)


def _avg(d, *keys):
    v = [d.get(k) for k in keys]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _follow(arms, key):
    """The mean of one follow-rate over the arms that actually move pixels around.

    Restricted to arms whose grid correspondence is NOT the identity: for `donut` and the
    resolution ladder the content never leaves its patch, so their follow rates are 1.0
    by construction and would drown the arms that carry information.
    """
    v = [a[key] for name, a in arms.items()
         if not a.get("identity_frame") and name not in SL.SPECIAL_ARMS
         and a.get(key) is not None and np.isfinite(a[key])]
    return float(np.mean(v)) if v else None


def _cmp(got, want, op, margin):
    if got is None or want is None or not np.isfinite(got) or not np.isfinite(want):
        return None
    ok = got > want + margin if op == ">" else got < want - margin
    return ok, f"{got:+.3f} vs {want:+.3f}"


def _near(got, want, tol):
    if got is None or not np.isfinite(got):
        return None
    return abs(got - want) <= tol, f"{got:+.3f} (within {tol} of {want:+.0f}?)"


def report_verdict(facts):
    """Score the four hypotheses against what was measured. Mechanically, from the table.

    This is not a substitute for reading sections 1 to 10. It is a guard against reading
    them selectively: the predictions were written down in the design document before the
    run, they are evaluated here in the order they were written, and a hypothesis that
    passes on one leg and fails on three says so out loud.
    """
    print("\n" + "=" * 78)
    print("11. WHAT IT SAYS -- the pre-registered predictions, scored")
    for name, why, preds in HYPOTHESES:
        rows = [(text, reader(facts)) for text, reader in preds]
        live = [r for _t, r in rows if r is not None]
        score = sum(1 for r in live if r[0])
        print(f"\n  {name}   {score}/{len(live)} predictions met"
              + ("" if len(live) == len(rows) else
                 f"  ({len(rows) - len(live)} not measurable from this run)"))
        print(f"     {why}")
        for text, r in rows:
            mark = "not run" if r is None else ("  MET  " if r[0] else "  no   ")
            print(f"       [{mark}] {text:<52} {'' if r is None else r[1]}")
    print("\n  A hypothesis is not chosen by having the most ticks. Read which prediction")
    print("  failed: `rot180 does not move it` and `the peak follows the embedding under")
    print("  permutation` are the two that separate content, slot and stamp, and no amount")
    print("  of agreement on the others substitutes for them.")
    print("\n  H1 to H4 were written before the run. H5 was NOT -- it is what H1 and H3")
    print("  become once the permutation arm is read, so it is a hypothesis this run")
    print("  GENERATED and cannot also confirm. Its predictions are written in the same")
    print("  falsifiable form so the next model, or the next resolution, can fail them.")


#: The grid the question is actually about: which heads, and which tokens were asking.
def head_sets(family):
    """Every head, and -- on Qwen3-VL only -- the pair this project's reward trained.

    On another family L22 h28/31 names two arbitrary heads. Printing them beside the
    Qwen3-VL row under the same label is how a table invents a cross-model comparison
    that was never made.
    """
    sets = [("all heads", None)]
    if family == TRAINED_FAMILY:
        sets.append(("trained L22 h28,31", [(TRAINED_LAYER, h) for h in TRAINED_HEADS]))
    return tuple(sets)
TOKEN_SETS = (("query tokens", "stats"),
              ("generated tokens", "stats_gen"),
              ("all tokens", "stats_all"))
LOCATIONS = (("ring", "ring_share", "ring"), ("depth1", "depth1_share", "depth1"),
             ("middle", "deep_share", "deep"), ("top", "top_share", "top"),
             ("bottom", "bottom_share", "bottom"), ("left", "left_share", "left"),
             ("right", "right_share", "right"),
             ("topleft", "first_patch_share", None),
             ("botright", "last_patch_share", None))


def _enrich(stats, field_cells, stat, area, min_mass):
    """One picture's enrichment for one location. NaN when nothing is measurable.

    `field_cells` is None for "every head that clears the image-mass floor", or a list of
    (layer, head). The floor matters only for the all-head average: a head that puts no
    weight on the picture still has a ring share, and it is noise wearing a statistic's
    name. The two trained cells are never floored -- they are named, not selected.
    """
    a = np.asarray(stats, dtype=float)
    idx = SL.STAT_INDEX[stat]
    if field_cells is not None:
        v = np.asarray([a[l, h, idx] for l, h in field_cells if l < a.shape[0]])
    else:
        live = a[..., SL.STAT_INDEX["image_mass"]] >= min_mass
        v = np.where(live, a[..., idx], np.nan).reshape(-1)
    if not np.isfinite(v).any() or not area:
        return float("nan")
    return float(np.nanmean(v)) / area


def report_grid(meta, arrays, args):
    """Every location, per image type, for each (head set) x (token set).

    This is the table the question was asked in. Nothing here is head-SELECTED: the two
    blocks are "every head in the model" and "the two heads the reward trained", so no
    number in it was picked for being large.
    """
    types = sorted({m["type"] for m in meta})
    have_gen = any("stats_gen" in arrays.get(m["unit"], {}) for m in meta)
    n_gen = [m.get("n_generated") for m in meta if m.get("n_generated")]
    print("\n" + "=" * 78)
    print("12. THE GRID -- head set x token set, every location, per type")
    print("   Enrichment: that location's share of the picture's attention divided by its")
    print("   share of the patches. 1.00 = its fair share. `topleft`/`botright` are single")
    print("   patches against a flat map's 1/N. `middle` is depth 3 or more from the edge.")
    if have_gen:
        print(f"   `generated tokens` comes from one teacher-forced forward over prompt ++ "
              f"answer,\n   the same construction the reward used; median answer length "
              f"{int(np.median(n_gen)) if n_gen else 0} tokens.")
    else:
        print("   NO GENERATED TOKENS in this run (--max-new-tokens was 0), so two of the")
        print("   three token sets are empty. Re-run the scan with --max-new-tokens > 0.")
    summary = {}
    for tok_name, field in TOKEN_SETS:
        for head_name, head_cells in head_sets(_family_of(meta)):
            rows = []
            for t in types:
                vals = {loc: [] for loc, _s, _k in LOCATIONS}
                for m in meta:
                    if m["type"] != t:
                        continue
                    a = arrays.get(m["unit"], {}).get(field)
                    if a is None:
                        continue
                    gh, gw = m["grid"]
                    sets = SL.named_sets(gh, gw)
                    for loc, stat, key in LOCATIONS:
                        area = 1.0 / (gh * gw) if key is None else sets[key].mean()
                        vals[loc].append(_enrich(a, head_cells, stat, area, args.min_mass))
                n = max(len(v) for v in vals.values())
                if n:
                    rows.append((t, n, {k: float(np.nanmean(v)) if v else float("nan")
                                        for k, v in vals.items()}))
            if not rows:
                continue
            print(f"\n  --- {tok_name}, {head_name} ---")
            print(f"    {'type':<18} {'n':>4} " +
                  " ".join(f"{loc:>8}" for loc, _s, _k in LOCATIONS))
            for t, n, v in rows:
                print(f"    {t:<18} {n:>4} " +
                      " ".join(f"{v[loc]:>8.2f}" for loc, _s, _k in LOCATIONS))
            summary[(tok_name, head_name)] = {
                loc: float(np.nanmean([r[2][loc] for r in rows])) for loc, _s, _k in LOCATIONS}

    if summary:
        print("\n  Pooled over the twelve types:")
        print(f"    {'token set':<18} {'head set':<20} " +
              " ".join(f"{loc:>8}" for loc, _s, _k in LOCATIONS))
        for (tok, head), v in summary.items():
            print(f"    {tok:<18} {head:<20} " +
                  " ".join(f"{v[loc]:>8.2f}" for loc, _s, _k in LOCATIONS))
    return summary


def _family_of(meta):
    """The one family this output directory holds, or a hard stop.

    Two families in one directory would be pooled by every table below into an average
    over two geometries, and the average of a 24x24 centre crop and a native-resolution
    grid is not a quantity. One output directory per model is the contract; this is what
    enforces it.
    """
    fams = sorted({m.get("family") or "(unrecorded)" for m in meta})
    if len(fams) > 1:
        raise SystemExit(
            f"results from {fams} are mixed in one output directory. Every table here "
            "pools over pictures, and pooling two geometries produces a number with no "
            "referent. Run one model per --out-dir.")
    return fams[0] if fams else "(unrecorded)"


def _pooled_locations(meta, arrays, min_mass, field="stats"):
    """Every location's enrichment, pooled over pictures, over all heads. -> dict.

    All heads, never a selected set: the cross-model table must not be allowed to pick
    each model's most border-leaning cells and then report that all three lean on the
    border. The image-mass floor is the only filter, and it removes heads that put no
    weight on the picture at all.
    """
    vals = {loc: [] for loc, _s, _k in LOCATIONS}
    for m in meta:
        a = arrays.get(m["unit"], {}).get(field)
        if a is None:
            continue
        gh, gw = m["grid"]
        sets = SL.named_sets(gh, gw)
        for loc, stat, key in LOCATIONS:
            area = 1.0 / (gh * gw) if key is None else sets[key].mean()
            vals[loc].append(_enrich(a, None, stat, area, min_mass))
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in vals.items()}


def stage_crossmodel(args):
    """The table the port exists to produce: the same corpus, three models, side by side.

    `docs/sink-location-cross-model.md` asks whether the one-patch outer ring and the
    upper-left peak are a property of Qwen3-VL or of VLMs. That is not a question any one
    output directory can answer, and it is not a question that survives pooling them --
    the grids differ, and on LLaVA-1.5 the grid covers a centre crop rather than the
    picture. So every directory is reduced on its own geometry first, and only the
    dimensionless enrichments are put beside each other.
    """
    dirs = [d for d in (args.dirs or "").split(",") if d]
    if len(dirs) < 2:
        raise SystemExit("--stage crossmodel needs --dirs a,b[,c]")
    runs = []
    for d in dirs:
        meta, arrays = read_stage(d, "scan")
        if not meta:
            print(f"(skipping {d}: no scan results)")
            continue
        runs.append({"dir": d, "meta": meta, "arrays": arrays,
                     "family": _family_of(meta), "n": len(meta)})
    if len(runs) < 2:
        raise SystemExit("fewer than two directories have results")

    print("=" * 78)
    print("0. WHAT IS BEING COMPARED")
    print("   The same pictures through three models. Nothing below is a raw percentage:")
    print("   the ring is 23% of a 16x16 grid and 16% of a 24x24 one, so every number is")
    print("   an enrichment -- that location's share of the picture's attention over its")
    print("   share of the patches. 1.00 is no effect, in every column.")
    print(f"\n    {'family':<12} {'n':>5} {'modal grid':>11} {'ring area':>10} "
          f"{'grid covers':>28} {chr(34) + 'the picture' + chr(34) + ' of a row':>26}")
    for r in runs:
        grids = [tuple(m["grid"]) for m in r["meta"]]
        modal = max(set(grids), key=grids.count)
        views = {tuple(round(x, 3) for x in (m.get("view") or SL.FULL_VIEW))
                 for m in r["meta"]}
        cover = ("the whole picture" if views == {(0.0, 0.0, 1.0, 1.0)}
                 else f"a centre crop ({len(views)} boxes)")
        img = [float(np.nanmean(a[..., SL.SPANS.index("image")]))
               for m in r["meta"] if (a := r["arrays"].get(m["unit"], {}).get("spans")) is not None]
        r["image_share"] = float(np.mean(img)) if img else float("nan")
        print(f"    {r['family']:<12} {r['n']:>5} {f'{modal[0]}x{modal[1]}':>11} "
              f"{SL.ring_area_frac(*modal):>10.3f} {cover:>28} {r['image_share']:>26.4f}")
    print("\n  The picture's share of a row is the budget every ring number is a division")
    print("  of. A model that gives the picture 1% of a row and one that gives it 15% are")
    print("  not making the same claim with the same enrichment.")

    print("\n" + "=" * 78)
    print("1. WHERE THE PICTURE'S ATTENTION GOES -- pooled over the twelve types, ALL HEADS")
    for r in runs:
        r["loc"] = _pooled_locations(r["meta"], r["arrays"], args.min_mass)
    print(f"\n    {'family':<12} " + " ".join(f"{loc:>8}" for loc, _s, _k in LOCATIONS))
    for r in runs:
        print(f"    {r['family']:<12} " +
              " ".join(f"{r['loc'][loc]:>8.2f}" for loc, _s, _k in LOCATIONS))
    print("\n  `topleft`/`botright` are SINGLE patches against a flat map's 1/N, so they")
    print("  run on a different scale from the block columns beside them. A ring above 1")
    print("  with top >> bottom and left >> right is the raster-order signature; a ring")
    print("  above 1 with the four edges level is a 2-D border effect and a different")
    print("  claim. Read those two together or neither.")

    print("\n" + "=" * 78)
    print("2. THE RING, PER IMAGE TYPE")
    types = sorted({m["type"] for r in runs for m in r["meta"]})
    print(f"\n    {'type':<18} " + " ".join(f"{r['family']:>12}" for r in runs))
    for t in types:
        cells = []
        for r in runs:
            sub = [m for m in r["meta"] if m["type"] == t]
            cells.append(_pooled_locations(sub, r["arrays"], args.min_mass)["ring"]
                         if sub else float("nan"))
        print(f"    {t:<18} " + " ".join(f"{c:>12.2f}" for c in cells))
    print(f"\n    {'ALL':<18} " +
          " ".join(f"{r['loc']['ring']:>12.2f}" for r in runs))
    print("\n  Pre-registered in the original design: E_ring >= 1.5 is 'the sink")
    print("  concentrates on the ring', 1.2-1.5 is weak, below 1.2 is a failure for that")
    print("  type. The same thresholds are applied here, to every model.")

    print("\n" + "=" * 78)
    print("3. IS IT A SINK, OR ONLY A PEAK -- at the strongest cell in each model")
    print(f"\n    {'family':<12} {'peak / uniform':>16} {'peak CV across queries':>24} "
          f"{'verdict':>28}")
    for r in runs:
        mag, cv = [], []
        mass_i = SL.STAT_INDEX["image_mass"]
        mag_i, cv_i = SL.STAT_INDEX["peak_uniform_x"], SL.STAT_INDEX["peak_cv"]
        for m in r["meta"]:
            a = r["arrays"].get(m["unit"], {}).get("stats")
            if a is None:
                continue
            v = np.where(np.asarray(a[..., mass_i], dtype=float) >= args.min_mass,
                         np.asarray(a[..., mag_i], dtype=float), np.nan)
            if not np.isfinite(v).any():
                continue
            l, h = divmod(int(np.nanargmax(v)), a.shape[1])
            mag.append(float(a[l, h, mag_i]))
            cv.append(float(a[l, h, cv_i]))
        mm = float(np.nanmean(mag)) if mag else float("nan")
        cc = float(np.nanmean(cv)) if cv else float("nan")
        verdict = ("a sink by both legs" if mm >= args.sink_x and cc <= args.sink_cv
                   else "a peak, not a sink" if mm < args.sink_x
                   else "large but query-DEPENDENT")
        print(f"    {r['family']:<12} {mm:>16.1f} {cc:>24.2f} {verdict:>28}")
    print(f"\n  Pre-registered: >= {args.sink_x}x uniform AND CV <= {args.sink_cv}. A model")
    print("  that fails the CV leg has a peak that moves with the query, which is not what")
    print("  the sink literature describes, however large the peak is.")

    print("\n" + "=" * 78)
    print("4. THE ARMS -- the causal half, per model")
    print("   `follow content` is the share of pictures whose peak patch SHOWS the")
    print("   baseline's peak patch; `follow slot` is the share whose peak sits at the")
    print("   same grid position. `permute` moves the encoder's OUTPUT rows and says")
    print("   whether the mark is in the embedding or in the language model's slot;")
    print("   `permute_pixels` (A10) moves the pixels BEFORE the encoder and says whether")
    print("   the encoder's own position embedding is what writes it.")
    want = ("rot180", "permute", "permute_identity", "permute_pixels", "tiled")
    for r in runs:
        cells, _n = choose_cells(r["meta"], r["arrays"], args.n_cells, args.min_mass)
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r["arms"] = report_arms(r["dir"], cells, args) or {}
    print(f"\n    {'arm':<20} " +
          " ".join(f"{r['family'] + ' dE_ring':>20}" for r in runs))
    for arm in want:
        if not any(arm in r["arms"] for r in runs):
            continue
        print(f"    {arm:<20} " + " ".join(
            f"{r['arms'].get(arm, {}).get('dE_ring', float('nan')):>+20.3f}" for r in runs))
    for col, label in (("follow_content", "follow content"), ("follow_slot", "follow slot")):
        print(f"\n    {label:<20} " +
              " ".join(f"{r['family']:>20}" for r in runs))
        for arm in want:
            if not any(arm in r["arms"] for r in runs):
                continue
            print(f"      {arm:<18} " + " ".join(
                f"{r['arms'].get(arm, {}).get(col, float('nan')):>20.3f}" for r in runs))

    print("\n" + "=" * 78)
    print("5. HOW TO READ IT  (written before the run, in docs/sink-location-cross-model.md)")
    print("   ring + upper-left peak in all three   a VLM-wide property, and A10 is what")
    print("                                         has to explain it")
    print("   holds in InternVL, fails in LLaVA     tie it to native-resolution encoders;")
    print("                                         scope the claim and cite the LLaVA")
    print("                                         bottom-bias literature as the contrast")
    print("   fails in both                         a Qwen3-VL property -- still")
    print("                                         publishable, and it makes the")
    print("                                         reward-design conclusion MORE")
    print("                                         interesting, not less")
    print("   ring without the upper-left peak      two mechanisms, not one. Split the")
    print("     (or the other way round)            claim")
    return 0


def stage_verify(args):
    """Did the refactor move the baseline? Compare two scan directories, unit by unit.

    Porting this experiment to a second and a third model meant putting an adapter under
    the one place it knew it was talking to Qwen3-VL. A refactor that silently changes
    what the Qwen3-VL scan measures would invalidate the comparison the port exists to
    make, and it would do so invisibly: every table would still print, and every number
    would still look like a number. So the port is gated on reproducing the published run
    on the SAME pictures, and this is the gate.

    Reported per statistic: the largest absolute difference over shared units, and the
    share of units whose peak patch is unchanged. The stats are stored as float16, so
    exact equality is the expectation and anything above the float16 step is a change.
    """
    meta_a, arr_a = read_stage(args.out_dir, "scan")
    meta_b, arr_b = read_stage(args.against, "scan")
    if not meta_a or not meta_b:
        print(f"nothing to compare: {len(meta_a)} units here, {len(meta_b)} there")
        return 1
    by_b = {m["unit"]: m for m in meta_b}
    shared = [m for m in meta_a if m["unit"] in by_b]
    print(f"{len(shared)} shared units ({len(meta_a)} here, {len(meta_b)} in "
          f"{args.against})")
    if not shared:
        return 1
    bad_grid = [m["unit"] for m in shared if m["grid"] != by_b[m["unit"]]["grid"]]
    print(f"grids identical on {len(shared) - len(bad_grid)}/{len(shared)} units"
          + (f"   MISMATCHED: {bad_grid[:5]}" if bad_grid else ""))
    worst = {n: 0.0 for n in SL.STAT_NAMES}
    peak_same, peak_n, map_worst = 0, 0, 0.0
    for m in shared:
        a = arr_a.get(m["unit"], {}).get("stats")
        b = arr_b.get(by_b[m["unit"]]["unit"], {}).get("stats")
        if a is None or b is None or a.shape != b.shape:
            continue
        d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        d = np.where(np.isfinite(d), d, 0.0)
        for i, n in enumerate(SL.STAT_NAMES):
            worst[n] = max(worst[n], float(d[..., i].max()))
        pa, pb = arr_a[m["unit"]].get("peak"), arr_b[m["unit"]].get("peak")
        if pa is not None and pb is not None and pa.shape == pb.shape:
            peak_same += int((pa == pb).sum())
            peak_n += int(pa.size)
        ma, mb = arr_a[m["unit"]].get("maps"), arr_b[m["unit"]].get("maps")
        if ma is not None and mb is not None and ma.shape == mb.shape:
            map_worst = max(map_worst, float(np.nanmax(np.abs(
                np.asarray(ma, dtype=np.float64) - np.asarray(mb, dtype=np.float64)))))
    print(f"\n    {'statistic':<22} {'largest |difference|':>22}")
    for n in SL.STAT_NAMES:
        print(f"    {n:<22} {worst[n]:>22.3e}")
    print(f"\n    peak patch identical in {peak_same}/{peak_n} cells "
          f"({peak_same / max(1, peak_n):.6f})")
    print(f"    largest |difference| in the layer-mean maps: {map_worst:.3e}")
    ok = (not bad_grid and max(worst.values()) <= args.verify_tol
          and peak_same == peak_n)
    print(f"\n  {'VERIFY PASS' if ok else 'VERIFY FAIL'}  "
          f"(tolerance {args.verify_tol:g} on every statistic, and every peak identical)")
    if not ok:
        print("  The refactored scan is not measuring what the published run measured.")
        print("  Fix that before reading a single cross-model number: the whole point of")
        print("  the port is a comparison, and a comparison needs a fixed reference.")
    return 0 if ok else 1


def stage_report(args):
    meta, arrays = read_stage(args.out_dir, "scan")
    if not meta:
        print(f"no scan results under {args.out_dir}")
        return 1
    n_dev = sum(1 for m in meta if m.get("dev"))
    fam_name = _family_of(meta)
    grids = sorted({tuple(m["grid"]) for m in meta})
    views = sorted({tuple(round(x, 4) for x in (m.get("view") or SL.FULL_VIEW))
                    for m in meta})
    print(f"{len(meta)} pictures from {args.out_dir}   "
          f"({n_dev} dev, {len(meta) - n_dev} test)")
    print(f"family {fam_name}   {len(grids)} distinct grid shape(s), modal "
          f"{max(grids, key=lambda g: sum(tuple(m['grid']) == g for m in meta))}")
    if views != [(0.0, 0.0, 1.0, 1.0)]:
        print(f"the grid covers a SUB-RECTANGLE of the picture ({len(views)} distinct "
              "view boxes): every 'ring' below\nis the border of what the encoder saw, "
              "which on this family is a centre crop of the picture.")
    types = sorted({m["type"] for m in meta})
    thin = [t for t in types
            if sum(1 for m in meta if m["type"] == t and not m.get("dev")) < 60]
    if thin:
        print(f"UNDER-POWERED (fewer than 60 test pictures): {', '.join(thin)} -- their "
              "CIs are wide\nand their verdicts should not be read as settled.")

    cells, n_dev_used = choose_cells(meta, arrays, args.n_cells, args.min_mass)
    print(f"\nheads chosen on {n_dev_used} DEV pictures, reported on the test split: "
          f"{cells}")

    report_budget(meta, arrays)
    report_types(meta, arrays, cells, args)
    facts = {"shape": report_shape(meta, arrays, cells, args) or {}}
    report_cells(meta, arrays, args)
    report_s3(meta, arrays, cells, args)
    facts["blank"] = report_blank(meta, arrays, cells, args) or {}
    facts["content"] = report_content(meta, arrays, args) or {}
    facts["norms"] = report_norms(meta, arrays, args) or {}
    facts["keys"] = report_key_split(meta, arrays, cells, args) or {}
    facts["arms"] = report_arms(args.out_dir, cells, args) or {}
    report_verdict(facts)
    report_grid(meta, arrays, args)

    print("\n" + "=" * 78)
    print("Read section 1 before anything else. If the picture's share of a row is a")
    print("fraction of a percent, every ring number above is a statement about how the")
    print("model divides up that fraction -- true, and small.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["corpus", "selftest", "scan", "arms", "report", "monitor",
                             "verify", "crossmodel"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dirs", default=None,
                    help="crossmodel: the scan directories to put side by side")
    ap.add_argument("--against", default=None,
                    help="verify: the scan directory this one must reproduce")
    ap.add_argument("--verify-tol", type=float, default=1e-3,
                    help="verify: the largest difference any statistic may show. The "
                         "stats are float16, so this is loose on purpose and a real "
                         "change clears it by orders of magnitude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--types", default="", help="comma-separated subset of the corpus")
    ap.add_argument("--per-type", type=int, default=150)
    ap.add_argument("--scan-rows-per-type", type=int, default=0,
                    help="scan only the first N pictures of each type (0 = all). Use it "
                         "with --max-new-tokens, which costs a generation per picture")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = prefill only, and the `generated` query set is empty. Above "
                         "0 the model writes an answer at full speed and ONE teacher-forced "
                         "forward over prompt ++ answer is measured, which is the same "
                         "construction the training reward used")
    ap.add_argument("--system-prompt", default="auto",
                    help="auto = each family's own (Qwen3-VL gets the project's trainer "
                         "prompt, so the published numbers reproduce; the others get "
                         "none). `none` puts every family on the same footing and is "
                         "what the cross-model tables are run under. Anything else is "
                         "used verbatim")
    ap.add_argument("--max-tiles", type=int, default=13,
                    help="the `tiled` arm: how many of a picture's tiles to score. "
                         "InternVL's max_dynamic_patch is 12 plus a thumbnail")
    ap.add_argument("--val-only", action="store_true",
                    help="never top up from set_a/set_b. Required for any checkpoint "
                         "that was GRPO-trained on them")
    ap.add_argument("--rebuild", action="store_true", help="corpus: ignore what exists")
    ap.add_argument("--arms", default=",".join(SL.DEFAULT_ARMS),
                    help=f"any of {','.join(SL.ARMS + SL.SPECIAL_ARMS)}")
    ap.add_argument("--arm-rows-per-type", type=int, default=40)
    ap.add_argument("--no-hidden", action="store_true",
                    help="skip the hidden-state norms (M1); saves memory on a big model")
    ap.add_argument("--no-key-stats", action="store_true", help="skip M2's key statistics")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--n-cells", type=int, default=16,
                    help="how many (layer, head) cells the headline averages over")
    ap.add_argument("--min-mass", type=float, default=0.002,
                    help="a cell below this image mass is noise wearing a statistic's name")
    ap.add_argument("--blank-min", type=float, default=0.15,
                    help="how much of the grid an interior blank region must cover")
    ap.add_argument("--sink-x", type=float, default=10.0,
                    help="S3: times uniform the top column must be to be called a sink")
    ap.add_argument("--sink-cv", type=float, default=0.5,
                    help="S3: the largest across-query CV a sink may have")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--selftest-rows", type=int, default=4)
    ap.add_argument("--selftest-tokens", type=int, default=16)
    ap.add_argument("--tie-back", action="store_true",
                    help="also run stage 1a/1b: the generated-token readout, and the "
                         "agreement that licenses the prefill proxy")
    ap.add_argument("--tieback-rows", type=int, default=24)
    ap.add_argument("--tieback-tokens", type=int, default=192)
    ap.add_argument("--tieback-rho", type=float, default=0.8)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.types = [t for t in args.types.split(",") if t] or list(CORPUS)
    bad = [t for t in args.types if t not in CORPUS]
    if bad:
        raise SystemExit(f"unknown type(s) {bad}; have {sorted(CORPUS)}")
    args.arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in args.arms if a not in SL.ARMS + SL.SPECIAL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; have {sorted(SL.ARMS + SL.SPECIAL_ARMS)}")

    if args.stage == "corpus":
        return stage_corpus(args)
    if args.stage == "report":
        return stage_report(args)
    if args.stage == "crossmodel":
        return stage_crossmodel(args)
    if args.stage == "verify":
        if not args.against:
            raise SystemExit("--stage verify needs --against DIR")
        return stage_verify(args)
    if args.stage == "monitor":
        IV.monitor(Path(args.out_dir), args.interval, args.once, "")
        return 0
    if not args.model:
        raise SystemExit("--model is required for this stage")
    if args.stage == "selftest":
        return stage_selftest(args)
    if args.stage == "scan":
        return stage_scan(args)
    return stage_arms(args)


if __name__ == "__main__":
    sys.exit(main())
