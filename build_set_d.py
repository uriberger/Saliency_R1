#!/usr/bin/env python
"""Build set_d -- a fresh corpus the same shape as saliency-r1-8k, sharing no image
with it -- and the two validation sets that go with it.

    set_d             8,080 rows over 6,953 images, all Visual-CoT
    val_d_natural       256 rows / 256 images, set_d's natural source proportions
    val_d_nonnatural    256 rows / 256 images, set_d's non-natural proportions

WHY THIS SET EXISTS
-------------------
set_c doubled saliency-r1-8k's rows but could only find 3.2% more images, so it packs
2.25 questions per image against the 8k's 1.16. The two runs' `completions/mean_length`
curves diverge in a way the LR schedule does not explain, and questions-per-image is the
leading suspect. set_d is the control that isolates it: the 8k's row count, the 8k's
source mix, the 8k's per-source IMAGE count -- hence the 8k's questions-per-image -- but
not one of the 8k's pictures. If its length curve tracks the 8k's, the corpus SHAPE is
what matters and set_c's gap is the packing. If it tracks set_c's, the shape is not the
story and the difference lives in which pictures were drawn.

Being 8k-sized also makes it LR-comparable for free: `num_train_epochs=3` puts max_steps
at 3,990 for 8,080 rows, exactly the old run's, so no step-vs-step correction is needed.

DISJOINTNESS, AND WHY IT IS NOT ONE HASH OF THE STORED BYTES
------------------------------------------------------------
The usual check here -- compare the SHA-256 of the stored JPEG bytes, exact because
every set of ours re-encodes through the same resize + quality-95 path -- does not
reach saliency-r1-8k: it was not built by that path, and it is not even internally
uniform. Measured on all 8,080 of its rows against a SHA-256 index of the archive:

    5,287 rows  image.path set, bytes byte-identical to the Visual-CoT ARCHIVE file
                (flickr30k, gqa, v7w, cub, vsr, and 47 docvqa/infographicsvqa rows)
    2,793 rows  image.path null, stored as a PNG with the long side capped at 512
                (all of openimages, textcap, textvqa; 1,282 docvqa/infographicsvqa)

So the 8k's images are addressed two ways, and both are used:

  * the raw rows give their archive identity directly. `--index` walks the archive once
    and records every member's SHA-256, which turns their hashes into the exact set of
    archive PATHS the 8k occupies -- including paths its own metadata never mentions
    (5,761 paths for 5,130 pictures, i.e. 631 stored twice under different names).
  * the re-encoded rows are found by CONTENT in the Visual-CoT metadata: question,
    answer and normalized box, the same string `union_bbox` writes here. All 8,080 rows
    resolve that way, 5,287 of them pinned by the filename they carry; where a row still
    matches more than one image (821 rows, 805 of them openimages) EVERY candidate is
    excluded. Over-excluding costs pool depth the recipe does not need; under-excluding
    would put one of the 8k's pictures in the set built to avoid them.

set_d excludes the union of the two. Against set_c and the validation sets the ordinary
stored-bytes check still applies, and `--verify` uses it: those were built here.

One difference set_d cannot erase: the 8k's docvqa and infographicsvqa pages are 512px
PNGs and set_d's are JPEG q95, so text imagery carries compression artifacts the 8k's
does not (12% of rows). Everything else is 512-capped either way, and the trainer caps
again at load, so the rest of the corpus is presented to the model the same way.

Usage:
    python build_set_d.py --index                        # archive SHA-256 index (once)
    python build_set_d.py --report                       # feasibility against the pools
    python build_set_d.py --build --out-dir DIR          # all three sets
    python build_set_d.py --verify --out-dir DIR         # prove the disjointness claims
"""

import argparse
import collections
import gc
import glob
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_grpo_sets as B
import build_set_c as C

B.CHUNK_ROWS = 1000


# ---------------------------------------------------------------------------
# Recipe -- exactly half of set_c's rows, i.e. exactly what the 8k holds
# ---------------------------------------------------------------------------
RECIPE_ROWS = {s: n // 2 for s, n in C.RECIPE_ROWS.items()}

# The 8k's own per-source unique-image counts, measured from its stored bytes.
# set_c had to raise gqa/openimages/textvqa above these because it wanted twice the
# rows; set_d wants the 8k's row count, so it can hold every one of them exactly.
IMAGE_BUDGET = {
    "flickr30k": 2618,
    "gqa": 1752,
    "openimages": 712,
    "docvqa": 282,
    "textcap": 441,
    "v7w": 595,
    "textvqa": 307,
    "infographicsvqa": 103,
    "cub": 80,
    "vsr": 63,
}

SEED = 2027           # not set_c's 2026: a different draw over a disjoint pool
VAL_SEED = 20270

EIGHTK_REPO = "peterant330/saliency-r1-8k"
INDEX_NAME = ".archive_sha256.tsv.gz"


def _check_recipe():
    assert sum(RECIPE_ROWS.values()) == 8080, sum(RECIPE_ROWS.values())
    assert sum(IMAGE_BUDGET.values()) == 6953, sum(IMAGE_BUDGET.values())
    assert set(RECIPE_ROWS) == set(IMAGE_BUDGET)


# ---------------------------------------------------------------------------
# saliency-r1-8k
# ---------------------------------------------------------------------------
def eightk_parquets():
    """The 8k's parquet shards, from whichever HF cache holds them."""
    pats = []
    for home in filter(None, [os.environ.get("HF_HOME"), "~/.cache/huggingface"]):
        pats.append(os.path.join(os.path.expanduser(home), "hub",
                                 "datasets--peterant330--saliency-r1-8k",
                                 "snapshots", "*", "data", "*.parquet"))
    for pat in pats:
        found = sorted(glob.glob(pat))
        if found:
            return found
    # Not cached yet -- fetch it.
    snap = B.hf_snapshot(EIGHTK_REPO, ["data/*.parquet"])
    found = sorted(glob.glob(str(snap / "data" / "*.parquet")))
    if not found:
        raise SystemExit(f"no parquet shards for {EIGHTK_REPO}")
    return found


def eightk_image_hashes():
    """SHA-256 of every image saliency-r1-8k stores, i.e. of the archive file itself.

    Streamed in batches: the three shards are 1.1 GB and the login node's address
    space is capped at 8 GB.
    """
    import pyarrow.parquet as pq

    per_source, everything = collections.defaultdict(set), set()
    rows = 0
    for path in eightk_parquets():
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=200, columns=["dataset", "image"]):
            d = batch.to_pydict()
            for source, image in zip(d["dataset"], d["image"]):
                h = hashlib.sha256(image["bytes"]).hexdigest()
                per_source[source].add(h)
                everything.add(h)
                rows += 1
    return rows, per_source, everything


def eightk_rows():
    """The 8k's non-image columns, which is what identifies its Visual-CoT rows."""
    import pyarrow.parquet as pq

    t = pq.read_table(eightk_parquets(),
                      columns=["dataset", "problem", "solution", "bbox",
                               "image.path"]).to_pydict()
    return list(zip(t["dataset"], t["problem"], t["solution"], t["bbox"], t["path"]))


def eightk_basenames_by_metadata(meta_dir, pools=None):
    """Every archive basename the 8k's rows COULD have come from.

    A fallback for when no archive index has been built. The 8k does not record which
    Visual-CoT row each of its rows is, so the row is found by its content -- question,
    answer and normalized box, which is the same string `union_bbox` writes here. Where
    that still matches more than one image, EVERY candidate is excluded: over-excluding
    costs a few hundred images out of pools of tens of thousands, under-excluding would
    put one of the 8k's pictures in the set that exists to avoid them.

    Returns (basenames, report) where report counts how each row resolved.
    """
    rows_by_source = collections.defaultdict(list)
    for row in eightk_rows():
        rows_by_source[row[0]].append(row)

    names, report = set(), {}
    for source in sorted(rows_by_source):
        pool = (pools or {}).get(source) or C.load_pool(source, meta_dir)[0]
        exact, by_qa = collections.defaultdict(list), collections.defaultdict(list)
        for p in pool:
            exact[(p["problem"], p["solution"], p["bbox"])].append(p)
            by_qa[(p["problem"], p["solution"])].append(p)

        stat = collections.Counter()
        for (_, problem, solution, bbox, path) in rows_by_source[source]:
            key = (problem.strip(), solution.strip())
            cands = exact.get(key + (bbox,)) or by_qa.get(key) or []
            if not cands:
                stat["unmatched"] += 1
                continue
            bases = {os.path.basename(c["_ref"][1]) for c in cands}
            if path and path in bases:
                bases = {path}          # the row names its own file: no ambiguity left
                stat["pinned_by_path"] += 1
            elif len(bases) > 1:
                stat["ambiguous"] += 1
            else:
                stat["unique"] += 1
            names |= bases
        report[source] = dict(stat)
    return names, report


def load_index(cache_dir):
    """{archive path: sha256} from a previous --index pass, or None."""
    path = Path(cache_dir) / INDEX_NAME
    if not path.exists():
        return None
    index = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            name, digest = line.rstrip("\n").split("\t")
            index[name] = digest
    return index


def eightk_basenames(meta_dir, cache_dir, pools=None):
    """The basenames set_d must not touch, and how they were established.

    Both routes, unioned. Neither is sufficient alone: the index cannot see the 2,793
    rows the 8k re-encoded (their bytes are in no archive member), and the metadata
    cannot see a picture stored under a second name in another sub-dataset's directory.
    """
    names, report = eightk_basenames_by_metadata(meta_dir, pools)
    by_meta = len(names)
    how = "metadata match"

    index = load_index(cache_dir)
    if index is not None:
        _, _, hashes = eightk_image_hashes()
        paths = [p for p, h in index.items() if h in hashes]
        found = {index[p] for p in paths}
        names |= {os.path.basename(p) for p in paths}
        report["_index"] = {
            "8k pictures": len(hashes), "of which are archive files": len(found),
            "archive paths carrying them": len(paths),
            "basenames added": len(names) - by_meta,
        }
        how = "metadata match + archive SHA-256 index"
    report["_total_basenames"] = len(names)
    return names, how, report


# ---------------------------------------------------------------------------
# Everything else already spoken for
# ---------------------------------------------------------------------------
def set_c_basenames(meta_dir):
    """set_c's images, by re-running its draw -- it is seeded, so this is exact."""
    legacy_val, _ = C.legacy_val_reference_keys(meta_dir)
    records, _ = C.draw_set_c(meta_dir, legacy_val)
    train = {os.path.basename(r["_ref"][1]) for r in records}
    used_keys = {r["_ref"][1] for r in records}
    val = C.draw_val_c(meta_dir, used_keys, legacy_val)
    val_names = {os.path.basename(r["_ref"][1])
                 for drawn, _ in val.values() for recs in drawn.values() for r in recs}
    return train, val_names, legacy_val


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------
def draw_set_d(meta_dir, excluded, pools=None):
    records, report = [], {}
    for source in sorted(RECIPE_ROWS):
        pool = (pools or {}).get(source) or C.load_pool(source, meta_dir)[0]
        groups = C.group_by_image(pool)
        rng = C.source_rng(SEED, source)
        picked = C.choose_images(groups, IMAGE_BUDGET[source], RECIPE_ROWS[source],
                                 rng, excluded)

        drawn = []
        for key in sorted(picked):
            rows = sorted(groups[key], key=lambda r: r["question_id"])
            take = picked[key]
            drawn.extend(rows if take >= len(rows) else rng.sample(rows, take))
        assert len(drawn) == RECIPE_ROWS[source], (source, len(drawn))
        rng.shuffle(drawn)
        records.extend(drawn)

        degs = collections.Counter(picked.values())
        report[source] = dict(rows=len(drawn), images=len(picked), pool_rows=len(pool),
                              pool_images=len(groups), per_image=dict(sorted(degs.items())))
    import random
    random.Random(SEED).shuffle(records)
    return records, report


def draw_val_d(meta_dir, used_keys, excluded, pools=None):
    """val_d_natural / val_d_nonnatural candidates, on set_d's source proportions."""
    nat_recipe = {s: n for s, n in RECIPE_ROWS.items() if s not in C.NONNATURAL}
    non_recipe = {s: n for s, n in RECIPE_ROWS.items() if s in C.NONNATURAL}
    nat_counts = B.allocate(nat_recipe, C.VAL_SIZE)
    non_counts = B.allocate(non_recipe, C.VAL_SIZE)

    taken = {os.path.basename(k) for k in used_keys} | set(excluded)
    out = {}
    for name, counts, seed in (("val_d_natural", nat_counts, VAL_SEED),
                               ("val_d_nonnatural", non_counts, VAL_SEED + 1)):
        drawn = {}
        for source, n in counts.items():
            if n <= 0:
                continue
            pool = (pools or {}).get(source) or C.load_pool(source, meta_dir)[0]
            rng = C.source_rng(seed, source)
            eligible, seen = [], set()
            for r in pool:
                base = os.path.basename(r["_ref"][1])
                if base in taken or base in seen:
                    continue
                seen.add(base)
                eligible.append(r)
            rng.shuffle(eligible)
            picked = eligible[: n * C.VAL_OVERSAMPLE]
            if len(picked) < n:
                raise SystemExit(f"{name}/{source}: only {len(picked)} free images "
                                 f"for a target of {n}")
            drawn[source] = picked
            taken.update(os.path.basename(r["_ref"][1]) for r in picked)
        out[name] = (drawn, counts)
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def do_index(args):
    """One pass over the archive, writing every member's path and SHA-256.

    ~139 GB of sequential reads for a ~15 MB index. Everything downstream that needs
    to know WHICH archive file a set of bytes is -- above all, which paths carry the
    pictures saliency-r1-8k stores -- is a lookup in this file afterwards.
    """
    cache_dir = Path(args.image_cache or (Path(args.out_dir) / "_viscot_paths"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / INDEX_NAME
    if dest.exists() and not args.force:
        n = sum(1 for _ in gzip.open(dest, "rt"))
        print(f"{dest} already holds {n} members; --force to rebuild")
        return 0

    shards = sorted((B.hf_snapshot(B.VISCOT_REPO, ["cot_images_tar_split/*"])
                     / "cot_images_tar_split").glob("cot_images_*"))
    if len(shards) != 13:
        raise SystemExit(f"expected 13 Visual-CoT shards, found {len(shards)}")

    tmp = dest.with_suffix(".partial")
    n, by_dir = 0, collections.Counter()
    with gzip.open(tmp, "wt") as out, B.open_chained_tar(shards) as tf:
        for member in tf:
            if not member.isfile():
                continue
            data = tf.extractfile(member)
            if data is None:
                continue
            digest = hashlib.sha256()
            while True:
                chunk = data.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            name = member.name.lstrip("./")
            out.write(f"{name}\t{digest.hexdigest()}\n")
            n += 1
            parts = name.split("/")
            if len(parts) > 2:
                by_dir[parts[1]] += 1
            if n % 20000 == 0:
                print(f"    {n} members", end="\r", flush=True)
    tmp.replace(dest)
    print(f"  indexed {n} members -> {dest}")
    print("  by directory: " + ", ".join(f"{d}({c})" for d, c in by_dir.most_common()))
    return 0


def do_report(args):
    """Is the recipe reachable once every spoken-for image is removed?"""
    _check_recipe()
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
    cache_dir = Path(args.image_cache or (Path(args.out_dir) / "_viscot_paths"))

    print("Loading the Visual-CoT pools ...", flush=True)
    pools = {s: C.load_pool(s, meta_dir)[0] for s in RECIPE_ROWS}

    print("Resolving saliency-r1-8k's images ...", flush=True)
    eightk, how, rep8k = eightk_basenames(meta_dir, cache_dir, pools)
    print(f"  {len(eightk)} basenames, by {how}")
    for k, v in sorted(rep8k.items()):
        print(f"    {k}: {v}")

    print("\nReproducing set_c and the validation draws ...", flush=True)
    set_c, val_c, legacy_val = set_c_basenames(meta_dir)
    print(f"  set_c {len(set_c)}   val_c candidates {len(val_c)}   "
          f"legacy val candidates {len(legacy_val)}")

    excluded = eightk | set_c | val_c | set(legacy_val)
    print(f"  {len(excluded)} basenames excluded in total\n")

    print(f"{'source':16s} {'rows':>5s} {'imgs':>5s} {'q/img':>6s} {'pool imgs':>10s} "
          f"{'free':>8s} {'need>=base':>11s} {'need>=base+1':>13s}")
    ok = True
    for s in sorted(RECIPE_ROWS):
        groups = C.group_by_image(pools[s])
        n, r = IMAGE_BUDGET[s], RECIPE_ROWS[s]
        base, extra = divmod(r, n)
        free = {k: v for k, v in groups.items()
                if os.path.basename(k) not in excluded}
        lo = sum(1 for v in free.values() if len(v) >= base)
        hi = sum(1 for v in free.values() if len(v) >= base + 1)
        fits = lo >= n and hi >= extra
        ok &= fits
        print(f"{s:16s} {r:5d} {n:5d} {r / n:6.2f} {len(groups):10d} {len(free):8d} "
              f"{lo:6d}/{n:<4d} {hi:7d}/{extra:<5d}  {'OK' if fits else 'INFEASIBLE'}")
    print(f"{'TOTAL':16s} {sum(RECIPE_ROWS.values()):5d} {sum(IMAGE_BUDGET.values()):5d} "
          f"{sum(RECIPE_ROWS.values()) / sum(IMAGE_BUDGET.values()):6.2f}")
    print("\nFEASIBLE" if ok else "\nINFEASIBLE")
    return 0 if ok else 1


def do_build(args):
    _check_recipe()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache or (out_dir / "_viscot_paths"))
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"

    pools = {s: C.load_pool(s, meta_dir)[0] for s in RECIPE_ROWS}
    eightk, how, _ = eightk_basenames(meta_dir, cache_dir, pools)
    print(f"saliency-r1-8k occupies {len(eightk)} basenames ({how})")
    set_c, val_c, legacy_val = set_c_basenames(meta_dir)
    excluded = eightk | set_c | val_c | set(legacy_val)
    print(f"excluding {len(excluded)} basenames "
          f"(8k {len(eightk)}, set_c {len(set_c)}, val {len(val_c | set(legacy_val))})")

    print("\nDrawing set_d ...")
    records, report = draw_set_d(meta_dir, excluded, pools)
    for s in sorted(report):
        r = report[s]
        spread = " ".join(f"{k}q x{v}" for k, v in r["per_image"].items())
        print(f"  {s:16s} {r['rows']:6d} rows / {r['images']:5d} images   [{spread}]"
              f"   pool {r['pool_rows']}/{r['pool_images']}")
    used_keys = {r["_ref"][1] for r in records}
    print(f"  total: {len(records)} rows over {len(used_keys)} images")

    print("\nDrawing the validation candidates ...")
    val = draw_val_d(meta_dir, used_keys, excluded, pools)
    for name, (drawn, counts) in val.items():
        print(f"  {name}: " + "  ".join(f"{s}={counts[s]}"
                                        for s in sorted(counts) if counts[s]))

    val_records = [r for drawn, _ in val.values()
                   for recs in drawn.values() for r in recs]
    wanted = used_keys | {r["_ref"][1] for r in val_records}

    if not args.skip_extract:
        shards = sorted((B.hf_snapshot(B.VISCOT_REPO, ["cot_images_tar_split/*"])
                         / "cot_images_tar_split").glob("cot_images_*"))
        if len(shards) != 13:
            raise SystemExit(f"expected 13 Visual-CoT shards, found {len(shards)}")
        print(f"\nStreaming {len(shards)} shards for {len(wanted)} images ...", flush=True)
        missing, seen_dirs, manifest = C.stage_images(wanted, shards, cache_dir)
        fallback = collections.Counter(
            (k.split("/")[1], manifest[k].split("/")[1])
            for k in wanted if k in manifest and manifest[k] != k)
        for (src, got), n in fallback.most_common():
            print(f"  {n} {src} images were served from the archive's {got}/ directory")
        if missing:
            raise SystemExit(f"{len(missing)} images were not in the archive, e.g. "
                             f"{sorted(missing)[:3]}")
    else:
        missing = {k for k in wanted if not (cache_dir / k).exists()}
        if missing:
            raise SystemExit(f"--skip-extract but {len(missing)} images are absent "
                             f"from {cache_dir}, e.g. {sorted(missing)[:3]}")

    # The archive bytes as they sit on disk, which is the space saliency-r1-8k's own
    # stored images live in -- checked here, before the re-encode makes them ours.
    raw = {}
    for key in sorted(used_keys):
        raw[key] = hashlib.sha256((cache_dir / key).read_bytes()).hexdigest()
    _, _, eightk_hashes = eightk_image_hashes()
    clash = {k for k, h in raw.items() if h in eightk_hashes}
    print(f"\nset_d holds {len(set(raw.values()))} distinct archive files; "
          f"{len(clash)} of them are also in saliency-r1-8k")
    if clash:
        raise SystemExit("the draw picked images the 8k already holds: "
                         f"{sorted(clash)[:3]}")

    resolver = C.PathResolver(cache_dir)
    del val_records, wanted, pools
    gc.collect()
    _, set_d_hashes = C.save_chunked("set_d", records, resolver, out_dir, "train")

    train_hashes = set(set_d_hashes)
    print(f"\nset_d holds {len(train_hashes)} distinct images by content "
          f"({len(set_d_hashes) - len(train_hashes)} rows share a picture)")

    excluded_h = set(train_hashes)
    for name, (drawn, counts) in val.items():
        print(f"\nSelecting {name} ...", flush=True)
        rows, hashes, rep = B.pick_clean(drawn, counts, resolver, "validation", excluded_h)
        excluded_h |= hashes
        short = False
        for source, r in rep.items():
            note = ""
            if r["dup_train"] or r["dup_val"] or r["unreadable"]:
                note = (f"   rejected: {r['dup_train']} already in set_d, "
                        f"{r['dup_val']} repeat, {r['unreadable']} unreadable")
            print(f"    {source:18s} {r['kept']:4d}/{r['want']:<4d} of {r['pool']} "
                  f"candidates{note}")
            short |= r["kept"] < r["want"]
        if short:
            raise SystemExit(f"{name}: could not fill every source. Raise "
                             f"VAL_OVERSAMPLE (currently {C.VAL_OVERSAMPLE}).")
        B.save_rows(name, rows, out_dir)

    print(f"\nNow verify what was actually written:")
    print(f"  python build_set_d.py --verify --out-dir {out_dir}")


def do_verify(args):
    """Check the saved artifacts.

    Two different notions of identity, because two different pipelines wrote them:

      * against set_c and every validation set -- built here -- a shared picture is a
        byte-identical STORED image, so the stored SHA-256 is exact;
      * against saliency-r1-8k -- not built here -- the stored bytes are its ARCHIVE
        bytes, so the comparison is between the 8k's stored hashes and the hashes of
        the archive files set_d was materialized from, read out of the image cache.
    """
    if os.environ.get("MALLOC_ARENA_MAX") != "2":
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.image_cache or (out_dir / "_viscot_paths"))
    if not (out_dir / "set_d").exists():
        raise SystemExit(f"missing {out_dir / 'set_d'}")

    ok = True
    print("Hashing set_d ...")
    train = B.stored_image_hashes(out_dir / "set_d")
    train_set = set(train)
    print(f"  set_d: {len(train)} rows, {len(train_set)} distinct images\n")

    seen = {}
    for name in ("val_d_natural", "val_d_nonnatural", "set_c",
                 "val_c_natural", "val_c_nonnatural", "val_natural", "val_nonnatural"):
        path = out_dir / name
        if not path.exists():
            print(f"{name}: absent, skipped")
            continue
        print(f"Hashing {name} ...")
        hashes = B.stored_image_hashes(path)
        leaked = train_set.intersection(hashes)
        print(f"  {name}: {len(hashes)} rows, {len(set(hashes))} distinct images")
        if leaked:
            print(f"  FAIL: {len(leaked)} image(s) also appear in set_d")
            ok = False
        if name.startswith("val"):
            dupes = len(hashes) - len(set(hashes))
            cross = set(hashes).intersection(seen)
            seen.update({h: name for h in hashes})
            if dupes:
                print(f"  FAIL: {dupes} row(s) repeat an image within the set")
                ok = False
            if cross:
                print(f"  FAIL: {len(cross)} image(s) shared with another validation set")
                ok = False

    print("\nChecking against saliency-r1-8k ...")
    rows8k, per8k, hashes8k = eightk_image_hashes()
    print(f"  {rows8k} rows, {len(hashes8k)} distinct pictures "
          f"({sum(len(v) for v in per8k.values())} counted per source)")
    staged = sorted(p for p in cache_dir.rglob("*") if p.is_file()
                    and not p.name.startswith("."))
    if not staged:
        print("  FAIL: the image cache is empty, so set_d's source files cannot be "
              "checked. Re-run --build (or point --image-cache at the staged copy).")
        ok = False
    else:
        # Leg 1, exact but partial: the 5,130 pictures the 8k stores as archive bytes.
        mine = {hashlib.sha256(p.read_bytes()).hexdigest() for p in staged}
        shared = mine & hashes8k
        print(f"  set_d staged {len(staged)} archive files, {len(mine)} distinct")
        if shared:
            print(f"  FAIL: {len(shared)} of them are byte-identical to an 8k image")
            ok = False
        else:
            print("  no staged file is byte-identical to an 8k image")

        # Leg 2, covers the rest: the 8k re-encoded 2,793 of its rows, so those can
        # only be caught by name -- against the same basename set the draw excluded.
        meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
        eightk, how, _ = eightk_basenames(meta_dir, cache_dir)
        used = {p.name for p in staged}
        collide = used & eightk
        print(f"  8k basenames ({how}): {len(eightk)}")
        if collide:
            print(f"  FAIL: {len(collide)} staged file(s) carry an 8k basename, e.g. "
                  f"{sorted(collide)[:3]}")
            ok = False
        else:
            print("  no staged file carries a basename the 8k occupies")

    from datasets import load_from_disk

    ds = load_from_disk(str(out_dir / "set_d"))
    by_src = collections.Counter(ds["dataset"])
    print("\nset_d composition:")
    for s, n in by_src.most_common():
        want = RECIPE_ROWS.get(s)
        flag = "" if want == n else f"   EXPECTED {want}"
        print(f"    {s:18s} {n:6d}{flag}")
        ok &= want == n
    nat = sum(ds["natural"])
    print(f"    {'-- natural':18s} {nat:6d}  ({100 * nat / len(ds):.1f}%)")
    print(f"    questions per image: {len(ds) / max(1, len(train_set)):.2f}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", action="store_true",
                   help="walk the archive once and record every member's SHA-256")
    p.add_argument("--report", action="store_true",
                   help="recipe feasibility once every spoken-for image is removed")
    p.add_argument("--build", action="store_true",
                   help="build set_d and both validation sets")
    p.add_argument("--verify", action="store_true", help="check the saved artifacts")
    p.add_argument("--out-dir", default="cold_data/grpo_sets")
    p.add_argument("--image-cache", default=None,
                   help="where extracted Visual-CoT images live "
                        "(default OUT_DIR/_viscot_paths)")
    p.add_argument("--skip-extract", action="store_true",
                   help="assume the image cache is already populated")
    p.add_argument("--force", action="store_true", help="rebuild the index even if present")
    args = p.parse_args()

    B.require_deps()
    if args.index:
        return do_index(args)
    if args.report:
        return do_report(args)
    if args.build:
        return do_build(args)
    if args.verify:
        return do_verify(args)
    p.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
