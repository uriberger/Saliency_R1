#!/usr/bin/env python
"""Build set_c -- twice saliency-r1-8k, drawn entirely from Visual-CoT -- and the two
validation sets that go with it.

    set_c            16,160 rows over 7,177 images, all Visual-CoT
    val_c_natural       256 rows / 256 images, set_c's natural source proportions
    val_c_nonnatural    256 rows / 256 images, set_c's non-natural proportions

WHAT "THE SAME CHARACTERISTICS" MEANS HERE
------------------------------------------
Measured on peterant330/saliency-r1-8k: 8,080 rows over 6,953 per-source-distinct
images, drawn from ten Visual-CoT sub-datasets. set_c doubles the row count of every
source exactly, which fixes both the source mix and the natural/non-natural split
(docvqa + infographicsvqa = 12.0% of rows in both sets).

The unique-image count cannot be held at 8k's level. Visual-CoT never stores more
than two questions per image for gqa, openimages, textvqa or cub, and the 8k already
sits at ~1.0-1.2 questions per image for those sources, so doubling the rows needs
essentially one fresh image per fresh row:

    source        8k rows  8k imgs  set_c rows  best possible from 8k's image count
    gqa              1765     1752        3530  3504  (26 short)
    openimages        860      712        1720  1424 (296 short)
    textvqa           370      307         740   614 (126 short)

Those three sources are given the minimum image budget that admits their row target
(rows / 2, since their pool caps at 2 questions per image); every other source keeps
the 8k's image count exactly. Total 7,177 images against the 8k's 6,953 -- 3.2% more
images for 100% more rows. See IMAGE_BUDGET below.

WHY IMAGES ARE ADDRESSED BY SOURCE-SCOPED PATH
-----------------------------------------------
build_grpo_sets.py resolves Visual-CoT images by bare basename and therefore has to
drop the 13,225 basenames that more than one sub-dataset claims. That is affordable
when only openimages is drawn from; it is not affordable here, where textcap and
textvqa are core sources and lose 83% and 94% of their rows to the collision. The
archive nests as cot_image_data/<source>/<path> and the metadata's `image` field is
exactly the <path> part, so this builder matches on the full source-scoped path
instead. Nothing is ambiguous and nothing has to be dropped.

Usage:
    python build_set_c.py --report                       # 8k stats and the recipe check
    python build_set_c.py --build --out-dir DIR          # all three sets, one archive pass
    python build_set_c.py --verify --out-dir DIR         # prove the disjointness claims
"""

import argparse
import collections
import gc
import gzip
import json
import os
import random
import sys
import zlib
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_grpo_sets as B

# The login node caps virtual address space at 8 GB (ulimit -v), and the draw leaves a
# couple of GB of freed-but-unreturned pool behind it. Arrow copies a whole chunk while
# converting it, so build_grpo_sets' default of 4,000 rows of 512px JPEG is enough to
# hit the ceiling; 1,000 keeps the conversion's peak allocation near 100 MB.
B.CHUNK_ROWS = 1000


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
# Rows: exactly twice what saliency-r1-8k holds per source (measured, not quoted).
RECIPE_ROWS = {
    "flickr30k": 5430,
    "gqa": 3530,
    "openimages": 1720,
    "docvqa": 1340,
    "textcap": 1280,
    "v7w": 1220,
    "textvqa": 740,
    "infographicsvqa": 600,
    "cub": 160,
    "vsr": 140,
}

# Images: the 8k's per-source unique-image count, except for the three sources whose
# Visual-CoT pool caps at two questions per image, which are raised to the minimum
# that admits their row target (8k counts were gqa 1752, openimages 712, textvqa 307).
IMAGE_BUDGET = {
    "flickr30k": 2618,
    "gqa": 1765,          # raised from 1752
    "openimages": 860,    # raised from 712
    "docvqa": 282,
    "textcap": 441,
    "v7w": 595,
    "textvqa": 370,       # raised from 307
    "infographicsvqa": 103,
    "cub": 80,
    "vsr": 63,
}

# saliency-r1-8k's label -> Visual-CoT's metadata file stem. The 8k calls Visual7W
# "v7w"; only the metadata file uses the long name.
SOURCE_META = {s: s for s in RECIPE_ROWS}
SOURCE_META["v7w"] = "visual7w"

# ... and -> the archive directories that may hold its images, best first. Measured
# from the archive itself (.archive_index.txt.gz): it has no textcap/ directory at all,
# because
# TextCaps is built on OpenImages and its pictures are simply the OpenImages ones. The
# same is true of TextVQA, whose own directory holds only part of what it references.
# Where two directories offer the same name they offer the same picture, so a fallback
# is a rename, not a substitution -- and the manifest records which one was used.
SOURCE_DIRS = {s: [s] for s in RECIPE_ROWS}
SOURCE_DIRS["textcap"] = ["textcap", "openimages", "textvqa"]
SOURCE_DIRS["textvqa"] = ["textvqa", "openimages", "textcap"]
SOURCE_DIRS["openimages"] = ["openimages", "textvqa", "textcap"]
SOURCE_DIRS["v7w"] = ["v7w", "visual7w"]

# Scanned documents and infographics: Grounding-DINO detections on those are noise, so
# the overlap reward is zeroed there. Same rule as build_grpo_sets.py.
NONNATURAL = {"docvqa", "infographicsvqa"}

ARCHIVE_ROOT = "cot_image_data"

SEED = 2026
VAL_SEED = 20260
VAL_SIZE = 256
VAL_OVERSAMPLE = 6

EIGHTK_REPO = "peterant330/saliency-r1-8k"


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------
def archive_key(source, image_field):
    """A record's canonical image identity, and its path in the image cache.

    Named for the source that references it, which is not always the directory the
    bytes came from -- see SOURCE_DIRS. Keeping the canonical name stable is what lets
    a record be resolved without consulting the archive again.
    """
    return f"{ARCHIVE_ROOT}/{source}/{image_field}"


def archive_candidates(key):
    """Where in the archive `key` might actually live, best first."""
    _, source, rel = key.split("/", 2)
    return [f"{ARCHIVE_ROOT}/{d}/{rel}" for d in SOURCE_DIRS[source]]


def load_pool(source, meta_dir):
    """Every Visual-CoT train row of one sub-dataset that carries a usable box.

    No answer-length filter, deliberately. build_grpo_sets.is_verifiable() keeps
    answers of at most three words, which would delete flickr30k -- a third of the
    8k, whose answers average 12 words. set_c mirrors the 8k, so it mirrors that too.
    """
    path = meta_dir / f"{SOURCE_META[source]}_cot_train.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; fetch the Visual-CoT metadata first")

    out, no_box = [], 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            bbox = B.union_bbox(r.get("bboxs"), r.get("width"), r.get("height"))
            if not bbox:
                no_box += 1
                continue
            out.append(
                {
                    "dataset": source,
                    "question_id": i,
                    "problem": r["question"].strip(),
                    "solution": r["answer"].strip(),
                    "bbox": bbox,
                    "natural": source not in NONNATURAL,
                    "_ref": ("viscot_path", archive_key(source, r["image"])),
                }
            )
    return out, no_box


def group_by_image(records):
    """{archive path: [records]}, each image's rows in metadata order."""
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["_ref"][1]].append(r)
    return groups


def source_rng(seed, source):
    # crc32, not hash(): Python randomizes string hashing per process, and this draw
    # has to be reproducible across runs.
    return random.Random(seed + zlib.crc32(source.encode()) % 10_000)


def choose_images(groups, n_images, n_rows, rng, excluded_basenames=frozenset()):
    """Pick n_images images and how many questions to take from each, totalling n_rows.

    The counts are as flat as the pool allows: n_rows % n_images images contribute
    base+1 questions and the rest contribute base, where base = n_rows // n_images.
    Among the images that are large enough, the smallest are preferred -- taking four
    questions off a 12-question scanned document when four suffice would skew the set
    towards a handful of unusually dense images.

    `excluded_basenames` is matched on the basename rather than the full path on
    purpose: sub-datasets built on OpenImages store the same picture under the same
    name in two directories, so a name spent elsewhere is spent here too.
    """
    base, extra = divmod(n_rows, n_images)
    if base == 0:
        raise ValueError("more images than rows requested")

    eligible = [(len(v), k) for k, v in groups.items()
                if len(v) >= base and os.path.basename(k) not in excluded_basenames]
    # Ascending degree, shuffled within each tier so the choice among equals is random
    # rather than an artifact of metadata order.
    rng.shuffle(eligible)
    eligible.sort(key=lambda t: t[0])

    hi = [k for deg, k in eligible if deg >= base + 1]
    if len(hi) < extra:
        raise SystemExit(
            f"only {len(hi)} images carry {base + 1}+ questions, need {extra}"
        )
    if len(eligible) < n_images:
        raise SystemExit(
            f"only {len(eligible)} images carry {base}+ questions, need {n_images}"
        )

    picked = {}
    for k in hi[:extra]:
        picked[k] = base + 1
    for deg, k in eligible:
        if len(picked) == n_images:
            break
        if k not in picked:
            picked[k] = base
    if len(picked) != n_images:
        raise SystemExit(f"could only fill {len(picked)} of {n_images} image slots")
    return picked


def draw_set_c(meta_dir, excluded_keys):
    """Select set_c's rows. Returns (records, per-source report)."""
    records, report = [], {}
    for source in sorted(RECIPE_ROWS):
        pool, no_box = load_pool(source, meta_dir)
        groups = group_by_image(pool)
        rng = source_rng(SEED, source)
        picked = choose_images(groups, IMAGE_BUDGET[source], RECIPE_ROWS[source], rng,
                               excluded_keys)

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
                              pool_images=len(groups), no_box=no_box,
                              per_image=dict(sorted(degs.items())))
    random.Random(SEED).shuffle(records)
    return records, report


# ---------------------------------------------------------------------------
# Images already spent by the existing validation sets
# ---------------------------------------------------------------------------
def legacy_val_reference_keys(meta_dir, oversample=4, val_size=256):
    """Basenames the existing val_natural / val_nonnatural draw could have taken.

    Those two sets were built to be image-disjoint from set_a and set_b, and they stay
    useful only if set_c avoids them too. Their rows do not record an image path, so
    the draw is reproduced instead: build_grpo_sets draws candidates per source with a
    per-source seed, then spends the surplus on a content check. Reproducing the
    candidate list (not the final pick) gives a superset of what they hold, which is
    the safe direction to err in. Restricting the recipes to their Visual-CoT sources
    is exact -- every source is drawn under its own seed, so dropping the others
    changes nothing about the ones kept.
    """
    viscot_nat = {s: n for s, n in B.RECIPE_NATURAL.items() if s in B.VISCOT_SOURCES}
    viscot_non = {s: n for s, n in B.RECIPE_NONNATURAL.items() if s in B.VISCOT_SOURCES}

    ambiguous = B.viscot_ambiguous_basenames(meta_dir)
    pools = {s: B.load_viscot(s, meta_dir, ambiguous) for s in viscot_nat | viscot_non}

    # What set_a / set_b themselves hold -- excluded from the val draw over there, so
    # it has to be excluded here too or the reproduction diverges.
    set_a, _ = B.draw(pools, viscot_nat, B.SEED)
    set_b, _ = B.draw(pools, viscot_non, B.SEED + 1)
    used = {B.image_key(r) for g in (set_a, set_b) for recs in g.values() for r in recs}

    nat_counts = {s: c for s, c in B.allocate(B.RECIPE_NATURAL, val_size).items()
                  if s in viscot_nat}
    non_counts = {s: c for s, c in B.allocate(B.RECIPE_NONNATURAL, val_size).items()
                  if s in viscot_non}
    nat_cand, _ = B.draw_val(pools, nat_counts, used, B.VAL_SEED, oversample)
    nat_keys = {B.image_key(r) for recs in nat_cand.values() for r in recs}
    non_cand, _ = B.draw_val(pools, non_counts, used | nat_keys, B.VAL_SEED + 1, oversample)
    non_keys = {B.image_key(r) for recs in non_cand.values() for r in recs}

    return {base for _, base in nat_keys | non_keys}, {base for _, base in used}


# ---------------------------------------------------------------------------
# Validation draw
# ---------------------------------------------------------------------------
def draw_val_c(meta_dir, used_keys, excluded_basenames, oversample=VAL_OVERSAMPLE,
               val_size=VAL_SIZE):
    """Candidates for val_c_natural and val_c_nonnatural, one row per unused image.

    Source proportions come from set_c's own row counts, split by imagery type. Only
    images set_c did not take are admissible, and the surplus (`oversample`) is what
    pick_clean spends when it rejects a candidate on image CONTENT rather than name.
    """
    nat_recipe = {s: n for s, n in RECIPE_ROWS.items() if s not in NONNATURAL}
    non_recipe = {s: n for s, n in RECIPE_ROWS.items() if s in NONNATURAL}
    nat_counts = B.allocate(nat_recipe, val_size)
    non_counts = B.allocate(non_recipe, val_size)

    # Basenames, not paths: a picture set_c took as textcap/x.jpg is the same picture
    # as openimages/x.jpg, and one of those in validation would be a leak.
    taken = {os.path.basename(k) for k in used_keys} | set(excluded_basenames)
    out = {}
    for name, counts, seed in (("val_c_natural", nat_counts, VAL_SEED),
                               ("val_c_nonnatural", non_counts, VAL_SEED + 1)):
        drawn = {}
        for source, n in counts.items():
            if n <= 0:
                continue
            pool, _ = load_pool(source, meta_dir)
            rng = source_rng(seed, source)
            eligible, seen = [], set()
            for r in pool:
                base = os.path.basename(r["_ref"][1])
                if base in taken or base in seen:
                    continue
                seen.add(base)
                eligible.append(r)
            rng.shuffle(eligible)
            picked = eligible[: n * oversample]
            if len(picked) < n:
                raise SystemExit(f"{name}/{source}: only {len(picked)} free images "
                                 f"for a target of {n}")
            drawn[source] = picked
            taken.update(os.path.basename(r["_ref"][1]) for r in picked)
        out[name] = (drawn, counts)
    return out


# ---------------------------------------------------------------------------
# Image staging
# ---------------------------------------------------------------------------
class PathResolver:
    """Resolves ("viscot_path", "cot_image_data/<source>/<path>") to a PIL image."""

    def __init__(self, root):
        self.root = Path(root)

    def get(self, ref):
        kind, key = ref
        if kind != "viscot_path":
            raise ValueError(f"unknown image ref kind: {kind}")
        return Image.open(self.root / key)


def stage_images(wanted, shards, cache_dir):
    """One sequential pass over the archive, writing out only the wanted members.

    The 13 shards concatenate into a single uncompressed tar, so members cannot be
    seeked to individually; the pass is ~139 GB of reads for ~1.5 GB of writes. Files
    already on disk are skipped, which makes a re-run after a failure cheap.

    The pass runs to the end even once every image has been found, because it also
    writes the archive's full member list to `.archive_index.txt.gz`. That index is
    what turns "which directory holds TextCaps?" into an offline question -- the
    alternative is another 139 GB read per guess.

    Each image is cached under its CANONICAL key, whatever directory supplied the
    bytes, so a record resolves the same way no matter how the archive is laid out.
    `.manifest.json` records the archive path each one actually came from.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / ".archive_index.txt.gz"
    manifest_path = cache_dir / ".manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    remaining = [k for k in wanted if not (cache_dir / k).exists()]
    print(f"  {len(wanted) - len(remaining)} of {len(wanted)} images already cached, "
          f"{len(remaining)} to extract", flush=True)
    if not remaining and index_path.exists():
        return set(), collections.Counter(), manifest

    # archive path -> the canonical keys that would accept it, best candidate first
    accepts = collections.defaultdict(list)
    for key in remaining:
        for rank, cand in enumerate(archive_candidates(key)):
            accepts[cand].append((rank, key))
    pending = set(remaining)

    seen_dirs = collections.Counter()
    left = len(pending)
    with gzip.open(index_path, "wt") as index, B.open_chained_tar(shards) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name.lstrip("./")
            index.write(name + "\n")
            parts = name.split("/")
            if len(parts) > 2:
                seen_dirs[parts[1]] += 1
            takers = [k for _, k in sorted(accepts.get(name, ())) if k in pending]
            if not takers:
                continue
            data = tf.extractfile(member)
            if data is None:
                continue
            payload = data.read()
            for key in takers:
                dest = cache_dir / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(payload)
                manifest[key] = name
                pending.discard(key)
            if left - len(pending) >= 500:
                left = len(pending)
                print(f"    {left} left", end="\r", flush=True)

    manifest_path.write_text(json.dumps(manifest))
    return pending, seen_dirs, manifest


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def do_report(args):
    """Print what the 8k holds and check the recipe against the live pools."""
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
    print(f"{'source':16s} {'rows':>6s} {'imgs':>6s} {'q/img':>6s} {'pool rows':>10s} "
          f"{'pool imgs':>10s} {'plan':>22s}")
    for s in sorted(RECIPE_ROWS):
        pool, _ = load_pool(s, meta_dir)
        groups = group_by_image(pool)
        n, r = IMAGE_BUDGET[s], RECIPE_ROWS[s]
        base, extra = divmod(r, n)
        hi = sum(1 for v in groups.values() if len(v) >= base + 1)
        lo = sum(1 for v in groups.values() if len(v) >= base)
        ok = hi >= extra and lo >= n
        print(f"{s:16s} {r:6d} {n:6d} {r / n:6.2f} {len(pool):10d} {len(groups):10d} "
              f"  {extra}x{base + 1}+{n - extra}x{base} {'OK' if ok else 'INFEASIBLE'}")
    print(f"{'TOTAL':16s} {sum(RECIPE_ROWS.values()):6d} {sum(IMAGE_BUDGET.values()):6d}")


def do_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache or (out_dir / "_viscot_paths"))
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"

    print("Reproducing the existing validation draw, so set_c can avoid its images ...")
    legacy_val, legacy_train = legacy_val_reference_keys(meta_dir)
    print(f"  {len(legacy_val)} basenames reachable by val_natural / val_nonnatural "
          f"-- excluded from set_c")
    print(f"  ({len(legacy_train)} basenames held by set_a / set_b are NOT excluded: "
          f"two training sets may share images)")

    print("\nDrawing set_c ...")
    records, report = draw_set_c(meta_dir, legacy_val)
    for s in sorted(report):
        r = report[s]
        spread = " ".join(f"{k}q x{v}" for k, v in r["per_image"].items())
        print(f"  {s:16s} {r['rows']:6d} rows / {r['images']:5d} images   [{spread}]"
              f"   pool {r['pool_rows']}/{r['pool_images']}"
              + (f"  ({r['no_box']} rows had no box)" if r["no_box"] else ""))
    used_keys = {r["_ref"][1] for r in records}
    print(f"  total: {len(records)} rows over {len(used_keys)} images")

    print("\nDrawing the validation candidates ...")
    val = draw_val_c(meta_dir, used_keys, legacy_val)
    for name, (drawn, counts) in val.items():
        print(f"  {name}: " + "  ".join(f"{s}={counts[s]}" for s in sorted(counts) if counts[s]))

    val_records = [r for drawn, _ in val.values() for recs in drawn.values() for r in recs]
    wanted = used_keys | {r["_ref"][1] for r in val_records}

    if not args.skip_extract:
        shards = sorted((B.hf_snapshot(B.VISCOT_REPO, ["cot_images_tar_split/*"])
                         / "cot_images_tar_split").glob("cot_images_*"))
        if len(shards) != 13:
            raise SystemExit(f"expected 13 Visual-CoT shards, found {len(shards)}")
        print(f"\nStreaming {len(shards)} shards for {len(wanted)} images ...", flush=True)
        missing, seen_dirs, manifest = stage_images(wanted, shards, cache_dir)
        if seen_dirs:
            print("  archive source directories: "
                  + ", ".join(f"{d}({n})" for d, n in seen_dirs.most_common()))
        fallback = collections.Counter(
            (k.split("/")[1], manifest[k].split("/")[1])
            for k in wanted if k in manifest and manifest[k] != k
        )
        for (src, got), n in fallback.most_common():
            print(f"  {n} {src} images were served from the archive's {got}/ directory")
        if missing:
            raise SystemExit(
                f"{len(missing)} images were not in the archive, e.g. "
                f"{sorted(missing)[:3]}. The directory list above shows what the archive "
                f"actually holds; {cache_dir / '.archive_index.txt.gz'} has every member "
                f"path, so the right SOURCE_DIRS entry can be found without re-reading "
                f"139 GB.")
    else:
        missing = {k for k in wanted if not (cache_dir / k).exists()}
        if missing:
            raise SystemExit(f"--skip-extract but {len(missing)} images are absent from "
                             f"{cache_dir}, e.g. {sorted(missing)[:3]}")

    resolver = PathResolver(cache_dir)
    del val_records, wanted
    gc.collect()  # the draw held every source's pool; none of it is needed from here
    B.save_records("set_c", records, resolver, out_dir, "train")

    print("\nHashing set_c's images, so the validation sets can exclude them by content ...",
          flush=True)
    train_hashes = set(B.stored_image_hashes(out_dir / "set_c"))
    print(f"  {len(train_hashes)} distinct images")

    excluded = set(train_hashes)
    for name, (drawn, counts) in val.items():
        print(f"\nSelecting {name} ...", flush=True)
        rows, hashes, rep = B.pick_clean(drawn, counts, resolver, "validation", excluded)
        excluded |= hashes
        short = False
        for source, r in rep.items():
            note = ""
            if r["dup_train"] or r["dup_val"] or r["unreadable"]:
                note = (f"   rejected: {r['dup_train']} already in set_c, "
                        f"{r['dup_val']} repeat, {r['unreadable']} unreadable")
            print(f"    {source:18s} {r['kept']:4d}/{r['want']:<4d} of {r['pool']} "
                  f"candidates{note}")
            short |= r["kept"] < r["want"]
        if short:
            raise SystemExit(f"{name}: could not fill every source with distinct, unused "
                             f"images. Raise VAL_OVERSAMPLE (currently {VAL_OVERSAMPLE}).")
        B.save_rows(name, rows, out_dir)

    print(f"\nNow verify what was actually written:")
    print(f"  python build_set_c.py --verify --out-dir {out_dir}")


def do_verify(args):
    """Check the saved artifacts, not the recipe.

    Every set resizes and re-encodes through the same path, so one source picture
    yields byte-identical output wherever it lands and a leak shows up as a hash
    collision. The existing val_natural / val_nonnatural are checked too when present:
    set_c is only a drop-in for the 8k if those stay usable against it.
    """
    out_dir = Path(args.out_dir)
    if not (out_dir / "set_c").exists():
        raise SystemExit(f"missing {out_dir / 'set_c'}")

    print("Hashing set_c ...")
    train = B.stored_image_hashes(out_dir / "set_c")
    train_set = set(train)
    print(f"  set_c: {len(train)} rows, {len(train_set)} distinct images\n")

    ok = True
    seen = {}
    for name in ("val_c_natural", "val_c_nonnatural", "val_natural", "val_nonnatural"):
        path = out_dir / name
        if not path.exists():
            print(f"{name}: absent, skipped")
            continue
        print(f"Hashing {name} ...")
        hashes = B.stored_image_hashes(path)
        leaked = train_set.intersection(hashes)
        dupes = len(hashes) - len(set(hashes))
        cross = set(hashes).intersection(seen)
        seen.update({h: name for h in hashes})
        print(f"  {name}: {len(hashes)} rows, {len(set(hashes))} distinct images")
        if leaked:
            print(f"  FAIL: {len(leaked)} image(s) also appear in set_c")
            ok = False
        if dupes:
            print(f"  FAIL: {dupes} row(s) repeat an image within the set")
            ok = False
        if cross:
            print(f"  FAIL: {len(cross)} image(s) shared with another validation set")
            ok = False

    from datasets import load_from_disk

    ds = load_from_disk(str(out_dir / "set_c"))
    by_src = collections.Counter(ds["dataset"])
    print("\nset_c composition:")
    for s, n in by_src.most_common():
        want = RECIPE_ROWS.get(s)
        flag = "" if want == n else f"   EXPECTED {want}"
        print(f"    {s:18s} {n:6d}{flag}")
        ok &= want == n
    nat = sum(ds["natural"])
    print(f"    {'-- natural':18s} {nat:6d}  ({100 * nat / len(ds):.1f}%)")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="recipe feasibility against the pools")
    p.add_argument("--build", action="store_true", help="build set_c and both validation sets")
    p.add_argument("--verify", action="store_true", help="check the saved artifacts")
    p.add_argument("--out-dir", default="cold_data/grpo_sets")
    p.add_argument("--image-cache", default=None,
                   help="where extracted Visual-CoT images live (default OUT_DIR/_viscot_paths)")
    p.add_argument("--skip-extract", action="store_true",
                   help="assume the image cache is already populated")
    args = p.parse_args()

    B.require_deps()
    if args.report:
        return do_report(args)
    if args.build:
        return do_build(args)
    if args.verify:
        return do_verify(args)
    p.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
