#!/usr/bin/env python
"""Build set_e -- twice saliency-r1-8k's rows at the 8k's exact questions-per-image --
and the two validation sets that go with it.

    set_e             16,160 rows over 13,906 images, all Visual-CoT
    val_e_natural        256 rows / 256 images, set_e's natural source proportions
    val_e_nonnatural     256 rows / 256 images, set_e's non-natural proportions

WHY THIS SET EXISTS
-------------------
docs/bigger-training-set.md is the full argument; the short form is that a 2x corpus at
the 8k's shape does not exist on disk. set_c has the rows (16,160) but found only 3.2%
more pictures than the 8k, so it packs 2.26 questions per image against the 8k's 1.16 --
and it hacked the overlap reward at ~step 2,400. set_d has the shape (1.16 q/img) but
only the 8k's row count. set_e is set_c's row counts with set_d's packing discipline:

    saliency-r1-8k    8,080 rows /  6,953 images   1.16 q/img   86.0% unique
    set_c            16,160 rows /  7,160 images   2.26 q/img   44.3% unique
    set_d             8,080 rows /  6,946 images   1.16 q/img   86.0% unique
    set_e            16,160 rows / 13,906 images   1.16 q/img   86.0% unique

Every per-source row count is exactly 2x the 8k's and every per-source IMAGE count is
exactly 2x the 8k's, so each source keeps its own packing rather than being flattened to
the corpus mean -- docvqa is 2.38 q/img in the 8k and stays 2.38 here. That is what makes
set_e differ from the 8k in one dimension (size) and only one.

DIFFICULTY IS MATCHED BY CONSTRUCTION, NOT BY FILTERING
-------------------------------------------------------
set_a was both larger and easier than the 8k (cold-start accuracy 0.40 against the 8k's
0.26) and it hacked first. set_e holds the 8k's source mix at the 8k's proportions and
draws from the same Visual-CoT pools, so its difficulty is the 8k's up to sampling noise.
No difficulty filter is applied: docs/bigger-training-set.md §5 shows the 8k is *more*
group-saturated than set_a and did not hack, so "keep the hard prompts" is not a rule the
runs support. Matching the mix is the claim that is supported.

WHAT IS EXCLUDED, AND WHAT IS MERELY DE-PRIORITISED
----------------------------------------------------
Hard exclusion (a drawn image may not be any of these): saliency-r1-8k, set_c, set_d, and
all three validation candidate pools (val_natural/val_nonnatural, val_c, val_d). The 8k
is resolved the two ways build_set_d documents at length -- archive SHA-256 index plus a
content match in the metadata -- because the 8k re-encoded 2,793 of its own rows and a
basename alone does not reach them.

Soft de-prioritisation: set_a / set_b's Visual-CoT images. Excluding them outright is
infeasible on exactly one source -- vsr's whole pool is 1,765 images and set_a/set_b hold
1,676 of them, leaving 27 against the 126 set_e needs. So they are ordered last in every
source's candidate list and taken only where the pool runs dry, which is ~0% incidental
overlap everywhere except vsr instead of the ~22% a blind draw gives. build_set_c made
the same call in its own words: "two training sets may share images".

TWO CONSTANTS, TWO JOBS
-----------------------
`natural` on a TRAINING row means "the overlap reward can grade this", and here it is
driven by OVERLAP_UNGRADEABLE -- the four OCR/document sources that carry 2.5-4x the
within-group overlap spread of every photographic source and the highest share of the
advantage. Those are the rows `--overlap_natural_only` masks, and they are the hack's raw
material. C.NONNATURAL keeps its own job of partitioning the VALIDATION draw by imagery
type, so val_e_natural / val_e_nonnatural stay directly comparable to val_c_* and val_d_*.
The two sets differ: textvqa and textcap are photographs (natural imagery) whose step text
is about written words (not overlap-gradeable).

Usage:
    python build_set_e.py --report                       # feasibility against the pools
    python build_set_e.py --build --out-dir DIR          # all three sets
    python build_set_e.py --verify --out-dir DIR         # prove the disjointness claims

--index is not repeated here: build_set_d's archive SHA-256 index is read from the same
cache directory, and it is a property of the archive, not of a set.
"""

import argparse
import collections
import gc
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_grpo_sets as B
import build_set_c as C
import build_set_d as D

B.CHUNK_ROWS = 1000


# ---------------------------------------------------------------------------
# Recipe -- set_c's rows over twice set_d's images
# ---------------------------------------------------------------------------
# set_c's per-source row counts already ARE 2x the 8k's (build_set_d derives its own by
# halving them), so this is the 8k's source mix at twice the size.
RECIPE_ROWS = dict(C.RECIPE_ROWS)

# Twice the 8k's own per-source unique-image counts, which is what build_set_d holds at
# 1x. Note this is NOT set_c's IMAGE_BUDGET: set_c kept the 8k's image counts while
# doubling the rows, which is the packing this set exists to undo.
IMAGE_BUDGET = {s: n * 2 for s, n in D.IMAGE_BUDGET.items()}

# The overlap reward's gate, distinct from C.NONNATURAL -- see the module docstring.
# docvqa and infographicsvqa are scanned pages and infographics; textvqa and textcap are
# photographs whose questions are about the written words in them. Grounding-DINO is
# answering a question it was not trained for on all four.
OVERLAP_UNGRADEABLE = {"docvqa", "infographicsvqa", "textvqa", "textcap"}

SEED = 2028           # not set_c's 2026 and not set_d's 2027
VAL_SEED = 20280


def _check_recipe():
    assert sum(RECIPE_ROWS.values()) == 16160, sum(RECIPE_ROWS.values())
    assert sum(IMAGE_BUDGET.values()) == 13906, sum(IMAGE_BUDGET.values())
    assert set(RECIPE_ROWS) == set(IMAGE_BUDGET)
    # Every source must be exactly 2x the 8k, in both dimensions, or the mix has moved.
    for s in RECIPE_ROWS:
        assert RECIPE_ROWS[s] == 2 * D.RECIPE_ROWS[s], s
        assert IMAGE_BUDGET[s] == 2 * D.IMAGE_BUDGET[s], s
    assert OVERLAP_UNGRADEABLE >= C.NONNATURAL, "the gate must not un-mask a NONNATURAL source"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def choose_images_ranked(groups, n_images, n_rows, rng, excluded=frozenset(),
                         deprioritised=frozenset()):
    """C.choose_images, plus a soft preference against `deprioritised` basenames.

    Identical to the original in every other respect: counts are as flat as the pool
    allows (`n_rows % n_images` images give base+1 questions, the rest give base), and
    among the images large enough the smallest are preferred so the set is not skewed
    towards a handful of unusually dense scanned documents.

    The de-prioritised images sort last but stay eligible, which is the whole point --
    a hard filter makes vsr infeasible. Within each (priority, degree) tier the order is
    still the shuffle, so the choice among equals stays random rather than an artifact
    of metadata order.
    """
    base, extra = divmod(n_rows, n_images)
    if base == 0:
        raise ValueError("more images than rows requested")

    eligible = [(os.path.basename(k) in deprioritised, len(v), k)
                for k, v in groups.items()
                if len(v) >= base and os.path.basename(k) not in excluded]
    rng.shuffle(eligible)
    eligible.sort(key=lambda t: (t[0], t[1]))

    hi = [k for _, deg, k in eligible if deg >= base + 1]
    if len(hi) < extra:
        raise SystemExit(f"only {len(hi)} images carry {base + 1}+ questions, need {extra}")
    if len(eligible) < n_images:
        raise SystemExit(f"only {len(eligible)} images carry {base}+ questions, "
                         f"need {n_images}")

    picked = {}
    for k in hi[:extra]:
        picked[k] = base + 1
    for _, deg, k in eligible:
        if len(picked) == n_images:
            break
        if k not in picked:
            picked[k] = base
    if len(picked) != n_images:
        raise SystemExit(f"could only fill {len(picked)} of {n_images} image slots")

    used_dep = sum(1 for k in picked if os.path.basename(k) in deprioritised)
    return picked, used_dep


def draw_set_e(meta_dir, excluded, deprioritised, pools=None):
    """Select set_e's rows. Returns (records, per-source report)."""
    import random

    records, report = [], {}
    for source in sorted(RECIPE_ROWS):
        pool = (pools or {}).get(source) or C.load_pool(source, meta_dir)[0]
        groups = C.group_by_image(pool)
        rng = C.source_rng(SEED, source)
        picked, used_dep = choose_images_ranked(
            groups, IMAGE_BUDGET[source], RECIPE_ROWS[source], rng,
            excluded, deprioritised)

        drawn = []
        for key in sorted(picked):
            rows = sorted(groups[key], key=lambda r: r["question_id"])
            take = picked[key]
            drawn.extend(rows if take >= len(rows) else rng.sample(rows, take))
        assert len(drawn) == RECIPE_ROWS[source], (source, len(drawn))

        # The overlap gate, not the imagery-type label -- see the module docstring.
        natural = source not in OVERLAP_UNGRADEABLE
        for r in drawn:
            r["natural"] = natural

        rng.shuffle(drawn)
        records.extend(drawn)

        degs = collections.Counter(picked.values())
        report[source] = dict(rows=len(drawn), images=len(picked), pool_rows=len(pool),
                              pool_images=len(groups), from_legacy=used_dep,
                              per_image=dict(sorted(degs.items())))
    random.Random(SEED).shuffle(records)
    return records, report


def draw_val_e(meta_dir, used_keys, excluded, pools=None):
    """val_e_natural / val_e_nonnatural candidates, on set_e's source proportions.

    Split by C.NONNATURAL (imagery type), exactly as val_c_* and val_d_* were, so the
    three validation pairs measure the same two things. Their `natural` column therefore
    keeps build_set_c's meaning, which is deliberate and is why OVERLAP_UNGRADEABLE is
    applied in draw_set_e rather than in load_pool.
    """
    nat_recipe = {s: n for s, n in RECIPE_ROWS.items() if s not in C.NONNATURAL}
    non_recipe = {s: n for s, n in RECIPE_ROWS.items() if s in C.NONNATURAL}
    nat_counts = B.allocate(nat_recipe, C.VAL_SIZE)
    non_counts = B.allocate(non_recipe, C.VAL_SIZE)

    taken = {os.path.basename(k) for k in used_keys} | set(excluded)
    out = {}
    for name, counts, seed in (("val_e_natural", nat_counts, VAL_SEED),
                               ("val_e_nonnatural", non_counts, VAL_SEED + 1)):
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
# What every earlier set has already spent
# ---------------------------------------------------------------------------
def spent_basenames(meta_dir, cache_dir, pools=None):
    """Replay every draw on record. Returns (hard_excluded, deprioritised, breakdown).

    set_c, set_d and all three validation pools are seeded draws, so replaying them is
    exact and costs no disk. The 8k is resolved by build_set_d's two mechanisms.
    """
    eightk, how, _ = D.eightk_basenames(meta_dir, cache_dir, pools)
    set_c, val_c, legacy_val = D.set_c_basenames(meta_dir)
    _, legacy_train = C.legacy_val_reference_keys(meta_dir)

    # set_d's own draw, replayed under the exclusions it was built with.
    excl_d = eightk | set_c | val_c | set(legacy_val)
    recs_d, _ = D.draw_set_d(meta_dir, excl_d, pools)
    set_d = {os.path.basename(r["_ref"][1]) for r in recs_d}
    val_d = D.draw_val_d(meta_dir, {r["_ref"][1] for r in recs_d}, excl_d, pools)
    val_d_names = {os.path.basename(r["_ref"][1])
                   for drawn, _ in val_d.values() for recs in drawn.values() for r in recs}

    breakdown = collections.OrderedDict([
        ("saliency-r1-8k", eightk),
        ("set_c", set_c),
        ("set_d", set_d),
        ("val_natural/nonnatural pool", set(legacy_val)),
        ("val_c pool", val_c),
        ("val_d pool", val_d_names),
    ])
    hard = set().union(*breakdown.values())
    breakdown["set_a / set_b (de-prioritised)"] = set(legacy_train)
    return hard, set(legacy_train), breakdown, how


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def do_report(args):
    """Is the recipe reachable once every spoken-for image is removed?"""
    _check_recipe()
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
    cache_dir = Path(args.image_cache or (Path(args.out_dir) / "_viscot_paths"))

    print("Loading the Visual-CoT pools ...", flush=True)
    pools = {s: C.load_pool(s, meta_dir)[0] for s in RECIPE_ROWS}

    print("Replaying every set on record ...", flush=True)
    excluded, deprioritised, breakdown, how = spent_basenames(meta_dir, cache_dir, pools)
    for name, names in breakdown.items():
        print(f"  {name:34s} {len(names):7d}")
    print(f"  {'HARD EXCLUDED':34s} {len(excluded):7d}   (8k resolved by {how})")

    print(f"\n{'source':16s} {'rows':>6s} {'imgs':>6s} {'q/img':>6s} {'pool':>7s} "
          f"{'free':>7s} {'>=base':>12s} {'>=base+1':>12s}  verdict")
    ok = True
    for s in sorted(RECIPE_ROWS):
        groups = C.group_by_image(pools[s])
        n, r = IMAGE_BUDGET[s], RECIPE_ROWS[s]
        base, extra = divmod(r, n)
        free = {k: v for k, v in groups.items()
                if os.path.basename(k) not in excluded}
        lo = sum(1 for v in free.values() if len(v) >= base)
        hi = sum(1 for v in free.values() if len(v) >= base + 1)
        # How much of the free pool is untouched by set_a/set_b, i.e. how much of the
        # draw can avoid them entirely.
        clean = sum(1 for k, v in free.items()
                    if len(v) >= base and os.path.basename(k) not in deprioritised)
        fits = lo >= n and hi >= extra
        ok &= fits
        note = "" if clean >= n else f"   {n - clean} must come from set_a/set_b"
        print(f"{s:16s} {r:6d} {n:6d} {r / n:6.2f} {len(groups):7d} {len(free):7d} "
              f"{lo:7d}/{n:<4d} {hi:7d}/{extra:<4d}  "
              f"{'OK' if fits else 'INFEASIBLE'}{note}")
    tot_r, tot_n = sum(RECIPE_ROWS.values()), sum(IMAGE_BUDGET.values())
    print(f"{'TOTAL':16s} {tot_r:6d} {tot_n:6d} {tot_r / tot_n:6.2f}")
    print(f"\nunique images: {100 * tot_n / tot_r:.1f}%  "
          f"(saliency-r1-8k: {100 * 6953 / 8080:.1f}%)")
    print("\nFEASIBLE" if ok else "\nINFEASIBLE")
    return 0 if ok else 1


def do_build(args):
    _check_recipe()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache or (out_dir / "_viscot_paths"))
    meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"

    pools = {s: C.load_pool(s, meta_dir)[0] for s in RECIPE_ROWS}
    print("Replaying every set on record ...", flush=True)
    excluded, deprioritised, breakdown, how = spent_basenames(meta_dir, cache_dir, pools)
    for name, names in breakdown.items():
        print(f"  {name:34s} {len(names):7d}")
    print(f"  excluding {len(excluded)} basenames (8k resolved by {how}); "
          f"{len(deprioritised)} de-prioritised")

    print("\nDrawing set_e ...")
    records, report = draw_set_e(meta_dir, excluded, deprioritised, pools)
    for s in sorted(report):
        r = report[s]
        spread = " ".join(f"{k}q x{v}" for k, v in r["per_image"].items())
        legacy = f"   {r['from_legacy']} from set_a/set_b" if r["from_legacy"] else ""
        print(f"  {s:16s} {r['rows']:6d} rows / {r['images']:5d} images   [{spread}]"
              f"   pool {r['pool_rows']}/{r['pool_images']}{legacy}")
    used_keys = {r["_ref"][1] for r in records}
    print(f"  total: {len(records)} rows over {len(used_keys)} images "
          f"({len(used_keys) / len(records) * 100:.1f}% unique)")

    print("\nDrawing the validation candidates ...")
    val = draw_val_e(meta_dir, used_keys, excluded, pools)
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
    _, _, eightk_hashes = D.eightk_image_hashes()
    clash = {k for k, h in raw.items() if h in eightk_hashes}
    print(f"\nset_e holds {len(set(raw.values()))} distinct archive files; "
          f"{len(clash)} of them are also in saliency-r1-8k")
    if clash:
        raise SystemExit("the draw picked images the 8k already holds: "
                         f"{sorted(clash)[:3]}")

    resolver = C.PathResolver(cache_dir)
    del val_records, wanted, pools
    gc.collect()
    _, set_e_hashes = C.save_chunked("set_e", records, resolver, out_dir, "train")

    train_hashes = set(set_e_hashes)
    print(f"\nset_e holds {len(train_hashes)} distinct images by content "
          f"({len(set_e_hashes) - len(train_hashes)} rows share a picture)")

    excluded_h = set(train_hashes)
    for name, (drawn, counts) in val.items():
        print(f"\nSelecting {name} ...", flush=True)
        rows, hashes, rep = B.pick_clean(drawn, counts, resolver, "validation", excluded_h)
        excluded_h |= hashes
        short = False
        for source, r in rep.items():
            note = ""
            if r["dup_train"] or r["dup_val"] or r["unreadable"]:
                note = (f"   rejected: {r['dup_train']} already in set_e, "
                        f"{r['dup_val']} repeat, {r['unreadable']} unreadable")
            print(f"    {source:18s} {r['kept']:4d}/{r['want']:<4d} of {r['pool']} "
                  f"candidates{note}")
            short |= r["kept"] < r["want"]
        if short:
            raise SystemExit(f"{name}: could not fill every source. Raise "
                             f"VAL_OVERSAMPLE (currently {C.VAL_OVERSAMPLE}).")
        B.save_rows(name, rows, out_dir)

    link_val_dir(out_dir)
    print(f"\nNow verify what was actually written:")
    print(f"  python build_set_e.py --verify --out-dir {out_dir}")


def link_val_dir(out_dir):
    """OUT_DIR/val_e/{val_natural,val_nonnatural} -> the val_e_* sets.

    The launcher's --val-sets-dir looks for those two names, so each corpus needs its
    own directory of aliases. Same shape as val_c/ and val_d/.
    """
    out_dir = Path(out_dir)
    link_dir = out_dir / "val_e"
    link_dir.mkdir(exist_ok=True)
    for alias, target in (("val_natural", "val_e_natural"),
                          ("val_nonnatural", "val_e_nonnatural")):
        link = link_dir / alias
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / target)
    print(f"\nvalidation aliases: {link_dir}/{{val_natural,val_nonnatural}}")


def do_verify(args):
    """Check the saved artifacts, on the same two notions of identity build_set_d uses."""
    if os.environ.get("MALLOC_ARENA_MAX") != "2":
        os.environ["MALLOC_ARENA_MAX"] = "2"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.image_cache or (out_dir / "_viscot_paths"))
    if not (out_dir / "set_e").exists():
        raise SystemExit(f"missing {out_dir / 'set_e'}")

    ok = True
    print("Hashing set_e ...")
    train = B.stored_image_hashes(out_dir / "set_e")
    train_set = set(train)
    print(f"  set_e: {len(train)} rows, {len(train_set)} distinct images "
          f"({100 * len(train_set) / max(1, len(train)):.1f}% unique)\n")

    seen = {}
    for name in ("val_e_natural", "val_e_nonnatural", "set_c", "set_d",
                 "val_c_natural", "val_c_nonnatural", "val_d_natural",
                 "val_d_nonnatural", "val_natural", "val_nonnatural"):
        path = out_dir / name
        if not path.exists():
            print(f"{name}: absent, skipped")
            continue
        print(f"Hashing {name} ...")
        hashes = B.stored_image_hashes(path)
        leaked = train_set.intersection(hashes)
        print(f"  {name}: {len(hashes)} rows, {len(set(hashes))} distinct images")
        if leaked:
            print(f"  FAIL: {len(leaked)} image(s) also appear in set_e")
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
    rows8k, per8k, hashes8k = D.eightk_image_hashes()
    print(f"  {rows8k} rows, {len(hashes8k)} distinct pictures "
          f"({sum(len(v) for v in per8k.values())} counted per source)")
    staged = sorted(p for p in cache_dir.rglob("*") if p.is_file()
                    and not p.name.startswith("."))
    if not staged:
        print("  FAIL: the image cache is empty, so set_e's source files cannot be "
              "checked. Re-run --build (or point --image-cache at the staged copy).")
        ok = False
    else:
        # Leg 1, exact but partial: the pictures the 8k stores as archive bytes.
        mine = {hashlib.sha256(p.read_bytes()).hexdigest() for p in staged}
        shared = mine & hashes8k
        print(f"  the cache holds {len(staged)} archive files, {len(mine)} distinct")
        if shared:
            print(f"  FAIL: {len(shared)} of them are byte-identical to an 8k image")
            ok = False
        else:
            print("  no staged file is byte-identical to an 8k image")

        # Leg 2, covers the rest: the 8k re-encoded 2,793 of its rows, so those can
        # only be caught by name -- against the same basename set the draw excluded.
        meta_dir = B.hf_snapshot(B.VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
        eightk, how, _ = D.eightk_basenames(meta_dir, cache_dir)
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

    ds = load_from_disk(str(out_dir / "set_e"))
    by_src = collections.Counter(ds["dataset"])
    print("\nset_e composition:")
    for s, n in by_src.most_common():
        want = RECIPE_ROWS.get(s)
        flag = "" if want == n else f"   EXPECTED {want}"
        print(f"    {s:18s} {n:6d}{flag}")
        ok &= want == n
    nat = sum(ds["natural"])
    want_nat = sum(n for s, n in RECIPE_ROWS.items() if s not in OVERLAP_UNGRADEABLE)
    print(f"    {'-- natural':18s} {nat:6d}  ({100 * nat / len(ds):.1f}%)"
          + ("" if nat == want_nat else f"   EXPECTED {want_nat}"))
    ok &= nat == want_nat
    print(f"    questions per image: {len(ds) / max(1, len(train_set)):.2f}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true",
                   help="recipe feasibility once every spoken-for image is removed")
    p.add_argument("--build", action="store_true",
                   help="build set_e and both validation sets")
    p.add_argument("--verify", action="store_true", help="check the saved artifacts")
    p.add_argument("--out-dir", default="cold_data/grpo_sets")
    p.add_argument("--image-cache", default=None,
                   help="where extracted Visual-CoT images live "
                        "(default OUT_DIR/_viscot_paths)")
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
