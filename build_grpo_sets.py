#!/usr/bin/env python
"""Build the two 50K GRPO training sets for the natural-vs-mixed imagery experiment.

    set_a   50,000 rows, 100% natural images
    set_b   50,000 rows, 40,000 natural + 10,000 non-natural (80/20)

set_b's natural half is drawn as a strict 80% subset of set_a's per-source draw, so
the two sets differ *only* in the 10K swap. Any downstream gap is attributable to the
imagery mix rather than to a source-composition confound.

Both sets keep only reasoning-style questions with verifiable short answers, and are
emitted in the saliency-r1-8k column layout so the trainer needs no changes:

    dataset, split, question_id, problem, bbox, solution, image

plus one extra column:

    natural (bool)  -- lets the overlap reward be zeroed on non-natural rows, where
                       Grounding-DINO detections are noise rather than signal.

`bbox` is the normalized "[x1, y1, x2, y2]" string the saliency reward parses, and is
empty for sources that ship no boxes (A-OKVQA, ViRL39K). Rows with an empty bbox are
usable with --reward_variant ours|none only.

Usage:
    python build_grpo_sets.py --probe                 # inspect archive layouts first
    python build_grpo_sets.py --download              # fetch missing sources
    python build_grpo_sets.py --build --out-dir DIR   # construct and save both sets
"""

import argparse
import io
import json
import os
import random
import re
import sys
import tarfile
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
# Natural draw for set_a. set_b's natural half is these counts * NATURAL_KEEP.
RECIPE_NATURAL = {
    "gqa": 18000,          # compositional multi-hop, 100% short answers
    "aokvqa": 17000,       # outside-knowledge MCQ, 100% verifiable
    "visual7w": 6000,      # why/how questions preferred
    "openimages": 5000,    # relation reasoning
    "vsr": 3000,           # spatial relations
    "visdrone": 1000,      # counting / enumeration (TreeVGR MCQ rows only)
}

NATURAL_KEEP = 0.8  # set_b keeps 80% of each natural source

# The 10K non-natural block that set_b swaps in.
RECIPE_NONNATURAL = {
    "virl_math_geo": 3500,     # math figures + geometry diagrams
    "virl_charts": 2500,       # tables / diagrams / charts
    "docvqa": 2000,            # scanned documents
    "virl_science": 1000,      # science diagrams
    "infographicsvqa": 1000,   # infographics
}

VIRL_CATEGORIES = {
    "virl_math_geo": ["(GradeSchool) Non-Geo Math", "(GradeSchool) Geometric"],
    "virl_charts": ["Tables/Diagrams/Charts"],
    "virl_science": ["(GradeSchool) Science", "Broader STEM Topics"],
}

VISCOT_SOURCES = {"gqa", "visual7w", "openimages", "vsr", "docvqa", "infographicsvqa"}

SEED = 42
MAX_IMAGE_SIDE = 512  # matches prepare_image() in trl/grpo_vlm_qwen3.py

VISCOT_REPO = "deepcs233/Visual-CoT"
AOKVQA_REPO = "HuggingFaceM4/A-OKVQA"
TREEVGR_REPO = "HaochenWang/TreeVGR-RL-37K"
VIRL_REPO = "TIGER-Lab/ViRL39K"

TREEVGR_PARQUET = "vstar30k_visdrone6k_x1y1x2y2.parquet"
VIRL_PARQUET = "39Krelease.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_verifiable(answer):
    """Short enough for accuracy_reward to grade by math_verify or exact match.

    Measured on Visual-CoT: this keeps 100% of gqa/openimages/vsr/cub, ~84% of
    visual7w and docvqa, and ~0% of flickr30k (whose answers are full sentences).
    """
    a = (answer or "").strip()
    return 0 < len(a.split()) <= 3


def union_bbox(boxes, width, height):
    """Collapse a list of absolute [x1,y1,x2,y2] boxes into one normalized string.

    Returns "" when the boxes are missing or degenerate. Normalizing here means the
    value survives the downstream resize unchanged.
    """
    if not boxes or not width or not height:
        return ""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for b in boxes:
        b = list(b)
        if len(b) != 4:
            continue
        xs1.append(b[0]); ys1.append(b[1]); xs2.append(b[2]); ys2.append(b[3])
    if not xs1:
        return ""
    x1, y1 = min(xs1) / width, min(ys1) / height
    x2, y2 = max(xs2) / width, max(ys2) / height
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return ""
    return f"[{round(x1, 3)}, {round(y1, 3)}, {round(x2, 3)}, {round(y2, 3)}]"


def prepare_image(img):
    """Cap the long side at MAX_IMAGE_SIDE and force RGB, as the trainer does."""
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def hf_snapshot(repo_id, allow_patterns=None, local_only=False):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_files_only=local_only,
        )
    )


def require_deps():
    """Fail loudly if run with an interpreter that lacks the project packages.

    The login node's default python is /cm/local/apps/python3, which happens to
    have PIL but not huggingface_hub or datasets -- enough for this module to
    import cleanly and then fail deep inside a helper.
    """
    import importlib

    missing = [m for m in ("huggingface_hub", "datasets", "pandas", "PIL")
               if not importlib.util.find_spec(m)]
    if missing:
        raise SystemExit(
            f"missing required package(s): {', '.join(missing)}\n"
            f"current interpreter: {sys.executable}\n"
            f"This needs the project environment:\n"
            f"    conda activate saliency_r1_qwen3_vllm"
        )


def resolve_cached(repo_id, allow_patterns, glob_pat):
    """Locate already-downloaded files WITHOUT triggering a download.

    Returns (paths, error). A probe that downloads 6 GB to tell you what it found
    is not a probe, so this is strictly local-cache lookup.
    """
    try:
        root = hf_snapshot(repo_id, allow_patterns, local_only=True)
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:200]}"
    paths = sorted(root.glob(glob_pat))
    if not paths:
        return [], f"not in local cache (no {glob_pat!r} under {root}) -- run --download"
    return paths, None


# ---------------------------------------------------------------------------
# Candidate loaders -- each returns a list of dicts WITHOUT the decoded image.
# `_ref` carries whatever the image resolver needs to fetch the pixels later.
# ---------------------------------------------------------------------------
def viscot_ambiguous_basenames(meta_dir):
    """Basenames claimed by more than one Visual-CoT sub-dataset.

    The metadata references images by bare basename, and the image archive holds
    all 12 sub-datasets, so a name owned by two sources cannot be resolved
    unambiguously by basename alone. 181 such names touch the sources we use --
    all openimages names also referenced by textcap/textvqa, which are built on
    OpenImages and so almost certainly point at the same picture. "Almost
    certainly" is not a basis for silently picking one, and 181 rows out of 43K
    candidates is not worth the risk, so they are dropped.
    """
    owners = defaultdict(set)
    for path in sorted(meta_dir.glob("*_cot_train.jsonl")):
        src = path.name.replace("_cot_train.jsonl", "")
        with open(path) as fh:
            for line in fh:
                owners[os.path.basename(json.loads(line)["image"])].add(src)
    return {name for name, srcs in owners.items() if len(srcs) > 1}


def load_viscot(source, meta_dir, ambiguous=frozenset()):
    """Read one Visual-CoT metadata jsonl and keep the verifiable rows."""
    path = meta_dir / f"{source}_cot_train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run --download first")

    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if not is_verifiable(r.get("answer")):
                continue
            if os.path.basename(r["image"]) in ambiguous:
                continue
            out.append(
                {
                    "dataset": source,
                    "question_id": i,
                    "problem": r["question"].strip(),
                    "solution": r["answer"].strip(),
                    "bbox": union_bbox(r.get("bboxs"), r.get("width"), r.get("height")),
                    "natural": source not in ("docvqa", "infographicsvqa"),
                    "_ref": ("viscot", r["image"]),
                }
            )
    return out


def rank_visual7w(records):
    """Put why/how questions first -- those are the ones that need reasoning."""
    reasoning, other = [], []
    for r in records:
        (reasoning if re.match(r"^(why|how)\b", r["problem"], re.I) else other).append(r)
    return reasoning, other


def load_aokvqa():
    """A-OKVQA as multiple choice: the letter is the verifiable target."""
    from datasets import load_dataset

    ds = load_dataset(AOKVQA_REPO, split="train")
    letters = "ABCD"
    out = []
    # Iterate without the image column: indexing a row decodes every column, and
    # decoding 17K images just to read the question text is pure waste. The
    # resolver pulls pixels from `ds` lazily, one row at a time, later on.
    for i, r in enumerate(ds.remove_columns(["image"])):
        choices = list(r["choices"])
        idx = r["correct_choice_idx"]
        if idx is None or idx >= len(letters) or len(choices) > len(letters):
            continue
        rendered = " ".join(f"({letters[j]}) {c}" for j, c in enumerate(choices))
        out.append(
            {
                "dataset": "aokvqa",
                "question_id": int(r["question_id"]) if str(r["question_id"]).isdigit() else i,
                "problem": f"{r['question'].strip()} Choices: {rendered}",
                "solution": letters[idx],
                "bbox": "",  # A-OKVQA ships no boxes
                "natural": True,
                "_ref": ("inline", i),
            }
        )
    return out, ds


def load_visdrone(parquet_path):
    """TreeVGR's MCQ rows only -- the V* half is free-form and fails verifiability."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out = []
    for i, r in df.iterrows():
        answer = str(r["answer"]).strip()
        if not re.fullmatch(r"\(?[A-Da-d]\)?", answer):
            continue
        imgs = list(r["images"])
        if len(imgs) != 1:
            continue
        boxes = [list(t["bbox"]) for t in r["target_instances"] if t is not None]
        out.append(
            {
                "dataset": "visdrone",
                "question_id": int(i),
                "problem": re.sub(r"^<image>\s*", "", str(r["problem"])).strip(),
                "solution": answer.strip("()").upper(),
                # Absolute boxes; normalized once the image is opened (no w/h in parquet).
                "_boxes_abs": boxes,
                "bbox": "",
                "natural": True,
                "_ref": ("treevgr", imgs[0]),
            }
        )
    return out


def load_virl(group, parquet_path):
    """ViRL39K rows for one non-natural category group. Answers are natively \\boxed{}."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    cats = VIRL_CATEGORIES[group]
    out = []
    for i, r in df.iterrows():
        if r["category"] not in cats:
            continue
        imgs = list(r["image"])
        if len(imgs) != 1:  # trainer takes exactly one image per sample
            continue
        out.append(
            {
                "dataset": group,
                "question_id": int(i),
                "problem": re.sub(r"^<image>\s*/?n?\s*", "", str(r["question"])).strip(),
                "solution": str(r["answer"]).strip(),
                "bbox": "",  # ViRL39K ships no boxes
                "natural": False,
                "_ref": ("virl", imgs[0]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------
class ChainedReader(io.RawIOBase):
    """Concatenate several shard files into one continuous readable stream."""

    def __init__(self, paths):
        self.paths, self.idx = list(paths), 0
        self.fh = open(self.paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.fh.readinto(b)
            if n:
                return n
            self.fh.close()
            self.idx += 1
            if self.idx >= len(self.paths):
                return 0
            self.fh = open(self.paths[self.idx], "rb")


def probe_archives(entries, n_show=8):
    """Print the first members of each archive so layouts can be confirmed up front.

    `entries` maps a label to the (paths, error) pair returned by resolve_cached().
    Errors are printed rather than swallowed -- the whole point is diagnosis.
    """
    for label, (paths, err) in entries.items():
        print(f"\n=== {label} ===")
        if err:
            print(f"  UNAVAILABLE: {err}")
            continue
        total = sum(Path(p).stat().st_size for p in paths)
        print(f"  {len(paths)} file(s), {total / 1e9:.2f} GB")

        first = str(paths[0])
        if first.endswith(".zip"):
            with zipfile.ZipFile(first) as z:
                names = z.namelist()
            print(f"  {len(names)} members. First {n_show}:")
            for n in names[:n_show]:
                print(f"    {n}")
            continue

        # Single .tar.gz, or a set of shards that concatenate into one gzip stream.
        mode = "r|gz" if len(paths) > 1 else "r|*"
        fobj = io.BufferedReader(ChainedReader(paths)) if len(paths) > 1 else None
        try:
            tf = tarfile.open(fileobj=fobj, mode=mode) if fobj else tarfile.open(first, mode=mode)
            with tf:
                shown = 0
                for m in tf:
                    if not m.isfile():
                        continue
                    print(f"    {m.name}    (basename: {os.path.basename(m.name)})")
                    shown += 1
                    if shown >= n_show:
                        break
        except Exception as e:
            print(f"  FAILED to read archive: {type(e).__name__}: {str(e)[:200]}")


def extract_from_tar_stream(part_paths, wanted, out_dir):
    """Stream a (possibly split) tar.gz once, writing out only the wanted basenames.

    Visual-CoT ships 13 shards that concatenate into a single gzip stream, so members
    cannot be seeked to individually -- but one sequential pass writing ~2 GB beats
    extracting the full 139 GB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = dict(wanted)  # basename -> destination filename
    found = {}

    stream = io.BufferedReader(ChainedReader(part_paths))
    with tarfile.open(fileobj=stream, mode="r|gz") as tf:
        for member in tf:
            if not remaining:
                break
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base not in remaining:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            dest = out_dir / remaining.pop(base)
            with open(dest, "wb") as out:
                out.write(fh.read())
            found[base] = dest
    return found, remaining


class ImageResolver:
    """Resolves each record's `_ref` to a PIL image."""

    def __init__(self, viscot_dir=None, treevgr_dir=None, virl_zip=None, aokvqa_ds=None):
        self.viscot_dir = viscot_dir
        self.treevgr_dir = treevgr_dir
        self.virl_zip = zipfile.ZipFile(virl_zip) if virl_zip else None
        self.aokvqa_ds = aokvqa_ds

    def get(self, ref):
        kind, key = ref
        if kind == "inline":
            return self.aokvqa_ds[key]["image"]
        if kind == "virl":
            with self.virl_zip.open(key) as fh:
                return Image.open(io.BytesIO(fh.read()))
        if kind == "viscot":
            return Image.open(self.viscot_dir / os.path.basename(key))
        if kind == "treevgr":
            return Image.open(self.treevgr_dir / os.path.basename(key))
        raise ValueError(f"unknown image ref kind: {kind}")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def gather_candidates(meta_dir, treevgr_parquet, virl_parquet):
    """Load every candidate pool, keyed by the recipe name that consumes it."""
    pools, aokvqa_ds = {}, None

    ambiguous = viscot_ambiguous_basenames(meta_dir)
    print(f"  dropping {len(ambiguous)} Visual-CoT basenames owned by >1 sub-dataset")
    for src in VISCOT_SOURCES:
        pools[src] = load_viscot(src, meta_dir, ambiguous)

    aok, aokvqa_ds = load_aokvqa()
    pools["aokvqa"] = aok
    pools["visdrone"] = load_visdrone(treevgr_parquet)
    for group in VIRL_CATEGORIES:
        pools[group] = load_virl(group, virl_parquet)

    return pools, aokvqa_ds


def draw(pools, recipe, rng_seed):
    """Sample each source to its recipe count. Returns {source: [records]}."""
    drawn, shortfalls = {}, []
    for source, n in recipe.items():
        pool = list(pools[source])
        # crc32, not hash(): Python randomizes string hashing per process, which would
        # make the draw differ between runs and break the set_a / set_b pairing.
        rng = random.Random(rng_seed + zlib.crc32(source.encode()) % 10_000)

        if source == "visual7w":
            # Prefer why/how, then top up from the rest so the count is still met.
            reasoning, other = rank_visual7w(pool)
            rng.shuffle(reasoning); rng.shuffle(other)
            pool = reasoning + other
        else:
            rng.shuffle(pool)

        if len(pool) < n:
            shortfalls.append((source, len(pool), n))
        drawn[source] = pool[:n]
    return drawn, shortfalls


def materialize(records, resolver, split_name):
    """Decode, resize and finalize rows into the output schema.

    Images are re-encoded to JPEG bytes rather than kept as PIL objects. Holding
    50K decoded 512px images costs ~18 GB of RSS (measured); as encoded bytes the
    same set is ~2.5 GB, which keeps the build inside a login-node memory budget.
    The Image feature accepts the {"bytes", "path"} form directly.
    """
    rows, failed = [], 0
    for r in records:
        try:
            img = resolver.get(r["_ref"])
            bbox = r["bbox"]
            if not bbox and r.get("_boxes_abs"):
                # VisDrone boxes are absolute and the parquet has no width/height,
                # so they can only be normalized once the image is open.
                bbox = union_bbox(r["_boxes_abs"], img.size[0], img.size[1])
            img = prepare_image(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            img.close()
        except Exception as e:  # a missing or corrupt image must not kill the build
            failed += 1
            if failed <= 5:
                print(f"  [warn] dropping {r['dataset']}#{r['question_id']}: "
                      f"{type(e).__name__}: {str(e)[:100]}")
            continue
        rows.append(
            {
                "dataset": r["dataset"],
                "split": split_name,
                "question_id": r["question_id"],
                "problem": r["problem"],
                "bbox": bbox,
                "solution": r["solution"],
                "image": {"bytes": buf.getvalue(), "path": None},
                "natural": r["natural"],
            }
        )
    return rows, failed


def summarize(name, rows):
    by_src = Counter(r["dataset"] for r in rows)
    nat = sum(1 for r in rows if r["natural"])
    boxed = sum(1 for r in rows if r["bbox"])
    print(f"\n=== {name}: {len(rows)} rows ===")
    for s, n in by_src.most_common():
        print(f"    {s:18s} {n:6d}")
    print(f"    {'-- natural':18s} {nat:6d}  ({100 * nat / max(1, len(rows)):.1f}%)")
    print(f"    {'-- with bbox':18s} {boxed:6d}  ({100 * boxed / max(1, len(rows)):.1f}%)")


# ---------------------------------------------------------------------------
def do_download(args):
    print("Downloading Visual-CoT metadata (small) ...")
    meta = hf_snapshot(VISCOT_REPO, ["metadata/*.jsonl"])
    print(f"  {meta}")

    print(f"Downloading ViRL39K ({VIRL_PARQUET} + images.zip, ~1.7 GB) ...")
    print(f"  {hf_snapshot(VIRL_REPO, [VIRL_PARQUET, 'images.zip'])}")

    print("Downloading TreeVGR-RL-37K (~6.1 GB) ...")
    print(f"  {hf_snapshot(TREEVGR_REPO, [TREEVGR_PARQUET, 'images.tar.gz'])}")

    print("Downloading A-OKVQA (~1.3 GB) ...")
    from datasets import load_dataset

    load_dataset(AOKVQA_REPO, split="train")

    if args.with_viscot_images:
        print("Downloading Visual-CoT images (~139 GB, 13 shards) ...")
        print(f"  {hf_snapshot(VISCOT_REPO, ['cot_images_tar_split/*'])}")
    else:
        print("\nSkipped Visual-CoT images (~139 GB). Re-run with --with-viscot-images "
              "when you have the disk; --build needs them.")


def do_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = hf_snapshot(VISCOT_REPO, ["metadata/*.jsonl"]) / "metadata"
    virl_root = hf_snapshot(VIRL_REPO, [VIRL_PARQUET, "images.zip"])
    treevgr_root = hf_snapshot(TREEVGR_REPO, [TREEVGR_PARQUET, "images.tar.gz"])

    print("Loading candidate pools ...")
    pools, aokvqa_ds = gather_candidates(
        meta_dir, treevgr_root / TREEVGR_PARQUET, virl_root / VIRL_PARQUET
    )
    for k, v in sorted(pools.items()):
        print(f"  {k:18s} {len(v):7d} verifiable candidates")

    # --- draw set_a, then derive set_b's natural half as a strict subset ---
    set_a_draw, short_a = draw(pools, RECIPE_NATURAL, SEED)
    set_b_natural = {
        src: recs[: int(round(len(recs) * NATURAL_KEEP))] for src, recs in set_a_draw.items()
    }
    set_b_nonnat, short_b = draw(pools, RECIPE_NONNATURAL, SEED + 1)

    for src, have, want in short_a + short_b:
        print(f"  [warn] {src}: only {have} candidates for a target of {want}")

    # --- resolve images ---
    viscot_cache = Path(args.viscot_cache or (out_dir / "_viscot_images"))
    treevgr_cache = Path(args.treevgr_cache or (out_dir / "_treevgr_images"))

    all_records = (
        [r for v in set_a_draw.values() for r in v]
        + [r for v in set_b_nonnat.values() for r in v]
    )

    viscot_wanted = {
        os.path.basename(r["_ref"][1]): os.path.basename(r["_ref"][1])
        for r in all_records
        if r["_ref"][0] == "viscot"
    }
    if viscot_wanted and not args.skip_viscot_extract:
        shards = sorted((hf_snapshot(VISCOT_REPO, ["cot_images_tar_split/*"])
                         / "cot_images_tar_split").glob("cot_images_*"))
        if not shards:
            raise SystemExit("Visual-CoT image shards absent; run --download --with-viscot-images")
        print(f"Streaming {len(shards)} Visual-CoT shards for {len(viscot_wanted)} images ...")
        _, missing = extract_from_tar_stream(shards, viscot_wanted, viscot_cache)
        if missing:
            print(f"  [warn] {len(missing)} images not found in the archive "
                  f"(e.g. {list(missing)[:3]}); those rows will be dropped")

    treevgr_wanted = {
        os.path.basename(r["_ref"][1]): os.path.basename(r["_ref"][1])
        for r in all_records
        if r["_ref"][0] == "treevgr"
    }
    if treevgr_wanted:
        tgz = treevgr_root / "images.tar.gz"
        print(f"Streaming TreeVGR images for {len(treevgr_wanted)} rows ...")
        _, missing = extract_from_tar_stream([tgz], treevgr_wanted, treevgr_cache)
        if missing:
            print(f"  [warn] {len(missing)} TreeVGR images not found; rows dropped")

    resolver = ImageResolver(
        viscot_dir=viscot_cache,
        treevgr_dir=treevgr_cache,
        virl_zip=virl_root / "images.zip",
        aokvqa_ds=aokvqa_ds,
    )

    # --- materialize and save ---
    from datasets import Dataset, Features, Image as HFImage, Value

    features = Features(
        {
            "dataset": Value("string"),
            "split": Value("string"),
            "question_id": Value("int64"),
            "problem": Value("string"),
            "bbox": Value("string"),
            "solution": Value("string"),
            "image": HFImage(),
            "natural": Value("bool"),
        }
    )

    for name, groups in (
        ("set_a", [set_a_draw]),
        ("set_b", [set_b_natural, set_b_nonnat]),
    ):
        records = [r for g in groups for v in g.values() for r in v]
        print(f"\nMaterializing {name} ({len(records)} records) ...")
        rows, failed = materialize(records, resolver, "train")
        if failed:
            print(f"  dropped {failed} rows with unreadable images")
        summarize(name, rows)
        ds = Dataset.from_list(rows, features=features)
        dest = out_dir / name
        ds.save_to_disk(str(dest))
        print(f"  saved -> {dest}")

    print("\nTrain with:")
    print(f"  --dataset_name {out_dir}/set_a --reward_variant ours")
    print(f"  --dataset_name {out_dir}/set_b --reward_variant ours")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="print archive layouts and exit")
    p.add_argument("--download", action="store_true", help="fetch missing source datasets")
    p.add_argument("--with-viscot-images", action="store_true",
                   help="also fetch the 139 GB Visual-CoT image shards")
    p.add_argument("--build", action="store_true", help="construct and save both sets")
    p.add_argument("--out-dir", default="cold_data/grpo_sets")
    p.add_argument("--viscot-cache", default=None)
    p.add_argument("--treevgr-cache", default=None)
    p.add_argument("--skip-viscot-extract", action="store_true",
                   help="assume --viscot-cache is already populated")
    args = p.parse_args()

    require_deps()

    if args.probe:
        probe_archives(
            {
                "ViRL39K images.zip": resolve_cached(VIRL_REPO, ["images.zip"], "images.zip"),
                "TreeVGR images.tar.gz": resolve_cached(
                    TREEVGR_REPO, ["images.tar.gz"], "images.tar.gz"
                ),
                "Visual-CoT image shards": resolve_cached(
                    VISCOT_REPO,
                    ["cot_images_tar_split/*"],
                    "cot_images_tar_split/cot_images_*",
                ),
            }
        )
        return
    if args.download:
        do_download(args)
    if args.build:
        do_build(args)
    if not (args.download or args.build):
        p.print_help()


if __name__ == "__main__":
    main()
