#!/usr/bin/env python
"""Build the donor bank for the mismatched-box control (--mismatch_bank).

The control scores a completion's observe steps against Grounding-DINO boxes that were
computed for a DIFFERENT question about a DIFFERENT picture. See
trl/rewards/mismatch_rewards.py for why, and docs/mismatch-boxes.md for the numbers. This
script produces the boxes, once, offline, so that training loads no DINO at all.

What a bank holds:

    index    every row key ("<dataset>/<question_id>") of every corpus given to
             --index-dataset, mapped to a hash of its encoded image bytes. The reward
             needs this to reject a donor that shares the row's PICTURE and not just its
             question -- 793 of saliency-r1-8k's 6714 images carry more than one
             question, up to 10.
    donors   --n-donors rows, each with --n-generations cold-start chains grouped by
             observe-step count. A chain is a list of per-step box lists, in relative
             coordinates, exactly as overlap_rewards._dino_boxes returns them. A step
             DINO could not ground is stored as an EMPTY list, so the control inherits
             the reference's skip rate instead of scoring every step (measured on the
             cold start: 3.4% of steps ungrounded, 0.5% of completions left with nothing).

Everything that decides what a chain IS matches the training path, because a bank built
off-distribution would make the control differ from its reference in a second way: the
same SYSTEM_PROMPT, the same 512px image cap, the same sampling parameters, the same
FLAN-T5 observe-step segmentation, the same format regex, the same --box_threshold. All
of it is reused from overlap_probe.py rather than reimplemented.

Three phases, because only the middle one wants a GPU:

    # 1. plan (CPU, ~2 min): hash every image, pick the donor rows
    python build_mismatch_bank.py --plan --out-dir outputs/mismatch_bank/8k \
        --dataset peterant330/saliency-r1-8k --n-donors 256

    # 2. generate + ground (GPU, one shard per card)
    CUDA_VISIBLE_DEVICES=0 python build_mismatch_bank.py --shard 0 --num-shards 8 \
        --out-dir outputs/mismatch_bank/8k --model <cold-start checkpoint>

    # 3. merge (CPU): one bank.json for --mismatch_bank
    python build_mismatch_bank.py --merge --out-dir outputs/mismatch_bank/8k

    # and then, any time
    python build_mismatch_bank.py --verify --out-dir outputs/mismatch_bank/8k

launch_mismatch_bank.sh runs all three.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    """Import a leaf module by path. Same trick, and the same reason, as
    overlap_probe._load_module: keep out of trl/__init__.py's heavy imports."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# overlap_probe owns the training-path reproduction (SYSTEM_PROMPT, prepare_image,
# load_model, generate, the think-span regexes, the DINO config). Importing it rather
# than copying those keeps this script correct when that one is fixed. It loads torch
# and a handful of leaf modules at import time; nothing touches a GPU until asked.
PROBE = _load_module("_bank_overlap_probe", "overlap_probe.py")
OSTEPS = PROBE.OSTEPS
OREW = PROBE.OREW

BANK_VERSION = 1
PLAN_NAME = "bank_plan.json"
BANK_NAME = "bank.json"


# ---------------------------------------------------------------------------
# phase 1: plan -- index every image, choose the donor rows
# ---------------------------------------------------------------------------
def _open_dataset(path: str, decode: bool = False):
    """Load a corpus by hub name or by save_to_disk directory. Always the WHOLE corpus.

    decode=False is what makes the index cheap: the encoded bytes are what gets hashed,
    so no image is ever rasterised for it. The generation phase passes decode=True.

    The trainer's holdout carve is NOT applied here -- see _carve. The index deliberately
    covers every row of the corpus, the 100 held-out ones included: the reward raises on a
    row it cannot look up, and an evaluation pass over the holdout sees those rows.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset, load_from_disk

    p = PROBE.repo_path(path) if not os.path.isabs(path) else Path(path)
    if os.path.isfile(os.path.join(p, "state.json")) or os.path.isfile(
        os.path.join(p, "dataset_dict.json")
    ):
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys"):
            ds = ds["train"]
    else:
        ds = load_dataset(path, split="train")
    return ds if decode else ds.cast_column("image", HFImage(decode=False))


def _carve(ds, split: str):
    """The trainer's 100-row seed-42 holdout, reproduced.

    `train` is the side the policy is optimised on, and donors are drawn from it so a
    donor chain comes from a picture the model saw exactly as often as the rows scored
    against it. ONE definition, called by both the planning and the generation phase,
    because the donor list is a list of INDICES into whatever this returns -- two
    spellings of the carve would mean the shards generate for different rows than the
    plan chose, silently.
    """
    if split == "all" or len(ds) <= 100:
        return ds
    parts = ds.train_test_split(test_size=100, seed=42)
    return parts["train" if split == "train" else "test"]


def image_group(rec) -> str:
    """A stable id for the PICTURE behind a row.

    A hash of the encoded bytes, so two rows carrying the same file land in the same
    group and the reward can reject a donor that shares the row's image. Corpora that
    reference images by path rather than embedding them fall back to the path, which is
    the same identity by another spelling.
    """
    b = rec.get("bytes")
    if b:
        return hashlib.blake2b(b, digest_size=8).hexdigest()
    return "path:" + str(rec.get("path"))


def build_index(paths, verbose=True) -> dict[str, str]:
    """{row key: image group} over every corpus in `paths`, holdout rows included."""
    index = {}
    for path in paths:
        ds = _open_dataset(path)
        cols = ds.column_names
        for need in ("dataset", "question_id", "image"):
            if need not in cols:
                raise SystemExit(
                    f"{path} has no '{need}' column (has {cols}). The mismatched-box "
                    "control identifies a row by (dataset, question_id) and its picture by "
                    "the image bytes; a corpus without those cannot be indexed."
                )
        n_before = len(index)
        for rec in ds.select_columns(["dataset", "question_id", "image"]):
            index[f"{rec['dataset']}/{rec['question_id']}"] = image_group(rec["image"])
        if verbose:
            print(f"  indexed {len(index) - n_before:6d} rows from {path}")
    return index


def choose_donors(dataset_path: str, split: str, index: dict, n_donors: int, seed: int):
    """Pick `n_donors` rows, all from DIFFERENT pictures.

    Distinct pictures because two donors sharing an image contribute nearly the same
    union field -- measured, the spread of the score handed out is 0.0036 between donor
    images against 0.0024 between chains of one image -- so a duplicate image buys about
    as much diversity as a second chain of a donor we already have, at the price of a
    whole donor slot.

    The trainer holds out 100 rows with seed 42 before training; drawing from the same
    `train` side keeps donors on the corpus the policy actually sees.
    """
    ds = _carve(_open_dataset(dataset_path), split)
    keys, groups = [], []
    for rec in ds.select_columns(["dataset", "question_id", "image"]):
        keys.append(f"{rec['dataset']}/{rec['question_id']}")
        groups.append(image_group(rec["image"]))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    picked, seen = [], set()
    for i in order:
        g = groups[i]
        if g in seen:
            continue
        seen.add(g)
        picked.append({"row_index": int(i), "key": keys[i], "image_group": g})
        if len(picked) == n_donors:
            break
    if len(picked) < n_donors:
        raise SystemExit(
            f"{dataset_path} has only {len(picked)} distinct images on split '{split}'; "
            f"asked for --n-donors {n_donors}"
        )
    # Sorted by row_index so a shard reads the corpus front to back.
    picked.sort(key=lambda d: d["row_index"])
    return picked


def phase_plan(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The donor corpus is always indexed, whatever else was asked for: the reward looks up
    # the ROW it is scoring, and every row of --dataset is one.
    index_paths = list(dict.fromkeys([args.dataset] + (args.index_dataset or [])))
    print(f"indexing {len(index_paths)} corpus/corpora ...")
    index = build_index(index_paths)
    print(f"  {len(index)} rows, {len(set(index.values()))} distinct images")
    print(f"choosing {args.n_donors} donor rows from {args.dataset} ...")
    donors = choose_donors(args.dataset, args.split, index, args.n_donors, args.seed)
    plan = {
        "version": BANK_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "index_datasets": index_paths,
        "seed": args.seed,
        "n_donors": args.n_donors,
        "n_generations": args.n_generations,
        "index": index,
        "donors": donors,
    }
    path = out / PLAN_NAME
    path.write_text(json.dumps(plan))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# phase 2: generate the chains and ground them
# ---------------------------------------------------------------------------
def observe_step_texts(output_text, clean_text, question, tok, clf):
    """The observe-step sentences of one batch of completions, the trainer's way.

    Returns a list (per completion) of step-text lists, or None for a completion whose
    format the reward would reject -- the trainer builds no observe-step maps for those,
    so they are not chains and must not enter the bank.

    Two decodings of the same completion, for the same reason overlap_probe keeps two:
    `output_text` keeps the special tokens and is the string the char offsets and the
    token map are built against; `clean_text` drops them and is what the format gate
    reads. Uses overlap_probe's own think-span regexes and OSTEPS.segment_observe_steps,
    so the step COUNT here means the same thing it will mean at reward time. That matters
    more than it looks: the count is what a completion is matched on.
    """
    out = tok(output_text)
    ts_idx = [re.search(r"<think>\s*(\S\S*)", t, re.DOTALL | re.MULTILINE) for t in output_text]
    te_idx = [re.search(r"(\S)\s*</think>", t, re.DOTALL | re.MULTILINE) for t in output_text]
    ts_idx = [m.start(1) if m else -1 for m in ts_idx]
    te_idx = [m.start(1) if m else -1 for m in te_idx]
    think_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(ts_idx)]
    think_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(te_idx)]

    result = []
    for c, text in enumerate(output_text):
        # The format gate the reward applies, on the same string the trainer decodes.
        valid = PROBE.judge_format(clean_text[c])
        # min(start, end): the same guard overlap_probe applies before segmenting, so a
        # completion whose </think> tokenises ahead of its <think> yields no step rather
        # than an inverted span.
        te = think_end[c] if valid and think_end[c] else 0
        ts = min(think_start[c], te) if valid and think_start[c] is not None else 0
        if not valid or ts_idx[c] < 0 or te_idx[c] < 0 or te <= ts:
            result.append(None)
            continue
        steps = OSTEPS.segment_observe_steps(
            text, ts_idx[c], te_idx[c], out, c, ts, te, question, clf
        )
        result.append([s for s, _, _ in steps])
    return result


def phase_shard(args):
    import torch

    out = Path(args.out_dir)
    plan = json.loads((out / PLAN_NAME).read_text())
    donors = plan["donors"]
    mine = [d for i, d in enumerate(donors) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(donors)} donor rows")

    OREW.configure(box_threshold=args.box_threshold, dino_batch_size=args.dino_batch_size,
                   dino_device=args.dino_device)

    device = args.device
    processor, model = PROBE.load_model(args.model, None, device, args.attn_impl)
    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=args.steps_device)
    tok = processor.tokenizer

    # The rows themselves, decoded this time -- the plan holds only their indices, into
    # exactly this carve. Same two functions the plan used, so the indices cannot drift.
    ds = _carve(_open_dataset(plan["dataset"], decode=True), plan["split"])
    # The plan is also where the chain count comes from. Passing --n-generations here as
    # well would let a shard silently disagree with the meta the merge writes.
    n_generations = int(plan["n_generations"])
    if d0 := [d["key"] for d in mine][:1]:
        print(f"  first donor {d0[0]}, {n_generations} chains each")

    written = []
    for di, d in enumerate(mine):
        row = ds[d["row_index"]]
        assert f"{row['dataset']}/{row['question_id']}" == d["key"], (
            f"donor {d['key']} is at index {d['row_index']} in the plan but that index now "
            f"holds {row['dataset']}/{row['question_id']} -- the corpus or the carve moved"
        )
        image = PROBE.prepare_image(row["image"])
        question = row["problem"]

        # Generate in chunks: 64 sequences x 1024 new tokens in one call is a large KV
        # cache next to an 8B model, and the chunk size is the only knob that keeps this
        # on one card. The chunks are independent samples, so this is not a different
        # distribution from one big call.
        texts, clean = [], []
        left = n_generations
        while left > 0:
            k = min(args.gen_batch, left)
            _inputs, prompt_len, seqs = PROBE.generate(
                processor, model, image, question, k,
                args.max_new_tokens, args.temperature, device,
            )
            ids = [s[0] for s in seqs]
            texts.extend(tok.batch_decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False))
            clean.extend(tok.batch_decode(ids, skip_special_tokens=True))
            left -= k
            del _inputs
        torch.cuda.empty_cache()

        per_completion = observe_step_texts(texts, clean, question, tok, clf)

        # One batched DINO call over every (this image, step text) of this donor row --
        # the same flattening think_overlap_reward does over a training batch.
        flat_texts, owner = [], []
        for c, steps in enumerate(per_completion):
            if not steps:
                continue
            for si, s in enumerate(steps):
                flat_texts.append(s)
                owner.append((c, si))
        boxes = OREW._dino_boxes([image] * len(flat_texts), flat_texts) if flat_texts else []

        grounded = defaultdict(dict)
        for (c, si), bx in zip(owner, boxes):
            grounded[c][si] = [[round(float(v), 5) for v in b] for b in (bx or [])]

        chains = defaultdict(list)
        for c, steps in enumerate(per_completion):
            if not steps:
                continue  # format-invalid or no observe step: not a chain
            chain = [grounded[c].get(si, []) for si in range(len(steps))]
            chains[len(steps)].append(chain)
        # Cap the variants kept per length. Beyond a handful they add file size rather
        # than diversity: the reward picks ONE per (training row, length), and chains of
        # one donor row differ by 0.0024 against the 0.0115 the whole reward spans.
        kept = {str(L): v[: args.max_per_length] for L, v in sorted(chains.items())}
        written.append({"key": d["key"], "image_group": d["image_group"], "chains": kept})
        n_ch = sum(len(v) for v in kept.values())
        print(f"  [{di + 1}/{len(mine)}] {d['key']:>28s}  {n_ch:3d} chains  "
              f"lengths {sorted(int(k) for k in kept)}", flush=True)

    # The shard records the config it ACTUALLY generated and grounded under, and the merge
    # copies that into the bank's meta rather than re-reading its own flags. Otherwise a
    # merge invoked with a different --model or --box-threshold than the shards ran with
    # would write a meta that describes a bank nobody built -- and box_threshold in
    # particular is load-bearing: the reward refuses a run whose threshold differs from it.
    cfg = {"model": args.model, "steps_ckpt": args.steps_ckpt,
           "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
           "box_threshold": args.box_threshold, "max_per_length": args.max_per_length,
           "n_generations": n_generations}
    path = out / f"bank_shard{args.shard:02d}.json"
    path.write_text(json.dumps({"config": cfg, "donors": written}))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# phase 3: merge
# ---------------------------------------------------------------------------
def phase_merge(args):
    out = Path(args.out_dir)
    plan = json.loads((out / PLAN_NAME).read_text())
    shards = sorted(out.glob("bank_shard*.json"))
    if not shards:
        raise SystemExit(f"no bank_shard*.json in {out}; run --shard first")
    donors, cfgs = [], []
    for s in shards:
        blob = json.loads(s.read_text())
        donors.extend(blob["donors"])
        cfgs.append((s.name, blob.get("config", {})))
    # Every shard must have run the same model, sampling and threshold, or the bank is two
    # different experiments in one file.
    base_name, base = cfgs[0]
    for name, c in cfgs[1:]:
        if c != base:
            raise SystemExit(
                f"{name} was generated with a different config than {base_name}:\n"
                f"  {base_name}: {base}\n  {name}: {c}\n"
                "Re-run the shards, or delete the odd one out and re-run just it."
            )
    seen = set()
    for d in donors:
        if d["key"] in seen:
            raise SystemExit(f"donor {d['key']} appears in two shards; re-run the shards")
        seen.add(d["key"])
    missing = [d["key"] for d in plan["donors"] if d["key"] not in seen]
    if missing:
        print(f"WARNING: {len(missing)} planned donor rows are in no shard "
              f"(first few: {missing[:5]})")
    donors = [d for d in donors if d["chains"]]
    bank = {
        "meta": {
            "version": BANK_VERSION,
            "dataset": plan["dataset"],
            "split": plan["split"],
            "index_datasets": plan["index_datasets"],
            "seed": plan["seed"],
            "n_donors": len(donors),
            **base,          # the config the shards actually ran under, not this call's
        },
        "index": plan["index"],
        "donors": donors,
    }
    path = out / BANK_NAME
    path.write_text(json.dumps(bank))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB), {len(donors)} donor rows")
    _report(bank)


# ---------------------------------------------------------------------------
# --verify
# ---------------------------------------------------------------------------
def _report(bank):
    donors = bank["donors"]
    lens = Counter()
    per_donor_lengths = []
    n_steps = n_ungrounded = 0
    for d in donors:
        ls = sorted(int(k) for k in d["chains"])
        per_donor_lengths.append(set(ls))
        for L, variants in d["chains"].items():
            lens[int(L)] += len(variants)
            for chain in variants:
                for step in chain:
                    n_steps += 1
                    n_ungrounded += 1 if not step else 0
    total = sum(lens.values())
    print(f"\ndonor rows            {len(donors)}")
    print(f"chains                {total}   ({total / max(1, len(donors)):.1f} per donor)")
    print(f"steps                 {n_steps}   ungrounded {n_ungrounded} "
          f"({100 * n_ungrounded / max(1, n_steps):.1f}%; the cold start's own rate is 3.4%)")
    print("\nchains by observe-step count, and the share of donor rows that can serve it")
    print("  n   chains   donors covering n")
    for n in sorted(lens):
        cov = sum(1 for s in per_donor_lengths if n in s)
        print(f"{n:3d}  {lens[n]:7d}   {100 * cov / max(1, len(donors)):5.1f}%")
    # The rate that matters at reward time: a completion whose step count its donor row
    # cannot match is served that donor's nearest length instead (0.21x the reward's
    # within-group spread, against 1.02x for switching donor). This says how often that
    # is, at the cold start's own mix of observe-step counts -- 880 chains from the
    # val_natural and grad_spread probes, conditioned on having at least one observe step
    # (3.1% have none and are never scored by any of these rewards). It is a floor on the
    # drift, not a forecast: a training run walks up the length distribution, and the tail
    # past the bank's longest chain is unservable at any bank size.
    cold = {1: .108, 2: .217, 3: .235, 4: .174, 5: .105, 6: .063, 7: .031, 8: .015,
            9: .007, 10: .005, 11: .005, 12: .003, 13: .001, 14: .002}
    mass = sum(cold.values())
    exact = sum(p * (sum(1 for s in per_donor_lengths if n in s) / max(1, len(donors)))
                for n, p in cold.items()) / mass
    print(f"\nexpected exact-length service at the cold-start length mix: {100 * exact:.1f}%"
          f"\n(the rest takes the nearest length the SAME donor row has -- see "
          f"docs/mismatch-boxes.md)")


def phase_verify(args):
    out = Path(args.out_dir)
    bank = json.loads((out / BANK_NAME).read_text())
    print(f"{out / BANK_NAME}\n")
    for k, v in bank["meta"].items():
        print(f"  {k:18s} {v}")
    idx = bank["index"]
    print(f"\nindex                 {len(idx)} rows, {len(set(idx.values()))} distinct images")

    donor_keys = {d["key"] for d in bank["donors"]}
    donor_groups = {d["image_group"] for d in bank["donors"]}
    ok = True
    if len(donor_groups) != len(bank["donors"]):
        print("  FAIL two donor rows share an image")
        ok = False
    missing = [k for k in donor_keys if k not in idx]
    if missing:
        print(f"  FAIL {len(missing)} donor rows are not in the index")
        ok = False
    empty = [d["key"] for d in bank["donors"] if not d["chains"]]
    if empty:
        print(f"  FAIL {len(empty)} donor rows hold no chain")
        ok = False

    # Every indexed row must actually be servable: walk the real donor resolution for a
    # sample of them and confirm it terminates on a donor with a different question AND a
    # different picture. This is the property the control is named for, so it is checked
    # rather than assumed.
    # Loaded under its DOTTED name so its `from . import overlap_rewards` resolves to the
    # copy overlap_probe already registered -- a second copy would carry its own _CFG and
    # the check would not be against the boxes the reward will actually rasterise.
    MM = _load_module("trl.rewards.mismatch_rewards", "trl/rewards/mismatch_rewards.py")
    MM._CFG["bank"] = str(out / BANK_NAME)
    MM._CFG["seed"] = args.mismatch_seed
    rng = np.random.default_rng(0)
    keys = list(idx)
    sample = [keys[i] for i in rng.choice(len(keys), size=min(2000, len(keys)), replace=False)]
    bad = 0
    for k in sample:
        d, h = MM.donor_for(k)
        if d["key"] == k or d["image_group"] == idx[k]:
            bad += 1
    if bad:
        print(f"  FAIL {bad}/{len(sample)} rows resolve to a donor sharing their question or image")
        ok = False
    else:
        print(f"  ok   {len(sample)} sampled rows all resolve to a different question and picture")

    # And that every step count the cold start can produce is servable, including the
    # ones no donor row holds -- the nearest-length ladder must never raise.
    for n in list(range(1, 20)) + [40, 85, 200]:
        for k in sample[:200]:
            d, h = MM.donor_for(k)
            chain, L = MM.chain_for(d, n, h)
            assert len(chain) == L and L >= 1
    print(f"  ok   step counts 1..19, 40, 85, 200 all resolve to a chain")

    _report(bank)
    print("\n" + ("VERIFY OK" if ok else "VERIFY FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="phase 1: index images, pick donors")
    ap.add_argument("--merge", action="store_true", help="phase 3: shards -> bank.json")
    ap.add_argument("--verify", action="store_true", help="check and describe an existing bank")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--dataset", default="peterant330/saliency-r1-8k",
                    help="corpus the donor rows come from")
    ap.add_argument("--index-dataset", action="append",
                    help="corpus to index (repeatable). Every corpus the reward will "
                         "see must be here, or it raises on the first unindexed row. "
                         "Defaults to --dataset.")
    ap.add_argument("--split", default="train", choices=["train", "holdout", "all"],
                    help="'train' reproduces the trainer's 100-row seed-42 carve-out")
    ap.add_argument("--n-donors", type=int, default=256)
    ap.add_argument("--n-generations", type=int, default=64,
                    help="chains per donor row; sets how many observe-step counts it covers")
    ap.add_argument("--max-per-length", type=int, default=8,
                    help="cap on stored chains per (donor, length)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="chooses WHICH rows become donors (bank-build time)")
    ap.add_argument("--mismatch-seed", type=int, default=0,
                    help="--verify only: the --mismatch_seed a run will use, i.e. which "
                         "donor each row is paired with. Separate from --seed because the "
                         "two are set at different times by different people.")

    ap.add_argument("--shard", type=int, default=None, help="phase 2: this shard's index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=str(REPO / "checkpoint" /
                                           "coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"))
    ap.add_argument("--steps-ckpt", default=str(REPO / "checkpoint" / "steps_classifier" / "best"))
    ap.add_argument("--steps-device", default="cuda:0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch", type=int, default=8,
                    help="sequences per generate() call; the KV-cache knob")
    ap.add_argument("--box-threshold", type=float, default=0.10,
                    help="DINO confidence floor. Baked into the bank: the reward refuses "
                         "a bank whose threshold differs from the run's.")
    ap.add_argument("--dino-batch-size", type=int, default=16)
    ap.add_argument("--dino-device", default=None)
    args = ap.parse_args()

    if args.plan:
        return phase_plan(args)
    if args.merge:
        return phase_merge(args)
    if args.verify:
        return phase_verify(args)
    if args.shard is None:
        raise SystemExit("pass one of --plan / --shard N / --merge / --verify")
    return phase_shard(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
