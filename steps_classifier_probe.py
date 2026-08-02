#!/usr/bin/env python
"""Does the FLAN-T5 step classifier label a sentence 'observe' because of quoting?

The overlap reward only scores sentences the classifier calls "observe", so if wrapping
text in quotes (or in a "Looking back at the image given: ..." retrieval frame) pushes
the label toward observe, the quoting behaviour the GRPO run developed is partly an
artifact of the classifier rather than of the attention metric.

Content is held fixed and only the framing varies, which is what the probe data cannot
do: there, quoted sentences are also *actually* scene descriptions, so the two are
confounded.

    python steps_classifier_probe.py --device cuda:0 [--from-probe <probe_merged.json>]
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OSTEPS = _load_module("_probe_overlap_steps", "trl/overlap_steps.py")

# Framings applied to the identical sentence. "bare" is the control.
FRAMINGS = {
    "bare":              lambda s: s,
    "quoted":            lambda s: f'"{s}"',
    "retrieval_quoted":  lambda s: f'Looking back at the image given: "{s}"',
    "retrieval_bare":    lambda s: f"Looking back at the image given, {s[0].lower() + s[1:]}",
    "reasoning_frame":   lambda s: f"So this means {s[0].lower() + s[1:]}",
}

FALLBACK = [
    # scene descriptions (expected observe under any framing)
    "The sky is blue with white clouds.",
    "In the background there are buildings and trees.",
    "The floor is tiled and light-coloured.",
    # deductions / plans (the interesting cases: does framing flip them?)
    "So the answer should be the man on the left.",
    "Therefore the most likely location is a kitchen.",
    "Let me check each option in turn before deciding.",
    "This rules out the banquet hall because it is too informal.",
    "The question is asking which animal is in the enclosure.",
    "I need to compare the two objects before answering.",
    "Since the sign is circular, it is probably a traffic sign.",
]


def sentences_from_probe(path: Path, per_label: int, seed: int):
    d = json.loads(Path(path).read_text())
    by = collections.defaultdict(list)
    for m in d["models"].values():
        for s in m["samples"]:
            for c in s["completions"]:
                for sn in c["all_sentences"]:
                    t = sn["text"].strip()
                    # skip anything already quoted: the framing must be ours alone
                    if 15 < len(t) < 160 and '"' not in t:
                        by[sn["label"]].append(t)
    rng = random.Random(seed)
    out = []
    for lab, v in sorted(by.items()):
        uniq = sorted(set(v))
        rng.shuffle(uniq)
        out += [(lab, t) for t in uniq[:per_label]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps-ckpt", default=str(REPO / "checkpoint/steps_classifier/best"))
    p.add_argument("--from-probe", default=None, help="probe_merged.json to draw real sentences from")
    p.add_argument("--per-label", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--question", default="What is the animal in the zoo?")
    args = p.parse_args()

    if args.from_probe:
        items = sentences_from_probe(Path(args.from_probe), args.per_label, args.seed)
    else:
        items = [("(fallback)", s) for s in FALLBACK]
    print(f"{len(items)} sentences, {len(FRAMINGS)} framings\n")

    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=args.device)
    # A neutral chain: the classifier also sees the surrounding chain, so hold it fixed
    # and vary only the step text. Caveat: in a real completion the quoting also changes
    # the chain, an effect this design deliberately excludes.
    chain = "The assistant is reasoning about the image to answer the question."

    labels = {}
    for fname, fn in FRAMINGS.items():
        labels[fname] = clf.predict_many([fn(t) for _, t in items], chain, args.question)

    print(f"{'framing':20s} " + "  ".join(f"{l:>8s}" for l in OSTEPS.LABELS) + "   observe%")
    for fname in FRAMINGS:
        c = collections.Counter(labels[fname])
        row = "  ".join(f"{c.get(l, 0):8d}" for l in OSTEPS.LABELS)
        print(f"{fname:20s} {row}   {100 * c.get('observe', 0) / len(items):6.1f}%")

    base = labels["bare"]
    print("\nflips relative to the bare sentence (same content, framing only):")
    for fname in FRAMINGS:
        if fname == "bare":
            continue
        to_obs = sum(1 for b, f in zip(base, labels[fname]) if b != "observe" and f == "observe")
        off_obs = sum(1 for b, f in zip(base, labels[fname]) if b == "observe" and f != "observe")
        print(f"  {fname:20s} ->observe {to_obs:4d}   observe-> {off_obs:4d}   net {to_obs - off_obs:+4d}")

    if args.from_probe:
        print("\nby the label the sentence had in its original completion:")
        for lab in sorted({l for l, _ in items}):
            idx = [i for i, (l, _) in enumerate(items) if l == lab]
            print(f"  original={lab:8s} n={len(idx):4d} " + "  ".join(
                f"{f}={100 * sum(labels[f][i] == 'observe' for i in idx) / len(idx):5.1f}%"
                for f in FRAMINGS))

    print("\nexamples that flipped to observe when quoted:")
    shown = 0
    for i, (lab, t) in enumerate(items):
        if base[i] != "observe" and labels["quoted"][i] == "observe" and shown < 8:
            print(f"  [{lab} -> observe] {t[:100]}")
            shown += 1
    if shown == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
