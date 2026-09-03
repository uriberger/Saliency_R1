# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MISMATCHED-BOX control (--mismatch_bank): the overlap reward with real DINO unions
that were computed for a DIFFERENT question about a DIFFERENT picture.

WHY THIS EXISTS. `think_overlap_reward` runs Grounding-DINO on the text of each observe
step and rewards the policy for attending inside the boxes that come back. Everything
that reward can be is downstream of one assumption: that running DINO on THAT SENTENCE
matters. The offline evidence says it may not --

    step_box_similarity.py (docs/step-box-similarity.md)
        two steps of one chain get masks no more alike than two steps of two DIFFERENT
        chains about the same picture (closeness 0.72 vs 0.70), and a step's map scores
        higher on its OWN mask than on another step's only 52.6% of the time.
    maskfree_rewards.py
        the best single predictor of mean_in is mean(m)/max(m), which never sees a box.

-- but both are static: they re-score maps that a DINO-trained policy produced. They
cannot say what a policy TRAINED against wrong boxes would do. This reward is that run.

It is not a placebo. The boxes are real Grounding-DINO output on a real photograph for a
real sentence written by the cold-start model; only the pairing is wrong. So it sits
between `--placebo roll` (the step's own union, moved) and `--maskfree` (no union at
all): the union keeps the size, shape and object-like structure DINO actually produces,
and loses only its relationship to this image and this sentence.

No Grounding-DINO is loaded. The boxes come from a bank built offline by
build_mismatch_bank.py, so the reward is a dict lookup plus the same rasterisation and
the same metric as the real one. That removes 16.6 s from a measured 40.5 s optimizer
step (see placebo_rewards' docstring for where that number is from).

WHICH DONOR, AND WHY IT IS NOT PER COMPLETION. This is the decision the whole experiment
turns on. Measured on 538 cold-start chains from the val_natural probes, as the spread
of the completion-level score ACROSS THE 8 ROLLOUTS OF ONE PROMPT -- which is the only
thing GRPO sees, since the group mean is subtracted before anything else:

    real reward (each completion on its own boxes)              0.0115
    this reward, all 8 rollouts sharing ONE donor chain         0.0094
    changing only which donor chain a completion drew           0.0117

The obvious implementation -- every completion looks up a chain with its own observe-step
count -- puts the third row inside the group and roughly 60% of the reward's within-group
variance becomes which donor a rollout happened to draw. That is `--placebo random` with
a box-shaped distribution, and it is already a run. So the donor is fixed PER PROMPT ROW:
all rollouts of a row are scored against the same donor row, and the surviving 0.0094 is
0.82x the real reward's tie-breaking strength with none of it donor noise.

The donor row is chosen by hashing the row's identity, not by drawing at random, so it is
the same in every epoch, on every rank and across a restart. Identity is `overlap_rewards.qbox_key`,
the (dataset, split, question_id) triple --  the same function --overlap_question_boxes
keys its own offline box cache by, not a second spelling of it. The trainer forwards every
dataset column to the reward, and the triple is unique in every corpus this trainer
accepts (`problem` is not -- 6844 distinct texts for saliency-r1-8k's 8080 rows, one
question repeated 41 times).

DIFFERENT QUESTION **AND** DIFFERENT PICTURE. 793 of the 6714 images in saliency-r1-8k
carry more than one question (up to 10), so excluding the row itself is not enough. The
bank ships an `index` mapping every row key to a hash of its encoded image bytes, and a
donor is rejected if it shares either the key or the image hash. A row the index does not
cover raises rather than silently falling back to a question-only exclusion -- rebuild the
bank with that corpus in --index-dataset.

MATCHING THE OBSERVE-STEP COUNT, AND WHAT HAPPENS WHEN IT CANNOT BE MATCHED. A completion
with n observe steps takes the donor row's chain that also has n, step i against donor
step i. When the donor row has no chain of length n, the nearest length it does have is
used, wrapped when shorter (step i takes donor step i % L) and cut short when longer.
The completion is never left unscored.

That ladder is not a compromise, it is the measurement:

    a wrong-length chain from the SAME donor row costs         0.0024   (0.21x)
    hopping to another donor row to find length n costs        0.0117   (1.02x)

(x = the real reward's 0.0115 within-group spread; the first number is the spread across
chains of one donor row, the second across donor rows, out of a total donor spread of
0.0051 = 0.0036 between rows + 0.0024 within one.) Chasing an exact length across donors
costs five times what accepting the length mismatch costs, which is why a thin or empty
length pool is never consulted: the donor row is picked first and the length second.

The third option -- leave the completion unscored when no chain of length n exists -- is
the one that must not be taken. The policy CHOOSES how many observe steps it writes, so a
step count that is never scored is a free exit from the reward, and it would be found.
Compare the ordinary escape (write ungroundable text), which the real reward has too.

The length ladder also disposes of the tail without a special case. Cold-start chains run
to at most 14 observe steps (880 chains, median 3, 96% at 7 or fewer), while the trained
checkpoints in the same probes reach 70, 83, 85, 87. No bank built from the cold-start
model can hold a chain that long, however large it is, so "no chain of this length" is
guaranteed to happen and to get more common as a run drifts. n=85 is the donor's longest
chain, wrapped, at the same 0.0024.

POSITIONAL, NOT BY TEXT. Step i takes donor step i (mod L). Assigning by a hash of the
step's text would give a repeated sentence a repeated box, which is the property that
makes the duplicate-step hack work on the real reward. Keeping it would be re-attaching
the text to the boxes, i.e. exactly the dependence this control exists to sever.

The consequence is worth stating, because it is not a defect but it does constrain the
reading: this control CANNOT be hacked the way its reference can. Repeating a
trivially-groundable sentence and describing the background both stop working when the
boxes do not respond to the text. So a divergence between the two runs is either "the
sentence mattered" or "the reference was hacking", and the diagnostics that separate them
(the duplicate-sentence fraction and the union area, already logged) are what settle it.

WHAT IS HELD EQUAL TO THE REFERENCE, so only the pairing differs:

    the metric      --overlap_metric, the mass floor, --box_threshold, --max_box_area and
                    --max_union_area are all read from overlap_rewards' own config, and
                    the union is built by its `_union_mask` and scored by its
                    `_step_score`. One switch, not two that have to agree. configure()
                    refuses a bank whose box_threshold differs from the run's, since that
                    filter was applied when the bank was built and cannot be re-applied.
    the natural gate --overlap_natural_only is read from the same place.
    the skipped set  a step whose donor boxes do not form a usable union is SKIPPED, not
                    scored 0, and a completion with no usable step returns None (masked,
                    neutral in the advantage) -- the same three exits the real reward
                    takes. The bank stores an EMPTY box list for a donor step DINO could
                    not ground, so the control inherits the reference's skip rate rather
                    than scoring everything: measured on the cold start, DINO leaves 3.4%
                    of steps ungrounded and 0.5% of completions with at least one observe
                    step entirely unscored.

The one thing that is NOT equal, and cannot be without loading DINO, is WHICH completions
those skips land on: here they follow the donor, there they follow the completion's own
text. `placebo_rewards` closes that gap by running the real pipeline as a gate; this
reward cannot, because not running DINO is the point. The 0.5% above is the size of it.

w_mismatch is applied by the trainer via --reward_weights. The control's within-group
spread is 0.82x the reference's, so holding tie-breaking strength constant (the placebo
convention) wants ~1.22x its weight -- re-measure on a probe run rather than assuming.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import overlap_rewards as _ORW

_CFG = {
    # Path to the bank written by build_mismatch_bank.py. None = disabled, and
    # think_mismatch_reward is then never installed by the launcher.
    "bank": None,
    # Chooses the donor row and the chain within it. Two runs differing only in it are
    # independent replicates of the same control over a different random pairing.
    "seed": 0,
}

# Lazily-loaded bank (one per training process). Loading is deferred so that importing
# this module -- which grpo_trainer_qwen3.py does unconditionally, for is_active() --
# costs nothing on a run that is not using it.
_BANK = {"donors": None, "index": None, "meta": None, "path": None}
_DONOR_CACHE: dict[str, tuple] = {}

# FIXED key set, always all of it, for the reason grad_rewards.DIAG_KEYS documents: the
# trainer gathers these across ranks, so a key set that depended on what a rank happened
# to see would mean a rank-dependent number of collectives, which hangs rather than fails.
#
# `exact_len_frac` is the one to watch. It is the share of scored completions whose
# observe-step count the donor row could match exactly, and it falls as the policy's
# chains outgrow the cold-start lengths the bank was built from. It is a description of
# how far the run has drifted, not a failure: a wrong-length chain from the right donor
# costs 0.21x the reward's within-group spread (see the module docstring).
DIAG_KEYS = ("exact_len_frac", "len_delta", "wrap_frac", "union_frac", "step_skip_frac")
_DIAG: dict[str, list[float]] = {}


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each mismatch diagnostic since the last call, then clear.

    Always all of DIAG_KEYS; NaN for a key nothing was recorded under.
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


def is_active() -> bool:
    """True when the mismatched-box control replaces the overlap reward. Rank-uniform (it
    is set from the CLI on every process), so the trainer may branch its logging
    collectives on it."""
    return _CFG["bank"] is not None


def configure(**kwargs):
    """Set the control's config from the CLI flags. None values are ignored.

    Call AFTER overlap_rewards.configure(): the box_threshold check below reads the
    resolved value out of that module, so there is one switch rather than two that have
    to agree. Loads the bank eagerly, so a missing or incompatible file fails at startup
    rather than at the first reward call of step 0 on eight ranks at once.
    """
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    if _CFG["bank"] is None:
        return
    meta = _load_bank()["meta"]
    # box_threshold was applied by DINO when the bank was built and is not recoverable
    # from the stored boxes: a bank built at 0.10 cannot be re-filtered to 0.25, and
    # scoring it as if it had been is a silently different reward. (max_box_area,
    # max_union_area and the metric ARE applied at scoring time, from this run's config,
    # so those are free to differ and are deliberately not checked here.)
    want = float(_ORW._CFG["box_threshold"])
    got = float(meta.get("box_threshold", float("nan")))
    if not np.isclose(want, got):
        raise ValueError(
            f"--mismatch_bank {_CFG['bank']} was built with --box_threshold {got}, but this "
            f"run configures {want}. The threshold was applied by Grounding-DINO when the "
            f"bank was written and cannot be re-applied to the stored boxes. Rebuild the "
            f"bank at {want}, or run at {got}."
        )


def _load_bank():
    """Read the bank once per process. -> {"donors", "index", "meta"}."""
    if _BANK["donors"] is not None:
        return _BANK
    path = _CFG["bank"]
    if path is None:
        raise ValueError("mismatch_rewards used with no --mismatch_bank configured")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--mismatch_bank {path} does not exist. Build one with:\n"
            f"    python build_mismatch_bank.py --out {path} ..."
        )
    with open(path) as fh:
        raw = json.load(fh)
    donors = []
    for d in raw["donors"]:
        # JSON object keys are strings; the observe-step count is an int everywhere else.
        donors.append({
            "key": d["key"],
            "image_group": d["image_group"],
            "chains": {int(k): v for k, v in d["chains"].items()},
        })
    if not donors:
        raise ValueError(f"--mismatch_bank {path} holds no donor rows")
    _BANK.update(donors=donors, index=raw["index"], meta=raw["meta"], path=path)
    return _BANK


def _blake_u64(*parts) -> int:
    """Stable 64-bit digest of the parts. Stable across processes, ranks and restarts,
    unlike Python's own hash(), which is salted per interpreter."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def row_key(dataset, split, question_id) -> str:
    """The row identity the bank is keyed by.

    Deliberately `overlap_rewards.qbox_key` itself and not a second spelling of it:
    --overlap_question_boxes keys its offline box cache the same way, and two offline-box
    features that disagreed about what a row IS would be a trap nobody would find. The
    triple (dataset, split, question_id) is unique in every corpus this trainer accepts
    and `problem` is deliberately not part of it -- questions repeat across images (35343
    distinct strings over set_a's 50000 rows), so keying on the text would collapse
    different pictures onto one entry.
    """
    return _ORW.qbox_key(dataset, split, question_id)


def donor_for(key: str):
    """The donor row and chain-selection hash for one training row. -> (donor, h).

    Deterministic in (seed, key), so every rollout of the row -- in every epoch, on every
    rank, after any restart -- is scored against the same donor. Scans forward from a
    hashed start until a donor passes the exclusion, which is what makes the result
    independent of how many donors happen to be excluded.
    """
    hit = _DONOR_CACHE.get(key)
    if hit is not None:
        return hit
    bank = _load_bank()
    donors, index = bank["donors"], bank["index"]
    if key not in index:
        raise KeyError(
            f"row {key!r} is not in --mismatch_bank {bank['path']}'s index, so this reward "
            f"cannot tell which donors share its image -- and 793 of saliency-r1-8k's 6714 "
            f"images carry more than one question, so excluding the row itself is not "
            f"enough. Rebuild the bank with this corpus in --index-dataset."
        )
    group = index[key]
    n = len(donors)
    start = _blake_u64("mismatch-donor", _CFG["seed"], key) % n
    for j in range(n):
        d = donors[(start + j) % n]
        if d["key"] == key or d["image_group"] == group:
            continue  # same question, or another question about the same picture
        out = (d, _blake_u64("mismatch-chain", _CFG["seed"], key))
        _DONOR_CACHE[key] = out
        return out
    raise RuntimeError(
        f"every one of the {n} donor rows in {bank['path']} is excluded for row {key!r} "
        f"(same question or same image). Build the bank with more --n-donors."
    )


def chain_for(donor, n_steps: int, h: int):
    """The donor chain a completion with `n_steps` observe steps is scored against.

    -> (chain, L) where chain is a list of L per-step box lists. Exact length when the
    donor row has one; otherwise the nearest length it does have, ties going to the
    LONGER chain (which needs no wrap, so more of the completion sees a distinct union).
    """
    chains = donor["chains"]
    L = n_steps if n_steps in chains else min(chains, key=lambda l: (abs(l - n_steps), -l))
    variants = chains[L]
    return variants[h % len(variants)], L


def think_mismatch_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None,
    dataset=None, split=None, question_id=None, **kwargs
):
    """Per-completion mismatched-box reward. See module docstring.

    Structurally identical to `think_overlap_reward` -- same natural-only gate, same
    `_union_mask`, same `_step_score`, same mean over the completion's scored steps, same
    multiplicative format gate, same None for a completion with nothing to score -- with
    exactly one substitution: where the boxes come from.

    `image` is accepted and never read. That is the control: the picture no longer takes
    part in choosing the region the policy is rewarded for attending to.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    missing = [n for n, v in zip(_ORW.QBOX_KEY_COLUMNS, (dataset, split, question_id))
               if v is None]
    if missing:
        raise KeyError(
            f"--mismatch_bank needs the {', '.join(repr(m) for m in missing)} column(s) to "
            "identify the row (the trainer forwards every dataset column to the reward), "
            "but they did not arrive. Every corpus this repo trains on carries all of "
            f"{_ORW.QBOX_KEY_COLUMNS} -- peterant330/saliency-r1-8k and everything "
            "build_grpo_sets.py writes."
        )

    # --overlap_natural_only, read from the overlap reward's config so the flag means the
    # same thing here as there and there is only one switch to set.
    if _ORW._CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "--overlap_natural_only requires a boolean 'natural' column in the dataset, "
                "but none reached the reward function. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    rewards = []
    for c in range(n):
        steps = saliency_map[c]
        if not scored[c]:
            rewards.append(None)  # non-natural under --overlap_natural_only -> mask
            continue
        if not steps:
            rewards.append(None)  # no observe step at all -> mask (neutral)
            continue

        donor, h = donor_for(row_key(dataset[c], split[c], question_id[c]))
        chain, L = chain_for(donor, len(steps), h)

        vals = []
        for si, st in enumerate(steps):
            step_map = st["map"]
            gh, gw = step_map.shape
            # i % L: wraps when the donor chain is shorter than the completion, and is a
            # no-op when it is longer (the loop stops at len(steps) either way).
            mask = _ORW._union_mask(chain[si % L], gh, gw)
            if mask is None:
                # The donor step's boxes do not form a usable union -- DINO grounded
                # nothing when the bank was built, or every box lost the area cap, or the
                # union is degenerate on this grid, or --max_union_area rejected it.
                # SKIPPED, not scored 0, exactly as in think_overlap_reward.
                _diag("step_skip_frac", 1.0)
                continue
            _diag("step_skip_frac", 0.0)
            _diag("union_frac", float(mask.mean()))
            s = _ORW._step_score(step_map, mask)
            if s is not None:
                vals.append(s)

        if not vals:
            rewards.append(None)  # zero usable observe steps -> mask (neutral)
            continue
        # Recorded per SCORED completion, so the rates read as "of the completions this
        # reward actually graded", which is the population the reward's level is over.
        _diag("exact_len_frac", 1.0 if L == len(steps) else 0.0)
        _diag("len_delta", float(L - len(steps)))
        _diag("wrap_frac", 1.0 if L < len(steps) else 0.0)
        # The format gate is multiplicative and kept identical to the overlap reward's.
        rewards.append(float(np.mean(vals)) * (1.0 if valid_list[c] else 0.0))
    return rewards
