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

"""Three PLACEBO rewards (--placebo roll|random|length): the overlap reward's
tie-breaking strength, with the grounding taken out. See docs/next-reward-experiments.md.

WHY THESE EXIST. Measured 2026-08-18/19, the attention-overlap reward does not identify
the correct completion within a group (r with accuracy_reward is -0.019 +- 0.051 at
mean_in w0.4), and it does not move the training objective -- yet it takes
`train/frac_reward_zero_std` from 0.547 (accuracy only) to 0.000, and the benchmark
does move. "It keeps the gradient alive" explains why something happens; it does not
explain WHICH DIRECTION the policy is pushed. A reward that broke ties by rewarding the
SHORTEST completion would also take frac_reward_zero_std to zero. These three rewards
are the controls that separate those explanations:

    roll     the configured metric on the SAME map, scored against the step's own box
             union MOVED to a wrong place. Same area, same shape, same everything except
             location. Isolates "is it grounding, or any same-shaped signal?"
    random   a deterministic hash of the completion text -> U(0, 1). Pure within-group
             variance, no direction at all.
    length   monotone decreasing in the completion's token count. Isolates "is the
             overlap reward a brevity reward in disguise?" -- the largest thing it is
             associated with within a group is brevity (r -0.042 / -0.105 / -0.035
             across the three trained runs, negative in every training tercile).

Each REPLACES `think_overlap_reward` in the same reward_funcs slot, so --reward_weights
lines up unchanged, and each is weighted to match mean_in w0.4's WITHIN-GROUP standard
deviation, so tie-breaking strength is held constant and only direction varies. The
launcher resolves that weight; see its --placebo block.

THE ONE THING THAT WOULD INVALIDATE THE COMPARISON, and how it is closed here.

`think_overlap_reward` returns None -- unscored, imputed to the group mean, neutral in
the advantage -- for a completion with no gradeable observe step (commit 8489767 exists
because scoring those 0 was wrong). `random` and `length` could trivially score EVERY
completion, and then a placebo run would differ from the mean_in run in two ways at
once: the direction of the signal AND which completions receive one.

So the scored set is not imitated here, it is TAKEN. This function runs the identical
pipeline -- the same --overlap_natural_only gate, the same flattening into one batched
Grounding-DINO call, the same `_union_mask`, and then the REAL configured metric via
`overlap_rewards._step_score` -- and uses that real score only as a boolean: a step
counts iff the real reward would have counted it. The placebo value replaces the number,
never the gate. `test_placebo_reward_cpu.py` pins the two unscored sets equal.

The consequence is that `length` is NOT cheap, and the doc's "no DINO, no attention
re-forward" is wrong on the DINO half: whether a completion has a gradeable observe step
is a question only DINO can answer, so every placebo pays the full grounding cost
(measured: 16.6 s of a 40.5 s optimizer step, against 1.0 s for the attention
re-forward). The re-forward is kept too -- at 2.5% of the step it is not worth a second
code path through the ZeRO-3 trainer, and keeping it means a placebo run is byte-for-byte
the same computation as its reference in everything except the reward value.

NOT AVAILABLE WITH --overlap_metric logratio, and the refusal is in `configure()`. The
roll-null IS a comparison against rolled copies of the union, so `--placebo roll` on it
asks for a rolled control of a rolled control; and its scorer DRAWS random placements and
writes into overlap_rewards' diagnostic buffer, so calling it as a parity gate would both
consume that draw and double-count the diagnostics. Use mean_in (the reference), or
mean_in_v2 / auroc.

DETERMINISM OF THE ROLL. The offset is drawn from an rng seeded by
blake2b(seed | prompt | step text), so the same observe step of the same prompt lands in
the same wrong place in every epoch, on every rank, and across a restart. It is NOT
seeded from a dataset row index: the probe writes a `row_index`, but no training corpus
carries one as a column, and the reward function only ever sees the columns the dataset
has. Hashing the text is the identity that actually survives shuffling -- and a fresh
roll per epoch would turn `roll` into a second, noisier copy of `random`, which is the
one confound this control cannot afford.

The offsets come from `roll_null.sample_offsets(mask, 1, rng, inframe=True)` -- the same
sampler the roll-null metric and the gradient reward use, not a second roller. In-frame
means the translated union stays inside the grid: a toroidal wrap splits the mask across
the image border and changes its SHAPE, which is exactly the thing this control is
supposed to hold fixed. `min_inframe=1` here rather than roll_null's default 4, because
the fallback is a last resort (a union whose bounding box spans the whole grid cannot
move at all in-frame) and every fallback is counted in `placebo/roll_toroidal_frac`.
Area is preserved either way -- np.roll is a permutation -- and it is asserted.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import overlap_rewards as _ORW
from . import roll_null as _RN

KINDS = ("roll", "random", "length")

_CFG = {
    # None = disabled; think_placebo_reward is then never installed by the launcher.
    "kind": None,
    # Seed for BOTH the roll offsets and the `random` draw, from --rollnull_seed. Two
    # runs differing only in it are independent replicates of the same control.
    "seed": 0,
    # Keep the translated union inside the grid (see the module docstring).
    "inframe": True,
    # --placebo length: score = -n_completion_tokens / length_scale. Linear on purpose --
    # the calibration multiplies the weight by a ratio of standard deviations, and a
    # non-linear map would make that ratio depend on where the length distribution sits.
    "length_scale": 1000.0,
}

# FIXED key set, always all of it, for the reason grad_rewards.DIAG_KEYS documents: the
# trainer gathers these across ranks, so a key set that depended on what a rank happened
# to see would mean a rank-dependent number of collectives, which hangs rather than fails.
#
# `roll_toroidal_frac` is the one to watch. It says the union's bounding box filled the
# grid, so the control wrapped across the image border and no longer has the union's
# shape -- which is the property that makes `roll` a control rather than noise.
DIAG_KEYS = ("roll_toroidal_frac", "roll_dist", "union_frac")
_DIAG: dict[str, list[float]] = {}


def _diag(key: str, value: float):
    _DIAG.setdefault(key, []).append(float(value))


def pop_diagnostics() -> dict[str, float]:
    """Mean of each placebo diagnostic since the last call, then clear.

    Always all of DIAG_KEYS; NaN for a key nothing was recorded under (every key for
    `random` and `length`, which have no roll).
    """
    out = {k: (float(np.mean(_DIAG[k])) if _DIAG.get(k) else float("nan")) for k in DIAG_KEYS}
    _DIAG.clear()
    return out


def is_active() -> bool:
    """True when a placebo replaces the overlap reward. Rank-uniform (it is set from the
    CLI on every process), so the trainer may branch its logging collectives on it."""
    return _CFG["kind"] is not None


def configure(**kwargs):
    """Set the placebo config from the CLI flags. None values are ignored.

    Call AFTER overlap_rewards.configure(): the metric compatibility check below reads
    the resolved metric out of that module, so there is one switch rather than two that
    have to agree.
    """
    for k, v in kwargs.items():
        if v is not None:
            _CFG[k] = v
    kind = _CFG["kind"]
    if kind is None:
        return
    if kind not in KINDS:
        raise ValueError(f"--placebo must be one of {'|'.join(KINDS)} (got {kind!r})")
    if _ORW._CFG.get("metric") == "logratio":
        raise ValueError(
            "--placebo is not available with --overlap_metric logratio. The roll-null is "
            "already scored against rolled copies of the union, so 'roll' would be a "
            "control of a control; and its scorer draws random placements and writes into "
            "overlap_rewards' diagnostic buffer, so using it as the scored/unscored parity "
            "gate would consume that draw and double-count the diagnostics. Use the "
            "reference metric (mean_in), or mean_in_v2 / auroc."
        )


# ---------------------------------------------------------------------------
# The three placebo values
# ---------------------------------------------------------------------------

def _completion_text(completion) -> str:
    """The assistant text of one completion, in either shape the trainer may hand over."""
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return completion if isinstance(completion, str) else ""


def _blake_u64(*parts: str) -> int:
    """Stable 64-bit digest of the parts. Stable across processes and restarts, unlike
    Python's own hash(), which is salted per interpreter."""
    payload = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def uniform01(text: str, seed: int = 0) -> float:
    """--placebo random: a deterministic U(0, 1) draw from the completion's own text.

    PER COMPLETION, not per prompt. A per-prompt value is constant inside a generation
    group, and GRPO subtracts the group mean before anything else, so it would give
    exactly zero advantage -- the opposite of the intended control, which is pure
    within-group variance.
    """
    return _blake_u64("placebo-random", str(seed), text) / 2.0**64


def length_score(n_tokens: int, scale: float = 1000.0) -> float:
    """--placebo length: monotone DECREASING in the completion's token count.

    `n_tokens` is the same count the trainer logs as train/completions/mean_length: the
    completion token ids up to and including the first EOS.
    """
    return -float(n_tokens) / float(scale)


def roll_mask(mask: np.ndarray, prompt: str, step_text: str, seed: int = 0,
              inframe: bool = True):
    """--placebo roll: the step's own union, moved. -> (rolled_mask, info) or (None, info).

    Deterministic in (seed, prompt, step text) -- see the module docstring.
    """
    info = {"toroidal": None, "dist": None, "offset": None}
    mask = np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(_blake_u64("placebo-roll", str(seed), prompt, step_text))
    # min_inframe=1: only fall back to a toroidal wrap when there is literally no in-frame
    # placement, i.e. the union's bounding box already spans the grid.
    offsets, toroidal = _RN.sample_offsets(mask, 1, rng, inframe=inframe, min_inframe=1)
    if not offsets:
        return None, info
    dy, dx = offsets[0]
    rolled = np.roll(mask, (dy, dx), axis=(0, 1))
    # Area- and (in-frame) shape-preserving by construction; np.roll is a permutation, so
    # this can only fail if the mask stopped being boolean.
    assert int(rolled.sum()) == int(mask.sum()), (int(rolled.sum()), int(mask.sum()))
    info.update(toroidal=bool(toroidal), dist=float(np.hypot(dy, dx)), offset=(int(dy), int(dx)))
    return rolled, info


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------

def think_placebo_reward(
    completions=None, saliency_map=None, valid_list=None, image=None, natural=None,
    prompts=None, completion_ids=None, **kwargs
):
    """Per-completion placebo reward. See module docstring.

    Structurally identical to `think_overlap_reward` -- same natural-only gate, same
    batched DINO call, same union mask, same mean over the completion's scored steps,
    same multiplicative format gate, same None for a completion with nothing to score --
    with exactly one substitution: the per-step VALUE. The scored/unscored decision is
    still taken by the real configured metric.
    """
    kind = _CFG["kind"]
    if kind not in KINDS:
        raise ValueError(
            f"think_placebo_reward was called with --placebo {kind!r}; "
            f"expected one of {'|'.join(KINDS)}."
        )

    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

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

    # Per-completion constants. `random` and `length` do not vary across a completion's
    # observe steps, so they are computed once here; the per-step loop below still decides
    # WHICH completions get one.
    const = [None] * n
    if kind == "random":
        texts = [_completion_text(c) for c in (completions or [None] * n)]
        const = [uniform01(t, _CFG["seed"]) for t in texts]
    elif kind == "length":
        if completion_ids is None:
            raise KeyError(
                "--placebo length needs the per-completion token ids (the trainer passes "
                "them as completion_ids), but none reached the reward function."
            )
        const = [length_score(len(ids), _CFG["length_scale"]) for ids in completion_ids]

    prompts = prompts if prompts is not None else [""] * n

    # One batched DINO call over every (completion, observe step). Identical to
    # think_overlap_reward, masked rows included: a masked row must not cost a grounding
    # call even if the trainer handed it maps.
    flat_images, flat_texts, flat_owner = [], [], []
    for c, steps in enumerate(saliency_map):
        if not steps or not scored[c]:
            continue
        img = image[c]
        for si, st in enumerate(steps):
            flat_images.append(img)
            flat_texts.append(st["text"])
            flat_owner.append((c, si))

    boxes_per_item = _ORW._dino_boxes(flat_images, flat_texts) if flat_images else []

    per_completion = [[] for _ in range(n)]
    for (c, si), boxes in zip(flat_owner, boxes_per_item):
        step_map = saliency_map[c][si]["map"]
        gh, gw = step_map.shape
        mask = _ORW._union_mask(boxes, gh, gw)
        if mask is None:
            continue  # DINO couldn't ground this step (or the union cap dropped it)
        # THE PARITY GATE. The real metric's value is thrown away; only whether it exists
        # is used, so a placebo scores a step if and only if the real reward would have.
        if _ORW._step_score(step_map, mask) is None:
            continue
        if kind == "roll":
            rolled, info = roll_mask(mask, prompts[c], saliency_map[c][si]["text"],
                                     seed=_CFG["seed"], inframe=_CFG["inframe"])
            if rolled is None:
                continue
            v = _ORW._step_score(step_map, rolled)
            if v is None:
                # Unreachable for mean_in / mean_in_v2 / auroc: none of them can reject a
                # mask the same size on the same map (mean_in and auroc depend on the
                # in/out COUNTS, which np.roll preserves; mean_in_v2's denominator is the
                # whole map). logratio can, and configure() refuses it.
                continue
            _diag("roll_toroidal_frac", 1.0 if info["toroidal"] else 0.0)
            _diag("roll_dist", info["dist"])
            _diag("union_frac", float(mask.mean()))
        else:
            v = const[c]
        per_completion[c].append(v)

    rewards = []
    for c in range(n):
        if not scored[c]:
            rewards.append(None)  # non-natural under --overlap_natural_only -> mask
            continue
        vals = per_completion[c]
        if not vals:
            rewards.append(None)  # zero scorable observe steps -> mask (neutral)
            continue
        # The format gate is multiplicative and kept identical to the overlap reward's,
        # because parity is the point. Note it is unreachable in training: the trainer
        # builds no observe-step maps for a format-invalid completion, so such a row
        # arrives with no steps and is masked above. If that ever changes, revisit it for
        # `length` -- a 0 there is the HIGHEST possible length score, not the lowest.
        rewards.append(float(np.mean(vals)) * (1.0 if valid_list[c] else 0.0))
    return rewards
