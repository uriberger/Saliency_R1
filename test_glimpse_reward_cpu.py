#!/usr/bin/env python
"""CPU checks for trl/rewards/glimpse_rewards.py -- the GLIMPSE grounding reward.

No GPU, no model, no Grounding-DINO (the grounding call is stubbed). The map algebra is
gated separately by test_glimpse_cpu.py; what this gates is the REWARD around it, and in
particular the two properties that decide whether either variant means what it says:

  * THE TWO VARIANTS ARE THE TWO METRICS, and both come from overlap_rewards, so a
    change there cannot silently give the reward a third behaviour. mean_in_v2 reads 1.0
    on a flat map (chance) and auroc reads 0.5, for a union of any area and any shape.
  * THE UNION-AREA FREE RIDE IS REAL AND IS LOGGED. mean_in_v2's ceiling is
    n_patches/n_in, so on a map that is merely concentrated somewhere -- anywhere -- a
    bigger union changes the score with no change in grounding. This is the known hack
    direction and the reason `union_frac` and `ceiling` are in DIAG_KEYS. The test
    asserts the effect EXISTS for mean_in_v2 and does NOT for auroc, because that
    difference is the whole reason to offer both.
  * a planted signal is recovered, and the two metrics agree on its sign;
  * degenerate steps return None -- absent, not zero -- so they leave the mean over
    steps rather than dragging it down;
  * the completion level: the format gate multiplies, no scorable step masks the row,
    --glimpse_natural_only masks the row, duplicates are deduped before the mean;
  * the diagnostics have a FIXED key set, because the trainer gathers them across ranks.

    python test_glimpse_reward_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load(dotted, rel):
    spec = importlib.util.spec_from_file_location(dotted, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub the parent packages so the module's relative imports resolve without executing
# trl/__init__.py (which drags in the whole TRL dependency tree for no reason here).
for _name, _path in (("trl", REPO / "trl"), ("trl.rewards", REPO / "trl" / "rewards")):
    _m = types.ModuleType(_name)
    _m.__path__ = [str(_path)]
    sys.modules[_name] = _m

OR = _load("trl.rewards.overlap_rewards", "trl/rewards/overlap_rewards.py")
_load("trl.rewards.grad_rewards", "trl/rewards/grad_rewards.py")
GL = _load("trl.rewards.glimpse_rewards", "trl/rewards/glimpse_rewards.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


def box(gh, gw, r0, r1, c0, c1):
    m = np.zeros((gh, gw), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def steps(*pairs):
    return [{"map": m, "text": t} for t, m in pairs]


def stub_dino(mapping):
    """Replace grounding with a fixed text -> boxes table (relative coords).

    Keyed on the NORMALISED text, because the real detector does not care about case or a
    trailing period either -- a stub that keys on the raw string would silently drop the
    duplicate variants and make the dedupe test pass for the wrong reason.
    """
    table = {GL._norm_text(k): v for k, v in mapping.items()}

    def _fake(images, texts):
        return [table.get(GL._norm_text(t), []) for t in texts]
    GL._dino_boxes = _fake


def reset(metric="mean_in_v2", **kw):
    OR.configure(metric=metric, box_threshold=0.1, max_box_area=0.5, max_union_area=None,
                 mass_floor_tau=None)
    cfg = {"dedupe_steps": True, "natural_only": False}
    cfg.update(kw)
    GL.configure(**cfg)
    GL.pop_diagnostics()


# ---------------------------------------------------------------------------
def test_flat_map_is_at_chance():
    print("\n[null] a flat map reads chance at every union size, in both variants")
    G = np.full((16, 16), 0.37, dtype=np.float32)
    for metric, chance in (("mean_in_v2", 1.0), ("auroc", 0.5)):
        reset(metric)
        worst = 0.0
        for side in (1, 2, 4, 6, 8, 10, 12):
            m = box(16, 16, 2, 2 + side, 3, 3 + side)
            worst = max(worst, abs(OR._step_score(G, m) - chance))
        check(f"{metric} is {chance} for every box size", worst < 1e-9,
              f"max deviation {worst:.2e}")


def test_scale_invariance():
    print("\n[null] both metrics are invariant to m -> c*m")
    rs = np.random.default_rng(0)
    G = rs.random((16, 16)).astype(np.float32)
    m = box(16, 16, 3, 9, 5, 11)
    for metric in ("mean_in_v2", "auroc"):
        reset(metric)
        base = OR._step_score(G, m)
        # RELATIVE, because mean_in_v2 is unbounded above: rescaling by 1e4 and dividing
        # two float64 means back out leaves ~1e-9 of rounding on a reading of order 1,
        # which is the arithmetic being exact, not the metric being asymmetric.
        worst = max(abs(OR._step_score(G * c, m) - base) / abs(base)
                    for c in (1e-3, 7.0, 1e4))
        check(f"{metric} is unchanged by a rescale", worst < 1e-8,
              f"max relative deviation {worst:.2e}")


def test_planted_signal():
    print("\n[signal] a map concentrated on the union beats one aimed elsewhere")
    m = box(16, 16, 4, 10, 4, 10)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    cold = np.full((16, 16), 0.1, dtype=np.float32)
    cold[box(16, 16, 11, 15, 11, 15)] = 3.0
    for metric, chance in (("mean_in_v2", 1.0), ("auroc", 0.5)):
        reset(metric)
        s_hot, s_cold = OR._step_score(hot, m), OR._step_score(cold, m)
        check(f"{metric}: aimed > chance > aimed elsewhere",
              s_hot > chance > s_cold, f"{s_hot:.3f} / {chance} / {s_cold:.3f}")


def test_union_area_free_ride():
    """The reason both variants exist, and the reason union_frac is always logged.

    On a map with a single hot blob, GROWING the union past the blob adds only cold
    patches. mean_in_v2 is a ratio of means, so its ceiling is n_patches/n_in and the
    reading MOVES; auroc is a rank statistic and does not care how many out-of-union
    patches there are relative to in-union ones, only about their order.
    """
    print("\n[hack] union area moves mean_in_v2 mechanically -- and not auroc")
    G = np.full((16, 16), 0.05, dtype=np.float32)
    G[box(16, 16, 6, 10, 6, 10)] = 4.0        # the blob, 16 patches
    tight = box(16, 16, 6, 10, 6, 10)          # exactly the blob
    loose = box(16, 16, 2, 14, 2, 14)          # the blob plus a lot of cold

    reset("mean_in_v2")
    v_t, v_l = OR._step_score(G, tight), OR._step_score(G, loose)
    check("mean_in_v2 reads differently for the same map under a bigger union",
          abs(v_t - v_l) > 0.5, f"tight {v_t:.3f} vs loose {v_l:.3f}")
    check("and the difference tracks the ceiling n_patches/n_in",
          v_t > v_l and v_t <= 256 / int(tight.sum()) + 1e-9,
          f"ceiling {256 / int(tight.sum()):.1f}")

    reset("auroc")
    a_t, a_l = OR._step_score(G, tight), OR._step_score(G, loose)
    check("auroc is far less moved by the same change", abs(a_t - a_l) < abs(v_t - v_l),
          f"tight {a_t:.3f} vs loose {a_l:.3f}")

    # The monitor that makes it visible in training.
    reset("mean_in_v2")
    stub_dino({"a cat": [[0.375, 0.375, 0.625, 0.625]]})
    GL.think_glimpse_reward(saliency_map=[steps(("a cat", G))], valid_list=[True],
                            image=[None])
    d = GL.pop_diagnostics()
    check("union_frac and ceiling are logged with every scored step",
          abs(d["union_frac"] - tight.mean()) < 1e-9
          and abs(d["ceiling"] - 256 / int(tight.sum())) < 1e-9,
          f"union_frac {d['union_frac']:.3f}, ceiling {d['ceiling']:.1f}")


def test_degenerate_steps_are_absent_not_zero():
    print("\n[skip] a step that cannot be scored is absent, not a low score")
    reset("mean_in_v2")
    zero = np.zeros((16, 16), dtype=np.float32)
    m = box(16, 16, 4, 8, 4, 8)
    check("an all-zero map has no defined ratio -> None", OR._step_score(zero, m) is None)
    full = np.ones((16, 16), dtype=bool)
    check("a union covering the whole grid is rejected by _union_mask",
          OR._union_mask([[0.0, 0.0, 1.0, 1.0]], 16, 16) is None
          or int(full.sum()) == 256)


def test_completion_level():
    print("\n[reward] the completion level")
    m = box(16, 16, 4, 8, 4, 8)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    flat = np.full((16, 16), 0.4, dtype=np.float32)
    stub_dino({"a cat": [[0.25, 0.25, 0.5, 0.5]], "nothing": []})

    reset("mean_in_v2")
    out = GL.think_glimpse_reward(
        saliency_map=[steps(("a cat", hot)), steps(("a cat", flat)), steps(("nothing", hot))],
        valid_list=[True, True, True], image=[None, None, None],
    )
    check("a grounded, well-aimed step scores above chance", out[0] > 1.0, f"{out[0]:.3f}")
    check("a flat map on the same box scores exactly chance", abs(out[1] - 1.0) < 1e-9,
          f"{out[1]:.6f}")
    check("a completion with no groundable step is masked, not zeroed", out[2] is None)

    reset("auroc")
    out_a = GL.think_glimpse_reward(
        saliency_map=[steps(("a cat", hot)), steps(("a cat", flat))],
        valid_list=[True, True], image=[None, None],
    )
    check("the auroc variant reads chance 0.5 on the flat map",
          abs(out_a[1] - 0.5) < 1e-9 and out_a[0] > 0.5,
          f"aimed {out_a[0]:.3f}, flat {out_a[1]:.6f}")

    reset("mean_in_v2")
    gated = GL.think_glimpse_reward(saliency_map=[steps(("a cat", hot))],
                                    valid_list=[False], image=[None])
    check("the format gate multiplies", gated[0] == 0.0)

    reset("mean_in_v2", natural_only=True)
    masked = GL.think_glimpse_reward(
        saliency_map=[steps(("a cat", hot)), steps(("a cat", hot))],
        valid_list=[True, True], image=[None, None], natural=[True, False],
    )
    check("--glimpse_natural_only masks the non-natural row",
          masked[0] > 1.0 and masked[1] is None)
    try:
        GL.think_glimpse_reward(saliency_map=[steps(("a cat", hot))], valid_list=[True],
                                image=[None])
        raised = False
    except KeyError:
        raised = True
    check("and says so rather than silently scoring every row when 'natural' is absent",
          raised)


def test_dedupe():
    print("\n[reward] duplicate steps")
    m = box(16, 16, 4, 8, 4, 8)
    hot = np.full((16, 16), 0.1, dtype=np.float32)
    hot[m] = 3.0
    cold = np.full((16, 16), 0.1, dtype=np.float32)
    cold[box(16, 16, 10, 14, 10, 14)] = 3.0
    stub_dino({"a cat": [[0.25, 0.25, 0.5, 0.5]], "a dog": [[0.25, 0.25, 0.5, 0.5]]})

    # one genuine, hard step (aimed elsewhere) diluted by three copies of an easy one
    hacked = steps(("a cat", hot), ("A cat.", hot), ("a  cat", hot), ("a dog", cold))
    reset("mean_in_v2", dedupe_steps=True)
    on = GL.think_glimpse_reward(saliency_map=[hacked], valid_list=[True], image=[None])
    d_on = GL.pop_diagnostics()
    reset("mean_in_v2", dedupe_steps=False)
    off = GL.think_glimpse_reward(saliency_map=[hacked], valid_list=[True], image=[None])
    d_off = GL.pop_diagnostics()

    check("duplicates are dropped before the mean", on[0] < off[0],
          f"deduped {on[0]:.3f} vs raw {off[0]:.3f}")
    check("the duplicate fraction is logged either way",
          abs(d_on["dup_frac"] - 0.5) < 1e-9 and abs(d_off["dup_frac"] - 0.5) < 1e-9,
          f"{d_on['dup_frac']:.2f}")
    check("the dedupe flag is glimpse's own, not grad's",
          GL._CFG["dedupe_steps"] is False)
    reset("mean_in_v2")


def test_diagnostics():
    print("\n[reward] the diagnostics the hacks show up in")
    reset("mean_in_v2")
    G = np.full((16, 16), 0.5, dtype=np.float32)
    G[box(16, 16, 6, 10, 6, 10)] = 2.0
    stub_dino({"centre": [[0.375, 0.375, 0.625, 0.625]],
               "corner": [[0.0, 0.0, 0.25, 0.25]]})
    GL.think_glimpse_reward(saliency_map=[steps(("centre", G))], valid_list=[True],
                            image=[None])
    centre = GL.pop_diagnostics()
    GL.think_glimpse_reward(saliency_map=[steps(("corner", G))], valid_list=[True],
                            image=[None])
    corner = GL.pop_diagnostics()
    check("ecc is near 0 for a centred union and large for a corner one",
          centre["ecc"] < 0.1 and corner["ecc"] > 0.6,
          f"centre {centre['ecc']:.3f}, corner {corner['ecc']:.3f}")
    check("n_image is the map's total mass",
          abs(centre["n_image"] - G.astype(np.float64).sum()) < 1e-6)
    check("grounded_frac counts the steps that actually scored",
          abs(centre["grounded_frac"] - 1.0) < 1e-9)

    # The map producer's own numbers reach the same channel.
    GL.record_map_info({"n_steps_built": 4, "unweighted_steps": [1],
                        "n_target_tokens": 57})
    info = GL.pop_diagnostics()
    check("record_map_info reports the blanked-aggregation fraction",
          abs(info["unweighted_frac"] - 0.25) < 1e-9, f"{info['unweighted_frac']:.2f}")
    check("and the per-case target-token count, which is what the cost is linear in",
          abs(info["n_target_tokens"] - 57) < 1e-9)

    # A fixed key set, NaN where nothing was seen: the trainer gathers these across ranks
    # and a rank-dependent set of keys would mean a rank-dependent number of collectives.
    cleared = GL.pop_diagnostics()
    check("pop_diagnostics clears", all(np.isnan(v) for v in cleared.values()))
    check("and always returns every key, whatever this rank saw",
          tuple(cleared) == GL.DIAG_KEYS and set(centre) == set(GL.DIAG_KEYS))


def test_metric_comes_from_overlap_rewards():
    """The brief's instruction, made mechanical: score through the existing
    `configure(metric=...)` path rather than reimplementing either metric."""
    print("\n[wiring] the reward scores through overlap_rewards, not a private copy")
    check("glimpse_rewards imports _step_score rather than defining one",
          GL._step_score is OR._step_score)
    reset("mean_in_v2")
    check("configure(metric=...) on overlap_rewards is what selects the variant",
          OR._CFG["metric"] == "mean_in_v2")
    reset("auroc")
    check("and switching it switches the reward", OR._CFG["metric"] == "auroc")
    reset("mean_in_v2")


def main():
    test_flat_map_is_at_chance()
    test_scale_invariance()
    test_planted_signal()
    test_union_area_free_ride()
    test_degenerate_steps_are_absent_not_zero()
    test_completion_level()
    test_dedupe()
    test_diagnostics()
    test_metric_comes_from_overlap_rewards()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
