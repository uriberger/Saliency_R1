#!/usr/bin/env python
"""CPU checks for impute_unscored_rewards in trl/grpo_trainer_qwen3.py.

No GPU, no model. What this gates is the one line that decides what an UNSCORED reward
does to a completion's advantage.

A reward func returns None -> NaN when it did not apply to a completion (no groundable
observe step, every step over --max_union_area, --overlap_natural_only masking the row).
The `nansum` this replaced turned that NaN into 0, which is neutral only for a metric
whose chance level happens to be 0. Measured on the four 8k runs on 2026-08-18: under
mean_in_v2 (level ~1.27) a masked completion took a -1.86 mean advantage in answer-tied
groups, and under the roll-null (level ~-0.28) a +1.05 one -- masking PAID. So the tests
below are about a sign and a magnitude that were both live in training, not about style.

The function is loaded by parsing the trainer source and exec'ing just this one
definition: grpo_trainer_qwen3.py imports its siblings relatively (`from ..data_utils
import ...`) and only resolves inside the installed trl_repo/ layout, which a worktree
must not touch. The extraction asserts it found exactly one definition, so a rename
fails the test loudly instead of silently testing nothing.

    python test_reward_masking_cpu.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
SRC = REPO / "trl" / "grpo_trainer_qwen3.py"
FUNC = "impute_unscored_rewards"


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(defs) == 1, f"expected exactly one module-level `{name}` in {path}, found {len(defs)}"
    ns: dict = {"torch": torch}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


impute = _load_function(SRC, FUNC)

NAN = float("nan")
G = 4  # num_generations, small enough to write groups out by hand


def check(label, got, want, tol=1e-6):
    got_t = torch.as_tensor(got, dtype=torch.float32)
    want_t = torch.as_tensor(want, dtype=torch.float32)
    ok = torch.allclose(got_t, want_t, atol=tol) and not torch.isnan(got_t).any()
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"        got  {got_t.tolist()}\n        want {want_t.tolist()}")
        sys.exit(1)


def advantages(rewards_per_func, weights, num_generations=G, scale=True):
    """The trainer's own advantage arithmetic, so the tests assert on what training sees."""
    r = (impute(rewards_per_func, num_generations) * torch.tensor(weights).unsqueeze(0)).sum(dim=1)
    g = r.view(-1, num_generations)
    a = (g - g.mean(dim=1, keepdim=True))
    if scale:
        a = a / (g.std(dim=1, keepdim=True) + 1e-4)
    return a.reshape(-1)


print(__doc__.splitlines()[0])
print("\n-- imputation --")

# 1. A scored group is untouched.
x = torch.tensor([[1.0, 0.2], [0.0, 0.4], [1.0, 0.6], [0.0, 0.8]])
check("all scored -> unchanged", impute(x, G), x)

# 2. One masked completion takes its group's mean of that func, and only that func.
x = torch.tensor([[1.0, 0.2], [0.0, NAN], [1.0, 0.6], [0.0, 0.4]])
check("one NaN -> group mean of the scored rows (0.2+0.6+0.4)/3",
      impute(x, G), [[1.0, 0.2], [0.0, 0.4], [1.0, 0.6], [0.0, 0.4]])

# 3. The chance level is irrelevant: a metric centred at 1.27 imputes 1.27, not 0.
x = torch.tensor([[1.0, 1.20], [1.0, NAN], [1.0, 1.30], [1.0, 1.31]])
check("mean_in_v2-like level imputes ~1.27, NOT 0",
      impute(x, G)[1, 1], (1.20 + 1.30 + 1.31) / 3)

# 4. Groups are independent -- one group's mean never leaks into the next.
x = torch.tensor([[NAN], [1.0], [1.0], [1.0],
                  [NAN], [9.0], [9.0], [9.0]])
check("per-group, not per-batch", impute(x, G).reshape(-1), [1, 1, 1, 1, 9, 9, 9, 9])

# 5. A wholly unscored group imputes 0 and stays flat (any constant would do).
x = torch.tensor([[1.0, NAN], [0.0, NAN], [1.0, NAN], [0.0, NAN]])
check("no scored completion in the group -> 0.0", impute(x, G)[:, 1], [0, 0, 0, 0])

# 6. Multiple funcs can be masked on different rows at once.
x = torch.tensor([[NAN, 0.2], [0.5, NAN], [0.1, 0.6], [0.0, 0.4]])
check("independent masks per func", impute(x, G),
      [[0.2, 0.2], [0.5, 0.4], [0.1, 0.6], [0.0, 0.4]])

print("\n-- what it does to the advantage (weights = [1.0 answer, 0.017 saliency]) --")
W = [1.0, 0.017]
# The imputed row's deviation is exactly 0 in exact arithmetic (checks 1-6 pin that on the
# reward itself). The advantage divides it by (group std + 1e-4), and on an answer-tied
# group that std is ~0.003, so float32 rounding in the mean comes back multiplied by ~350.
# NEUTRAL is the tolerance below; the groupmates' advantages are O(1) next to it.
NEUTRAL = 2e-4

# 7. THE regression. Answer side unanimous, so the saliency term is the whole advantage;
#    the masked completion must come out at exactly 0 instead of taking the group.
x = torch.tensor([[1.0, -0.10], [1.0, NAN], [1.0, -0.50], [1.0, -0.30]])
a = advantages(x, W)
check("answer-tied group: the unscored completion gets advantage 0", a[1], 0.0, tol=NEUTRAL)
assert a[0] > 0 > a[2], f"the scored completions must still rank against each other: {a.tolist()}"
print(f"  [ok] its groupmates still score against each other ({a[0]:+.3f}, {a[2]:+.3f})")

# 8. The old `nansum` behaviour, asserted as the thing that is now gone. Level ~-0.3 and
#    a masked completion scored 0 -> it BEAT every groupmate on a reward never measured.
old = torch.nan_to_num(x, nan=0.0)
old_r = (old * torch.tensor(W).unsqueeze(0)).sum(dim=1).view(1, G)
old_a = ((old_r - old_r.mean()) / (old_r.std() + 1e-4)).reshape(-1)
assert old_a[1] == old_a.max(), "expected the old behaviour to reward the masked row"
print(f"  [ok] the old nansum gave that same row {old_a[1]:+.3f} (the group's best), now {a[1]:+.3f}")

# 9. Same test with a chance-1.0 metric: the old code punished, the new one is neutral.
x = torch.tensor([[1.0, 1.20], [1.0, NAN], [1.0, 1.30], [1.0, 1.31]])
a = advantages(x, W)
old = torch.nan_to_num(x, nan=0.0)
old_r = (old * torch.tensor(W).unsqueeze(0)).sum(dim=1).view(1, G)
old_a = ((old_r - old_r.mean()) / (old_r.std() + 1e-4)).reshape(-1)
check("mean_in_v2-like, answer-tied: unscored completion gets 0", a[1], 0.0, tol=NEUTRAL)
assert old_a[1] == old_a.min(), "expected the old behaviour to punish the masked row"
print(f"  [ok] the old nansum gave it {old_a[1]:+.3f} (the group's worst), now {a[1]:+.3f}")

# 10. A group nobody scored contributes no saliency spread at all -- so a group that was
#     answer-tied is now genuinely zero-std, which is what frac_reward_zero_std reports.
x = torch.tensor([[1.0, NAN], [1.0, NAN], [1.0, NAN], [1.0, NAN]])
r = (impute(x, G) * torch.tensor(W).unsqueeze(0)).sum(dim=1)
check("wholly unscored + answer-tied -> zero-std group (honest frac_reward_zero_std)",
      r.std(), 0.0, tol=NEUTRAL)

# 11. The saliency term must not change an advantage it never measured anywhere.
x_scored = torch.tensor([[1.0, -0.10], [0.0, -0.20], [1.0, -0.50], [0.0, -0.30]])
x_masked = torch.tensor([[1.0, NAN], [0.0, NAN], [1.0, NAN], [0.0, NAN]])
a_answer_only = advantages(torch.stack([x_scored[:, 0], torch.zeros(G)], dim=1), [1.0, 0.0])
check("fully-unscored group falls back to the answer-side advantage",
      advantages(x_masked, W), a_answer_only, tol=NEUTRAL)

# ---------------------------------------------------------------------------
# 12-17. THE SWITCH. `impute` above is no longer unconditional: the trainer picks it
# only when SELFSAL_IMPUTE_UNSCORED is set, because the right answer depends on the
# metric's CHANCE LEVEL. mean_in sits at ~0.065 and tolerates an unapplied reward being
# scored 0 (~2.1 sd) -- and wants to, because that penalty is what keeps chains
# groundable; removing it collapsed completion length from 154-157 to 114-117 by step
# 1600 in the self-saliency release. mean_in_v2 sits at 1.0 (~9.5 sd) and does not.
#
# Checked on the SOURCE, not by importing the trainer, for the reason in the module
# docstring: it only resolves inside trl_repo/, which a worktree must not touch.
# ---------------------------------------------------------------------------
print("\n-- the SELFSAL_IMPUTE_UNSCORED switch --")

src = SRC.read_text()
has_env = "os.environ.get(\"SELFSAL_IMPUTE_UNSCORED\")" in src
check("trainer gates the weighted sum on SELFSAL_IMPUTE_UNSCORED", float(has_env), 1.0)
check("the nansum branch is present (unapplied -> 0)",
      float(".nansum(dim=1)" in src), 1.0)
check("the impute branch is still present (unapplied -> neutral)",
      float("impute_unscored_rewards(rewards_per_func, self.num_generations)" in src), 1.0)


def weighted(rewards_per_func, weights, impute_on):
    """Exactly the two branches the trainer now chooses between."""
    w = torch.tensor(weights).unsqueeze(0)
    if impute_on:
        return (impute(rewards_per_func, G) * w).sum(dim=1)
    return (rewards_per_func * w).nansum(dim=1)


# A mean_in-shaped group: answer-tied, so the saliency term IS the whole advantage,
# and completion 2 wrote no groundable observe step.
MI_W = [1.0, 0.2]
x = torch.tensor([[1.0, 0.060], [1.0, 0.040], [1.0, NAN], [1.0, 0.050]])

r_imp = weighted(x, MI_W, True)
check("impute: the unapplied completion deviates from its group by 0",
      r_imp[2] - r_imp.mean(), 0.0, tol=NEUTRAL)

r_nan = weighted(x, MI_W, False)
penalty = (r_nan[2] - r_nan.mean()).item()
check("nansum: the unapplied completion is PENALISED, not neutral",
      float(penalty < -1e-4), 1.0)
check("nansum: and penalised by exactly w_overlap x its groupmates' mean level",
      -penalty, 0.2 * (0.060 + 0.040 + 0.050) / 3 * (3 / 4) + 0.2 * 0.0 * 0, tol=5e-3)

# The scored completions must be untouched by the choice -- it only moves the masked row.
check("the choice moves only the unapplied row, not its groupmates",
      r_imp[[0, 1, 3]] - r_imp[[0, 1, 3]].mean(),
      r_nan[[0, 1, 3]] - r_nan[[0, 1, 3]].mean(), tol=NEUTRAL)

print("\nall reward-masking checks passed")
