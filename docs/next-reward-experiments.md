# Next session: three control rewards, to find out whether DIRECTION matters

**Your job is to implement three auxiliary rewards behind flags and stop. Uri runs
them.** Do not launch training. Do not re-litigate the findings below — they are
settled measurements, and the experiment exists because they do not add up to an
explanation.

---

## What is established

Measured 2026-08-18/19 (details in [sharpness-results.md](sharpness-results.md) and the
memory files `saliency-reward-is-a-tiebreaker`, `sharpness-beats-grounding`):

1. **The saliency reward does not identify the correct completion.** Within a group of
   8 completions for the same prompt — which is the only contrast GRPO acts on, since
   the advantage subtracts the group mean — `r(overlap reward, accuracy reward)` is
   −0.019 ± 0.051 (mean_in w0.4), +0.009 ± 0.074 (auroc w0.11), −0.004 ± 0.052
   (mean_in_v2 w0.033).

2. **It does not improve the training objective.** All four runs below are on the same
   default `peterant330/saliency-r1-8k`, and at steps 2000–3000 they are
   indistinguishable:

   | run | accuracy_reward | openai judge |
   |---|---|---|
   | no-sal (accuracy only) | 0.267 | 0.732 |
   | saliency-r1 | 0.267 | 0.723 |
   | overlap mean_in w0.4 | 0.262 | 0.729 |
   | overlap auroc w0.11 | 0.277 | 0.736 |

3. **It does keep the gradient alive.** 86–90% of groups have zero accuracy variance,
   and `train/frac_reward_zero_std` — the share of groups with zero advantage, i.e. no
   gradient at all — runs 0.271 → **0.547** for accuracy-only against **0.000** for
   every overlap run. That ordering matches the bench/natural/mean ordering exactly.

4. **The largest thing the reward is associated with, within a group, is brevity.**

   | run | r(overlap, accuracy) | r(overlap, **completion length**) |
   |---|---|---|
   | mean_in w0.4 | −0.019 | **−0.042** |
   | auroc w0.11 | +0.009 | **−0.105** |
   | mean_in_v2 w0.033 | −0.004 | −0.035 |

   All negative, in all three runs and all training terciles. Plausible mechanism:
   `token_reduction=mean` averages the metric over each observe step's tokens and over
   steps, so fewer and tighter observe steps dilute it less.

## What is NOT established, and why these runs exist

Uri's objection, which is correct: "tie-breaking" explains why gradient keeps flowing,
but not **which direction** it flows in. A reward that broke ties by rewarding the
*shortest* completion would also take `frac_reward_zero_std` to zero, and there is no
reason to expect it to help. Since the benchmark does move, the direction cannot be
irrelevant. Nothing measured so far identifies that direction.

Note the uncomfortable coincidence: auroc w0.11 drove length 172 → 71 and entropy
0.62 → 0.199, and has the **highest** bench/natural/mean of anything run (0.7544). The
"obviously bad" shortest-wins reward may be the one that works. mean_in w0.4 shortens
mildly (200 → 147) and is the only run that is best on natural *and* nonnatural.

---

## The experiment

Three auxiliary rewards, each replacing `think_overlap_reward`, each **matched to
mean_in w0.4's within-group standard deviation** so that tie-breaking strength is held
constant and only direction varies.

| flag | reward | what it isolates |
|---|---|---|
| `--placebo roll` | the same map and metric scored on a **rolled** union: same area, same shape, wrong location | is it grounding, or any same-shaped signal? |
| `--placebo random` | a deterministic hash of the completion → uniform score | pure variance, no direction at all |
| `--placebo length` | monotone decreasing in completion token count | is the reward a brevity reward in disguise? |

Readout, against `overlap mean_in w0.4` and against the accuracy-only control
(`--w-overlap 0`, which Uri is running separately):

| outcome | conclusion |
|---|---|
| all three ≈ mean_in | direction is irrelevant; the reward is a variance source |
| `length` ≈ mean_in, `random` does not | the overlap reward is a brevity reward in disguise — the current best guess |
| mean_in beats all three | grounding contributes something specific, and every null measured so far was looking in the wrong place |

`length` is the cheapest of the three — no DINO, no attention re-forward — so it runs at
roughly accuracy-only speed. Implement it first.

---

## Implementation

### Where the code lives

- **Edit `trl/`, not `trl_repo/`.** `trl/` is the tracked source; `trl_repo/` is the
  gitignored clone that actually executes, and it is **symlinked into every worktree and
  shared by every running job**. `patch_trl_qwen3.sh` copies `trl/` → `trl_repo/` and
  re-patches for every session and every job currently training. Say so before running
  it, and do not run it while Uri has training in flight.
- Reward implementations: `trl/rewards/overlap_rewards.py`, registered in
  `trl/rewards/__init__.py`.
- Roll machinery already exists: `trl/rewards/roll_null.py` gives
  `sample_offsets(mask, k, rng, inframe=True)` and `inframe_offsets(mask)`. Use it —
  do not write a new roller.
- Script args and reward wiring: `trl/grpo_vlm_qwen3.py` (see the existing
  `rollnull_offsets` / `rollnull_clip` / `rollnull_seed` plumbing around lines 550–585).
- Launcher: `launch_grpo_qwen3_overlap_colocated_job.sh`. Add `--placebo` to the arg
  loop (~line 279+), pass it through to the training invocation (~line 950+), and
  extend `SUFFIX` (~line 536) so a placebo run cannot share a checkpoint directory or a
  wandb run name with a real one.

### The three rewards, precisely

**`roll`** — identical to the configured metric in every respect except the mask.
Replace the step's union mask `m` with `np.roll(m, offset, axis=(0,1))` where `offset`
comes from `sample_offsets(m, 1, rng, inframe=True)`. The roll must be:
- **deterministic per (row_index, step index)**, seeded from `--rollnull-seed`, so the
  same step gets the same wrong place in every epoch. A fresh roll each epoch would make
  the reward pure noise and confound this with `random`.
- **area- and shape-preserving.** `inframe=True` keeps the mask inside the grid, which
  matters: a toroidal wrap splits the mask across edges and changes its shape. Assert
  `rolled.sum() == m.sum()`.

**`random`** — `score = U(0,1)` from a stable hash of the completion text (e.g.
`blake2b(completion.encode(), digest_size=8)` → int → /2**64). Per **completion**, not
per prompt: a per-prompt value is constant within a group and would give exactly zero
advantage, which is the opposite of the intended control. No DINO, no map, no GPU work.

**`length`** — `score = -n_completion_tokens / 1000.0` (or any monotone decreasing
function; keep it linear so the calibration below is interpretable). Use the same token
count the trainer already logs for `train/completions/mean_length`.

### Scored-vs-unscored parity — the trap that would invalidate the comparison

`mean_in` returns **unscored** for completions with no gradeable observe steps, and
commit `8489767` ("an unscored saliency reward must not be scored 0") exists because
scoring those 0 was wrong. `random` and `length` can score *every* completion, which
would change *which* completions receive an auxiliary signal, not just its direction —
a second variable.

**Every placebo must return unscored on exactly the completions where the configured
`mean_in` reward would be unscored.** That means running the step segmentation (and, for
`roll`, the DINO call) even for `random` and `length`, and masking their output by the
same validity flag. Yes, this makes `length` more expensive than it needs to be; a
one-variable comparison is worth it. Add a test that pins the two unscored sets equal on
a fixture.

### Calibration — match within-group sd, not sample sd

Effective pressure is `w × sd_within_group(reward)`, because GRPO centres within the
group before anything else. The existing `overlap_metric_spread.py` reports **sd per
sample**, which is the wrong quantity and is how the incumbent weights were set.

1. Extend `overlap_metric_spread.py` to also report **within-group sd** (group-mean-
   centre, then pool). `launch_overlap_probe.sh` already generates 8 generations per
   sample, so the data is there.
2. Measure `sd_within(mean_in)` on the same corpus.
3. For each placebo, set `w_placebo = 0.4 × sd_within(mean_in) / sd_within(placebo)`.
4. Print the resolved weight in the launcher banner, as the other metrics already do.

A cheaper cross-check that needs no GPU: the wandb `completions` tables carry
per-completion `think_overlap_reward` grouped by prompt, so `sd_within` for the
incumbent can be read straight off a finished run
(`api.run(...).history(keys=["completions"])`, then download the table json).

### What to log

Beyond the defaults, make sure these are on so the runs are comparable without new
tooling: `train/frac_reward_zero_std`, `train/entropy`,
`train/completions/mean_length`, `train/rewards/*/mean` and `/std`, and the
`completions` table (it is what every within-group number above came from). Add
`train/rewards/<name>/within_group_std` if it is cheap — it is the quantity the
calibration targets and nobody can currently see it during a run.

---

## Pre-registered predictions

Write these down before looking, because with four runs and two benchmark axes it is
easy to find a story afterwards.

- `random` reaches `frac_reward_zero_std` ≈ 0.000 like every other continuous reward.
- `length` produces the largest entropy and length collapse of the three.
- On bench/natural/mean, `length` ≥ mean_in > `random` ≈ `roll`.
- On bench/nonnatural/mean, `length` is the worst, mirroring auroc's 0.5067.
- Train accuracy and the judge score are flat across all of them, as they were for the
  four runs above.

## Traps

- **The benchmark still cannot resolve this.** se on a difference in
  bench/natural/mean was ≈ 0.028 at 5 tasks × 100 items, against effects of
  0.02–0.035. Pairing over items now halves it — use
  `python bench_samples.py --compare A B --suite natural`, which needs no GPU and
  is already harvested for every existing result. Even so, against `sft-coldstart`
  the best run to date is +0.0275 with a 95% CI of [-0.0050, +0.0600]: not
  significant. Raising the natural benchmarks to 300 is built and costed; see
  [bench-precision.md](bench-precision.md). Do not report a ranking of these four
  runs without a stated uncertainty.
- **Do not compare against `baseline/grpo-no-saliency`.** It starts from vanilla
  Qwen3-VL-8B-Instruct, not from `coldstart_..._merged` (check
  `bench_eval/base_model.txt`). The correct control is Uri's `--w-overlap 0` run.
  This is not hypothetical: against `grpo-no-saliency` the same paired data puts
  `auroc w0.11` at +0.0325 [+0.0025, +0.0625], a CI that excludes zero, and
  against `sft-coldstart` it does not.
- Match the reference run's adapter: `--lora-targets q_proj,v_proj`. The launcher's
  current default is `q_proj,k_proj,v_proj`, which is a different experiment and adds
  `_loraqkv` to the name.
- `/lustre/fs1` project quota for `nvr_israel_rlop` has been hitting 100T/100T. Check
  `df -h /home/uberger/scratch` before starting anything that writes checkpoints.
