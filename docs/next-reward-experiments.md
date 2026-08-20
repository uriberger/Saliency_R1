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

~~`length` is the cheapest of the three — no DINO, no attention re-forward — so it runs at
roughly accuracy-only speed.~~ **Wrong, and it contradicts the parity rule two sections
down.** Whether a completion has a gradeable observe step is a question only Grounding-DINO
can answer, so `length` pays the full grounding cost — 16.6 s of a 40.5 s optimizer step,
against 1.0 s for the attention re-forward. The re-forward is kept as well (2.5% of the
step is not worth a second path through the ZeRO-3 trainer), so a placebo run is the same
computation as its reference in everything except the reward's value. `length` is still
the one to run first, but because its calibration is the most stable of the three
(w 0.031 on both measured corpora), not because it is cheap.

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
group before anything else. ~~The existing `overlap_metric_spread.py` reports **sd per
sample**, which is the wrong quantity and is how the incumbent weights were set.~~
**Half wrong.** Its `sd/sample` column is already a within-group quantity — the MEAN over
prompts of each prompt's own sd — so the incumbent weights are not measuring the wrong
thing. What it was missing is the POOLED version, and the two differ by only a few percent
at n=8 (`E[s] < sqrt(E[s²])`), which is well inside the ±25% those weights already carry.
The column that really is the wrong one, `sd all`, was never used for a weight.

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

---

## As built — 2026-08-20

Implemented, CPU-tested, **not run**. Branch `feat/placebo-rewards`.

### Flags

```bash
bash launch_grpo_qwen3_overlap_colocated_job.sh --placebo length --lora-targets q_proj,v_proj
```

`--placebo roll|random|length` replaces `think_overlap_reward` with
`think_placebo_reward` in the same `reward_funcs` slot, so `--reward_weights` is
unchanged. It requires the attention map and refuses `--overlap-metric logratio` (the
roll-null is already scored against rolled unions, and its scorer draws randomness, so
using it as the parity gate would consume its own draw). The run name gains
`_placebo<kind>`, so a placebo can never share a checkpoint directory or a wandb run with
a real one.

### Resolved weights, and where they came from

`w = 0.4 × sd_within(mean_in) / sd_within(placebo)`, with `sd_within` the pooled
within-group sd. Measured on the cold-start policy this experiment starts from,
temperature 1, 8 generations, with `overlap_metric_spread.py` — which computes the three
placebo rows by importing `trl/rewards/placebo_rewards.py` itself, so the number a run is
launched with comes from the function the run will use.

| placebo | set_a 40×8 (315 compl) | val_natural 30×8 (231 compl) | launcher default |
|---|---|---|---|
| `roll` | — (that probe kept no maps) | 0.320 | **0.32** |
| `random` | 0.013 | 0.010 | **0.013** |
| `length` | 0.031 | 0.031 | **0.031** |

`sd_within(mean_in)` is 0.0098 on set_a and 0.0071 on val_natural, and that difference is
where almost all of the spread in the table comes from — `random` is analytic given it
(U(0,1) has sd 0.2887; measured 0.2917 and 0.2930, which also says the hash is uniform).
`roll` has one cold-start measurement; the trained checkpoint `mean_in_v2_cp_1700` on
set_a puts it at 0.52, so read 0.32 as bracketed by [0.32, 0.52]. It sits near the
reference's own 0.4 for a structural reason: it is the same metric on the same map with an
equal-area mask, and only the mask's location differs. To re-measure on the corpus you
actually train on:

```bash
bash launch_overlap_probe.sh --n-samples 40 --no-judge \
    --out-dir outputs/overlap_probe/placebo_spread --dataset <dataset>
python overlap_metric_spread.py outputs/overlap_probe/placebo_spread
```

### Scored-vs-unscored parity

Not imitated — taken. `think_placebo_reward` runs the identical pipeline (the same
`--overlap_natural_only` gate, the same batched Grounding-DINO call, the same
`_union_mask`, then the real configured metric via `overlap_rewards._step_score`) and uses
the real score **only as a boolean**. `test_placebo_reward_cpu.py` pins the two unscored
sets equal on a hand-built fixture covering every way the real reward can decline to score
a completion, and on 300 randomised batches × 3 metrics × 3 placebos.

### New logging

`train/rewards/<func>/within_group_std` — the quantity the calibration targets, for every
reward term, live. It is not a new collective: `rewards_per_func` is already gathered.
Plus `placebo/roll_toroidal_frac`, `placebo/roll_dist`, `placebo/union_frac` when
`--placebo roll` is on; `roll_toroidal_frac` is the one to read, because it says the
control wrapped across the image border and stopped having the union's shape.

### Which trainer fold you run under — decide this deliberately

`trl_repo/` is deliberately behind `main` on `trl/grpo_trainer_qwen3.py`: `main` imputes
each group's mean for an unscored reward (commit 8489767), `trl_repo` still folds with
`.nansum(dim=1)`, which reads an unscored reward as **0**. The four reference runs were
trained under `nansum`.

`--placebo` needs no trainer change to work — only the two new logging lines live there —
so the placebos can run under either fold, and the choice is an experimental one:

- **`nansum` (what runs today)** matches the reference exactly, and scores the ~2–4% of
  completions with no gradeable observe step as 0 on the auxiliary dimension rather than
  leaving them neutral. For `mean_in` (level 0.04, `sd_within` 0.0098) that 0 is a ~4σ
  penalty. It is why `--placebo length` is anchored at the completion cap rather than
  written as `-n/1000`: under the latter, 0 would be the *highest* possible length score,
  so "produce no groundable observe step" would become the best move on the very
  dimension this experiment is measuring. The anchor is a constant, invisible to the
  advantage, and makes the unscored read a floor instead of a ceiling.
- **`impute` (main)** is the correct semantics, and makes the placebo runs differ from
  the reference in one more way than direction — unless the reference is re-run too.

Shipping `--placebo` to `trl_repo/` does **not** require the full `patch_trl_qwen3.sh`
(which is all-or-nothing and would carry 8489767 with it). Four files suffice:
`trl/rewards/placebo_rewards.py`, `trl/rewards/__init__.py`, `trl/scripts/utils.py` →
`trl_repo/trl/...`, and `trl/grpo_vlm_qwen3.py` →
`trl_repo/examples/scripts/grpo_vlm_qwen3.py`. Adding
`trl/grpo_trainer_qwen3.py` is what buys the new logging *and* the `impute` fold — one
decision, not two.

### Four GPUs

`--placebo` still needs DINO, but DINO peaks at 1.6 GB, so it fits on vLLM's GPU:

```bash
bash launch_grpo_qwen3_overlap_colocated_job.sh --placebo length \
    --num-gpus 4 --share-sidecar-gpu --grad-accum 16 --lora-targets q_proj,v_proj
```

`--grad-accum 16` is load-bearing. The generation batch is
`per_device × train_procs × grad_accum`, so 3 training procs at grad_accum 8 put 24
sequences (3 prompts × 8 generations) behind each optimizer step instead of 48 — half the
prompts per step, and a different meaning for the LR schedule, against an 8-GPU reference.
16 restores 48. The banner prints `gen_batch`; check it reads 48.
