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

> **Superseded 2026-09-01.** `trl_repo/` was deliberately behind `main` on
> `trl/grpo_trainer_qwen3.py` when this was written. It no longer is: all fourteen files
> `patch_trl_qwen3.sh` copies are byte-identical to their tracked sources, the trainer
> included, so the patch script is a no-op with respect to this choice and carries no
> hidden change in fold semantics. Run it rather than hand-copying — it is also how the
> second cluster gets this code (`git pull` + patch), and hand-copying is exactly the
> 069bd32 failure the script's §2f comment records. The paragraph below is kept because
> the reference runs' semantics are still what it says.

`trl_repo/` was deliberately behind `main` on `trl/grpo_trainer_qwen3.py`: `main` imputes
each group's mean for an unscored reward (commit 8489767), `trl_repo` still folded with
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

---

# The mask-free rewards — 2026-08-31

Branch `feat/maskfree-rewards`. **`flatness` has since run to 3,990 steps — results at
the bottom of this page, and they split the prediction in half. `mass` has not run.**

## Why, in one table

`mean_in = mean_U(m) / max(m)`, and the denominator is over the *whole* map. Measured on
the val_natural probe, on the quantity GRPO sees (per-completion reward, group mean
removed):

| mask fed to `mean_in` | all unions | union < 0.30 |
|---|---|---|
| the real DINO union | 1.000 | 1.000 |
| in-frame roll (what `--placebo roll` did) | +0.538 | +0.449 |
| **uniform toroidal roll (a genuine relocation)** | **+0.353** | **−0.015** |
| random box of the same area, anywhere | +0.505 | +0.457 |
| **`mean_all(m)/max(m)` — no mask at all** | **+0.690** | **+0.678** |

Once the mask is genuinely relocated its score stops predicting the reward. The best
single predictor of `mean_in` is a statistic that never sees a box. Corroboration from
two directions: [sharpness-results.md](sharpness-results.md) finds *"DINO survives in 0
of 4 families. SHARP in 2. MASS in 3"*, and the `mean_in`-trained 8k policy puts **2.3×**
the cold start's attention mass on the image (0.00393 → 0.00893) with an 18% flatter map,
while the `auroc`-trained one goes the other way (−8%) and is the run that damaged
non-natural.

Hypothesis: `mean_in` is an attention-flatness reward, flatness and image mass are
coupled through the softmax, and image mass is the one map property with controlled
evidence of predicting correctness. These two rewards are the experiment that can kill it.

## Flags

```bash
bash launch_grpo_qwen3_overlap_colocated_job.sh --maskfree flatness --num-gpus 7
bash launch_grpo_qwen3_overlap_colocated_job.sh --maskfree mass     --num-gpus 7
```

| flag | reward | isolates |
|---|---|---|
| `--maskfree flatness` | `mean(m)/max(m)` over the grid — `mean_in` with the union replaced by the image | shape only; **scale-invariant**, so it cannot be raised by attending to the image more |
| `--maskfree mass` | `log(sum(m)) + anchor` — the softmax mass the step's think-tokens put on image patches | scale only; the variable `sharpness-results.md` says predicts correctness |

The pair is deliberate: `flatness` is scale-invariant and `mass` is scale-only, so between
them they span the two directions `mean_in` moved in. They correlate +0.38 within a group
at the cold start, so they are not redundant either.

Both take `think_overlap_reward`'s slot, so `--reward_weights` is unchanged. Attention map
only (the weights were measured on it). Refuses to combine with `--placebo`.

## Neither one calls Grounding-DINO

That is the point, and it is enforced by not starting the server rather than by trusting
the reward: with `--maskfree`, the launcher skips `serve_grounding_dino.py`, omits
`--dino_api_base`, and gives vLLM GPU 0 with training on 1..N−1. `test_maskfree_reward_cpu.py`
runs every case with `overlap_rewards._dino_boxes` replaced by a bomb.

Cost removed: **16.6 s of a 40.5 s optimizer step**, plus a GPU. The layer-22 re-forward
(1.0 s) and the T5 observe-step segmentation are *kept*, so the maps scored are the ones
`think_overlap_reward` would have scored and this stays a control on the reference rather
than a third experiment.

**`--num-gpus 7` is load-bearing**, for the reason `--share-sidecar-gpu` needs
`--grad-accum 16`: `gen_batch = per_device × train_procs × grad_accum`, and 8 GPUs with no
DINO give 7 training procs and `gen_batch` 56 against the reference's 48. 7 GPUs give 6
procs and 48. The banner prints `gen_batch` and warns when `train_procs != 6`.

## The scored set — the one place this is not a single-variable change

The overlap reward is unscored where DINO grounds nothing; without DINO that question
cannot be asked, so a completion counts here when it has any observe step with a
positive-maximum map. A superset in principle. Measured on val_natural:

| | |
|---|---|
| observe steps with a map | 874 |
| ... that DINO grounded | 859 (98.3%) |
| completions the DINO reward scored | 231 / 240 (96.2%) |
| completions scored **mask-free** | 231 / 240 (96.2%) — **+0** |

Identical. `--maskfree-parity` re-imposes the DINO gate (value stays mask-free, only the
gate changes) at the full DINO cost; off by default because the measurement says it buys
nothing.

## Weights

`w = 0.4 × sd_within(mean_in) / sd_within(variant)`, from `overlap_metric_spread.py`,
which now prints both rows by importing `trl/rewards/maskfree_rewards.py` itself:

| variant | level | sd_within | w at w_ref=0.4 | launcher default |
|---|---|---|---|---|
| `mean_in` (reference) | 0.0388 | 0.0071 | 0.400 | — |
| `flatness` | 0.0513 | 0.0064 | 0.447 | **0.45** |
| `mass` | 12.1621 | 0.4586 | 0.006 | **0.006** |

`flatness` landing next to `mean_in`'s own 0.4 is structural: it is the same statistic on
a superset of the same patches. One-corpus caveat applies as everywhere else — read these
as ±25%, the way `roll`'s 0.32 turned out to be bracketed by [0.32, 0.52].

Two traps this cost a debugging pass each, both now closed in code:

- **`mass` must not read `_decode_step_map`.** That decoder returns the map normalised to
  its own peak, which is correct for every metric offered until now (all four are
  scale-invariant) and wrong for the first one that is not: a peak-normalised map sums to
  ~8, not to attention's ~0.004. The spread tool reads the probe's stored `image_mass`.
- **The anchor has to clear the measured minimum.** `log(sum(m))` is negative, and a
  `.nansum` fold reads an unscored reward as 0 — which would make "produce no gradeable
  observe step" the best move. The first draft's anchor of 8.0 needed 8.87 and went
  negative on the lowest ~1% of real steps. It is **18.0**, against min 1.40e-04 over
  13,648 observe steps: four orders of magnitude of margin. The guarantee is empirical,
  not structural.

## What to log

`maskfree/flatness`, `maskfree/mass`, `maskfree/peak`, `maskfree/mean`, `maskfree/n_steps`.
**Both statistics are recorded under either kind**, on purpose: a `--maskfree flatness`
run whose *mass* rises is the finding this experiment exists to produce, and it is
invisible if only the scored quantity is logged.

## Pre-registered predictions

- Both keep `train/frac_reward_zero_std` ≈ 0.000, like every other continuous reward.
- `flatness` raises `maskfree/mass` without being rewarded for it — the coupling claim.
- If the mechanism is right, `flatness` ≈ `mean_in` on both benchmark suites, at ~60% of
  the step time and one fewer GPU.
- `mass` is the sharper test and the one that could go wrong loudly: it has an unbounded
  degenerate direction (dump the whole softmax row on the image), which `flatness` does
  not. Watch `maskfree/peak` and `train/entropy` for it.
- If **neither** moves the benchmark, the union's area/shape mattered after all and the
  flatness reading of `mean_in` is wrong.

---

# Result: `flatness` bought the natural gain and none of the non-natural one — 2026-09-03

`--maskfree flatness` ran to 3,990 steps. Scored against the two 8k runs that bracket it,
all three at the **same** `n300_100` sample profile, which is the only way these are
comparable — the top-level `bench_eval/step-*.json` files are an older profile and mixing
the two is the trap [bench-precision.md](bench-precision.md) exists to name.

| run | bench_eval dir | natural early → late | Δ | non-natural early → late | Δ |
|---|---|---|---|---|---|
| `mean_in` wov0.4 | `…-overlap__wov0.4_2head_trmean_saliency_r1_8k/` | 0.7266 → 0.7494 ±0.0042 | **+0.0227** | 0.5184 → 0.5354 ±0.0045 | +0.0170 |
| `maskfree flatness` w0.45 | `checkpoint/…-maskfree-flatness/` | 0.7265 → 0.7427 ±0.0064 | +0.0162 | 0.5186 → 0.5232 ±0.0046 | **+0.0045** |
| `mean_in_v2` wov0.033 | `checkpoint/…_saliency_r1_8k_mean_in_v2/` | 0.7266 → 0.7276 ±0.0040 | **+0.0009** | 0.5168 → 0.5457 ±0.0080 | **+0.0289** |

early = steps ≤ 600, late = steps ≥ 2400; ± is the spread **across late checkpoints**, not
a seed CI. All three start from the same 0.7266 natural, which is the check that the
profile really is matched.

**Verdict on the pre-registered predictions.**

| prediction | outcome |
|---|---|
| `frac_reward_zero_std` ≈ 0 | **held** — logged 0.000 at step 3990 |
| `flatness` raises `maskfree/mass` unrewarded | **held, strongly** — see the coupling note below |
| `flatness` ≈ `mean_in` **on both suites** | **half** — natural yes, non-natural no |
| neither moves the benchmark | **refuted** — `flatness` moved natural |
| `mass` is the sharper test | **unrun** |

The half that failed is the informative half. On natural, `flatness` recovers 71% of
`mean_in`'s gain and the 0.0065 shortfall is **below** the ~0.013 seed variance
[bench-precision.md](bench-precision.md) measured between two runs of one identical
config — i.e. indistinguishable. On non-natural it recovers 26%, and the 0.0225 gap to
`mean_in_v2` is well above that floor.

So the two factors of `mean_in` are not redundant, they are **specialised**:

- `flatness` = `mean(m)/max(m)`, box-blind → the **natural** gain
- `mean_in_v2` = `mean_U(m)/mean(m)`, the box-aware half → the **non-natural** gain, and it
  is the best of the three there while doing nothing at all on natural
- `mean_in` = their exact product → the only one that gets both, and the best at neither

## `mean_in` is already a balanced blend, and nobody chose the balance

`mean_in = mean_in_v2 × flatness` holds step by step (see
[peak-location-results.md](peak-location-results.md)), so in logs the two channels are
additive. Within-group variance decomposition of `log mean_in` on the cross-run probe,
per completion with the prompt mean removed — the quantity the advantage keeps:

| model | sd(log v2) | sd(log flat) | corr | var share v2 | var share flat | 2·cov |
|---|---|---|---|---|---|---|
| cold start | 0.1273 | 0.1150 | **+0.034** | 0.53 | 0.43 | 0.03 |
| `mean_in` 8k | 0.1276 | 0.1221 | +0.163 | 0.45 | 0.41 | 0.14 |
| `mean_in_v2` set_a cp1000 | 0.1297 | 0.1179 | +0.165 | 0.47 | 0.39 | 0.14 |

Two nearly **orthogonal** channels (r = +0.03 at the cold start) carrying nearly **equal**
pressure. `mean_in` hard-codes exponents (1, 1) and that split is an accident of the
arithmetic, not a choice anyone made.

Which makes the obvious next reward a **re-weighting, not a new metric**:

    R(α, β) = mean_in_v2^α · flatness^β        reward ∝ α·log v2 + β·log flat

`(1,1)` = `mean_in`, `(1,0)` = `mean_in_v2`, `(0,1)` = `flatness` — three points already
measured. Both sds are in hand, so the weights follow immediately. Minimal
implementation: run `mean_in_v2` and `maskfree flatness` as **two reward terms with
independent weights** (0.033 and 0.45, both already calibrated); the only code change is
lifting the launcher's refusal to combine `--maskfree` with the DINO reward.

Where to aim, from the dose–response: `mean_in` runs the `v2` channel at ~half dose and
gets +0.0170 non-natural against `v2`'s +0.0289 — close to linear, so **α > 1 should buy
non-natural**. On natural `mean_in`'s +0.0227 beats both specialists *and* beats the
linear prediction (0.5×0.0162 + 0.5×0.0009 = 0.0086), which suggests an interaction — but
that gap is about one seed-sd, so it is a hypothesis, not a finding.

Pre-register the next run: if `R(1.5, 1)` moves non-natural toward +0.029 while natural
holds near +0.023, α is a free lunch. If natural collapses to `v2`'s +0.001 the moment α
rises, the equal balance was load-bearing and the interaction reading was right.

A variant worth naming and rejecting: denominator = `max` over patches **outside** the
box. It de-peaks only where a peak is unwanted and lets the map stay sharp inside, which
sounds right — but it is *more* hackable, not less: covering the current peak with a box
collapses the denominator, and that is exactly the ring migration `mean_in` set_a cp2000
found (union coverage of the outer ring 0.374 → 0.527 at unchanged total area).

## The coupling claim held

Logged at the end of the `flatness` run: `maskfree/flatness` 0.0716, `maskfree/mass`
13.664 (= `log(sum m) + 18`, so image mass 0.0131), `maskfree/peak` 0.00139,
`frac_reward_zero_std` 0.000. The reward is scale-**invariant** and mass rose anyway —
against the cold start's 0.00393 that is ×3.3, where `mean_in` 8k reached ×2.3. Read the
ratio, not the level: the logged numbers are on the training corpus and the cold-start
references are on `val_natural`.

**Caveat on matched pressure.** `w = 0.45` was set from a cold-start `sd_within(flatness)`
of 0.0064. The run's own `think_maskfree_reward/within_group_std` at step 3990 is
**0.01077**, 1.7× that, so the realised pressure was not exactly `mean_in`'s. The
comparator runs' wandb directories are not on this machine, so the same figure could not
be read for them. This does not touch the *direction* of any result above, but a 1.7×
pressure difference is inside the range that could move a 0.006 benchmark gap.

## What this does and does not settle about DINO

It does not license dropping DINO. The box term is what carried non-natural transfer, and
`flatness` is the run that lost it. What it opens instead: a mask-free reward can be
trained on **non-natural images**, which no run has done — every run above trained on the
natural 8k and reached non-natural by transfer. That is the experiment only a DINO-free
reward can run, and it is a cleaner test of whether the non-natural gain needs a box or
just needs the data.
