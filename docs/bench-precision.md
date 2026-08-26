# Making the mini-benchmark able to resolve a 0.03 effect

Supersedes `next-bench-precision.md`, whose three tasks are built. This is the
reference for what exists, what it costs, and the one decision left.

`bench/natural/mean` averaged 5 benchmarks scored on 100 documents each, giving
se ≈ 0.028 on a difference between two checkpoints against effects of 0.02–0.035 —
so nothing in the table below was distinguishable from noise:

```
sft-coldstart      0.7177        overlap-8k - grpo-no-saliency = +0.0342  ->  1.22 sigma
grpo-no-saliency   0.7162        two identical configs gave 0.750 and 0.737
saliency-r1        0.7390
overlap-8k         0.7504
auroc w0.11        0.7544
```

Three things were built. **The first one is free and already done.**

---

## 1. Pairing — done, no GPU time, no new measurement

Every checkpoint is scored on the *same* fixed-seed sample, so the comparison was
always paired; it just was not being analysed that way. And lmms-eval has been
writing one row per document all along, in `--log_samples` files beside each
`results.json`. So this was harvesting, not generating.

```fish
python bench_samples.py --retro          # done: 186/186 results, 25 s, 24 MB
python bench_samples.py --index          # what is stored, and what can be joined
python bench_samples.py --compare grpo-no-saliency overlap-8k --suite natural
python bench_samples.py --compare A B --flips 10     # which documents changed
```

se on a difference is now **0.015–0.017 against 0.028 unpaired** — the whole gain
the plan hoped n=300 would buy, at the sample size already on disk. Discordance is
7–11%, better than the 15% assumed.

The table above, redone paired over the 400 natural items, against
**`sft-coldstart`** — the model all three overlap runs actually start from
(2026-08-19):

```
model                pooled item acc     diff              95% CI    disc   McNemar p
sft-coldstart                 0.7225        —                   —       —           —
grpo-no-saliency              0.7175  -0.0050  [-0.0325, +0.0225]    8.5%      0.8642
saliency-r1                   0.7375  +0.0150  [-0.0175, +0.0475]   10.5%      0.4408
overlap-8k                    0.7475  +0.0250  [-0.0075, +0.0575]   11.5%      0.1839
auroc w0.11 @3990             0.7500  +0.0275  [-0.0050, +0.0600]   10.8%      0.1263
```

**Nothing here is significant.** Halving the standard error was not enough: the
best run's advantage over the model it started from is +0.0275 with a CI that
still contains zero. That is the honest reading, and it makes the case for n=300
stronger rather than weaker — at 300 the same effect would have a half-width of
~0.019 and would separate.

The reference matters, and it is easy to get wrong. Against `grpo-no-saliency` the
same data gives `overlap-8k +0.0300 [+0.0000, +0.0600] p=0.073` and
`auroc w0.11 +0.0325 [+0.0025, +0.0625] p=0.053` — a CI that excludes zero. But
`grpo-no-saliency` is GRPO from **vanilla Qwen3-VL-8B-Instruct**, not from the
cold start (check `bench_eval/base_model.txt`), so that comparison carries the
cold-start SFT as well as the reward, and the trap is already written down in
[next-reward-experiments.md](next-reward-experiments.md). It is quoted here only
because the difference between the two tables is the whole finding.

(These accuracies differ slightly from `bench/natural/mean` — 0.7475 against
0.7504 for overlap-8k — because this pools items and weights each equally, while
the curve averages four benchmark means, and because pope contributes per-item
accuracy here against corpus F1 there. `--compare` prints both.)

The store is `<run>/bench_eval/[<profile>/]samples/step-<N>.jsonl.gz`: one header
line, then `{task, doc_id, item, metric, score, pred, target, resp}` per document,
with the response truncated to 400 characters.

### Three corrections to the original plan

**`doc_hash` does not work.** It is the constant `74234e98…` on every row of every
task — lmms-eval never computes it for these. The key here is `item`, a digest of
the document's own content (question_id, prompt, media, target). It was checked to
be *identical across checkpoints* for all five natural benchmarks, which is what
makes the pairing legitimate; `--compare` joins on `doc_id` and asserts `item`
agrees, and refuses the comparison outright if it does not.

**Three benchmarks' headline is not the mean of their rows.** pope and p3 report a
corpus-level F1; MME sums per-category averages. A paired test over their items is
a test of *per-item accuracy*, which is a fine endpoint but is not the number the
curve plots. `item_decomposes=False` marks them in `benchmarks.py` and `--compare`
says so in its output.

**Four tasks record no per-row score at all** — mme-realworld, mathvision,
mmmu_pro and p3 store the prediction and the answer and compare them only during
aggregation. `dict_score()` reproduces the comparison, which is exact for the
multiple-choice tasks these are.

**Some item keys are degenerate.** illusionvqa asks the same question of 1000
different images and records neither the image nor an id, so its `item` collapses
to 4 distinct values in 100 rows; scienceqa and pope have one duplicate each.
These are still joinable by `doc_id` at a fixed sample size, which is the only
comparison they are used for — all of them are in the non-natural suite, which
stays at 100. `--index` prints them.

---

## 2. Sample profiles — the natural suite can now be 300

Sizes are per suite and overridable per task. `process_docs` receives only the
dataset, so one sampler function is emitted per distinct size and each task's yaml
names the one it wants (`take_100`, `take_300`, `take_for_mme`).

```fish
python eval_mini/make_mini_tasks.py --out-dir DIR --natural-n 300 --nonnatural-n 100
python eval_mini/make_mini_tasks.py --natural-n 300 --task-n mmstar=200 --print-sample-n
```

A **sample profile** is the pair of suite sizes, and it is the one thing that must
never be mixed within a curve. It decides:

| | 100/100 (default) | 300/100 |
|---|---|---|
| results | `bench_eval/step-N.json` | `bench_eval/n300_100/step-N.json` |
| banked units | `bench_eval/partial/` | `bench_eval/n300_100/partial/` |
| per-item rows | `bench_eval/samples/` | `bench_eval/n300_100/samples/` |
| WandB keys | `bench/*` | `bench_n300_100/*` |
| baseline run id | `bench-baseline-<name>` | `bench-baseline-<name>-n300_100` |

The default profile keeps the flat layout on purpose: 258 existing results live
there, the trainer's `BenchmarkResultsCallback` globs `bench_eval/step-*.json`
non-recursively while a run is training, and an eval job was in flight when this
was written. Nothing had to move.

The namespaces are separate *top levels* rather than a deeper path segment
(`bench/n300/natural/mean`) precisely so that a `bench/*` panel cannot reach both
sizes at once. Every step file records `sample_n`, and `bench_eval.py --collect`
checks it against the `n-samples` each results file reports — a 100-document
result cannot land on a 300-document curve **in either direction**, and neither
can the reverse.

---

## 3. Re-scoring the history

```fish
python rerun_bench_evals.py                       # the work list and the bill
python rerun_bench_evals.py --every 500 --dispatch # submit one job
python rerun_bench_evals.py --every 500 --watch 300
```

It resolves the model each existing result would need, reports the ones whose
adapter `CKPT_KEEP_EVERY` has pruned, orders baselines first (`--first SUBSTR` to
promote a run), and dispatches through `watch_bench_evals.sh` — the same 1 GPU,
1 hour, resumable job that produced every other point. It is idempotent: work
already redone is skipped.

**The non-natural suite is carried over, not re-run.** It stays at 100 documents,
so its result at the new profile is the *same* evaluation of the same model on the
same documents. `--carry-over` (on by default) writes `run_bench_eval.sh`'s own
banking markers pointing at the existing `results.json`, and the step file records
`carried_over` so a repeated measurement is never read as an independent one.
This removes about a third of the sweep. Where a checkpoint never got a
non-natural result — 13 of set_c's 53 — it is generated, filling a hole that has
been on the curve all along.

---

## What it costs, measured

Over the 295 mini-suite `results.json` on disk, single-GPU wall clock fits

```
minutes = 1.5 + output_tokens / 3310
```

with the same token rate for all three suites (natural 3323, mme-realworld 3304,
non-natural 3311). Which gives, on 1 GPU:

| unit | n=100 | n=300 | measured at n=100 |
|---|---|---|---|
| natural suite (4 tasks) | 40 min | **115 min** | median 29.8, p90 39.6 |
| mme-realworld (3.2 MP) | 14 min | 37 min | median 11.2, p90 22.1 |
| non-natural suite (8 tasks) | 32 min | — | median 23.7 |
| mmstar alone | 22 min | **66 min** | |

Budgets carry a ×1.3 margin, which puts them at the measured p90: a unit started
and killed at the wall clock banks nothing and is repeated from scratch, while a
unit not started costs only a resubmission.

**The natural suite at 300 cannot finish in a one-hour job**, so it would be
started, killed and repeated forever. The banking unit is therefore derived from
the cost and the job's own wall clock rather than set by hand:

```
--bank auto   (default)  suite while the suites fit, task once they do not
--bank suite             3 units    right at n=100, and at n=300 on 8 GPUs
--bank task              13 units   required at n=300 on 1 GPU, costs 4 extra loads (~6 min)
```

`auto` exists because the pair (sample size, granularity) is coupled, and setting
it wrong is silent and fatal — a job that cannot finish its own unit banks
nothing, writes no step file, and is resubmitted forever with nothing in the log
to say why.

Units are attempted largest-first, and one that does not fit the remaining clock
is **skipped so the next is tried**, rather than ending the job — so the merge is
paid once and the small units still fill the tail of a window.

**One unit is over even at task granularity: mmstar at 300 is budgeted 66 min
against a 55 min window.** A guard that took that literally would decline to start
it on every job — the same forever-loop. So a unit's requirement is capped at the
whole window: it is attempted at the start of a fresh job, where it has the full
allocation, and either finishes (median ~50 min) or is killed and retried. The
banner warns when this is happening. `--task-n mmstar=200` (44 min) stops paying
for the retries, at the cost of a benchmark measured on fewer documents than its
neighbours.

---

## Where 300 is now the default

Every path that scores a checkpoint is at **natural = 300, non-natural = 100**:

| | default | cost per checkpoint |
|---|---|---|
| a finished run (`watch_bench_evals.sh`, pointed at its output dir) | 300/100, `--bank auto` → per task | ~3.6 GPU-h, ~4 one-hour jobs |
| named checkpoints (`run_bench_eval_steps.sh`, 8 GPUs) | 300/100, per suite | ~28 min |
| the historical sweep (`rerun_bench_evals.py`) | 300/100, per task | ~2.9 GPU-h |

`--natural-n 100` restores the old size anywhere.

**Nothing runs during training.** The colocated launcher used to start the
dispatcher alongside the run; it no longer starts anything, and no flag turns that
back on. Benchmarks are scored afterwards, by pointing `watch_bench_evals.sh` at a
run's output directory — its state is on disk (a checkpoint is done once
`bench_eval/step-<N>.json` exists), so starting it after the fact loses nothing and
it backfills the whole run.

That also removes the failure mode this section used to warn about. At 300 an eval
is ~3.6 GPU-h per checkpoint against ~1.6 at 100, and the dispatcher runs one job
at a time — alongside training, checkpoints arriving faster than ~4 hours apart
left the curve permanently behind. Run after the fact and the only cost is wall
clock you are not waiting on. Raise `--every` to score a sparser set of steps.

## The decision left

**How much history to re-score.** The full sweep is 537 single-GPU hours,
~590 one-hour jobs:

```
all 186 checkpoints   537 GPU-hours   586 jobs
--every 200    99      285 GPU-hours   311 jobs
--every 500    52      148 GPU-hours   162 jobs
--every 1000   32       90 GPU-hours    99 jobs
```

Step 0 and each run's final checkpoint are always kept. Most of the difference
buys resolution along the training-step axis, which is not the axis the precision
work is about.

---

## The measurement worth more than all of this

Evaluation noise still cannot be separated from **run-to-run (seed) variance**.
Two runs of an identical config scored 0.750 and 0.737; at n=100 that gap is fully
explained by eval noise, but if seed variance is genuinely ~0.013 then n=300 buys
nothing, because the floor is the training run and not the eval.

Scoring the `wov0.4 8k` replicate answers it. Note it is **not** in the re-run
work list, because that list is checkpoints that already have a result and
`checkpoint/…-overlap__wov0.4_2head_trmean/` has none — it has a single checkpoint
that was merged into the `overlap-8k` baseline. Score it as a baseline, or point
`watch_bench_evals.sh` at it directly with `--natural-n 300 --bank task`.

Once both replicates are harvested, the paired test answers it directly:

```fish
python bench_samples.py --compare overlap-8k <replicate> --natural-n 300
```

Two runs of the same config differing by more than the paired CI is seed variance,
and it is the number that decides whether any of the rest is worth doing.

---

## Housekeeping

- `/lustre/fs1` was at 51T/100T when this was written. The per-item store is 24 MB
  for all 186 harvests; the merged models are 16 GB each and are kept while a
  checkpoint has units still owed, so finer banking holds one longer.
- `test_bench_precision_cpu.py` — 47 checks, no GPU, no lmms-eval clone, no
  cluster. Run it after touching any of this.
