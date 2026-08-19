# Next session: make the mini-benchmark able to resolve a 0.03 effect

Three tasks, in this order:

1. **Raise the sample size to 300 on the NATURAL benchmarks only.** Non-natural stays
   at 100.
2. **Write a tool that re-runs every checkpoint for which a result already exists**, so
   the historical curve is regenerated at the new sample size.
3. **Save per-sample results** and bind them to (run, step) so paired tests are
   possible.

Uri runs the jobs. Your job is the code.

---

## Why

`bench/natural/mean` averages 5 benchmarks, each scored on `SAMPLE_N = 100` items. That
gives se ≈ 0.020 on the mean and **se ≈ 0.028 on a difference between two checkpoints**.
The effects being chased are 0.02–0.035:

```
sft-coldstart      0.7177        overlap-8k - grpo-no-saliency = +0.0342  ->  1.22 sigma
grpo-no-saliency   0.7162        two identical configs gave 0.750 and 0.737
saliency-r1        0.7390
overlap-8k         0.7504
auroc w0.11        0.7544
```

Nothing in that table is distinguishable from noise. At n=300 (paired, ~15% discordance)
se on a difference falls to roughly 0.010, which makes a 0.03 effect a 3σ result instead
of a 1.2σ one.

**Constraint that shapes every design choice below: 1 GPU, 1 hour, backfill.** This is
deliberate — longer or wider jobs wait hours in the queue. Do not "fix" it by raising
`--duration`. `run_bench_eval.sh` is already built for it: it banks finished *suites*
under `bench_eval/partial/step-<N>/` and keeps the merged model, so the next job resumes
at the first suite still owed. A TIMEOUT is the designed drain, not a failure.

---

## Task 1 — natural to 300

### Where the number lives

`eval_mini/make_mini_tasks.py` writes `minisample.py` next to the generated task yamls,
with a single module-level `SAMPLE_N = {n}` (line ~49) consumed by `take()` and
`take_groups()`. `run_bench_eval.sh` regenerates those configs into
`<run>/bench_eval/tasks/` on every job, so there is one place to change and no stale
copies to chase.

The natural / non-natural split is `make_mini_tasks.py --print-tasks natural` and
`--print-tasks nonnatural`; `run_bench_eval.sh` reads both (lines 170–176) and pulls
`mmerealworld_mini` into its own suite with `--max-pixels 3211264`.

Natural: `mme`, `mmerealworld`, `mmstar`, `pope`, `realworldqa`.

### The problem

`process_docs: !function minisample.take` passes no task name, so a single global
`SAMPLE_N` cannot differ per task. Do **not** solve this with a global that the yaml
sets — `process_docs` gets only the dataset.

Simplest approach that keeps the generated file readable: emit one sampler function per
size and reference the right one from each yaml, e.g. `take_100` / `take_300` (and
`take_groups_300_for_mme` for the grouped one). `make_mini_tasks.py` already emits a
per-task `take_for_{task}` for grouped benchmarks (line ~131), so the pattern exists.
Drive it from a `SUITE_N = {"natural": 300, "nonnatural": 100}` dict with a
`--natural-n` / `--nonnatural-n` CLI override.

### Two things to know

- **MME is grouped.** `take_groups` draws whole yes/no pairs and counts in rows, so
  `SAMPLE_N=300` means 150 images. That is intended; keep the row-based counting so a
  grouped benchmark costs the same as an ungrouped one.
- **The 300-item sample is a strict superset of the 100-item sample.** Both are
  `dataset.shuffle(seed=1234).select(range(N))` with the same fixed seed, so the first
  100 items are identical. This matters twice: old per-sample results remain valid for
  those items, and any item-level comparison between an old N=100 run and a new N=300
  run can be restricted to the shared 100 without re-running anything.

### Cost, so the estimate is honest

Items go 1300 → 5×300 + 8×100 = 2300 (1.8×). Generation is ~20 min of a ~30 min job (the
rest is weight load and LoRA merge), and the natural tasks are the short-answer half, so
expect the natural suite to grow from ~8 min to ~25 min of generation. It still fits a
1 h slot **only because the merge is banked across jobs** for the same checkpoint.

If it turns out not to fit, do not lengthen the job — **shard the banking unit from
suite to task**. `run_suite` (line ~250) already takes an arbitrary task list and banks
by tag, so the change is a loop over tasks with `run_suite "$MERGED" "$task" "$task"`,
plus retuning `SUITE_MINUTES` down to a per-task estimate. That turns more items into
*more jobs of the same length*, which is what the scheduling situation rewards.

---

## Task 2 — re-run everything that already has a result

Changing `SAMPLE_N` invalidates every existing aggregate, so the historical curve has to
be regenerated or the old and new numbers will be silently mixed.

### Find the work

Results live in three shapes:

```
checkpoint/<run>/bench_eval/step-<N>.json              # in-training curve
outputs/bench_one/<run>-cp<step>/bench_eval/step-<step>.json   # one-off checkpoint evals
outputs/bench_baselines/<name>/bench_eval/step-0.json  # the 5 baselines
```

Each is `{"metrics": {...}, "step": N}`. `outputs/bench_baselines/<name>/bench_eval/
base_model.txt` names the model a baseline was scored from.

### What the tool must do

- Enumerate every existing `step-*.json` and resolve, for each, the model it would need:
  a LoRA checkpoint dir (`checkpoint/<run>/checkpoint-<step>`) or a merged model path.
- **Report which are no longer re-runnable.** `CKPT_KEEP_EVERY` (default 100) prunes
  checkpoints, so some steps that have results no longer have weights. Print that list
  explicitly rather than silently skipping — a curve with holes that nobody was told
  about is worse than a shorter curve.
- Emit a work list and dispatch through the existing path
  (`watch_bench_evals.sh` / `run_bench_eval.sh`, 1 GPU, 1 h, resumable). Do not write a
  second dispatcher.
- Be resumable and idempotent: re-running the tool must skip work already redone.
- **Never overwrite the N=100 results.** Write the new ones so both survive — either
  `step-<N>.json` gaining an `"sample_n"` field and living under
  `bench_eval_n300/`, or a filename that encodes N. Whatever you choose, make
  `bench_eval.py --collect` refuse to mix sample sizes into one curve, and make every
  consumer (`report_bench_evals.sh`, `watch_bench_evals.sh`, the wandb push) state which
  N it is reading. The single most likely way this work goes wrong is a plot with 100-
  and 300-item points on the same line.

Order the work list newest-checkpoint-first within each run, and put the five baselines
first overall — without a re-scored baseline none of the new numbers can be compared to
anything.

---

## Task 3 — per-sample results

**lmms-eval is already writing them.** `launch_lmms_eval_job.sh` passes `--log_samples`
and `--output_path` (lines 274–276), and the files are on disk now:

```
$VLM_REASONING/results/lmms_eval/<MODEL_SLUG><MNT><PX><TAG>/checkpoint__<name>/
    <timestamp>_samples_<task>_mini.jsonl        # one row per item
```

Schema (verified on `pope_mini`):

```
doc_id, doc_hash, target, input,
resps            full model response
filtered_resps   the extracted answer
<metric>         {"question_id":…, "score": 0.0/1.0, "prediction":…, "ground_truth":…}
```

So this task is **harvesting and indexing, not generating**. Nothing new needs to run on
a GPU.

### What to build

- At collect time, bind the sample files to `(run, step)`. `run_bench_eval.sh` calls
  `bench_eval.py --collect --run-dir --step --results <output dirs>` with exactly the
  directories the suites wrote to, so that is the hook — the mapping is unambiguous
  there and irrecoverable later, because the output dir is keyed by model slug and
  accumulates timestamps from every re-run.
- Store a reduced per-item record next to `step-<N>.json`:
  `{task, doc_hash, doc_id, score, prediction, target}` plus the response **truncated**
  (say 400 chars) — gzipped. Full responses across 13 tasks × 300 items × 30 checkpoints
  × several runs will not survive the quota; the truncation keeps enough to see *what*
  changed on items that flipped.
- **Key on `doc_hash`, not `doc_id`.** `doc_id` is a position in the sampled subset and
  shifts when `SAMPLE_N` changes; `doc_hash` is stable, which is what makes an N=100
  result poolable with an N=300 one over the shared items.
- Retro-harvest what already exists where the mapping can still be recovered, and be
  explicit about what cannot.

### The payoff, which is the point of all three tasks

With per-item scores keyed by `doc_hash`:

- **Paired tests.** The same fixed-seed items are used for every checkpoint, so McNemar
  or a paired bootstrap over items applies. That alone takes se on a difference from
  0.028 to ~0.017 at the current n, for zero GPU cost — do it before anything else.
- **A pooled item-level endpoint.** `bench/natural/mean` currently averages five task
  means with different item counts (MME is 150 images behind a nonstandard
  normalisation), which makes its standard error hard to state honestly. Pool items and
  weight each equally, and report the mean-of-means alongside for continuity.
- **Which items flipped.** This is the only route to the mechanism question that is
  actually open — whether the natural-bench gain is grounding, brevity, or entropy. See
  [next-reward-experiments.md](next-reward-experiments.md).

---

## One measurement worth more than all of this

You cannot currently separate evaluation noise from **run-to-run (seed) variance**. Uri
has two runs of an identical config that scored 0.750 and 0.737. At n=100 that gap is
fully explained by eval noise — but if seed variance is genuinely ~0.013, then n=300
buys nothing, because the floor is set by the training run and not by the eval.

Scoring the existing `wov04 8k` replicate at the new sample size answers it, and the
checkpoints already exist. Put it early in the Task 2 work list.

## Housekeeping

- `/lustre/fs1` project quota for `nvr_israel_rlop` has been hitting 100T/100T, which
  breaks running jobs. Check `df -h /home/uberger/scratch` before a re-run sweep, and
  remember the merged model is 16 GB and is kept on disk for as long as a checkpoint has
  suites still owed — finer sharding holds it longer.
- Work in a worktree (`./worktree.sh new ...`), never in the central tree; `outputs/` and
  `checkpoint/` are symlinks shared with every other session and every running job.
