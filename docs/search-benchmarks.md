# The visual-search benchmarks: V*, HR-Bench, SpatialTree — 2026-11-08

`eval_mini/benchmarks.py` carried thirteen benchmarks and none of the ones the
visual-search literature is headlined by. This adds four — `vstar_bench`,
`hrbench_4k`, `hrbench_8k`, `spatialtreebench` — so that a DeepEyes / ERGO /
Mini-o3 / Monet row has something of ours to sit next to, and so that our own
effect can be read on the half of the split where it lives.

It is Prompt 8 of [deepeyes-build-prompts.md](deepeyes-build-prompts.md) and
Prompt 4 of [ergo-build-prompts.md](ergo-build-prompts.md), done once. It is worth
having whether or not either baseline is built.

**Everything here is built, tested on CPU and committed. No GPU job has run,
because this cluster has none** — but three of the four benchmarks have *already*
been scored on every arm at full split, by an earlier, separate path, and §3 is
that table.

---

## 1. The point of this is the resolution, and there were two caps

V\*, HR-Bench 4K and HR-Bench 8K exist to ask whether a model can find a small
thing in a large picture. Scored at a reduced resolution they measure something
else, and they do it without complaining. The images are bigger than their names
suggest — measured off the rows, not read off the paper:

| benchmark | rows | pixels | at the 2.1 M default | at 3.2 M | at 16.78 M |
|---|---|---|---|---|---|
| V\* | 191 | 3.4 MP median, 8.3 max | every image reduced | 89.5% reduced | **not resized at all** |
| HR-Bench 4K | 800 | **16.3 MP** | ÷7.7 | ÷5.1 | **not resized at all** |
| HR-Bench 8K | 800 | **38.8 MP** median, 59.0 max | ÷18.5 | ÷12.1 | ÷2.3 (the model's own limit) |

Two caps sit on the `make_mini_tasks.py` → `run_bench_eval.sh` → `bench_eval.py`
path, and only one of them is visible from this repo:

1. **`lmms_eval/models/simple/qwen3_vl.py:89`** — `max_pixels: int = 2048*32*32`,
   i.e. **2,097,152**. This is the one that bites. It applies to every task that
   does not ask for something else, and it is never mentioned anywhere in our
   launchers.
2. **`vlm_reasoning/scripts/slurm/launch_lmms_eval_job.sh:318`** raises it to
   3,211,264 for `--tasks mmerealworld|hrbench4k|hrbench8k|vstar_bench` — matched
   against the **whole `--tasks` string**, so it fires only when one of those four
   is run alone under exactly that name. A mini task is called `vstar_bench_mini`,
   so it would never have fired; a combined `--tasks a,b` call never fires either.

### How it is disabled

Each of the three image benchmarks carries `max_pixels = NATIVE_MAX_PIXELS =
16,777,216` in the table. That number is **Qwen3-VL-8B-Instruct's own
`preprocessor_config.json: size.longest_edge`** — the largest input the model can
represent (16,384 visual tokens at its 32×32 merged patch). Passing it means "do
not cap below what the model can read", which is the only defensible meaning of
"native resolution" for a model whose processor caps the input itself.

It is **not** 12,845,056. That figure appears in `launch_lmms_eval_job.sh`'s
comments as "Qwen's native cap" and belongs to Qwen2.5-VL's 28×28 patch.

Because `--max-pixels` is per lmms-eval **invocation** and applies to every task in
it, each of the four also gets a banking unit of its own — the same reason
mme-realworld has had one since the beginning. `plan_units()` now **refuses** to
build a unit whose tasks disagree about resolution, rather than silently serving
one of them at the other's setting.

### A third cap, not on this path, that has already produced misread numbers

`build_bench_candidates.py` grades the Figure-1 chain scans at
`--max-image-side 1024`. That is why `coldstart` reads 60.2 on `vstar_mini` there
while the same model reads 79.6 on real V\*. Those are a valid *relative*
ours-vs-coldstart signal under a fixed cap and nothing else; see
[deepeyes-gate.md](deepeyes-gate.md) §3. The mini tasks added here are deliberately
named `vstar_bench_mini` / `hrbench_4k_mini`, not `vstar_mini` / `hrbench4k_mini`,
so the two can never be confused in a filename, a WandB key or a conversation.

## 2. What is in the table, and why each field is what it is

| | `vstar_bench` | `hrbench_4k` / `hrbench_8k` | `spatialtreebench` |
|---|---|---|---|
| yaml | `vstar_bench/vstar_bench.yaml` | `hrbench/hrbench{4,8}k.yaml` | `spatialtreebench/spatialtreebench.yaml` |
| metric | `vstar_overall_acc,none` | `average,none` | `SpaTreeBench,none` |
| **scale** | **100.0** | **1.0** | **1.0** |
| in_mean | no | no | no |
| on_demand | yes | yes | yes |
| max_pixels | 16,777,216 | 16,777,216 | — (video) |
| split | 191 | 800 | 6,300 |

**`scale` was taken from a real `results.json`, and the two families disagree**:
`vstar_overall_acc` is a *percentage* (its aggregation multiplies by 100),
`hrbench`'s `average` is a *fraction*. Neither yaml says so. Both were then
re-derived independently by driving each task's own `process_results` and
aggregation over an all-correct and an all-wrong run — see
`test_search_benchmarks_cpu.py --lmms-eval`, which observes the top and the bottom
of the scale rather than inferring them from one mid-range number.
`spatialtreebench` has never been run anywhere in this repo, so its 1.0 comes from
the same direct exercise of its aggregation, not from a file.

**`in_mean=False`.** Adding four benchmarks to `bench/natural/mean` would redefine
the headline metric on 339 existing step files. The four are reported as their own
`bench/natural/<task>` curves, which is what a baseline row needs anyway.

**`on_demand=True`** keeps them out of every routine one-hour eval job — and, more
importantly, out of `as_suite_n`'s profile reduction. Without the second part,
V\*'s 191-row split inside a `--natural-n 300` suite would make the natural size
the *pair* `(191, 300)`, every run's profile would rename from `n300_100` to
`n191-300_100`, and 339 step files plus the `bench_n300_100/*` WandB namespace
would be orphaned by a benchmark nobody asked that job to run. `split_n=191`
clamps V\*'s asked-for size so `bench_eval.py`'s profile check still passes.

**`item_decomposes=False`** on HR-Bench: its headline `average` is the mean over
`cycle_category` of the per-cycle means, not the mean of the rows. Its per-row
verdict is called `gpt_score`, which `bench_samples.dict_score` now reads — without
that, every HR-Bench row would have been stored unscored and the per-item store
would silently have been empty.

### `spatialtreebench` comes with four caveats and one blocker

1. **It is not the TreeBench of ERGO's table.** lmms-eval's own README claims the
   name `TreeBench` maps to this task. For our purposes that is wrong: ERGO reports
   **HaochenWang/TreeBench** (arXiv:2507.07999, <1k rows, csv, image-only) and this
   task is **LongfeiLi/SpatialTree-Bench** (arXiv:2512.20617, 6,300 rows, 8.0 GB of
   images and video). lmms-eval has **no task for the former**. A number produced
   here must not be put in a TreeBench column.
2. **Its aggregate scores a missing leaf as 0.** The root is a weighted mean over
   27 leaves and `calculate_node_score` does `avg_leaf_scores.get(name, 0)`.
   Measured: a run that is *perfect* on every leaf it covers but misses 7 of 27
   reads **0.638**, not 1.000. That is a 36-point bias in the direction that looks
   like a worse model. Hence `stratify_json=("extra_info", "spatree")`, which
   balances the draw over the deepest `spatree*` key inside a JSON-encoded column —
   `stratify_by` cannot reach it, because it indexes `dataset[key]` directly.
3. **374 of its 6,300 rows are judged by an LLM** (`gpteval`). Without an API key
   those rows score 0 silently. `needs_judge=True` records it.
4. **814 rows are video** (up to 32 decoded frames each) and 5,486 are images, so
   `max_pixels` does not govern its sampling and is deliberately unset.

**Blocker:** its data is not staged. `video: True` in its yaml makes lmms-eval
`snapshot_download` the repo and unzip `images.zip` + `videos.zip` (2.08 + 5.94 GB,
~16 GB on disk after extraction) into `$HF_HOME/SpatialTreeBench/` on first use,
which a compute node with `HF_HUB_OFFLINE=1` cannot do. It has to be staged from a
login node (§5). Given (1), do that only if a SpatialTree row is actually wanted.

## 3. What is already measured — full splits, every arm

These were produced by `launch_lmms_eval_job.sh --tasks <one task>` before this
work, which is the one spelling that triggered the 3.2 M default. They are **full
splits** (191 / 800 / 800), `--r1-mode`, greedy, `max_new_tokens 4096`, and they
reduce through `bench_eval.py` unchanged now that `TASK_ALIASES` maps lmms-eval's
`hrbench4k` onto our `hrbench_4k`.

| arm | V\* | HR-4K | HR-8K | avg output tokens |
|---|---|---|---|---|
| **sft-coldstart** | **79.6** | **76.9** | **72.9** | 275 |
| grpo-no-saliency | 79.6 | 77.4 | 72.0 | 228 |
| saliency-r1 | 80.6 | 78.9 | 76.0 | 224 |
| **overlap-8k** (wov0.4 2head trmean) | **82.7** | **79.5** | **74.5** | 186 |
| ease-8k-v2 | 81.2 | 76.9 | 72.8 | 222 |
| Qwen3-VL-8B-Instruct (plain, **non-r1**) | 84.3 | 78.0 | 74.1 | **3.9** |

Three readings, and the third is the one that matters:

* **The acceptance threshold is met, in the opposite direction to the one
  expected.** The build brief asks for a cold-start number "within a few points of
  V\* ~77.5, HR-4K ~72.4, HR-8K ~68.1, expect lower". Our cold start is **above**
  all three, by +2.1 / +4.5 / +4.8. The published row is Qwen3-VL-8B-**Thinking**
  and, per [deepeyes-gate.md](deepeyes-gate.md) §2 footnote 1, is not verifiable in
  the tech report's arXiv HTML — treat it as secondary. Not 0, not 100, and in the
  right neighbourhood.
* **overlap-8k is ahead of the cold start on all three** (+3.1 / +2.6 / +1.6),
  which is the direction the search-vs-symbolic split
  ([[benchmark-table-splits-search-vs-symbolic]]) predicts. Unpaired, a single
  arm's standard error is ~2.9 pp on V\*'s 191 rows and ~1.5 pp on HR-Bench's 800,
  so none of these three gaps separates on its own — the honest reading is a
  consistent direction, not a result. These are the *same documents* for every
  arm, so the paired test is the right one and it has not been run here; and the
  placebo arms, which are what the split's own finding turns on, are not in this
  set at all. Nothing in this row should be claimed before both.
* **The plain Instruct row is not comparable** and is here only to anchor the
  scale. It answers in 3.9 tokens against the arms' ~200
  ([[plain-instruct-rows-are-not-comparable]]), so its 84.3 on V\* is a different
  measurement, not a better model.

**All of it is at 3.2 M pixels, not native.** Whether native moves these numbers is
the open question §5 exists to answer. The one hint on disk: four arms were also
scored at 4,194,304 and landed on *exactly* the same figures as their 3.2 M runs
(vstar 82.199 and 83.246, hrbench4k 0.7588, hrbench8k 0.7162), so the 3.2 → 4.2 M
step moved nothing. 3.2 → 16.8 M is a 5.2× step and a different question.

## 4. How to run it

### For a curve, alongside the other benchmarks

```fish
bash watch_bench_evals.sh --run-dir checkpoint/<run> --search --duration 2
```

`--search` is forwarded to `run_bench_eval.sh --search`, which passes
`--on-demand` to `bench_eval.py --plan`. Each of the four becomes its own unit
with its own `--max-pixels`. At `--natural-n 300` HR-Bench is budgeted at 60–64
minutes, above a one-hour window, so either give the job two hours or shrink them:

```fish
bash watch_bench_evals.sh --run-dir checkpoint/<run> --search \
    --task-n hrbench_4k=100 --task-n hrbench_8k=100
```

### For a baseline row in the paper table — use the full split

A 300-row sample of HR-Bench is comparable across our arms (same seed, same
documents) and **not** comparable to anybody's published number. For a table row,
run the real task whole, which is also how everything in §3 was produced:

```fish
cd /home/uberger/scratch/research/vlm_reasoning
# DURATION is an env var in that launcher, not a flag; it defaults to 4 hours.
for t in vstar_bench hrbench4k hrbench8k
    env DURATION=2 bash scripts/slurm/launch_lmms_eval_job.sh \
        --model <merged checkpoint> --tasks $t --max-pixels 16777216 \
        --max-new-tokens 4096 --r1-mode --tag r1_native --num-gpus 1
end
```

`--max-pixels` must be given **explicitly**: the implicit default for these three
task names is 3.2 M, and being explicit also suffixes the results directory
(`_px16777216`), so the native run cannot overwrite — or silently reuse the
response cache of — the 3.2 M runs in §3.

Then reduce and publish exactly as any other result:

```fish
python bench_eval.py --collect --run-dir <shadow dir> --step 0 \
    --results <each results.json>
```

### For the whole baseline list at once

```fish
bash run_bench_baselines.sh --search --only sft-coldstart --only overlap-8k
```

## 5. What still needs a GPU

| | what | cost |
|---|---|---|
| a | the four at native resolution on `sft-coldstart` — **does 16.8 M move §3's numbers?** | ~1 h, 1 GPU, V\* + HR-4K + HR-8K at n=100 |
| b | the same on `overlap-8k`, and on the two placebo arms | ~1 h each |
| c | full splits at native for the paper row | ~2–3 h per arm |
| d | spatialtreebench at all | stage 8 GB first, then ~4 h/arm + a judge key |

(a) is the one that decides whether §3 can be reported as a benchmark table or has
to be redone. The staged handoff is `SEARCH_BENCH_RUN_PROMPT.md`.

Staging the SpatialTree data, from a login node with internet:

```fish
conda activate lmms_eval
set -x HF_HOME /home/uberger/scratch/cache/hf_cache
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('LongfeiLi/SpatialTree-Bench', repo_type='dataset'))"
# then let lmms-eval unzip it once, also from a node with internet:
#   LMMS_EVAL_DATASETS_CACHE=$HF_HOME/datasets python -c "..." with HF_HUB_OFFLINE unset
```

Only the 15 MB `annotations_plain.parquet` is in the cache today.

## 6. One thing found along the way that is not about these benchmarks

lmms-eval's `_resolve_hf_datasets_cache_dir()` (added upstream 2026-07, merged into
our clone at `21a2b52` on 2026-09-24) treats `$HF_HOME/datasets` as unsafe for file
locks when it sits on a remote filesystem — which `/lustre` is — and silently
redirects the `datasets` build cache to node-local scratch
(`/tmp/lmms_eval_hf_datasets/$USER` when `/tmp` has ≥1 GB free). On a fresh node
that directory is empty, so `datasets` falls through to the hub, and with
`HF_HUB_OFFLINE=1` that is a `ConnectionError` wrapped in a `tenacity.RetryError`
**before the first document, for every benchmark**.

Reproduced on this login node for `pope_mini` and `realworldqa_mini`, not only the
new tasks. It has not bitten the compute nodes — evals ran successfully four days
after that merge — so their `/tmp` must fail the writable-or-1 GB test. But "so
far" is doing the work in that sentence, and the second cluster is a different
machine. `run_bench_eval.sh` now exports

```
LMMS_EVAL_DATASETS_CACHE=${HF_HOME}/datasets
```

which that function consults first and unconditionally, removing the branch. Every
build these jobs read is already in that directory and has been shared across jobs
since August.

## Files

| path | what |
|---|---|
| `eval_mini/benchmarks.py` | the table, `NATIVE_MAX_PIXELS`, `plan_units`' resolution guard |
| `eval_mini/make_mini_tasks.py` | `_take_json_strata`, the sampler spatialtreebench needs |
| `bench_eval.py` | `--plan --on-demand`; `expected` no longer counts absent on-demand |
| `bench_samples.py` | `dict_score` reads HR-Bench's `gpt_score` |
| `run_bench_eval.sh` | `--search`; the `LMMS_EVAL_DATASETS_CACHE` pin |
| `watch_bench_evals.sh`, `run_bench_baselines.sh` | `--search`, forwarded |
| `test_search_benchmarks_cpu.py` | 71 checks; `--lmms-eval` adds the offline task construction |
| `SEARCH_BENCH_RUN_PROMPT.md` | the cross-cluster handoff |

## Sources

* [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench) and
  `lmms-lab-encoder/vstar-bench` — the task's `dataset_path` is the latter; both cached
* [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench) — config
  `hrbench_version_split`, splits `hrbench_4k` / `hrbench_8k`, cached and built
* [LongfeiLi/SpatialTree-Bench](https://huggingface.co/datasets/LongfeiLi/SpatialTree-Bench)
  — arXiv:2512.20617, annotations only in cache
* [HaochenWang/TreeBench](https://huggingface.co/datasets/HaochenWang/TreeBench) —
  arXiv:2507.07999, ERGO's TreeBench; **no lmms-eval task exists**
* [deepeyes-gate.md](deepeyes-gate.md) §2–3 — the published baselines and the
  1024-px scan trap
