# Handoff prompt: score the visual-search benchmarks at native resolution

Paste this into a new Claude Code session started in
`/home/uberger/scratch/research/saliency_r1` on the **second cluster**
(`cs-oci-ord-login-01`). Written 2026-11-08 on `oci-nrt-cs-001`.

---

I want V\*, HR-Bench 4K and HR-Bench 8K scored at **native resolution** on our
arms. The harness is built, tested on CPU and merged; nothing has run on a GPU.

**Read `docs/search-benchmarks.md` first.** It is the reference for what the
benchmarks are, where the resolution cap lives, and what is already measured. Do
not re-derive any of it.

## Why this is happening on this cluster

`oci-nrt-cs-001` has **no GPU nodes at all**. Every GPU partition reports
`TotalNodes=0`, no node advertises a `gpu` GRES, and the pending jobs sit at
`Reason=PartitionConfig`. This is the same outage recorded in
`docs/deepeyes-gate.md` §4 and `docs/second-backbone-port.md`.

The two clusters **share the filesystem** — `/home/uberger/scratch/research/saliency_r1`
is the same directory on both — so nothing needs copying. Just `git log` to confirm
you have the commits.

## ⚠ Never run python on the login node here

`ulimit -v` is 8 GB and `ulimit -u` is **300 per user, shared across every
session**. Importing torch or transformers on `cs-oci-ord-login-01` hangs pinned
at the wall, holds thread slots forever, and kills unrelated Claude Code sessions.
Route **every** python through `srun`, including CPU-only checks. This is recorded
in `docs/glimpse-handoff.md` and has already cost two sessions.

## Step 0 — preflight, and stop if any of it fails

```bash
# 1. GPUs exist here
sinfo -h -N -o "%G" | grep -ci gpu                 # expect > 0
source cluster_env.sh; sr1_pick_partition 2        # resolves by asking the machine

# 2. the commits (HEAD should contain "the four visual-search benchmarks")
git -C /home/uberger/scratch/research/saliency_r1 log --oneline -3

# 3. the data, already in the shared HF cache -- no download needed
ls -d $HF_HOME/hub/datasets--lmms-lab-encoder--vstar-bench \
      $HF_HOME/hub/datasets--DreamMr--HR-Bench
ls -d $HF_HOME/datasets/lmms-lab-encoder___vstar-bench \
      $HF_HOME/datasets/DreamMr___hr-bench      # the BUILT copies; both must exist

# 4. the checkpoints
ls -d checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged \
      checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.4_2head_trmean_merged
```

Then the CPU test, **through `srun`, not on the login node**:

```bash
srun -A nvr_israel_rlop -p cpu_short -t 00:30:00 --pty bash -c '
  source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
  conda activate lmms_eval
  cd /home/uberger/scratch/research/saliency_r1
  python test_search_benchmarks_cpu.py --lmms-eval'
```

Expect `71 passed, 0 failed`. If the `--lmms-eval` half fails with a
`ConnectionError` / `RetryError` about reaching the Hub, that is
`docs/search-benchmarks.md` §6: export
`LMMS_EVAL_DATASETS_CACHE=$HF_HOME/datasets` and tell me, because it means this
cluster's compute nodes take the node-local-cache branch and **every** eval job
here needs that pin, not just these.

## Step 1 — the question that decides everything else

**Does native resolution move the numbers we already have at 3.2 M pixels?**
`docs/search-benchmarks.md` §3 is a complete table of all three benchmarks on all
five arms at full split, measured at 3,211,264 pixels. If native agrees with it,
that table is the paper table and steps 2–3 are unnecessary.

One GPU, ~1 hour, on the cold start only:

```bash
cd /home/uberger/scratch/research/vlm_reasoning
# DURATION is an ENV var in this launcher, not a flag. It defaults to 4 h.
for t in vstar_bench hrbench4k hrbench8k; do
  DURATION=2 bash scripts/slurm/launch_lmms_eval_job.sh \
      --model /home/uberger/scratch/research/saliency_r1/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged \
      --tasks $t --max-pixels 16777216 --max-new-tokens 4096 \
      --r1-mode --tag r1_native --num-gpus 1
done
```

`--max-pixels` **must** be explicit. The implicit default for these three task
names is 3.2 M, and being explicit is also what suffixes the results directory
(`_px16777216`) so the native run cannot overwrite — or silently reuse the response
cache of — the 3.2 M runs.

**Two things to watch and report rather than work around:**

* **OOM.** HR-Bench 8K at native is 16,384 visual tokens per image and 65,536 ViT
  patches. This has never been run; it is expected to fit on an 80 GB card but it
  is untested. If it OOMs, say so — do **not** quietly lower `--max-pixels` and
  produce a number that looks like a benchmark score. The fallback, if one is
  needed, is to report at 3.2 M *and label it*, which is what §3 already is.
* **Wall clock.** `eval_mini/benchmarks.py` budgets 15–16 min per 100 documents at
  native, extrapolated from 4.2–4.4 measured at 3.2 M. That is a guess with a
  quadratic term in it. Whatever the real figure is, put it in `MINUTES_PER_100`
  — the comment there says to.

Compare against §3's cold-start row: **V\* 79.6, HR-4K 76.9, HR-8K 72.9.**

* within ~1 point → native does not matter here; §3 stands, go to step 3
* materially higher → native matters; step 2 re-measures every arm
* materially lower → something is wrong, stop and tell me

## Step 2 — only if step 1 moved the numbers

The other four arms, same command with a different `--model`. Labels and paths are
in `run_bench_baselines.sh`'s `BASELINES` list. ~1 h per arm per benchmark at full
split, so budget ~6 GPU-hours.

Also add the **placebo arms** (`placebo-length`, `placebo-random`). §3 does not
have them and the search-vs-symbolic finding
([[benchmark-table-splits-search-vs-symbolic]]) turns on exactly that comparison:
the placebo buys the visual-search half too, and the driver is chain length.
Without a placebo row this table cannot be claimed as an effect of the reward.

## Step 3 — fold it into the curve, for runs still training

```bash
bash watch_bench_evals.sh --run-dir checkpoint/<run> --search --duration 2 \
    --task-n hrbench_4k=100 --task-n hrbench_8k=100
```

`--search` adds the four as units of their own. Without the `--task-n` caps,
HR-Bench is budgeted above a one-hour window and will be started, killed and
retried. The scores land under `bench/natural/<task>`; they are deliberately
**not** in `bench/natural/mean`, so no existing panel or step file changes.

## What I do NOT want done without asking

* **spatialtreebench.** It is in the table and the config generates, but it needs
  8 GB staged, an API key for 374 judged rows, and ~4 h per arm — and it is **not**
  the TreeBench of ERGO's paper (different dataset, different paper; lmms-eval has
  no task for the real one). `docs/search-benchmarks.md` §2 has the detail. Ask me
  before spending anything on it.
* **Lowering `--max-pixels` to make something fit.** Report the blocker instead.
  A number produced at a reduced resolution on these three benchmarks is not a
  benchmark number, and it looks exactly like one.

## When it is done

Update `docs/search-benchmarks.md` §3 with a native-resolution table beside the
3.2 M one — both, not one replacing the other — and `MINUTES_PER_100` in
`eval_mini/benchmarks.py` with what the jobs actually took.
