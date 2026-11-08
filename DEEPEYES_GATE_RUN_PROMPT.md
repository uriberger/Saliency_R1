# Handoff prompt: run the DeepEyes go/no-go gate on this cluster

Paste this into a new Claude Code session started in
`/home/uberger/scratch/research/saliency_r1` on the **second cluster**
(`cs-oci-ord-login-01`). Written 2026-11-07 on `oci-nrt-cs-001`.

---

I need to settle one decision: **is there enough headroom left on Qwen3-VL to justify
training a DeepEyes baseline?** The gate harness is built, every model and dataset it
touches is already on disk, and nothing has run on a GPU yet. This is a ~4-hour,
**single-GPU** job. It is not a training run.

**Read `docs/deepeyes-gate.md` first** — especially §6 (the decision as a cost), §7 (how
to run it) and §8 (the pre-registered thresholds). It records which claims were measured
and which were inferred. Do not re-derive what is in there.

## Why this is happening on this cluster

`oci-nrt-cs-001` has **no GPU nodes at all**. Every GPU partition — `backfill`,
`batch_block1`, `batch_large`, `batch_long`, `batch_short`, `batch_singlenode`,
`interactive`, `sniff-gpu-nodes` — reports `TotalNodes=0`, and no node anywhere
advertises a `gpu` GRES. Only the eight `cpu_*` partitions have hardware. That is a
hardware outage, not the `(PartitionConfig)` scheduling churn we have seen before. We
moved rather than waited.

**The two clusters do _not_ share a filesystem.** Both check the repo out at the same
absolute path, which makes it look like they do, but `$HOME/scratch` is a **per-cluster
symlink into that cluster's own lustre mount**. Nothing written on `oci-nrt-cs-001` is
visible here. This is recorded in `docs/ergo-gate-lowres.md`, where it already broke one
set of handoff instructions.

Three consequences, and they are the whole reason Step 0 exists:

1. **The branch has to travel through `origin`.** It cannot be read across.
2. **Gitignored artifacts are not here** unless this cluster already had them —
   `checkpoint/`, `outputs/`, `cold_data/` and `HF_HOME` are all per-cluster.
3. **The eval compute nodes have no network** and `HF_HUB_OFFLINE=1` is pinned on
   purpose, so anything missing from this cluster's HF cache fails the first unit with a
   `ConnectionError` instead of downloading. Pre-stage from a **login** node.

## Where the harness lives

The gate was built on branch **`probe/deepeyes-gate`** on the origin cluster. It must be
pushed there and fetched here:

```fish
# there (oci-nrt-cs-001), from the central tree -- I will have done this already
git push origin probe/deepeyes-gate

# here
git fetch origin
git log --oneline origin/probe/deepeyes-gate -2   # expect 0b39687 and 06ad1d4
./worktree.sh new probe/deepeyes-gate-ord          # or check the branch out directly
```

If `git fetch` does not show that branch, **stop and tell me** — it means the push did
not happen and there is nothing here to run.

## Do not run python on the login node

This has already cost two sessions. On `cs-oci-ord-login-01`, `ulimit -v` is 8 GB
(soft == hard) and `ulimit -u` is **300 per user, shared across every session**.
Importing torch/transformers on the login node spawns ~64 BLAS threads and then hangs
pinned at the 8 GB wall instead of exiting, holding those thread slots forever. A few of
those exhaust the 300-task limit, `clone()` starts returning `EAGAIN` for every process
you own, and Claude Code dies when it cannot spawn a thread — killing whichever session
needs one next, not necessarily the one that started the python.

So route **every** python through `srun`, including CPU-only checks. After any crash,
look for leftovers with
`ps -u $USER -o pid,etime,nlwp,vsz,args --sort=-nlwp | head` and kill them; they never
exit on their own. Compute nodes have `ulimit -v unlimited`. This is recorded in
`docs/glimpse-handoff.md`.

## Step 0 — preflight, before anything is submitted

**Stop and tell me if any of these fails** rather than working around it. Everything
below was staged on the *other* cluster and none of it is guaranteed here.

```fish
# 1. are there GPUs here at all, and how many
sinfo -o "%.20P %.10l %.6D %.6t %.14G" | grep -i gpu
sinfo -h -N -o "%G" | grep -ci gpu          # must NOT be 0

# 2. the harness arrived (see above -- it travels through origin)
git log --oneline origin/probe/deepeyes-gate -2     # expect 0b39687, 06ad1d4

# 3. the cold start -- 17 GB, and the ONLY artifact that cannot be re-downloaded
du -sh checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged

# 4. this cluster's HF cache (NOT the other cluster's)
echo $HF_HOME
du -sh $HF_HOME/hub/models--ChenShawn--DeepEyes-7B          # 16 GB, re-downloadable
ls -d $HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct \
      $HF_HOME/hub/datasets--craigwu--vstar_bench \
      $HF_HOME/hub/datasets--DreamMr--HR-Bench

# 5. the judge key
set -x NVIDIA_API_KEY (cat ~/.config/nvidia/api_key)
```

**What to do about each gap:**

| missing | fix |
|---|---|
| `ChenShawn/DeepEyes-7B`, `Qwen3-VL-8B-Instruct`, `vstar_bench`, `HR-Bench` | re-download from a **login** node (it has network; compute nodes do not). Cheaper than copying. |
| `outputs/deepeyes_gate/data/*.tsv` (6.2 GB) | **regenerate, do not copy** — `build_deepeyes_eval_data.py` rebuilds them from the HR-Bench parquet in ~5 min on `cpu_short`, and also clones and patches their repo. |
| the 17 GB cold start checkpoint | **this is the one real copy.** It is not on any hub. If it is absent, tell me before copying 17 GB across clusters — dropping the `coldstart` arm and running `--models deepeyes-7b,qwen3vl-8b` still answers most of §8, and I would rather make that trade deliberately. |

Then, once the gaps are closed:

```fish
# CPU, ~5 min, cpu_short. Patches their clone and converts HR-Bench parquet -> TSV.
python build_deepeyes_eval_data.py --out-dir outputs/deepeyes_gate/data
```

Remember the login-node rule above: send even this through `srun`.

## Step 1 — run the gate

```fish
cd .worktrees/probe-deepeyes-gate-ord    # the worktree you made in Step 0
set -x NVIDIA_API_KEY (cat ~/.config/nvidia/api_key)
bash launch_deepeyes_gate_job.sh --name deepeyes-gate --duration 4
```

Three arms, serially, inside one **1-GPU** allocation:

| arm | what |
|---|---|
| `deepeyes-7b` | `ChenShawn/DeepEyes-7B` — their released checkpoint, the upper end |
| `qwen3vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` — the untouched backbone |
| `coldstart` | `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` |

The last two are scored **through DeepEyes' own zoom loop**, not through lmms-eval. That
is deliberate and it is the whole point: their claim is about what the loop buys, so a
baseline measured without the loop concedes the question before it is asked.

If the queue is tight, `--bench vstar --duration 2` is the cheap version — 191 rows per
arm, ~10 min each, and V\* is the benchmark the decision actually turns on. The full
`vstar,hrbench` adds 1,600 HR-Bench rows per arm at ~1 h per arm.

The launcher **refuses to submit into a GPU-less cluster** rather than queueing into a
job that never starts and never errors. If it refuses here, that is check 1 failing and I
want to know.

## Step 2 — read the result against the pre-registered table

`docs/deepeyes-gate.md` §8 fixed these **before** the run, so the outcome is read rather
than argued:

| if the run shows | then |
|---|---|
| DeepEyes-7B reproduces at 90.1 ± 2 **and** our cold start is below ~78 on V\* | the ~12-point gap is real; reopen — but note ZoomEye gets it training-free |
| our cold start lands at or above ~85 on V\* | settled, no-go, with a measurement instead of an inference |
| DeepEyes-7B fails to reproduce (say < 85) | their eval release cannot be run as published (§9) — treat *all* their numbers with care, including the Qwen2.5-VL-7B baseline the +18.9 is measured from |
| our cold start is below ~70 on V\* | the backbone is worse than Qwen2.5-VL-7B on this axis and the premise of §2 is wrong |

Do not move these thresholds to fit the numbers. If the result lands between two rows,
say so plainly and quote both.

## Step 3 — report back

Give me:

1. The three-arm table, V\* (and HR-Bench if the full run was used), with row counts.
2. Which §8 row fired, verbatim.
3. Whether DeepEyes-7B reproduced its published 90.1 — this is load-bearing beyond our
   own decision, because §9 of the gate doc shows three import/argparse faults in their
   eval release, one of which (`judge_result_hrbench.py:39`) raises `AttributeError` at
   module level before argparse can print help. The published HR-Bench legs **cannot have
   been run from the published files**. If their checkpoint also fails to reproduce on
   V\*, that is a finding about the paper, not about us.
4. Anything that had to be changed to make it run here, so it goes back into the doc.

Then update `docs/deepeyes-gate.md` — replace §5's "what *was* measured" and the Verdict
with the measured table — and commit it on a worktree branch. Do not merge to `main`;
I confirm that.

## What is already decided, so you do not reopen it

- **Do not port their fork and do not clone verl at HEAD.** Their release is frozen on
  verl 0.2.0.dev0 / transformers 4.51.3 / vLLM 0.8.2, none of which load Qwen3-VL; verl at
  HEAD deleted `image_zoom_in_tool.py`. The base is `deepeyes_fork/` — verl 0.7.0.dev0,
  Qwen3-VL-native, already on disk and already proven. See `docs/deepeyes-build-prompts.md`.
- **The judge is ours, not their self-hosted Qwen2.5-72B.** The substitution is verified
  live, 17/17 through both of their graders. That removes 4 GPUs from any future run.
- **This gate is not the training run.** If §8 says reopen, the 80-step run is a separate
  decision costing 30–71 h on 8 A100s. Do not start it off the back of this job.
