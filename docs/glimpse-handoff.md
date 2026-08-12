# Handoff — GLIMPSE (map 5) memory rework, state as of 2026-08-12

Branch `feat/glimpse-map`, pushed to `origin`. Two commits on top of `main` at `10539ff`:

| commit | what |
|---|---|
| `feb1f4d` | the previous session's GLIMPSE implementation, committed unchanged as a baseline |
| `a8cd0e8` | the memory rework — `dz/dA` one layer at a time |

**Nothing in `a8cd0e8` has ever been executed.** It passes `py_compile` and nothing else.
The allocation expired before it could be run. Validating it is the whole remaining task.

## Why the rework exists

Map 5 died with CUDA OOM on any sample with a long chain. Cause: it ran the whole text
stack in eager so every layer's `[H, N, N]` attention sat in one graph. HF's eager path
saves an fp32 softmax output *and* the bf16 copy the `A·V` matmul consumed, so across 36
layers and 32 heads it costs **≈ 9 KB of GPU memory per (query, key) pair**.

Measured on an A100-80GB, peak allocated including the 16.3 GiB of weights:

| `N` | all-eager | `--glimpse-layer-frac 0.6` |
|---|---|---|
| 1200 | 32.5 GiB | — |
| 1600 | 42.3 | 32.6 |
| 2000 | 54.3 | — |
| 2400 | 68.6 | 49.1 |
| 3600 | **OOM** | — |

`N` is the whole teacher-forced sequence, so `--max-new-tokens 1024` puts a long chain
past 2000 unaided. Maps 1–4 never paid this: 1–3 re-run one layer in eager inside an
sdpa forward, 4 differentiates through sdpa.

`a8cd0e8` makes map 5 work the same way — sdpa forward, one backward per target token for
`∂z/∂h_l` at every layer, then each layer replayed alone in eager with `∂z/∂h_l` pushed
into it. Chain rule, so the same quantity by a different floating-point path. Full
reasoning in `docs/saliency-maps.md` §6 and the `GlimpseGradCache` docstring.

## The one task: run the two gates

```fish
# 1. get a node. ONE free 80 GB GPU is enough; the equivalence half runs the OLD
#    all-eager implementation too, so it needs ~25 GiB of genuinely free memory.
srun -A nvr_israel_rlop -p interactive --time=1:00:00 --gres=gpu:1 --pty bash

# 2. from the login shell, with $JOBID from `squeue -u $USER`:
srun --jobid=$JOBID --overlap -n1 bash -lc '
  cd <REPO>/.worktrees/feat-glimpse-map &&
  source <CONDA>/etc/profile.d/conda.sh && conda activate saliency_r1_qwen3_vllm &&
  export HF_HOME=<HF_HOME> HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
         CUDA_VISIBLE_DEVICES=0 &&
  python test_glimpse_cpu.py && python test_glimpse_gpu.py'
```

`test_glimpse_cpu.py` — pure algebra plus `test_grad_cache_identity`, which checks the
chain-rule claim against an all-in-one backward on the toy stack. No model, no GPU, but
it imports torch, so it still goes through `srun` (see below).

`test_glimpse_gpu.py` — the real gate, two halves:

- **equivalence**: new map vs the `feb1f4d` all-eager map on identical inputs from real
  `val_natural` samples. Passes at correlation ≥ 0.99 and max relative deviation ≤ 0.05.
  They are the same quantity, so anything below that is a real disagreement, not bf16
  noise. The baseline is materialised from git (`git show feb1f4d:saliency_viz.py`) so it
  cannot drift; `--max-new-tokens` defaults to 256 so the baseline survives to be compared.
- **scaling**: peak allocated at `N` = 1600 / 2400 / 3600 / 4800 against the table above.
  N=3600 OOM'd on the baseline and has to fit now. Rough estimate for N=4800 is ~40 GiB —
  an estimate, not a measurement, and it is the first thing that will fail if the rework
  is wrong about where the memory went.

If only a partly-occupied GPU is available: `--skip-scaling` drops the peak to ~25 GiB and
keeps the correctness gate, which is the half that decides whether the rework is right.

### Reading the result

- Both halves green → offer to merge: `./worktree.sh done feat/glimpse-map` **from the
  central tree**, after asking. Do not merge unasked.
- Equivalence fails → the rework is wrong, not the tolerance. Suspects, in order: the
  causal mask handed to the eager replay (`IV.causal_mask`, `_check_causal`), the recorded
  kwargs (`_check_replay` should have caught it), `--glimpse-layer-frac` < 1 changing which
  layers are cut.
- Scaling fails only at 4800 → not a defect; trim `--scale` and record the real ceiling in
  `docs/saliency-maps.md`.

## Cluster portability — check before running

Everything below differs per cluster and none of it is in git:

- `<REPO>`, `<CONDA>`, `<HF_HOME>` — `/home/uberger/scratch` is a **per-cluster Lustre
  mount**, so an identical path on another cluster is a different directory.
- `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` and
  `cold_data/grpo_sets/val_natural` are gitignored artifacts and must already be staged.
  `test_glimpse_gpu.py` takes `--base-model` and `--dataset` if they live elsewhere.
- The account and partition in the `srun` line (`-A nvr_israel_rlop -p interactive`) —
  confirm with `sinfo` / `sacctmgr`.
- The FLAN-T5 steps classifier is **not** needed: `test_glimpse_gpu.py` builds synthetic
  step spans on purpose, since comparing two implementations does not require real ones.

## Do not run python on a login node

This cost two killed sessions. On `cs-oci-ord-login-01`, `ulimit -v` is 8 GB (soft ==
hard) and `ulimit -u` is 300 **per user, shared across every session**. Importing
torch/transformers there spawns ~64 BLAS threads and then hangs pinned at the 8 GB wall
instead of exiting, holding those thread slots forever. A few of those exhaust the 300-task
limit, `clone()` starts returning `EAGAIN` for every process the user owns, and Claude
Code dies when it cannot spawn a thread — killing whichever session needs one next, not
necessarily the one that started the python.

So: route **every** python through `srun`, including CPU-only tests. After any crash,
check for leftovers with `ps -u $USER -o pid,etime,nlwp,vsz,args --sort=-nlwp | head` and
kill them; they never exit on their own. Compute nodes have `ulimit -v unlimited`.
