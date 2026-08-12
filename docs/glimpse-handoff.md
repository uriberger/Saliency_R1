# GLIMPSE (map 5) memory rework — RESOLVED 2026-08-12

**Done and merged.** Kept for the reasoning, and for the cluster-portability and
login-node sections at the bottom, which still apply to anything run here.

| commit | what |
|---|---|
| `feb1f4d` | the previous session's GLIMPSE implementation, committed unchanged as a baseline |
| `a8cd0e8` | the memory rework — `dz/dA` one layer at a time |
| `1c6ccb2` | equivalence gate moved to fp32, where the comparison is decidable |
| `fed5580` | `diag_glimpse_dev.py`, the harness that calibrated those thresholds |

Merged to `main` as `92cd34b`; branch `feat/glimpse-map` deleted local and remote.

## Outcome

`a8cd0e8` is correct. Equivalence against the all-eager baseline in fp32: **corr
1.000000, max deviation 1.3e-06**, on real `val_natural` samples. CPU gate 33/33,
including the chain-rule identity at exactly `0.000e+00`.

Peak allocated, H100-80GB, all 36 layers propagated:

| `N` | all-eager | grad cache |
|---|---|---|
| 1600 | 42.3 GiB | **26.17** |
| 2400 | 68.6 | **31.98** |
| 3600 | **OOM** | **42.03** |
| 4800 | — | **53.56** |

The pre-run estimate for N=4800 was ~40 GiB; the measurement is 53.56, and it still fits.
Confirmed end to end afterwards: a 20-sample, 8-GPU `launch_saliency_viz.sh` run drew
glimpse on every sample at `--max-new-tokens 1024` with no OOM.

**The one trap.** The gate originally compared in bf16 and failed at max dev 0.0707
against a 0.05 threshold. The threshold was wrong, not the code: this map carries
0.063–0.089 of its *own* bf16 rounding noise, so the baseline fails that check against
itself, and the reading moved 0.0707 → 0.02675 between two processes on one sample while
being bit-deterministic inside each. Hence fp32 — and TF32 must be off with it, since its
~10-bit mantissa is roughly bf16 and would silently restore the problem.

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

## Rerunning the two gates

```fish
# 1. get a node. ONE free 80 GB GPU is enough for the correctness half; the FULL run
#    needs ~54 GiB free, since --scale goes to N=4800. The equivalence half now runs
#    in fp32, which casts the 8B model and costs ~37 GiB on its own.
srun -A nvr_israel_rlop -p interactive --time=1:00:00 --gres=gpu:1 --pty bash

# 2. from the login shell, with $JOBID from `squeue -u $USER`:
srun --jobid=$JOBID --overlap -n1 bash -lc '
  cd <REPO> &&
  source <CONDA>/etc/profile.d/conda.sh && conda activate saliency_r1_qwen3_vllm &&
  export HF_HOME=<HF_HOME> HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
         CUDA_VISIBLE_DEVICES=0 &&
  python test_glimpse_cpu.py && python test_glimpse_gpu.py'
```

Budget ~20 minutes for the full run: fp32 equivalence is the slow half.

`test_glimpse_cpu.py` — pure algebra plus `test_grad_cache_identity`, which checks the
chain-rule claim against an all-in-one backward on the toy stack. No model, no GPU, but
it imports torch, so it still goes through `srun` (see below).

`test_glimpse_gpu.py` — the real gate, two halves:

- **equivalence**: new map vs the `feb1f4d` all-eager map on identical inputs from real
  `val_natural` samples, **in fp32** (`--equiv-dtype`). Passes at corr ≥ 0.9999 and max
  relative deviation ≤ 1e-3; measured 1.000000 / 1.3e-06, so those thresholds sit ~1000×
  above the noise and will still catch anything real. The baseline is materialised from
  git (`git show feb1f4d:saliency_viz.py`) so it cannot drift; `--max-new-tokens` defaults
  to 256 so the baseline survives to be compared.
- **scaling**: peak allocated at `N` = 1600 / 2400 / 3600 / 4800, in bf16, against the
  table above. This half is what makes the memory claim — the per-sample peaks printed by
  the equivalence half are informational, since at `N ≈ 400` in fp32 the peak is weights
  plus a graph too small to matter and the ordering flips on allocator noise.

If only a partly-occupied GPU is available: `--skip-scaling` keeps the correctness gate,
which is the half that decides whether the rework is right.

### If a gate goes red

- **Equivalence, in fp32** → the rework is wrong, and the threshold is not the suspect:
  the two paths agree to 1.3e-06 there. Look at the causal mask handed to the eager replay
  (`IV.causal_mask`, `_check_causal`), the recorded kwargs (`_check_replay` should have
  caught it), then `--glimpse-layer-frac` < 1 changing which layers are cut.
- **Equivalence, in bf16** (`--equiv-dtype bfloat16`) → almost certainly the dtype, not the
  code. Confirm with `diag_glimpse_dev.py --fp32` before touching `saliency_viz.py`; that
  is exactly the trap described above.
- **Scaling above 4800** → not a defect; trim `--scale` and record the real ceiling in
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
