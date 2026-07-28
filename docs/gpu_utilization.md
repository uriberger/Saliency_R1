# Measuring per-GPU utilization

Two questions, two different tools: what a *finished* job did, and what a *running* job is
doing right now.

## Finished jobs — already recorded, nothing to instrument

Every run launched with `--report_to wandb` (all the GRPO launchers) has a wandb
system-metrics sampler polling NVML about every 15 s and appending the samples to the run's
binary transaction log at `trl_repo/wandb/run-*/​*.wandb`. That file is written **whether
wandb is online or offline**, so per-GPU history for jobs that finished weeks ago is already
on disk. 145 runs' worth, currently.

```fish
python gpu_util_report.py --list                     # what's available
python gpu_util_report.py                            # newest run
python gpu_util_report.py -r wov0.8 --skip-warmup 10 # by name substring, drop model-load
python gpu_util_report.py -r run-20260727_234947 --timeline
python gpu_util_report.py -r <run> --csv util.csv    # raw samples for your own analysis
```

`--skip-warmup N` matters: model load plus vLLM warmup is several minutes of near-zero
utilization, and it drags the means down enough to mislead.

Same data is in the wandb UI under a run's **System** tab if the run was online — the script
exists for the offline runs and for diffing runs against each other.

### Reading the columns

| Column | Meaning | Trap |
|---|---|---|
| `SM util %` | fraction of sampled time ≥1 kernel was resident | **not** how well the SMs are fed; a single tiny kernel reads 100% |
| `MemBW %` | memory-controller read/write activity | low + high SM util ⇒ compute-bound or spinning |
| `HBM %` | peak HBM used | sizing headroom |
| `Power W` | board draw vs the 700 W limit | the honest occupancy proxy — a GPU spinning in an NCCL all-reduce reads 100% SM util but draws ~150 W |
| `busy frac` | % of samples above `--busy-threshold` (default 5%) | separates "slow" from "idle half the time" |

Trust `Power W` over `SM util %` when the two disagree. That is the whole reason the column
is there.

## Running jobs — attach to the allocation

```fish
srun --overlap --jobid=<JOBID> --ntasks=1 nvidia-smi \
    --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv
```

`--overlap` is the load-bearing flag: without it srun waits for the job's own steps to
release the resources, so it just hangs. Add `-l 5` to `nvidia-smi` for a repeating sample,
or `nvidia-smi pmon -c 1` to attribute utilization to individual PIDs — useful on the
colocated launcher for telling DINO and vLLM apart when they share GPU 0.

`sacct` does **not** carry GPU utilization here; `TRESUsageInAve` covers CPU/mem/disk only.
`ConsumedEnergy` is populated and is a crude whole-node proxy, nothing per-GPU.

## What the current runs actually look like

Two representative 8×H100 runs, warmup skipped:

**Non-colocated (`grpo-qwen3-vl-8b-instruct-no-sal`, 77 min)** — 8 symmetric training ranks,
79–83% SM util, ~290 W, spread of 4 points across ranks. Balanced; the ceiling here is
per-rank efficiency, not skew.

**Colocated (`overlap__wov0.8_2head_trmax`, 228 min)** — fleet mean 38.7%:

```
 GPU |  SM util % (mean) | Power W | busy frac | role
   0 |       27.3        |   249   |   29.6%   | Grounding-DINO
   1 |       11.5        |   164   |   16.4%   | vLLM generation
   2 |       29.3        |   152   |   38.0%   | training
 3-7 |      46.5-51.1    |  150-156|  ~56%     | training
```

Two things worth acting on, both visible only per-GPU:

1. **The sidecars are nearly idle.** GPU 1 (vLLM) averages 11.5% and 164 W of a 700 W limit;
   GPU 0 (DINO) 27%. This is the measured case for `--share-sidecar-gpu`, which collapses
   both onto GPU 0 and buys a seventh training rank. The launcher comment already argues this
   from DINO's 1.6 GB memory footprint; utilization says the same thing about compute.
2. **Training ranks sit near 150 W — 21% of the power limit — while reporting ~48% SM util.**
   That gap is the signature of ranks spinning in collectives rather than computing, i.e.
   waiting on the vLLM generation round-trip. It is a pipelining problem, not a kernel
   problem, and it is the larger of the two effects.

GPU 2 running ~18 points below ranks 3–7 is unexplained and worth a look — rank 0 carries the
LoRA merge and the NCCL weight push to vLLM, so it may simply be the odd rank out.
