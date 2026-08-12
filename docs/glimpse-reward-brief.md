# Brief — a GLIMPSE-based grounding reward

Build a GLIMPSE-based grounding reward for GRPO: use the GLIMPSE saliency map (map 5) as
the saliency source, and score it with the **`mean_in_v2`** metric — mean saliency inside
the DINO union mask, divided by mean saliency over the whole map.

Read `docs/saliency-maps.md` §6 first, then `trl/rewards/grad_rewards.py`.

## There is an exact precedent — mirror it

The roll-null **gradient** reward is the same shape of thing, already built and merged.
Copy its structure rather than inventing one:

| file | role |
|---|---|
| `trl/grad_maps.py` | produces `[n_steps, gh, gw]` maps (`step_grad_maps`) |
| `trl/rewards/grad_rewards.py` | turns maps into a reward (`think_grad_reward`) |
| `trl/rewards/overlap_rewards.py` | DINO boxes, `_union_mask`, `_mean_in_v2`, `_step_score` |
| `trl/grpo_trainer_qwen3.py` ~1757–1873 | where the trainer computes maps per case |
| `trl/grpo_vlm_qwen3.py` ~546–567 | where reward funcs are selected and registered |
| `trl/scripts/utils.py` ~127 | where the CLI flags live (`grad_target`, …) |
| `test_grad_reward_cpu.py` | the CPU gate to mirror |
| `patch_trl_qwen3.sh` ~84–100 | the `cp` list you **must** extend for new files |

The GLIMPSE map producer already exists in `saliency_viz.py`: `glimpse_map(...)` and
`GlimpseGradCache`. It returns one `[gh, gw]` map per step, already renormalised inside
the step — the same convention `_union_mask` / `_mean_in_v2` expect.

It was validated on GPU on 2026-08-12 (`main` at `8816fb3`): equivalence against the
all-eager implementation is exact in fp32 (corr 1.000000, max dev 1.3e-06), and peak
memory runs 26.2 GiB at `N`=1600 to 53.6 GiB at `N`=4800 on an H100-80GB. See
`docs/glimpse-handoff.md` for the full outcome, including the bf16 trap.

## Answer the feasibility question BEFORE writing the reward

GLIMPSE costs **one backward per target token**, plus a replay of all 36 layers in eager
for that token. The grad reward amortises its backward across steps with a chunked vmap
(`SPAN_CHUNK_DEFAULT` in `trl/grad_maps.py`); GLIMPSE as written does not. A GRPO step is
already ~40 s with DINO taking 16.6 s of it, and the colocated setup shares one
allocation between DINO, vLLM and training.

So the **first deliverable is a measurement, not code**: on one GPU, time and peak-memory
GLIMPSE map generation for a realistic case (8 completions × ~2–4 observe steps,
`--max-new-tokens 1024`), against `step_grad_maps` on the same input. Report seconds per
case and GiB. If it is 10× the grad reward, say so plainly and stop to discuss before
building the training path — a reward too slow to run is not a reward. `overlap_probe.py`
already drives maps over real cases and is the cheapest place to measure.

It must also work under ZeRO-3 with frozen params — see `frozen_params` in
`trl/grad_maps.py` and the trainer's `_no_deepspeed_zero3()` usage. Loading or
differentiating aux models naively under ZeRO-3 shards them to 1-D.

## The metric

`_mean_in_v2` (`trl/rewards/overlap_rewards.py:453`) = mean inside mask / mean over map.
Chance = 1.0, ceiling = `n_patches / n_in`, invariant to `m -> c*m`, returns `None` on an
all-zero map. Use it through the existing `configure(metric="mean_in_v2")` path rather
than reimplementing it.

**Known caveat, do not ignore:** union area moves `mean_in_v2` mechanically, since its
ceiling is `n_patches / n_in` — so a reward that can grow the union gets a free ride. See
`flow_correlation_probe.py:78-84` and the auroc union-growth writeup. Whatever you build,
report union area alongside the score.

## Be skeptical — prior evidence says attention-based grounding is weak here

Several probes in this repo found attention-derived saliency is **not** grounded:

- an intervention study found that forcing an observe step onto the objects it names has
  no causal effect beyond a wrong-place control;
- a 1152-head scan ranked the rewarded heads ~1100/1152 under auroc;
- rollout maps are anti-grounded at every layer and carry a strong reading-order prior
  (5.8–8.8× mass on the top row, monotone in sequence order).

GLIMPSE is gradient-*weighted* attention, so it is not identical to any of those, and
that is precisely why it is worth trying. But design the validation to **detect** "the
reward moves for text-side reasons" rather than assume grounding. Use per-step LOC
(roll-null on the step's own DINO union) and `ov_share`; never score against
question-level GT bboxes.

## Deliverables

1. The timing / memory measurement above, reported before building anything.
2. `trl/glimpse_maps.py` + `trl/rewards/glimpse_rewards.py` following the grad precedent,
   wired as a new reward variant, with CLI flags in `trl/scripts/utils.py` and the
   launcher, and the new files added to `patch_trl_qwen3.sh`'s `cp` list.
3. `test_glimpse_reward_cpu.py` mirroring `test_grad_reward_cpu.py` — no model, no GPU.
4. A weight calibration for `w_glimpse` the way `w_grad` was derived
   (`overlap_metric_spread.py:23-26` documents the arithmetic).

## Repo rules that will bite you

- **Work in a worktree**: `./worktree.sh new feat/glimpse-reward`. The launch directory is
  the shared central tree and is **read-only**. See `CLAUDE.md`.
- Tracked `trl/` is the **source**; gitignored `trl_repo/` is what actually **executes**.
  They drift silently. `patch_trl_qwen3.sh` copies `trl/` → `trl_repo/` and is **global**:
  it re-patches for every session and every running job. Say so before running it.
- **Never run python on the login node**, not even CPU tests. `ulimit -v` is 8 GB and
  `ulimit -u` is 300 shared across all your sessions; a torch import there wedges instead
  of exiting and has killed sessions. Route everything through
  `srun --jobid=$JOBID --overlap -n1 bash -lc '...'`.
- Env: `conda activate saliency_r1_qwen3_vllm`, `HF_HOME=/home/uberger/scratch/cache/hf_cache`,
  `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`.
- The user's shell is **fish** — write fish syntax.
- Do not merge to `main` without asking.
