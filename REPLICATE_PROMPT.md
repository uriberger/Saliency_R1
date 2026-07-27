# Replication prompt for a fresh Claude Code session (new cluster)

> Paste everything below the line into a new Claude Code session started on the
> target machine. Unlike the first version of this doc, it does **not** assume the
> target has the same paths as the origin box — Step 0 makes the agent verify and
> ask. Do the setup on a login node; run training later via `submit_job` or `salloc`.
>
> Last synced with the repo at the merge commit that reconciled this box with the
> commits pushed from the other cluster (setup_cuda_home.sh, layer-22 attention
> capture, GPU-batched T5 step classifier, parallel judge) — 2026-07-26.

---

I'm replicating a research repo, **Saliency-R1** (CVPR-2026 baseline I reproduce and then
swap their saliency reward for my attention-overlap reward), from another NVIDIA cluster
onto this one. It was fully set up there and pushed to my fork. **Actually run this setup
here** — clone, create the four conda envs, install, patch, and verify — don't just review
the steps. Work through it in order, checking each command's output before moving on.
**Do all setup on a login node — no GPU / no `nvcc` needed. Do NOT launch any training
jobs.** The end state is a working, import-verified environment. Give me a short status at
each env boundary, and stop and ask if a step fails or a version lands off-pin.

## Step 0 — Verify prerequisites and adapt paths BEFORE anything else

The origin machine used these paths; several are **hardcoded in the launch scripts**:

| what | origin path | hardcoded refs |
|---|---|---|
| repo | `/home/uberger/scratch/research/saliency_r1` (= `/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1`) | 16 |
| conda | `/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh` | 15 |
| HF cache | `/home/uberger/scratch/cache/hf_cache` | 17 |
| CUDA toolkit | `/cm/shared/apps/cuda12.4/toolkit/12.4.1` | 8 |
| `submit_job` (ADLR cluster-interface) | `/lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface/21.1_2026-04-15_21-25-57` | 3 |
| LLaMA-Factory | `/home/uberger/scratch/research/LLaMA-Factory` | 2 |

**First report to me:** which of these exist here, whether this cluster mounts the *same*
lustre filesystem as the origin (check whether the origin repo path above is readable), and
what `submit_job`, the SLURM account/partition (`nvr_israel_rlop` / `batch_singlenode`), and
the CUDA toolkit path look like here. **Stop and check with me before rewriting any
hardcoded path** — don't guess. Note `setup_cuda_home.sh` already auto-derives `CUDA_HOME`
from the active conda env, so CUDA may need no edit at all.

## Step 1 — Clone

```bash
git clone https://github.com/uriberger/Saliency_R1.git <REPO>
cd <REPO> && git checkout main
```

This is my fork; `main` is the working branch (NOT `saliency-r1-setup`, which is stale).
Read `SETUP_NOTES.md`, `README.md`, and `CLAUDE.md` in the repo first — they explain the
design, the reward seam, and the gotchas.

**Only if the paths differ** from the table in Step 0, rewrite the hardcoded ones and show
me `git diff --stat` before continuing:

```bash
for f in $(grep -rl '/home/uberger/scratch/research/saliency_r1' *.sh *.py train/cold_start/*/*.yaml); do
    sed -i "s|/home/uberger/scratch/research/saliency_r1|$REPO|g" "$f"; done
for f in $(grep -rl '/home/uberger/scratch/miniconda3' *.sh); do
    sed -i "s|/home/uberger/scratch/miniconda3|$CONDA_ROOT|g" "$f"; done
for f in $(grep -rl '/home/uberger/scratch/cache/hf_cache' *.sh); do
    sed -i "s|/home/uberger/scratch/cache/hf_cache|$HF_HOME|g" "$f"; done
```

Also check `ACCOUNT=nvr_israel_rlop` / `PARTITION=batch_singlenode` in the launchers against
this cluster's SLURM setup, and tell me if either needs changing (don't change them silently).

## What we're building — four conda envs

| env | purpose | key pins | requirements file |
|---|---|---|---|
| `saliency_r1` | Qwen2.5-VL GRPO (original baseline) | transformers 4.55.0, trl 0.21.0, torch 2.7.1+cu126, deepspeed 0.17.4 | `requirements_clean.txt` |
| `saliency_r1_qwen3` | Qwen3-VL GRPO, HF-generate path | transformers 5.13.0.dev0 @ git `612c371`, trl 0.21 editable, torch 2.7.1+cu126, peft 0.19.1, accelerate 1.10.0 | `requirements_qwen3.txt` |
| `saliency_r1_qwen3_vllm` | Qwen3-VL GRPO, colocated vLLM sidecar (fast path; default env of `launch_grpo_qwen3_overlap_colocated_job.sh`) | same as above **but** vllm 0.11.0, torch 2.8.0, xformers 0.0.32.post1 | `requirements_qwen3_vllm.txt` |
| `sr1_coldstart` | LLaMA-Factory cold-start SFT | LLaMA-Factory @ `76a0391` editable, transformers 5.7.0, trl 0.24.0, torch 2.6.0+cu124, deepspeed 0.19.2 | `requirements_coldstart.txt` |

All GRPO envs share one external TRL clone at `trl_repo/` (gitignored), installed editable
into each. `flash-attn` is intentionally skipped (source build, needs `nvcc`); GRPO runs use
`--attn_implementation sdpa`.

## Step 2 — `saliency_r1` (Qwen2.5-VL GRPO)

```bash
source <CONDA_SH>; conda create -y -n saliency_r1 python=3.10; conda activate saliency_r1
pip install -r requirements_clean.txt
bash patch_transformers.sh      # patches sdpa_attention.py + modeling_qwen2_5_vl.py, backs up *.orig
```

## Step 3 — Rebuild `trl_repo/` (shared by all GRPO envs)

```bash
git clone --branch v0.21-release --depth 1 --single-branch https://github.com/huggingface/trl.git trl_repo
cp trl/grpo_trainer.py trl_repo/trl/trainer/grpo_trainer.py
cp -r trl/rewards/*    trl_repo/trl/rewards/
cp trl/grpo_vlm.py     trl_repo/examples/scripts/grpo_vlm.py
cp trl/__init__.py     trl_repo/trl/__init__.py
conda activate saliency_r1 && pip install -e trl_repo --no-deps
```

## Step 4 — `saliency_r1_qwen3` (Qwen3-VL GRPO)

```bash
conda create -y -n saliency_r1_qwen3 python=3.10 && conda activate saliency_r1_qwen3
grep -v '#egg=trl' requirements_qwen3.txt > /tmp/req_qwen3.txt   # trl comes from trl_repo, not that line
pip install -r /tmp/req_qwen3.txt
bash patch_transformers_qwen3.sh saliency_r1_qwen3   # sdpa identity trick for transformers 5.x
bash patch_trl_qwen3.sh          saliency_r1_qwen3   # grpo_trainer_qwen3.py + overlap reward + import_utils shim
pip install -e trl_repo --no-deps
```

`patch_trl_qwen3.sh` installs the attention-overlap reward files (`overlap_steps.py`,
`overlap_rewards.py`, `grpo_vlm_qwen3.py`) behind the `reward_variant=ours` flag.
Verify: `python -c 'from trl import GRPOTrainerQwen3; print(GRPOTrainerQwen3.__module__)'`
(run it from `trl_repo/`, see gotcha 1).

## Step 5 — `saliency_r1_qwen3_vllm` (colocated vLLM path)

```bash
conda create -y -n saliency_r1_qwen3_vllm python=3.10 && conda activate saliency_r1_qwen3_vllm
grep -v '#egg=trl' requirements_qwen3_vllm.txt > /tmp/req_qwen3_vllm.txt
pip install -r /tmp/req_qwen3_vllm.txt
bash patch_transformers_qwen3.sh saliency_r1_qwen3_vllm
bash patch_trl_qwen3.sh          saliency_r1_qwen3_vllm
bash patch_vllm_qwen3.sh         saliency_r1_qwen3_vllm   # vLLM 0.11 vs transformers 5.x fixes
pip install -e trl_repo --no-deps
```

Expected end state: transformers 5.13.0.dev0, vllm 0.11.0, torch 2.8.0, trl editable.
Report any pin the resolver moved.

## Step 6 — `sr1_coldstart` (LLaMA-Factory SFT)

```bash
conda create -y -n sr1_coldstart python=3.10 && conda activate sr1_coldstart
git clone https://github.com/hiyouga/LLaMA-Factory.git <LF_DIR>
cd <LF_DIR> && git checkout 76a0391dddc07741ff3e8fa2c82ebed106508280
pip install -e ".[torch,metrics]" --no-build-isolation
cd <REPO>
grep -v '#egg=llamafactory' requirements_coldstart.txt > /tmp/req_cold.txt
pip install -r /tmp/req_cold.txt
```

This env needs **no** transformers attention patch (SFT emits no attention). SFT config:
`train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml`; before running it you'd point
`dataset_dir` at the cold-start data and register `saliency_r1_llava_cot_full` /
`saliency_r1_mulberry_sft_full` in LLaMA-Factory's `dataset_info.json`.

## Step 7 — Artifacts that are NOT in git (report what's missing; don't fetch 17 GB unasked)

- `checkpoint/steps_classifier/best/` (~422 MB; `encoder/ tokenizer/ head.pt cfg.json`) —
  FLAN-T5 step classifier, **required** by the overlap reward. The launchers hard-fail
  without it; override the path with `OVERLAP_STEPS_CKPT`. Master copy lives in my
  `vlm_reasoning` repo at `steps_classifier/checkpoints/best`.
- `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged/` (~17 GB) — default
  `--model` for both overlap launchers, i.e. the GRPO starting point. Either copy it or
  re-run the cold-start SFT here.
- `cold_data/` — only needed if re-running SFT.

If this cluster mounts the same lustre as the origin, prefer symlinking/pointing at the
existing paths over copying. Otherwise propose a transfer plan and let me approve it.

Auto-downloading (leave HF online, `HF_HOME` on fast shared storage):
`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen2.5-VL-{3B,7B}-Instruct`,
`peterant330/Saliency-R1-CI-v2`, GRPO dataset `peterant330/saliency-r1-8k`, and
`IDEA-Research/grounding-dino-base` for the DINO reward server.

## Verification checklist (report results; launch nothing)

1. `saliency_r1`: patched `GRPOTrainer` imports (has the attention-output hook), patched
   `modeling_qwen2_5_vl` loads, rewards import; transformers 4.55.0 / trl 0.21.0 /
   torch 2.7.1+cu126.
2. `saliency_r1_qwen3`: `from trl import GRPOTrainerQwen3` works; transformers 5.13.0.dev0;
   overlap reward files present in `trl_repo`.
3. `saliency_r1_qwen3_vllm`: same as (2) plus `import vllm` (0.11.0) and the tokenizer /
   `tie_word_embeddings` patches applied.
4. `sr1_coldstart`: `llamafactory-cli version` works; transformers 5.7.0.
5. `pip show trl` in all three GRPO envs → `Editable project location: .../trl_repo`.
6. All `patch_*.sh` created `.orig` backups (they're idempotent; safe to re-run).
7. `bash check_cuda_home.sh` passes, and `python test_overlap_reward_cpu.py` runs (CPU-only
   sanity check of the overlap reward).

## Gotchas (these bit me on the original setup)

1. **cwd shadowing:** run python from `trl_repo/`, never the repo root — the root has a
   `./trl/` *patch-source* folder that shadows the editable-installed `trl_repo/trl`. The
   launchers `cd trl_repo` for exactly this reason.
2. **`submit_job` beats a PATH stub** — it prepends itself to PATH inside the launcher, so a
   "stub on PATH" dry-run really submits a job. To dry-run safely, export `submit_job` as a
   bash *function* that just captures the `-c` body. Submit nothing during setup.
3. **`CUDA_HOME` / conda:** conda activate can override `CUDA_HOME`, and a conda env's `bin`
   can shadow the active env's python. `setup_cuda_home.sh` plus the `set +u/-u` wrappers in
   the launchers handle this — keep them intact.
4. **flash-attn skipped by design**; vLLM ships its own kernels. Only add flash-attn on a
   GPU node if I ask.
5. **My shell is fish** — write fish syntax for anything I run by hand (`set -x VAR val`),
   but the launch scripts themselves are bash.
6. **WandB** is hardcoded in the launchers to project `vlm_reasoning`, entity `nvr-israel`;
   online only if `WANDB_API_KEY` is set, else offline.
7. **LLM-as-judge** (`openai_reward`) routes to the NVIDIA inference API
   (`https://inference-api.nvidia.com`, `azure/openai/gpt-4o-mini`, key via `NVIDIA_API_KEY`,
   model must carry the provider prefix). Not needed for setup.

Start with Step 0 and report before touching anything.
