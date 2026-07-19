# Replication prompt for a fresh Claude Code session (new machine)

> Paste everything below the line into a new Claude Code session started on the
> target machine. It assumes the **same directory layout** as the original box
> (lustre home at `/home/uberger/scratch/...`) and that SLURM/`submit_job` is
> available. Do the setup on a login node; run training later via `submit_job`
> or `salloc`.

---

I'm replicating a research repo, **Saliency-R1** (a CVPR-2026 baseline I'm
reproducing so I can later swap their saliency reward for my attention-overlap
reward). It was fully set up on another machine and pushed to my fork. I want you
to reproduce that setup here, step by step, verifying as you go. **Do all setup
on this login node (no GPU / no `nvcc` needed for setup). Do NOT launch any
training jobs** — the goal is a working, import-verified environment.

## Target layout (must match — this machine has the same paths)

- Repo goes at: `/home/uberger/scratch/research/saliency_r1`
  (= `/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1`)
- Miniconda at: `/home/uberger/scratch/miniconda3`
  (activate via `source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh`)
- HF cache on lustre: `export HF_HOME=/home/uberger/scratch/cache/hf_cache`
- LLaMA-Factory clone (for SFT) at: `/home/uberger/scratch/research/LLaMA-Factory`

**First, verify these prerequisites exist** and stop and tell me if any are
missing: the miniconda install at that path, `submit_job` on PATH (ADLR cluster
interface), write access to the lustre paths above, and a CUDA toolkit under
`/cm/shared/apps/cuda12.4/toolkit/12.4.1` (used by the launch scripts). If
miniconda is at a different path, tell me before proceeding — many scripts
hardcode `/home/uberger/scratch/miniconda3`.

## Step 1 — Clone the repo

```bash
git clone git@github.com:uriberger/Saliency_R1.git /home/uberger/scratch/research/saliency_r1
cd /home/uberger/scratch/research/saliency_r1
git checkout saliency-r1-setup
```

`git@github.com:uriberger/Saliency_R1.git` is my fork (remote `myfork`); the
branch is `saliency-r1-setup`. Read `SETUP_NOTES.md` and `NEXT_SESSION_PROMPT.md`
in the repo first — they explain the design, the reward seam, and the gotchas
below in more detail.

## What we're building — three conda envs

| env | purpose | key pins | requirements file |
|-----|---------|----------|-------------------|
| `saliency_r1` | Qwen2.5-VL GRPO (original baseline) | transformers 4.55.0, trl 0.21.0, torch 2.7.1+cu126 | `requirements_clean.txt` |
| `saliency_r1_qwen3` | Qwen3-VL GRPO (current work) | transformers 5.13.0.dev0 @ git `612c371`, trl 0.21 editable, torch 2.7.1+cu126 | `requirements_qwen3.txt` |
| `sr1_coldstart` | LLaMA-Factory cold-start SFT | LLaMA-Factory @ git `76a0391`, transformers 5.7.0, trl 0.24.0, torch 2.6.0+cu124 | `requirements_coldstart.txt` |

All three share a single external TRL clone at `trl_repo/` (gitignored),
installed editable into both GRPO envs. `flash-attn` and `vllm` are intentionally
**skipped** (no `nvcc` on login node; GRPO runs use `--attn_implementation sdpa`).

## Step 2 — Env `saliency_r1` (Qwen2.5-VL GRPO)

```bash
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
conda create -y -n saliency_r1 python=3.10
conda activate saliency_r1
pip install -r requirements_clean.txt      # their pins minus flash-attn/vllm/conda-local lines
```

Then patch transformers for attention output (backs up `*.orig`):

```bash
bash patch_transformers.sh                 # patches sdpa_attention.py + modeling_qwen2_5_vl.py
```

## Step 3 — Reconstruct `trl_repo/` (shared by both GRPO envs)

`trl_repo/` is gitignored — recreate it by cloning upstream TRL and applying the
in-repo patch sources. Clone the version this repo was built against:

```bash
git clone --branch v0.21-release --depth 1 --single-branch \
    https://github.com/huggingface/trl.git trl_repo
```

**Qwen2.5-VL patch set** (README §GRPO step 2 — copy the tracked patch sources in):

```bash
cp trl/grpo_trainer.py        trl_repo/trl/trainer/grpo_trainer.py
cp -r trl/rewards/*           trl_repo/trl/rewards/
cp trl/grpo_vlm.py            trl_repo/examples/scripts/grpo_vlm.py
cp trl/__init__.py            trl_repo/trl/__init__.py
```

Then install editable into `saliency_r1`:

```bash
conda activate saliency_r1
pip install -e trl_repo --no-deps
```

Verify (from a neutral cwd — see gotcha #1): the patched `GRPOTrainer` imports and
has the attention-output hook.

## Step 4 — Env `saliency_r1_qwen3` (Qwen3-VL GRPO)

```bash
conda create -y -n saliency_r1_qwen3 python=3.10
conda activate saliency_r1_qwen3
```

Install from the frozen file, but **strip the editable trl line** (line with
`-e git+https://github.com/huggingface/trl.git...` / `#egg=trl`) — trl comes from
`trl_repo`, not that line. Everything else, including
`transformers @ git+...@612c371...`, installs as-is:

```bash
grep -v '#egg=trl' requirements_qwen3.txt > /tmp/req_qwen3.txt
pip install -r /tmp/req_qwen3.txt
```

Apply the Qwen3/transformers-5.x patches:

```bash
bash patch_transformers_qwen3.sh saliency_r1_qwen3   # ONLY the sdpa identity trick (tf5.x threads attn via record-outputs)
bash patch_trl_qwen3.sh          saliency_r1_qwen3   # adds grpo_trainer_qwen3.py, overlap reward, import_utils shim
pip install -e trl_repo --no-deps                    # editable trl into THIS env too
```

`patch_trl_qwen3.sh` also copies the attention-overlap reward files
(`overlap_steps.py`, `overlap_rewards.py`, `grpo_vlm_qwen3.py`, etc.) that back
the `reward_variant=ours` flag. Verify with:

```bash
python -c 'from trl import GRPOTrainerQwen3; print(GRPOTrainerQwen3.__module__)'
```

### DINO server (needed only for the `reward_variant=ours` overlap reward)

`serve_grounding_dino.py` / `serve_grounding_dino.sh` run a GroundingDINO server
(model auto-downloads to `HF_HOME`) used by the overlap reward. Same
`saliency_r1_qwen3` env — no extra install expected, but confirm
`transformers`' GroundingDINO import works. Don't start it now.

## Step 5 — Env `sr1_coldstart` (LLaMA-Factory SFT)

```bash
conda create -y -n sr1_coldstart python=3.10
conda activate sr1_coldstart
git clone https://github.com/hiyouga/LLaMA-Factory.git /home/uberger/scratch/research/LLaMA-Factory
cd /home/uberger/scratch/research/LLaMA-Factory
git checkout 76a0391dddc07741ff3e8fa2c82ebed106508280
pip install -e ".[torch,metrics]" --no-build-isolation
cd /home/uberger/scratch/research/saliency_r1
# reconcile remaining pins with the freeze (transformers 5.7.0, trl 0.24.0, etc.):
grep -v '#egg=llamafactory' requirements_coldstart.txt > /tmp/req_cold.txt
pip install -r /tmp/req_cold.txt
```

SFT config lives at
`train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml` (plus the original
Qwen2.5-VL 3B/7B yamls). Before running SFT you'll need to: point
`dataset_dir` at the downloaded cold-start data, and register the datasets
`saliency_r1_llava_cot_full` / `saliency_r1_mulberry_sft_full` in LLaMA-Factory's
`dataset_info.json`. The `cold_data/` scripts prep the LLaVA-CoT + Mulberry
sources. **This env does NOT need the transformers attention patch** (SFT doesn't
emit attention).

## Step 6 — Data & models (auto-download; not a setup blocker)

Models (`Qwen/Qwen2.5-VL-3B/7B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`,
`peterant330/Saliency-R1-CI-v2`) and the GRPO dataset (`peterant330/saliency-r1-8k`)
auto-download from HF at first run into `HF_HOME`. Cold-start data
(`peterant330/code_start_data`, unpacks to `saliency_r1_data_filt`) is the only
manual download, needed only for the SFT stage. Leave HF online. Don't pre-download
unless I ask.

## Verification checklist (do these; report results — don't launch training)

1. `saliency_r1`: patched `GRPOTrainer` imports, patched `modeling_qwen2_5_vl`
   loads, rewards import. Versions match: transformers 4.55.0, trl 0.21.0,
   torch 2.7.1+cu126.
2. `saliency_r1_qwen3`: `from trl import GRPOTrainerQwen3` works; transformers is
   `5.13.0.dev0`; overlap reward files present in `trl_repo`.
3. `sr1_coldstart`: `llamafactory-cli version` works; transformers 5.7.0.
4. `patch_transformers*.sh` created `*.orig` backups (idempotent, safe to re-run).
5. `trl_repo/` is editable-installed in both GRPO envs (`pip show trl` →
   `Editable project location: .../trl_repo`).

## Gotchas (these bit me on the original setup)

1. **cwd shadowing:** run python/verification from **`trl_repo/`** (or any dir
   that isn't the repo root). The repo root has a `./trl/` *patch-source* folder
   that shadows the editable-installed `trl_repo/trl` when you import from root.
   The launch scripts `cd trl_repo` for this reason.
2. **`submit_job` beats a PATH stub:** it prepends itself to PATH inside the
   launcher, so a "stub on PATH" dry-run will REALLY submit a job. To dry-run a
   launcher safely, define `submit_job` as an exported bash *function* that just
   captures the `-c` body. (Don't submit any jobs during setup.)
3. **`CUDA_HOME`:** conda activate can override it; the launch scripts re-export
   `CUDA_HOME=/cm/shared/apps/cuda12.4/toolkit/12.4.1` explicitly. Confirm that
   toolkit path exists on this machine (adjust if the CUDA module path differs).
4. **flash-attn / vllm skipped** by design. If ever wanted on a GPU node:
   `pip install flash-attn==2.8.2 vllm==0.10.0` (source build, needs `nvcc`).
5. **WandB** is hardcoded in launchers to project `vlm_reasoning`, entity
   `nvr-israel`; online only if `WANDB_API_KEY` is set, else offline.
6. **LLM-as-judge** (`openai_reward`) routes to the NVIDIA inference API
   (`https://inference-api.nvidia.com`, `azure/openai/gpt-4o-mini`, key via
   `NVIDIA_API_KEY`). Not needed for setup.

Please start by verifying prerequisites, then work through the steps, and give me
a short status at each env boundary. If anything about paths, CUDA, or `submit_job`
differs on this machine, stop and check with me rather than guessing.
