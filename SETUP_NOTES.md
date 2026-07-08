# Saliency-R1 setup notes (Uri)

Repo cloned from https://github.com/peterant330/Saliency_R1 for the purpose of
(1) reproducing their GRPO method, then (2) later swapping their saliency reward
for our attention-overlap reward. Kept **separate** from the main `vlm_reasoning`
repo (different framework: TRL+DeepSpeed vs. our hand-rolled trainer).

## What's installed (done on a no-GPU login node)

- **conda env `saliency_r1`** (python 3.10) at
  `/home/uberger/scratch/miniconda3/envs/saliency_r1`.
- Deps from `requirements_clean.txt` (= their `requirements.txt` minus the 92
  `@ file://` conda-local lines that can't pip-install, minus `flash-attn` and
  `vllm`). Exact pins matched: torch 2.7.1+cu126, transformers 4.55.0, trl 0.21.0,
  accelerate 1.10.0, deepspeed 0.17.4, datasets 4.0.0, peft 0.17.0.
  - **flash-attn / vllm were skipped**: no `nvcc` on the login node (flash-attn is
    a source build), and their GRPO run uses `--attn_implementation sdpa` (they
    patch `sdpa_attention.py`), so neither is needed for a first run. Add later on
    a GPU node if wanted: `pip install flash-attn==2.8.2 vllm==0.10.0`.
- **transformers patched** for attention output via `patch_transformers.sh`
  (backs up `*.orig`): `integrations/sdpa_attention.py` and
  `models/qwen2_5_vl/modeling_qwen2_5_vl.py`.
- **patched TRL** cloned to `trl_repo/` (v0.21-release), their files copied in
  (grpo_trainer.py, rewards/*, examples/scripts/grpo_vlm.py, trl/__init__.py),
  then `pip install -e trl_repo --no-deps` so it overrides the PyPI trl.

All imports verified on CPU: `GRPOTrainer` is the patched one (has the
`output_attentions=True` generation hook and builds `bbox_list`), patched
qwen2_5_vl loads, rewards import.

## How the reward gets its inputs (the seam for our reward swap)

`trl_repo/trl/trainer/grpo_trainer.py`:
- generation runs with `output_attentions=True` (~line 1639), `attentions = outputs.attentions`;
- the think-region attention is aggregated into a spatial `attn_map` over image patches (~1774–1828), yielding `attn_map` + `valid_list`;
- `_calculate_rewards(...)` (line 1306) sets `bbox_list = [i['bbox'] for i in inputs]` (question-level box from the `saliency-r1-8k` dataset) and calls rewards with `saliency_map=attn_map, valid_list=valid_list, bbox_list=bbox_list`.

`trl_repo/trl/rewards/saliency_rewards.py` — `think_saliency_reward` is just
**fraction of saliency mass inside the box**: `sum(saliency[in bbox]) / sum(saliency)`,
gated by `valid_list`. Our `mean_in` overlap reward will replace this scoring; the
harder part of the port is producing **per-step** boxes/segmentation instead of the
one question-level box.

## Running it (needs a GPU node — this login node has none)

First-run config (decided): model `Qwen/Qwen2.5-VL-3B-Instruct` (smoke test),
**keep** `openai_reward` (LLM-as-judge). GPUs via `--num-gpus N` (default 1;
single-node DeepSpeed Zero3, `--num_processes` auto-matched).

**LLM-as-judge → NVIDIA inference API.** `openai_rewards.py` now defaults to
`base_url=https://inference-api.nvidia.com` and `model=azure/openai/gpt-4o-mini`,
reading the key from **`NVIDIA_API_KEY`** (falls back to `OPENAI_API_KEY`). Override
endpoint/model with `OPENAI_BASE_URL` / `JUDGE_MODEL`. Pass the key as a run-time
variable: `NVIDIA_API_KEY=nvapi-... bash launch_grpo_job.sh ...`. The launchers only
export `OPENAI_BASE_URL`/`JUDGE_MODEL` when set, so an unset value can't clobber the
NVIDIA default with `""`.

**cwd gotcha:** run python from `trl_repo/` (the launchers `cd` there). Running from
the repo root imports the stale patch-*source* `./trl/` folder instead of the
editable-installed `trl_repo/trl`.

**WandB.** Both launchers export `WANDB_PROJECT=vlm_reasoning` and
`WANDB_ENTITY=nvr-israel` (hardcoded). Online only if `WANDB_API_KEY` is set;
otherwise `WANDB_MODE=offline` (local logs only, no upload).

**Reward metrics logged** (patched `grpo_trainer.py`, ~line 1904). Reward funcs are
`think_format_reward`, `think_saliency_reward`, `openai_reward` (the LLM-judge
correctness score = the "accuracy" reward — `accuracy_reward` in grpo_vlm.py is
defined but NOT used). Per function: `rewards/{name}/mean` and `rewards/{name}/std`
(nanmean/nanstd across all rollouts). Overall (weighted-sum) reward: `reward` (mean)
and — added by us — `rewards/overall/mean` + `rewards/overall/std` (mean/std across
all rollouts, matching the per-func semantics). Kept the stock `reward_std` too,
which is the mean *within-group* std used for GRPO advantage normalization (a
different quantity). The metric edit is applied to BOTH `trl/grpo_trainer.py` (patch
source) and `trl_repo/trl/trainer/grpo_trainer.py` (the live editable copy).

Three launchers:

- **`launch_grpo_job.sh`** — submits via ADLR `submit_job` on a **bare GPU node**
  (no container image; activates the `saliency_r1` conda env on lustre),
  `--duration 4 --autoresume_uninstrumented` for wall-limit auto-resume. The
  node-side command passes `--resume_from_checkpoint True` **only if** a
  `checkpoint-*` dir already exists in `--output-dir`, so the first run starts
  fresh and every auto-resume relaunch continues. `--num-gpus N` overrides
  accelerate's `--num_processes` (the zero3 config hardcodes 8).

      OPENAI_API_KEY=sk-... WANDB_API_KEY=... bash launch_grpo_job.sh
      # paper config:
      OPENAI_API_KEY=sk-... WANDB_API_KEY=... bash launch_grpo_job.sh \
          --model peterant330/Saliency-R1-CI-v2 --num-gpus 8

- **`launch_grpo_job_direct.sh`** — same config/args but runs in-place on an
  existing interactive allocation (`salloc`); no submit_job, no auto-resume, but
  still resumes from a checkpoint if present.

- **`run_grpo.sh`** — the original bare accelerate command (kept for reference).

Env knobs for both launchers: `MODEL` via `--model`, `--num-gpus`, `--output-dir`;
`PARTITION`, `DURATION`, `HF_TOKEN`, `WANDB_API_KEY` (omit → `WANDB_MODE=offline`),
`OPENAI_API_KEY`, `OPENAI_BASE_URL`. Any unrecognized flag is forwarded verbatim to
`grpo_vlm.py` (e.g. `--max_steps 5` for a quick smoke run). HF is left online for
first-time model/dataset download.
