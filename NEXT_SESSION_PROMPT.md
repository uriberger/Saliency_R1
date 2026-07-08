# Handoff prompt for a new Claude Code session

I'm working in the `vlm_reasoning` project (repo:
`/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/vlm_reasoning`).
Please start by reading the project wiki (`wiki/README.md` + `wiki/project_description.md`)
and your memory (`MEMORY.md`, especially the `saliency-r1-repo-setup` memory) to load context.

## Background / where we are

I'm reproducing the **Saliency-R1** baseline (CVPR 2026, arXiv 2604.04500,
github.com/peterant330/Saliency_R1) so I can later swap their saliency reward for my
attention-overlap (`mean_in`) reward. Their repo is cloned to
`~/scratch/research/saliency_r1/` (SEPARATE from the main repo; their stack is
TRL 0.21.0 + DeepSpeed, mine is a hand-rolled trainer).

The **GRPO side is already fully set up and validated** (details in
`~/scratch/research/saliency_r1/SETUP_NOTES.md`):
- conda env **`saliency_r1`** (py3.10) with their pins (torch 2.7.1+cu126,
  transformers 4.55.0, trl 0.21.0, deepspeed 0.17.4), transformers patched for
  attention output, patched TRL installed `-e` at `trl_repo/`.
- Launchers `launch_grpo_job.sh` (submit_job, bare GPU node + conda, auto-resume) and
  `launch_grpo_job_direct.sh` (interactive). WandB hardcoded to
  project `vlm_reasoning` / entity `nvr-israel`. LLM-as-judge routed to the NVIDIA
  inference API (`https://inference-api.nvidia.com`, model `azure/openai/gpt-4o-mini`,
  key via `NVIDIA_API_KEY`). Extra reward mean/std metrics added.

## New task: set up cold-start SFT for MY model

Saliency-R1's pipeline is: cold-start SFT → GRPO. They cold-start **Qwen2.5-VL-7B**
and publish the SFT'd checkpoint (`peterant330/Saliency-R1-CI-v2`). **I want to run the
comparison on my own model, `Qwen/Qwen3-VL-8B-Instruct`**, which they did NOT use — so I
need to run the cold-start SFT myself on that model, then GRPO from it.

What the repo has for SFT: only **LLaMA-Factory config YAMLs** at
`~/scratch/research/saliency_r1/train/cold_start/vision_r1_full_Qwen2.5-VL-{3B,7B}-Instruct_.../train.yaml`
plus `train/cold_start/examples/deepspeed/ds_z3_config.json`. The trainer itself is
**LLaMA-Factory** (external, not installed): run via `llamafactory-cli train <yaml>`.
Cold-start data is on HF (`peterant330/code_start_data` → unpacks to `saliency_r1_data_filt`);
the yaml references a hardcoded author path `dataset_dir: /data/qiyuan/saliency_r1/...`
and dataset names `saliency_r1_llava_cot_full`, `saliency_r1_mulberry_sft_full` that must
be registered in LLaMA-Factory's `dataset_info.json`.

**Please set up the cold-start SFT environment**, adapted for `Qwen/Qwen3-VL-8B-Instruct`:
1. Clone + `pip install -e` LLaMA-Factory (a dedicated conda env is fine — decide whether
   to reuse `saliency_r1` or make a new one; Qwen3-VL may need a newer transformers than
   4.55.0, so a separate env is likely cleaner — check LLaMA-Factory's Qwen3-VL support
   and required transformers version first).
2. Download the cold-start dataset to lustre (`HF_HOME=/home/uberger/scratch/cache/hf_cache`),
   fix `dataset_dir`, and register the two datasets in `dataset_info.json`.
3. Create a new SFT config for `Qwen/Qwen3-VL-8B-Instruct` (copy the 7B yaml, change
   `model_name_or_path`, and set the correct LLaMA-Factory `template` for Qwen3-VL — the
   existing yaml uses `template: qwen2_vl`, which is Qwen2.5-VL-specific and likely wrong
   for Qwen3-VL; verify).
4. Write `launch_sft_job.sh` + `launch_sft_job_direct.sh` mirroring the existing
   `launch_grpo_job*.sh` conventions (submit_job on a bare GPU node + conda activate;
   WandB project `vlm_reasoning` / entity `nvr-israel`; `--num-gpus N`; auto-resume).

## Environment gotchas (learned this session)

- **This login node has NO GPU and NO `nvcc`** — do all setup here, but the actual
  training must go to a GPU node via `submit_job` (ADLR cluster-interface;
  `launch_grpo_job.sh` shows the pattern) or an interactive `salloc`.
- **`submit_job` prepends itself to PATH inside the launcher**, so a naive
  "stub submit_job on PATH" test will bypass the stub and REALLY SUBMIT a job. To
  dry-run a launcher safely, define `submit_job` as an exported bash *function*
  (functions beat PATH) that just captures the `-c` body. (I accidentally queued a job
  this way and had to `scancel` it.)
- Shell is **fish** — use `venv/bin/activate.fish`; conda via
  `source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh; conda activate <env>`.
- `HF_HOME=/home/uberger/scratch/cache/hf_cache` (keep HF downloads on lustre).
- **Big caveat to flag, not necessarily solve now:** the Saliency-R1 GRPO attention-output
  patch is **Qwen2.5-VL-specific** (`transformer/modeling_qwen2_5_vl.py` + `sdpa_attention.py`).
  Running GRPO on Qwen3-VL-8B later will need an equivalent attention-output patch for the
  Qwen3-VL modeling file — the SFT step doesn't need it, but the downstream GRPO does, so
  keep it in mind.

Please read the wiki + `SETUP_NOTES.md` first, then propose a concrete plan for the SFT
env setup before making changes.
