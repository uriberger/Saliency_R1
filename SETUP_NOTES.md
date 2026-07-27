# Saliency-R1 setup notes (Uri)

Repo cloned from https://github.com/peterant330/Saliency_R1 for the purpose of
(1) reproducing their GRPO method, then (2) swapping their saliency reward for our
attention-overlap reward. Kept **separate** from the main `vlm_reasoning` repo
(different framework: TRL+DeepSpeed vs. our hand-rolled trainer).

To rebuild all of this on a new machine, see **`REPLICATE_PROMPT.md`** (a
paste-into-Claude-Code prompt). This file is the reference for *what* exists and *why*.

## The four conda envs

All live under `/home/uberger/scratch/miniconda3/envs/`, all python 3.10, all created on a
**no-GPU login node** (`flash-attn` is therefore skipped everywhere — it's a source build
needing `nvcc`; GRPO runs use `--attn_implementation sdpa`, and vLLM ships its own kernels).

| env | purpose | frozen pins | requirements file |
|-----|---------|-------------|-------------------|
| `saliency_r1` | Qwen2.5-VL GRPO — the paper's original baseline | transformers 4.55.0, trl 0.21.0 (editable), torch 2.7.1+cu126, accelerate 1.10.0, deepspeed 0.17.4, datasets 4.0.0, peft 0.17.0 | `requirements_clean.txt` |
| `saliency_r1_qwen3` | Qwen3-VL GRPO, HF-`generate()` path | transformers 5.13.0.dev0 @ git `612c371`, trl 0.21.0 (editable), torch 2.7.1+cu126, peft 0.19.1, accelerate 1.10.0, deepspeed 0.17.4 | `requirements_qwen3.txt` |
| `saliency_r1_qwen3_vllm` | Qwen3-VL GRPO with the vLLM generation sidecar (the fast path) | as above **but** vllm 0.11.0, torch 2.8.0, xformers 0.0.32.post1 | `requirements_qwen3_vllm.txt` |
| `sr1_coldstart` | cold-start SFT via LLaMA-Factory | llamafactory 0.9.6.dev0 @ git `76a0391` (editable), transformers 5.7.0, trl 0.24.0, torch 2.6.0+cu124, deepspeed 0.19.2, peft 0.18.1 | `requirements_coldstart.txt` |

Both `requirements_qwen3*.txt` files carry an `-e git+...#egg=trl` line for reference —
**strip it on install** (`grep -v '#egg=trl'`), because trl comes from the local `trl_repo/`
clone, not from git. Same for `#egg=llamafactory` in `requirements_coldstart.txt`.

### `saliency_r1` — Qwen2.5-VL GRPO (the paper's baseline)

`requirements_clean.txt` = their `requirements.txt` minus the 92 `@ file://` conda-local
lines that can't pip-install, minus `flash-attn` and `vllm`. **transformers patched** by
`patch_transformers.sh` (backs up `*.orig`): `integrations/sdpa_attention.py` **and**
`models/qwen2_5_vl/modeling_qwen2_5_vl.py` — both are needed on transformers 4.55 to get
attention weights out of generation. Patched TRL comes from `trl_repo/` (see below).

Verified on CPU: `GRPOTrainer` is the patched one (has the `output_attentions=True`
generation hook and builds `bbox_list`), patched `qwen2_5_vl` loads, rewards import.

### `saliency_r1_qwen3` — Qwen3-VL GRPO

Uses `GRPOTrainerQwen3` (`trl/grpo_trainer_qwen3.py` → `trl_repo/trl/trainer/`) and
`grpo_vlm_qwen3.py`. Only **one** transformers patch is needed here
(`patch_transformers_qwen3.sh`, the SDPA identity trick): transformers 5.x threads attention
weights up generically via the record-outputs hook system, so the whole-file modeling-file
replacement that Qwen2.5-VL needed does not apply. This is the env that carries the
attention-overlap reward (`patch_trl_qwen3.sh` installs it).

### `saliency_r1_qwen3_vllm` — same, plus the vLLM sidecar

Identical stack except `pip install vllm==0.11.0` drags torch to 2.8.0 (cu128), which is why
it is a **separate env** rather than an extra in `saliency_r1_qwen3`. `patch_vllm_qwen3.sh`
fixes two vLLM-0.11-vs-transformers-5.x breakages (`all_special_tokens_extended` removal;
`tie_word_embeddings` moving off `Qwen3VLTextConfig`). Default env of
`launch_grpo_qwen3_overlap_colocated_job.sh`, `run_smoketest_vllm.sh`,
`submit_smoketest_vllm.sh`. Note the cu126→cu128 bump: `run_smoketest_vllm.sh` Stage A
deliberately fails fast if the execution node's driver can't run a cu128 kernel.

### `sr1_coldstart` — cold-start SFT (LLaMA-Factory)

Saliency-R1's pipeline is cold-start SFT → GRPO. They publish an SFT'd **Qwen2.5-VL-7B**
(`peterant330/Saliency-R1-CI-v2`); we run the comparison on **Qwen3-VL-8B-Instruct**, which
they never SFT'd, so we run the cold start ourselves. Trainer is external LLaMA-Factory at
`/home/uberger/scratch/research/LLaMA-Factory` (`llamafactory-cli train <yaml>`), installed
editable. Config: `train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml` — LoRA rank 128,
ZeRO-3, `template: qwen3_vl_nothink`, `cutoff_len: 16384`, datasets
`saliency_r1_llava_cot_full` + `saliency_r1_mulberry_sft_full` with
`dataset_dir: cold_data/Saliency-R1-cold` (the two datasets must also be registered in
LLaMA-Factory's `dataset_info.json`). The original Qwen2.5-VL 3B/7B yamls are kept alongside.
**This env needs no transformers attention patch** — SFT emits no attention.

Output checkpoint used downstream:
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` (LoRA merged into the
base via `merge_lora.py`), which is the default `--model` of both overlap launchers.

## `trl_repo/` — the shared patched TRL

`trl_repo/` is a **gitignored** clone of TRL `v0.21-release`, installed `pip install -e
trl_repo --no-deps` into all three GRPO envs so it overrides the PyPI trl. The repo tracks
the *patch sources* under `./trl/` and copies them in:

- Qwen2.5-VL set (manual, README §GRPO step 2): `trl/grpo_trainer.py`, `trl/rewards/*`,
  `trl/grpo_vlm.py`, `trl/__init__.py`.
- Qwen3-VL set (`patch_trl_qwen3.sh`, idempotent): `import_utils.py` `_pkg_available()` shim
  (transformers 5.x changed `_is_package_available` to return a tuple, which made every
  availability flag truthy), `grpo_trainer_qwen3.py`, `overlap_steps.py`,
  `rewards/overlap_rewards.py`, `rewards/openai_rewards.py`, `rewards/__init__.py`,
  `scripts/utils.py`, `examples/scripts/grpo_vlm_qwen3.py`, plus lazy-import registration of
  `GRPOTrainerQwen3` in both `__init__.py`s.

The env argument to `patch_trl_qwen3.sh` is cosmetic — there is one `trl_repo/` shared by all
GRPO envs, so a single run patches all of them.

**Keep `./trl/` (tracked source) and `trl_repo/` (live) in sync in both directions**: edits
made live must be copied back into `./trl/` and committed, or the next `patch_trl_qwen3.sh`
silently reverts them.

## How the reward gets its inputs (the seam for our reward swap)

`--reward_variant` (in `trl/scripts/utils.py`) selects the mode:

- **`saliency_r1`** (default) — the paper's whole-completion rollout saliency reward.
- **`ours`** — our attention-overlap reward.
- **`none`** — accuracy + judge + format only; skips the attention re-forward entirely.

### `saliency_r1` path

`trl_repo/trl/trainer/grpo_trainer.py`: generation runs with `output_attentions=True`
(~line 1639); think-region attention is aggregated into a spatial `attn_map` over image
patches (~1774–1828) giving `attn_map` + `valid_list`; `_calculate_rewards(...)` (line 1306)
sets `bbox_list = [i['bbox'] for i in inputs]` (question-level box from `saliency-r1-8k`) and
calls rewards with `saliency_map=attn_map, valid_list=valid_list, bbox_list=bbox_list`.
`trl_repo/trl/rewards/saliency_rewards.py::think_saliency_reward` is then just the fraction
of saliency mass inside the box, gated by `valid_list`.

### `ours` path (attention-overlap)

`grpo_trainer_qwen3.py::_compute_overlap_step_maps` re-forwards the policy and captures raw
per-head observe→patch attention at **layer 22, heads 28+31** (`--overlap_layer`,
`--overlap_heads`), reduced within each observe step by `--token_reduction` (mean|max|min),
scored as `mean_in` against **per-step Grounding-DINO boxes** (`--box_threshold` 0.10,
`--max_box_area` 0.5; `--dino_api_base` points at a served endpoint, else DINO runs locally
per training process). Observe steps are identified by the FLAN-T5 step classifier in
`overlap_steps.py` (`OVERLAP_STEPS_CKPT`, default `checkpoint/steps_classifier/best`;
`OVERLAP_STEPS_DEVICE` defaults to `cuda`, batched via `predict_many()`). Reward weight is
`--w_overlap` (launcher default 0.2, and every swept HP appears in the model + wandb name).

Two performance fixes worth knowing (details in `docs/performance_grpo_overlap.md`): the
attention capture recomputes **only layer 22** via a forward hook instead of running eager
attention over all 36 layers, and the LLM judge runs through a `ThreadPoolExecutor`
(`JUDGE_MAX_WORKERS`, default 8) instead of a serial list-comp.

## Scripts

**Patches** (all idempotent, all back up `*.orig`): `patch_transformers.sh` (Qwen2.5-VL),
`patch_transformers_qwen3.sh <env>`, `patch_trl_qwen3.sh <env>`, `patch_vllm_qwen3.sh <env>`.

**GRPO launchers** — each submits via ADLR `submit_job` on a **bare GPU node** (no container;
activates the conda env on lustre) with `--duration N --autoresume_uninstrumented` for
wall-limit auto-resume, and passes `--resume_from_checkpoint True` only if a `checkpoint-*`
dir already exists in `--output-dir`. `--num-gpus N` overrides accelerate's `--num_processes`
(the zero3 config hardcodes 8). Unrecognized flags are forwarded verbatim to the training
script (e.g. `--max_steps 5` for a smoke run).

| script | env | what |
|---|---|---|
| `launch_grpo_job.sh` / `..._direct.sh` | `saliency_r1` | Qwen2.5-VL GRPO (paper config: `--model peterant330/Saliency-R1-CI-v2 --num-gpus 8`) |
| `launch_grpo_qwen3_job.sh` / `..._direct.sh` | `saliency_r1_qwen3` | Qwen3-VL GRPO, their reward |
| `launch_grpo_qwen3_overlap_job.sh` | `saliency_r1_qwen3` | overlap reward, non-colocated: all GPUs train, DINO local or via `--dino-api-base` |
| `launch_grpo_qwen3_overlap_colocated_job.sh` | `saliency_r1_qwen3_vllm` | overlap reward, sidecars on the same node: GPU0 DINO, GPU1 vLLM, GPU2..N-1 training |
| `launch_coldstart_job.sh` / `..._direct.sh` | `sr1_coldstart` | cold-start SFT via `llamafactory-cli train` |
| `run_grpo.sh` | `saliency_r1` | the original bare accelerate command (reference) |

`--direct` variants (and the `_direct.sh` scripts) run in-place on an existing interactive
allocation: no submit_job, no auto-resume, but still resume from a checkpoint if present.

**Helpers**: `serve_grounding_dino.sh` (batched DINO endpoint, `IDEA-Research/grounding-dino-base`,
run it on a GPU outside the training allocation), `merge_lora_grpo_qwen3.sh [adapter] [out]`
(reads the base model from the adapter's `adapter_config.json`, defaults output to
`<adapter>_merged`), `setup_cuda_home.sh` (derive `CUDA_HOME`, keep the toolkit and the conda
interpreter from shadowing each other), `check_cuda_home.sh` (validate `CUDA_HOME` on the
*execution* node before DeepSpeed JIT-compiles its ops), `preflight_overlap_gpu_node.py`,
`test_overlap_reward_cpu.py` (CPU sanity check of the overlap reward),
`run_smoketest_vllm.sh` / `submit_smoketest_vllm.sh`, `judge_probe.py` (diagnose judge 403s).

## Runtime knobs

**LLM-as-judge → NVIDIA inference API.** `openai_rewards.py` defaults to
`base_url=https://inference-api.nvidia.com` and `model=azure/openai/gpt-4o-mini`, reading the
key from **`NVIDIA_API_KEY`** (falls back to `OPENAI_API_KEY`; `NVIDIA_API_KEY` wins so a
stale OpenAI key in the shell isn't sent to the NVIDIA gateway). With neither set the module
imports with a placeholder key and a printed warning — it must not crash the run at import,
since it is imported regardless of which reward funcs are used; every judged sample then
masks to `None`. Override endpoint/model with `OPENAI_BASE_URL` / `JUDGE_MODEL`; the model
name must be provider-prefixed or the gateway returns 403 `key_model_access_denied`. Pass the
key at run time: `NVIDIA_API_KEY=nvapi-... bash launch_grpo_job.sh ...`. The launchers export
`OPENAI_BASE_URL`/`JUDGE_MODEL` only when set, so an unset value can't clobber the default
with `""`.

**WandB.** Launchers export `WANDB_PROJECT=vlm_reasoning` and `WANDB_ENTITY=nvr-israel`
(hardcoded). Online only if `WANDB_API_KEY` is set, else `WANDB_MODE=offline`.

**Reward metrics logged.** Reward funcs are `think_format_reward`, the saliency/overlap
reward, `openai_reward` (the LLM-judge correctness score — `accuracy_reward` in `grpo_vlm.py`
is defined but NOT used). Per function: `rewards/{name}/mean` and `rewards/{name}/std`
(nanmean/nanstd across all rollouts). Overall: `reward` plus — added by us —
`rewards/overall/mean` / `rewards/overall/std`. The stock `reward_std` is kept but is a
different quantity (mean *within-group* std used for GRPO advantage normalization).

**HF.** `HF_HOME=/home/uberger/scratch/cache/hf_cache`; left online so models/datasets
download on first use (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen2.5-VL-{3B,7B}-Instruct`,
`peterant330/Saliency-R1-CI-v2`, `peterant330/saliency-r1-8k`,
`IDEA-Research/grounding-dino-base`). Cold-start data (`peterant330/code_start_data`) is the
one manual download.

## Gotchas

1. **cwd shadowing** — run python from `trl_repo/`, never the repo root: the root's `./trl/`
   *patch-source* folder shadows the editable-installed `trl_repo/trl`. The launchers `cd`
   there for this reason.
2. **`submit_job` beats a PATH stub** — it prepends itself to PATH inside the launcher, so a
   "stub on PATH" dry-run really submits a job. To dry-run safely, export `submit_job` as a
   bash *function* that captures the `-c` body.
3. **`CUDA_HOME` / conda** — activation can override `CUDA_HOME`, and a conda env's `bin` can
   shadow the active env's python; `setup_cuda_home.sh` plus the `set +u/-u` wrappers around
   `conda activate` handle both. DeepSpeed JIT-compiles ops on the execution node, so this
   must be right *there*, not on the login node.
4. **flash-attn skipped by design** in every env.
5. **Shell is fish** — conda via `source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh;
   conda activate <env>`; the launch scripts themselves are bash.
6. **Non-git artifacts** that a new machine needs: `checkpoint/steps_classifier/best/`
   (~422 MB, required by the overlap reward; master copy in `vlm_reasoning` at
   `steps_classifier/checkpoints/best`) and
   `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged/` (~17 GB, the GRPO
   starting point) — plus `cold_data/` if re-running SFT.
