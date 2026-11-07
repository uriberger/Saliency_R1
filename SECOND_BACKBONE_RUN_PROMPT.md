# Handoff prompt: run the GLM + InternVL cold starts on this cluster

Paste this into a new Claude Code session started in
`/home/uberger/scratch/research/saliency_r1` on the **second cluster**
(`cs-oci-ord-login-01`). Written 2026-11-06 on `oci-nrt-cs-001`.

---

I want to train two second backbones for the Self-Saliency project — **GLM-4.1V-9B-Thinking**
and **InternVL3.5-8B** — starting with their cold-start SFT. The harness for both is
already built and merged on `main`; nothing has run on a GPU yet.

**Read `docs/second-backbone-port.md` first.** It is the end-to-end recipe and it records
which claims were measured rather than assumed. Do not re-derive what is in there.

## Why this is happening on this cluster

`oci-nrt-cs-001` has **no GPU nodes at all** — `batch_block1`, `batch_singlenode`,
`batch_short`, `batch_long`, `batch_large` and `backfill` all report `TotalNodes=0`, no
node anywhere advertises a `gpu` GRES, 31 jobs have been stuck at `Reason=PartitionConfig`
since 2026-10-30, and the last GPU job of mine there ended 2026-10-27. That is a hardware
outage, not the `(PartitionConfig)` scheduling churn we have seen before (which lasted
5–15 minutes with nodes present). We moved rather than waited.

## Step 0 — preflight, before anything is submitted

Four things, and **stop and tell me if any fails** rather than working around it:

```bash
# 1. are there GPUs here at all, and how many free
sinfo -o "%.20P %.10l %.6D %.6t %.14G" | grep -i gpu
sinfo -h -N -o "%G" | grep -ci gpu

# 2. the repo has the five commits this depends on (HEAD should be 34d96a2 or later)
git -C /home/uberger/scratch/research/saliency_r1 log --oneline -5

# 3. the cold-start corpus, 44 GB: 272,494 rows plus two image trees
du -sh cold_data/Saliency-R1-cold/            # expect ~44G
wc -l cold_data/Saliency-R1-cold/*.jsonl      # expect 95040 + 177454

# 4. the two checkpoints (20G + 16G). Absent is fine IF the hub is reachable --
#    they will download on first use. Absent AND offline is a blocker.
ls -d $HF_HOME/hub/models--zai-org--GLM-4.1V-9B-Thinking \
      $HF_HOME/hub/models--OpenGVLab--InternVL3_5-8B-HF
```

Also confirm the `sr1_coldstart` conda env exists (it is what runs LLaMA-Factory) and that
`/home/uberger/scratch/research/LLaMA-Factory` is present — the cold-start launcher `cd`s
into it. `./env_fingerprint.sh --env sr1_coldstart --no-imports` prints a diffable
fingerprint if anything looks off.

If the 44 GB corpus is missing, it has to come from `oci-nrt-cs-001`. Use a
`cpu_datamover` / `cpu` partition job for the copy, not a login node, and do not copy the
36 GB of weights — re-downloading them is cheaper than moving them.

## Step 1 — launch both cold starts

They are independent and should run side by side (8 GPUs each; the GPU-per-user cap on the
origin cluster was 32, check this one's):

```bash
bash launch_coldstart_job.sh --config train/cold_start/glm4v_9b_thinking_sft/train.yaml
bash launch_coldstart_job.sh --config train/cold_start/internvl3_5_8b_sft/train.yaml
```

Pass `WANDB_API_KEY=...` to get online logging; without it the run goes to
`WANDB_MODE=offline` and the only live record is `checkpoint/<run>/trainer_log.jsonl`.

**The 4-hour wall limit needs no manual restarts.** `submit_job
--autoresume_uninstrumented` relaunches and resumes from the highest `checkpoint-N`; the
Qwen3-VL cold start did exactly that in 3 segments with no intervention. Do not babysit it
with resubmissions.

Expected, from the Qwen3-VL run's own `trainer_log.jsonl`: **8,516 steps at ~1,124
steps/h ≈ 7.6 h of compute, about 3 chunks.** All three backbones sit at 250–320 image
tokens per picture so per-step cost should be comparable; GLM is 9B against 8B, so expect
it a little slower. `save_steps: 500` means a relaunch loses up to ~27 minutes.

If a job OOMs, halve `per_device_train_batch_size` and double
`gradient_accumulation_steps` — the effective batch must stay matched to the Qwen3-VL arm,
which is the whole point of the comparison. Pass them as
`per_device_train_batch_size=4 gradient_accumulation_steps=2` (forwarded verbatim to
LLaMA-Factory).

## Step 2 — track

Watch, and report to me rather than fixing silently:

* `squeue --me` state, and `outputs/logs/coldstart_<name>.<jobid>.out`;
* `checkpoint/<run>/trainer_log.jsonl` — `current_steps`/`total_steps` and the loss;
* that each run is on its **own** WandB id (`a0f941a6` for GLM, `85e85ffe` for InternVL,
  `cceaecd9` is Qwen3-VL's and must not be touched). That bug was fixed in 34d96a2; if two
  runs ever share a curve again, something reverted.

## Step 3 — what comes after, in this order

The order is forced by a dependency, not a preference:

1. merge each adapter —
   `bash merge_lora_grpo_qwen3.sh <checkpoint dir> <checkpoint dir>_merged`
   (it reads the base model out of `adapter_config.json`, so it is model-agnostic);
2. **head and layer selection** — `launch_head_correlation.sh --base-model <merged>`.
   This needs the merged checkpoint, so it cannot overlap step 1. `--overlap-layer 22
   --overlap-heads 28,31` are Qwen3-VL-8B numbers; on GLM (40x32) and InternVL (36x32)
   they name arbitrary heads, and the layer and heads **are the reward's definition**;
3. GRPO — `launch_grpo_qwen3_overlap_job.sh --model <merged> --overlap-layer L
   --overlap-heads a,b`. Read the `[overlap] first scored case:` line on the first step; a
   `MISMATCH` there means every map is being dropped and the reward will be nan.

## Things that will bite, all in docs/second-backbone-port.md

* **InternVL's processor tiles unless told not to**, and its own `crop_to_patches: False`
  attribute does not stop it — 13 tiles and 3,328 image tokens against the 1 tile and 256
  the geometry assumes. The trainer pins it and `after_processor` refuses a tiled batch,
  but the **eval** does not: lmms-eval's `internvl_hf_chat` defaults `max_patches=12`, so
  benchmark runs need `max_patches=1,min_patches=1` or the arm is scored off-distribution.
* **InternVL's vision tower uses bare `q_proj`/`k_proj`/`v_proj`** — 48 of its 120 suffix
  matches are in the InternViT. `vlm_family.InternVL.lora_target_modules` scopes it; GLM
  and Qwen3-VL need no scoping (fused `qkv` towers).
* **`trl_repo/` may lack `trl/trainer/vlm_family.py`** — it did on the origin cluster, and
  the trainer imports it at module scope. The non-colocated GRPO launcher now copies it in
  at launch; the colocated one still needs `patch_trl_qwen3.sh`, which re-patches a clone
  every session and every running job shares — so run that deliberately, never mid-step.
* `BENCH_MODEL_TYPE=glm4v` / `internvl_hf_chat` for benchmarks; `internvl3_5` is the
  wrapper for the `trust_remote_code` repo and is the wrong one here.

## What is NOT done

No GPU step has run on either backbone anywhere: no cold-start step, no GRPO step, no
`[overlap]` shape line, nothing about reward quality. §8 of
`docs/second-backbone-port.md` is the authoritative list.
