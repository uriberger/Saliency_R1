#!/bin/bash
# GRPO training for Qwen3-VL-8B with attention-overlap reward (colocated).
# Co-locates DINO + vLLM sidecars on the same node as training:
#     GPU 0         Grounding-DINO reward server   (127.0.0.1:$DINO_PORT)
#     GPU 1         vLLM generation server         (127.0.0.1:$VLLM_PORT)
#     GPU 2..N-1    GRPO training, DeepSpeed ZeRO-3  (N-2 processes)
#
# --share-sidecar-gpu puts DINO and vLLM together on GPU 0, giving N-1 training processes:
#     GPU 0         Grounding-DINO + vLLM  (vllm_gpu_mem default drops 0.90 -> 0.85)
#     GPU 1..N-1    GRPO training                    (N-1 processes)
# OFF by default. On a 4-GPU node this is 3 training procs instead of 2 (+50%). Justified by
# job 5627435 on GB200: DINO peaked at 1.6 GB of 185 GB (1%), so a dedicated GPU is waste.
#
# vLLM replaces the slow HF generate() path -- this is the main speedup over the
# non-colocated launcher. The policy is LoRA; the trainer merges the adapter and
# pushes weights to the vLLM server over NCCL every step.
#
# Default: submits to the cluster via submit_job (SLURM).
# --direct: runs on the current node immediately (no SLURM), e.g. on an interactive GPU node.
#
# Usage:
#   WANDB_API_KEY=... NVIDIA_API_KEY=... bash launch_grpo_qwen3_overlap_colocated_job.sh [OPTIONS]
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --direct --num-gpus 8
#
# THE HACK-RESISTANT REWARD -- one flag, nothing else to set:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --overlap-metric auroc
#
# Identical semantics to the non-colocated launcher (only the generation path differs),
# so the same defaults and the same evidence apply. Why: mean_in divides by the map's own
# PEAK, so a map that merely FLATTENS scores higher without attending the box any better
# -- 32x more movement under flattening than under real grounding, and that is what the
# wov0.2/wov0.4 runs did (their MMStar gain vanishes once chance-corrected). auroc depends
# only on patch ORDER, so that route is closed by construction. Derivation + evidence:
# vlm_reasoning/wiki/hack-resistant-overlap-reward-plan.md
#
# --overlap-metric auroc also switches two coupled defaults, because neither transfers
# from mean_in. Both are still overridable, and both appear in the run name, so a name
# always states what actually ran.
#
#   mass_floor_tau  ->  0.0022   Not optional: auroc is rank-based and so blind to a
#                                model withdrawing attention from the image. 0.0022 =
#                                p10 of the reference model's image_mass.
#   w_overlap       ->  0.11     GRPO normalises advantages within the group, so what
#                                matters is the reward term's SPREAD, and the auroc
#                                composite has ~3.6x the per-sample sd of mean_in
#                                (0.13 vs 0.036). 0.11 reproduces the pressure of
#                                mean_in's wov0.4. Reusing 0.2/0.4 here would apply
#                                ~3.6x the intended pressure.
#
# --overlap-metric mean_in_v2 is the third option: the same mean over the box, divided
# by the mean over the WHOLE map instead of by its peak. Chance is 1.0, and it is
# unbounded above only in principle -- measured over 1074 grounded steps of the
# cold-start policy on set_a it runs p10 0.41 / median 0.74 / p99 1.36 / max 2.33,
# because the median DINO box union already covers 56% of the image and the ceiling
# (n_patches/n_in) is ~1.8 there. No clamp needed.
#
# It carries ONE coupled default, and unlike auroc's it was measured here rather than
# taken from the offline screen (overlap_metric_spread.py on a 40-sample probe):
#
#   w_overlap       ->  0.033    Its per-sample sd is 0.105 vs mean_in's 0.0086, i.e.
#                                12x the spread, so 0.4 x 0.0086/0.105 ~ 0.033
#                                reproduces the pressure of mean_in's wov0.4. Reusing
#                                0.2/0.4 would apply ~6-12x the intended pressure. The
#                                same script re-derives auroc's weight as 0.089 against
#                                the documented 0.11, so treat 0.033 as +-25%.
#
# The mass floor stays OFF: a ratio of two means is blind to the model withdrawing
# attention from the image (the hole auroc's floor closes), but auroc's tau=0.0022 does
# not fit this corpus -- p10 of image_mass here is 0.00078 and 0.0022 bites on 30% of
# steps, well past the p25 the reward docstring warns about. If you enable it anyway,
# the floor RAISES the spread (0.105 -> 0.143), so pass --w-overlap 0.024 with it.
#
# Two caveats it does have and mean_in does not: it is more coupled to DINO box size
# (r +0.38 vs mean_in's +0.17), though that pull dies at 1.0 rather than diverging, and
# it has no offline attack/utility screen behind it.
#
# LORA TARGETS -- k_proj is now on by default:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --lora-targets q_proj,v_proj   # old default
#
# This reward pays for WHERE attention lands, and with k_proj frozen the image-token key
# directions are frozen with it: an attention logit here is cos(q, k_j) (Qwen3-VL RMS-
# normalises q and k per head, so a LoRA can only rotate them, never rescale), and two
# patches whose keys point the same way cannot be pulled apart by any query update.
# Adapting k_proj is what opens that route. Cheap under GQA -- 32 query heads over 8 KV
# heads makes k_proj 4096x1024 against q_proj's 4096x4096, so +38% adapter params over
# q+v -- and blunt in one way worth knowing: reward heads 28 and 31 share KV group 7, so
# a key update moves heads 28-31 together. The old q_proj,v_proj default is the LoRA
# paper's, chosen for downstream language quality at a fixed budget, not for attention
# placement. Changing this invalidates --resume from an existing adapter checkpoint.
#
# PLACEBO CONTROLS -- does the reward's DIRECTION matter at all?
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --placebo length --lora-targets q_proj,v_proj
#
# Measured 2026-08-18/19: the overlap reward does not identify the correct completion
# within a group (r with accuracy_reward -0.019 +- 0.051 at mean_in w0.4) and does not
# move the training objective -- but it takes train/frac_reward_zero_std from 0.547 to
# 0.000, and the benchmark moves. "It keeps the gradient alive" says why SOMETHING
# happens; it does not say which way the policy is pushed. These three replace the
# overlap reward with a control that has its tie-breaking strength and none of its
# grounding, each weighted to mean_in w0.4's WITHIN-GROUP sd so only direction varies:
#
#   --placebo roll     the same metric, the same map, the step's own box union MOVED to a
#                      deterministic wrong place. Same area, same shape. w 0.32.
#                      -> is it grounding, or any same-shaped signal?
#   --placebo random   a stable hash of the completion text -> U(0,1). w 0.013.
#                      -> pure within-group variance, no direction at all.
#   --placebo length   (max_completion_length - n_completion_tokens)/1000. w 0.031.
#                      -> is the overlap reward a brevity reward in disguise? Within a
#                         group, brevity is the largest thing it is associated with
#                         (r -0.042 / -0.105 / -0.035 across the three trained runs).
#
# The constant in `length` is not decoration. The doc specifies -n/1000, and a constant
# offset cancels in the GRPO advantage -- but NOT in the reward fold that trl_repo is
# currently running: it predates commit 8489767 and still uses `.nansum(dim=1)`, which
# reads an UNSCORED reward as 0. Under -n/1000 every scored completion is negative, so 0
# would be the BEST possible length score and "produce no groundable observe step" would
# become the winning move on the auxiliary dimension -- in the one experiment that exists
# to measure direction. Anchored at the completion cap the score is in [0, cap/1000] and
# an unscored completion reads as the longest possible one, which is the same kind of
# penalty an unscored mean_in already takes. Under the merged trainer the two are
# identical. See "WHICH TRAINER IS RUNNING" below.
#
# A FIXED RECTANGLE INSTEAD OF THE BOXES -- is the detector buying anything?
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --overlap-rect-frac 0.565
#
# Keeps think_overlap_reward and its metric, its slot and its weight, and swaps ONLY the
# mask: each step is scored against a centred rectangle covering that fraction of the
# patch grid instead of against its own Grounding-DINO union. NO DINO IS STARTED, so the
# sidecar's GPU goes to training (2 GPUs are enough, as with --maskfree).
#
# Measured motivation (centre_box_probe.py, outputs/centre_box_probe/report.txt, on the
# val_natural probe). Two findings that point opposite ways:
#
#   the rectangle is NOT DINO's mask   given the step's own union AREA, so that only
#                                      placement differs, closeness to that union is
#                                      0.230 -- against a 0.235 different-image floor.
#   but the reward cannot tell         the per-completion reward built on the rectangle
#                                      reproduces the real per-step-DINO reward's ranking
#                                      of a group at rho 0.651, vs 0.621 for DINO once
#                                      per chain -- and 0.638 for NO MASK AT ALL.
#
# GRPO only ever sees that ranking, so this run is the training arm: if the rectangle
# trains as well as the boxes, the detector was not buying the gradient. 0.565 is the
# fraction the probe used -- DINO's own mean union coverage -- so the arm differs from its
# reference in mask PLACEMENT alone and gives nothing away on mask SIZE.
#
# Two things to expect. Nothing is ever ungroundable, so EVERY observe step is scored and
# the scored set is larger than the reference's; and w_overlap was calibrated on DINO
# unions, so re-check it with overlap_metric_spread.py before calling the two runs
# pressure-matched. The run name carries _rect<frac> so it cannot share a checkpoint
# directory with the DINO run it is compared against.
#
# WHICH TRAINER IS RUNNING. As of 2026-08-20 trl_repo/ is deliberately behind main on
# trl/grpo_trainer_qwen3.py: main imputes each group's mean for an unscored reward
# (8489767), trl_repo still folds with nansum, and that was not shipped because it
# changes the GRPO advantage for every run. The four reference runs were trained under
# nansum. --placebo needs no trainer change to work -- only the two new logging lines
# (rewards/*/within_group_std and placebo/*) live there -- so the placebos can be run
# under either fold. Decide which one deliberately: matching the reference means nansum,
# and nansum means ~2-4% of completions (the ungroundable ones) are scored 0 on the
# auxiliary dimension rather than left neutral.
#
# Read against `overlap mean_in w0.4` and against the accuracy-only control
# (--w-overlap 0), NOT against baseline/grpo-no-saliency, which starts from vanilla
# Qwen3-VL-8B rather than the cold-start merge. All three ~= mean_in means direction is
# irrelevant; `length` ~= mean_in but `random` not means it is a brevity reward; mean_in
# beating all three means grounding contributes something specific.
#
# EVERY PLACEBO IS UNSCORED ON EXACTLY THE COMPLETIONS mean_in WOULD LEAVE UNSCORED.
# That is not imitated, it is taken: the run does the same segmentation, the same batched
# Grounding-DINO call and the same metric, and uses the real score only as a gate. So a
# placebo run is NOT cheap -- DINO is 16.6 s of a 40.5 s optimizer step and the parity
# rule needs all of it. The attention re-forward (1.0 s) is kept too, so a placebo run is
# the same computation as its reference in everything but the reward's value.
#
# --placebo appends _placebo<kind> to the run name, so it can never share a checkpoint
# directory or a wandb run with a real one. It requires the attention map and refuses
# --overlap-metric logratio (the roll-null is already a rolled control).
#
# On four GPUs. --placebo still needs DINO, but DINO peaks at 1.6 GB, so put it on vLLM's
# GPU and keep the other three for training:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --placebo length \
#       --num-gpus 4 --share-sidecar-gpu --grad-accum 16 --lora-targets q_proj,v_proj
#
# --grad-accum 16 is not optional if the run is to be compared step-for-step with an
# 8-GPU one: the generation batch is per_device x train_procs x grad_accum, so 3 procs at
# grad_accum 8 would put 24 sequences (3 prompts x 8 generations) behind each optimizer
# step instead of 48, halving the prompts per step and changing the LR schedule's meaning.
# 16 restores 48. The banner prints gen_batch -- check it reads 48.
#
# MISMATCHED BOXES -- does it matter that DINO read THAT SENTENCE?
#
#   bash launch_mismatch_bank.sh --out-dir outputs/mismatch_bank/8k        # once, ~1-2 h
#   bash launch_grpo_qwen3_overlap_colocated_job.sh \
#       --mismatch-bank outputs/mismatch_bank/8k/bank.json \
#       --num-gpus 7 --lora-targets q_proj,v_proj
#
# --num-gpus 7, NOT 8: no detector runs, so its GPU goes to training and eight cards give
# SEVEN training procs -- gen_batch 56 against the reference's 48. Seven cards restore six
# procs and 48, and --grad-accum cannot (48/7 is not an integer). The GEN BATCH: line in
# the banner says so whenever the number is not 48.
#
# Every reward in this launcher is downstream of one assumption: that grounding the STEP'S
# OWN TEXT is what makes the boxes meaningful. Two offline results say it may not be --
# two steps of one chain get masks no more alike than two steps of two different chains
# about the same picture (step_box_similarity.py), and the best single predictor of
# mean_in is a statistic that never sees a box (--maskfree). Neither can say what a policy
# TRAINED against wrong boxes would do, because both re-score maps a DINO-trained policy
# already produced. This flag is that run.
#
# The boxes are real Grounding-DINO output on a real photograph for a real cold-start
# sentence -- only the pairing is wrong. So it is the rung between --placebo roll (own
# union, moved) and --maskfree (no union): the union keeps DINO's size, shape and
# object-like structure and loses only its relation to this picture and this sentence.
#
# One donor row per PROMPT, shared by every rollout of it, and that is the experiment
# rather than an implementation detail. Measured on 538 cold-start chains, as the spread
# across the 8 rollouts of one prompt, which is all GRPO ever sees:
#
#     real reward, each completion on its own boxes         0.0115
#     this control, all 8 sharing ONE donor chain           0.0094   -> w 0.49
#     changing only which donor chain a completion drew     0.0117
#
# A donor drawn per rollout would put the third row inside the group, ~60% of the reward's
# within-group variance would be the draw, and the run would be --placebo random with a
# box-shaped distribution. Sharing it keeps 0.82 of the reference's tie-breaking strength
# with none of it donor noise.
#
# Matching the observe-step count therefore comes SECOND, after the donor: the donor row's
# chain of length n when it has one, its nearest length wrapped when it does not (0.0024,
# against the 0.0117 of hopping donors to chase an exact length). A completion is never
# left unscored for want of a length -- the policy picks its own step count, so an
# unservable count would be a free exit from the reward. Watch mismatch/exact_len_frac: it
# falls as the policy's chains outgrow the cold start's, which stop at 14 observe steps
# while the trained checkpoints reach 85.
#
# NO Grounding-DINO runs anywhere in the job, so no server is started, its GPU goes to
# training, and 16.6 s comes off a 40.5 s optimizer step. The price is the one thing that
# cannot be held equal without it: WHICH completions go unscored follows the donor here
# and the completion's own text there. That is 3.4% of steps and 0.5% of completions at
# the cold start -- the bank stores an ungroundable donor step as an empty box list, so
# the RATE is inherited even though the incidence is not.
#
# Read against `overlap mean_in w0.4` on the same corpus, and read a divergence carefully:
# this control cannot be hacked the way its reference can (repeating a trivially-groundable
# sentence and describing the background both stop working when the boxes do not respond
# to the text), so a difference is either "the sentence mattered" or "the reference was
# hacking". The duplicate-sentence fraction and the union area separate those.
#
# --mismatch-bank appends _mismatch<seed> to the run name. Run it twice at two seeds
# before believing it: a seed is a random pairing, and one pairing is not a result.
#
# MIXED CORPORA -- restrict the overlap reward to photographs:
#
#   bash launch_grpo_qwen3_overlap_colocated_job.sh --dataset_name cold_data/grpo_sets/set_b --natural-only
#
# set_b is 80% natural + 20% charts/documents/diagrams. Grounding-DINO grounds the
# observe-step phrase on the image, and it is a photograph detector, so on the non-natural
# fifth its boxes -- and hence the whole overlap score -- are noise. --natural-only masks
# the overlap term on those rows (their `natural` column is False); they keep format +
# accuracy + judge. Adds _natonly to the run name. Off by default.
#
# ONE GROUNDING PER QUESTION, DONE BEFORE THE RUN:
#
#   bash launch_precompute_question_boxes_job.sh --dataset cold_data/grpo_sets/set_a
#   bash launch_grpo_qwen3_overlap_colocated_job.sh \
#       --dataset_name cold_data/grpo_sets/set_a \
#       --question-boxes outputs/question_boxes/set_a_bt0.10.json
#
# Instead of calling Grounding-DINO once per observe step on that step's own sentence,
# ground once per dataset ROW on the row's QUESTION, offline, and score every step of that
# row against the one union. dino_text_sensitivity.py is why this is not a worse mask: the
# question recovers a step's real mask at IoU 0.649 / closeness 0.785, where a DIFFERENT
# REAL STEP OF THE SAME CHAIN gets 0.635 / 0.721, at the same mask size. No DINO server is
# started, so its GPU goes to training -- which changes TRAIN_N and gen_batch, so check the
# banner before comparing to a per-step run. Adds _qbox to the run name.
#
# WATCHING A RUN INSTEAD OF ONLY AUTOPSYING IT
#
# Two independent evaluations, both landing in the same WandB run -- one during
# training, one only after it:
#
#   validation   At step 0 and every --eval-steps (100) steps, val_natural and
#                val_nonnatural are scored on ANSWER ACCURACY ONLY -- 256 held-out
#                rows each, whose images appear in neither set_a nor set_b -- and
#                logged as val/<set>/accuracy. Build them once with
#                `python build_grpo_sets.py --build-val`; --no-eval turns it off.
#
#                One greedy completion per prompt in a single batched vLLM call: no
#                DINO, no saliency re-forward, no judge, no log-prob pass, and the
#                policy never runs a forward. Routing this through the Trainer's
#                evaluation loop instead re-runs the entire reward pipeline and
#                measured 21.6 min per set -- ~90% of training throughput at a
#                100-step cadence, which is what made the cheap path necessary.
#
#   benchmarks   NOT run during training. This launcher used to start
#                watch_bench_evals.sh alongside the run and let it submit a 1-GPU
#                eval job per kept checkpoint; it no longer starts anything. The
#                test suites are scored after the fact, on the checkpoints that
#                are worth scoring, by running the dispatcher by hand:
#
#                  bash watch_bench_evals.sh --run-dir <output-dir>
#
#                It backfills whatever has piled up -- its state is entirely on
#                disk (a checkpoint is done once bench_eval/step-<N>.json exists),
#                so starting it late loses nothing. Results are appended to the
#                training run's WandB history with
#                `python bench_eval.py --backfill --run-dir DIR --wandb-run-id ID`.
#
#                --auto-bench / --no-auto-bench are still accepted and ignored, so
#                the command lines of in-flight runs keep parsing across a requeue.
#
# Environment overrides:
#   PARTITION=batch_short        DURATION=1 (hours; see the note at the default -- the
#                                length decides which partitions are eligible)
#   NATURAL_ONLY=true            (same as --natural-only; --no-natural-only to force off)
#   SAVE_STEPS=10   CKPT_KEEP_EVERY=100
#   EVAL_STEPS=100   VAL_SETS_DIR=<dir>
#   DINO_PORT=8100   VLLM_PORT=8000   VLLM_MAX_MODEL_LEN=4096
#   VLLM_GPU_MEM     (default 0.90, or 0.85 with --share-sidecar-gpu)
#   SHARE_SIDECAR_GPU=true   (same as --share-sidecar-gpu; --no-share-sidecar-gpu to force off)
#   VLLM_ENFORCE_EAGER=False
#   OVERLAP_STEPS_DEVICE=cpu   OVERLAP_STEPS_CKPT=<path>
#   NVIDIA_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
#   WANDB_API_KEY   (omit -> offline)   HF_TOKEN
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
REPO=/home/uberger/scratch/research/saliency_r1
CONDA_SH=/home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
# This colocated job runs both the vLLM server and the trainer in ONE env, so it
# MUST use the vllm-enabled env. Hardcoded (not overridable) to prevent picking up
# a stray CONDA_ENV from the shell, which silently breaks the vLLM sidecar.
CONDA_ENV=saliency_r1_qwen3_vllm
HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

# ---------- SLURM defaults ----------
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/cluster_env.sh"
ACCOUNT=nvr_israel_rlop
# Resolved after argument parsing, not here: the set of partitions that can hold this job
# depends on DURATION, and --duration is not known yet. See RESOLVED_PARTITION below.
PARTITION=${PARTITION:-}
# 1 hour. The long-lived pools here cap at MaxTime=4:00:00, so a 4h request is the one
# length that can ONLY start on a fully idle node -- backfill needs a hole at least as
# long as the job, and a 4h hole exists only when something ran to the wall.
# Dropping to 2h was meant to fit the 2-4h gaps that open constantly, and it did not
# help: with polar4/polar3/polar at zero idle nodes the 2h requests still sat at
# Reason=Priority with StartTime=Unknown. 1h is the shortest useful chunk and fits the
# largest set of backfill windows, so it is what to try when 2h will not start.
#
# On oci-nrt-cs-001 it does something better than fit a gap: at 1h (or 2h) the job also
# becomes eligible for batch_short, whose MaxTime=2h put it out of reach at 4h. That is the
# highest-priority GPU partition open to this account (PriorityTier=40 against
# batch_block1's 20), it preempts the backfill pool rather than queueing behind it, and
# `sbatch --test-only` puts a 1h 8-GPU job there inside a minute against ~6.7 h on
# batch_block1. It is one more pool the scheduler can start us in, not a replacement --
# see the measured numbers in cluster_env.sh. sr1_pick_partition adds and removes it from
# DURATION automatically (SR1_JOB_HOURS), so raising DURATION back to 4 silently gives it
# up again. That is the real cost of a long chunk here now.
#
# What it costs: warm-up is ~6 min (DINO 30s, vLLM 75s, ~3 min to the first step) against
# ~50s/step, so overhead per allocation goes from ~3% at 4h to ~7% at 2h to ~10% here --
# about 273 steps per chunk, then ~132, now ~62. It costs throughput, not progress:
# --autoresume_uninstrumented requeues either way and the run picks up from the newest
# checkpoint (see RESUME_FLAG below), so the only real loss is re-paying warm-up more
# often. Watch that SAVE_STEPS still lands a checkpoint comfortably inside 62 steps --
# anything after the last save is redone on requeue.
#
# Raise it with DURATION=2 when the pool is idle enough that a longer hole is plausible --
# 2h is the longest chunk that keeps batch_short, so it is the one to try first. DURATION=4
# only pays off once the 4h pools themselves are idle.
DURATION=${DURATION:-1}

# ---------- training defaults ----------
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
NUM_GPUS=8
OUTPUT_DIR=""
MAX_COMPLETION_LENGTH=1024
NUM_GENERATIONS=8
GRAD_ACCUM=8
PER_DEVICE_BATCH=1
LEARNING_RATE=1e-5
# LoRA targets, comma-separated. k_proj is in the default because this run rewards
# WHERE attention lands: Qwen3-VL RMS-normalises q and k per head (q_norm/k_norm in
# modeling_qwen3_vl.py), so a LoRA can only ROTATE key/query directions, never scale
# them, and with k_proj frozen the image-token key directions are fixed -- two patches
# whose keys point the same way cannot be separated by any query update, which is
# exactly the move the overlap reward is asking for. It is cheap under GQA (32 query
# heads over 8 KV heads, so k_proj is 4096x1024 against q_proj's 4096x4096: +38%
# adapter params over q+v). It is also blunt: the reward heads 28 and 31 share KV
# group 7, so a key update moves heads 28-31 together.
# The upstream q_proj,v_proj default comes from the LoRA paper's ablation, which tuned
# downstream language quality at a fixed budget, not attention placement.
# NOTE: changing this invalidates --resume from an existing adapter checkpoint (the
# adapter shapes no longer match). It needs a fresh run.
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj}
SAVE_STEPS=${SAVE_STEPS:-10}
# Checkpoints are cheap -- 116 MB of LoRA adapter plus ZeRO state, not a full model
# -- and every kept one is a point on the benchmark curve, so keep one per 100
# steps rather than per 500. A 3,000-step run costs ~3.5 GB.
CKPT_KEEP_EVERY=${CKPT_KEEP_EVERY:-100}
# Held-out validation: score val_natural and val_nonnatural every EVAL_STEPS steps.
# Their images are disjoint from set_a and set_b, so this measures generalization
# rather than memorization. Set VAL_SETS_DIR="" to turn validation off entirely.
EVAL_STEPS=${EVAL_STEPS:-100}
VAL_SETS_DIR=${VAL_SETS_DIR:-$REPO/cold_data/grpo_sets}
# No benchmark eval runs during training: this launcher starts no dispatcher, and
# there is no flag that makes it start one. Score checkpoints afterwards by running
# watch_bench_evals.sh by hand -- see the printed hint at the end of startup.
#
# The BENCH_* values below no longer drive anything here. They survive because
# --bench-gpus is still on the command line of runs that are in flight (dropping
# the flag would send it to the training script as an unknown argument), and
# because the hint quotes them.
BENCH_GPUS=${BENCH_GPUS:-1}
BENCH_NATURAL_N=${BENCH_NATURAL_N:-300}
BENCH_NONNATURAL_N=${BENCH_NONNATURAL_N:-100}
EXTRA_ARGS=""
DIRECT=false

# ---------- overlap-reward defaults ----------
# W_OVERLAP and MASS_FLOOR_TAU are metric-dependent; the auroc values are applied
# after arg parsing, only if the user did not set them explicitly (see below).
W_OVERLAP=0.2
TOKEN_REDUCTION=mean
OVERLAP_HEADS="28,31"
OVERLAP_LAYER=22
BOX_THRESHOLD=0.10
MAX_BOX_AREA=0.5         # per-BOX area cap. Pass 0 to disable it and keep every box
                         # above --box-threshold.
MAX_UNION_AREA=""        # unset -> off. Per-STEP cap on the union of the kept boxes:
                         # a step whose union covers more than this fraction of the
                         # image is skipped (not scored 0), like an ungroundable one.
                         # --max-box-area does not bound the union -- N disjoint boxes
                         # each under the per-box cap can cover the whole image, and
                         # the measured median union is already 56% of it.
REWARD_VARIANT=ours      # ours (attention overlap) | grad (roll-null pixel gradient)
                         # | glimpse (GLIMPSE grounding; 55-59x grad, read the warning)
                         # NOTE: this default is what a relaunch that forgets
                         # --saliency-method silently falls back to. The resume gate
                         # below exists because that fallback is invisible otherwise.
ALLOW_MAP_CHANGE=false   # --allow-map-change: resume a checkpoint with a DIFFERENT
                         # saliency map than the one it was trained with. Off by default.
# ---------- regulators: keep GRPO from buying the auxiliary reward with degenerate text --
# Both are OFF by default, so a run that names neither is byte-identical to every run on
# record. They cover DIFFERENT failure modes and are meant to be readable separately:
#
#   --beta B        KL anchor to the base policy. Under LoRA the trainer needs no second
#                   model -- it disables the adapter to get the reference
#                   (grpo_trainer_qwen3.py:836-839, :2480) -- so this costs one extra
#                   forward per optimizer step and no GPU memory. Every run on record
#                   used beta=0, at which the trainer sets ref_model=None and there is no
#                   anchor at all. The one run that did not: 50k set_a, mean_in_v2,
#                   k_proj, beta 0.004, 5050 steps -- length 236->228 flat, entropy
#                   0.684->0.713, accuracy 0.434->0.586, overlap +11%, kl reaching 0.049.
#                   The set_a hack it would otherwise have produced never appeared. Note
#                   that run also added k_proj, so beta is not cleanly isolated in it.
#
#   --length-guard L  A leash on completion length, L tokens = the BASE policy's mean
#                   length on the corpus you are training on (read completions/mean_length
#                   at step 0; there is no safe default). Zero inside a band around L,
#                   quadratic outside. Catches set_a-style padding and all four
#                   collapse-to-short failures; it does NOT catch a set_c-style entropy
#                   collapse, whose mean length excursion (+17%) sits inside any band wide
#                   enough to allow the healthy shortening every good run does. Read
#                   lenguard/frac_penalized to see whether it is touching anything.
#                   See trl/rewards/length_guard_rewards.py for the shape and the
#                   calibration of --length-guard-weight.
BETA=0
LENGTH_GUARD_REF=""      # empty = guard off. In TOKENS.
LENGTH_GUARD_WEIGHT=0.20
LENGTH_GUARD_BAND_LO=0.30   # free window, as MULTIPLES of the reference length
LENGTH_GUARD_BAND_HI=3.0
LENGTH_GUARD_KNEE=1.0
ALLOW_REGULATOR_CHANGE=false  # --allow-regulator-change: resume a checkpoint whose
                              # regulator settings differ from this command line.
GRAD_TARGET=clogit       # clogit (default) | logit | logprob -- see trl/grad_maps.py
GRAD_NULL_OFFSETS=16
GRAD_LOGRATIO_CLIP=1.0
# GLIMPSE (reward_variant=glimpse). The two variants the reward offers are the metric:
# mean_in_v2 (chance 1.0, ceiling n_patches/n_in) and auroc (chance 0.5, rank-based).
GLIMPSE_TARGET=clogit
GLIMPSE_LAYER_FRAC=1.0   # cost dial 1: 0.6 is 1.64x cheaper, and a METHOD change
GLIMPSE_TOKEN_CAP=0      # cost dial 2: tokens scored per step, 0 = all. Cost is linear
GLIMPSE_DEPTH_TEMP=0.2   # the paper's text; 0.36 matches its shape on 36 layers
GLIMPSE_TEMP=0.5
GLIMPSE_TOKEN_WEIGHT=full
# Roll-null knobs, used only when the metric is 'logratio' (for --glimpse or for the
# attention reward). reward_variant=grad keeps its own GRAD_* copies of these.
ROLLNULL_OFFSETS=16
ROLLNULL_CLIP=1.0
ROLLNULL_SEED=0
OVERLAP_METRIC=""        # unset -> per-method default (mean_in for attention,
                         # logratio for grad, mean_in_v2 for glimpse). Otherwise
                         # mean_in | mean_in_v2 (/mean not /max; see
                         # below) | auroc (hack-resistant; see above)
MASS_FLOOR_TAU=""        # unset -> off for mean_in, 0.0022 for auroc (see below).
                         # Pass 0 to force it off explicitly.
# --natural-only: score the overlap reward only on rows with natural=True, leaving
# charts/documents/diagrams to format + accuracy + judge. Grounding-DINO is a
# photograph detector, so its boxes on non-natural imagery are noise, and a noisy
# overlap term is worse than none. Only meaningful on a mixed corpus with a `natural`
# column (cold_data/grpo_sets/set_b); OFF by default so existing runs are unchanged.
NATURAL_ONLY=${NATURAL_ONLY:-false}
# --question-boxes <file>: ground ONCE per dataset row, on the row's question, before the
# run, and reuse that union for every observe step -- instead of one Grounding-DINO call
# per step on the step's own sentence. Build the file with
# launch_precompute_question_boxes_job.sh. It must cover the corpus this run trains on and
# be built at the same --box-threshold; the trainer refuses a mismatch rather than training
# on the wrong boxes. Setting it also turns the DINO sidecar OFF and gives its GPU to
# training. Empty -> the incumbent per-step grounding, so a bare invocation is unchanged.
QUESTION_BOXES=""
# --placebo roll|random|length: REPLACE the overlap reward with a control that has its
# within-group spread but none of its grounding, to find out whether its DIRECTION
# matters. Empty = off (the real reward). See the PLACEBO block in the header.
PLACEBO=${PLACEBO:-}
# --maskfree flatness|mass: REPLACE the overlap reward with one that needs NO BOXES, to
# test whether mean_in's benefit was ever about grounding. Empty = off. This is the one
# reward here that does not start Grounding-DINO at all -- see the MASKFREE block in the
# header and the GPU layout below.
MASKFREE=${MASKFREE:-}
MASKFREE_PARITY=${MASKFREE_PARITY:-false}
# --overlap-rect-frac F: keep the overlap reward but score each step against a CENTRED
# RECTANGLE covering F of the patch grid instead of against its DINO boxes. Empty = off
# (the real union). Like --maskfree this starts no Grounding-DINO; unlike --maskfree it
# is still think_overlap_reward, so --overlap-metric and the reward weights carry over.
# 0.565 is the fraction to compare against a DINO run -- see the RECTANGLE block in
# trl/rewards/overlap_rewards.py and outputs/centre_box_probe/report.txt.
RECT_FRAC=${RECT_FRAC:-}
# --mismatch-bank <bank.json>: REPLACE the overlap reward with the MISMATCHED-BOX control
# -- the same metric on the same map, scored against real Grounding-DINO boxes computed
# for a DIFFERENT question about a DIFFERENT picture. Tests the assumption underneath
# every variant here: that running DINO on THAT SENTENCE matters. Empty = off. Like
# --maskfree it starts no Grounding-DINO at all; unlike it, the boxes are real. Build the
# bank with launch_mismatch_bank.sh; see docs/mismatch-boxes.md.
MISMATCH_BANK=${MISMATCH_BANK:-}
# Which donor row each training row is paired with. A second run at another seed is a
# replicate over a different pairing, which is how a result is told from a pairing.
MISMATCH_SEED=${MISMATCH_SEED:-0}

# ---------- sidecar defaults ----------
DINO_PORT=${DINO_PORT:-8100}
VLLM_PORT=${VLLM_PORT:-8000}
# VLLM_GPU_MEM default is resolved AFTER arg parsing -- it depends on whether the two
# sidecars share a GPU (see SHARE_SIDECAR_GPU below). Leaving it empty here preserves
# both override paths: the VLLM_GPU_MEM env var and the --vllm-gpu-mem flag.
VLLM_GPU_MEM=${VLLM_GPU_MEM:-}
# Put Grounding-DINO on the SAME GPU as vLLM, freeing one GPU for training. Measured on
# GB200 (job 5627435): DINO peaked at 1.6 GB of 185 GB (1%), while vLLM's 89% was simply
# gpu_memory_utilization=0.90 taking what it is allowed. Dedicating a whole GPU to DINO is
# therefore mostly waste -- on a 4-GPU node this turns 2 training procs into 3 (+50%).
# OFF by default: it changes GPU placement, so opt in explicitly.
SHARE_SIDECAR_GPU=${SHARE_SIDECAR_GPU:-false}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-False}
OVERLAP_STEPS_DEVICE=${OVERLAP_STEPS_DEVICE:-cuda}   # T5 step-classifier on the training GPU (CPU was the dominant per-step cost)
OVERLAP_STEPS_CKPT=${OVERLAP_STEPS_CKPT:-$REPO/checkpoint/steps_classifier/best}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --direct)                 DIRECT=true;                  shift ;;
        --model)                  MODEL="$2";                   shift 2 ;;
        --num-gpus)               NUM_GPUS="$2";                shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";              shift 2 ;;
        --partition)              PARTITION="$2";               shift 2 ;;
        --duration)               DURATION="$2";                shift 2 ;;
        --nvidia-api-key)         NVIDIA_API_KEY="$2";          shift 2 ;;
        --openai-api-key)         OPENAI_API_KEY="$2";          shift 2 ;;
        --wandb-api-key)          WANDB_API_KEY="$2";           shift 2 ;;
        --hf-token)               HF_TOKEN="$2";                shift 2 ;;
        --max-completion-length)  MAX_COMPLETION_LENGTH="$2";   shift 2 ;;
        --num-generations)        NUM_GENERATIONS="$2";         shift 2 ;;
        --grad-accum)             GRAD_ACCUM="$2";              shift 2 ;;
        --per-device-batch)       PER_DEVICE_BATCH="$2";        shift 2 ;;
        --learning-rate)          LEARNING_RATE="$2";           shift 2 ;;
        --w-overlap)              W_OVERLAP="$2"; W_OVERLAP_SET=1; shift 2 ;;
        --token-reduction)        TOKEN_REDUCTION="$2";         shift 2 ;;
        --lora-targets)           LORA_TARGETS="$2";            shift 2 ;;
        --overlap-heads)          OVERLAP_HEADS="$2";           shift 2 ;;
        --overlap-layer)          OVERLAP_LAYER="$2";           shift 2 ;;
        --box-threshold)          BOX_THRESHOLD="$2";           shift 2 ;;
        --max-box-area)           MAX_BOX_AREA="$2";            shift 2 ;;
        --max-union-area)         MAX_UNION_AREA="$2";          shift 2 ;;
        --overlap-metric)         OVERLAP_METRIC="$2";          shift 2 ;;
        --saliency-method)        SALIENCY_METHOD="$2";         shift 2 ;;
        --allow-map-change)       ALLOW_MAP_CHANGE=true;        shift 1 ;;
        --grad)                   REWARD_VARIANT=grad;          shift 1 ;;
        --grad-target)            GRAD_TARGET="$2";             shift 2 ;;
        --grad-null-offsets)      GRAD_NULL_OFFSETS="$2";       shift 2 ;;
        --grad-logratio-clip)     GRAD_LOGRATIO_CLIP="$2";      shift 2 ;;
        --glimpse)                REWARD_VARIANT=glimpse;       shift 1 ;;
        --glimpse-target)         GLIMPSE_TARGET="$2";          shift 2 ;;
        --glimpse-layer-frac)     GLIMPSE_LAYER_FRAC="$2";      shift 2 ;;
        --glimpse-token-cap)      GLIMPSE_TOKEN_CAP="$2";       shift 2 ;;
        --glimpse-depth-temp)     GLIMPSE_DEPTH_TEMP="$2";      shift 2 ;;
        --glimpse-temp)           GLIMPSE_TEMP="$2";            shift 2 ;;
        --glimpse-token-weight)   GLIMPSE_TOKEN_WEIGHT="$2";    shift 2 ;;
        --rollnull-offsets)       ROLLNULL_OFFSETS="$2";        shift 2 ;;
        --rollnull-clip)          ROLLNULL_CLIP="$2";           shift 2 ;;
        --rollnull-seed)          ROLLNULL_SEED="$2";           shift 2 ;;
        --mass-floor-tau)         MASS_FLOOR_TAU="$2";          shift 2 ;;
        --placebo)                PLACEBO="$2";                 shift 2 ;;
        --maskfree)               MASKFREE="$2";                shift 2 ;;
        --maskfree-parity)        MASKFREE_PARITY=true;         shift 1 ;;
        --overlap-rect-frac)      RECT_FRAC="$2";               shift 2 ;;
        # Absolutised here for the same reason as --question-boxes below, and it bites
        # harder: the existence check further down runs in the invocation cwd and PASSES
        # on a relative path, so the run gets all the way to six ranks importing torch
        # before mismatch_rewards._load_bank() looks for it under $REPO/trl_repo and does
        # not find it. The usage block above spells the flag relative.
        --mismatch-bank)
            MISMATCH_BANK="$2"
            [[ -e "$MISMATCH_BANK" ]] && MISMATCH_BANK="$(cd "$(dirname "$MISMATCH_BANK")" && pwd)/$(basename "$MISMATCH_BANK")"
            shift 2 ;;
        --mismatch-seed)          MISMATCH_SEED="$2";           shift 2 ;;
        # Regulators. --beta is a real flag rather than an EXTRA_ARGS passthrough
        # precisely so it can reach SUFFIX below: passed through, a beta run would share a
        # checkpoint directory and a wandb run with the beta=0 control it is meant to be
        # compared against.
        --beta)                   BETA="$2";                    shift 2 ;;
        --length-guard)           LENGTH_GUARD_REF="$2";        shift 2 ;;
        --length-guard-weight)    LENGTH_GUARD_WEIGHT="$2";     shift 2 ;;
        --length-guard-band-lo)   LENGTH_GUARD_BAND_LO="$2";    shift 2 ;;
        --length-guard-band-hi)   LENGTH_GUARD_BAND_HI="$2";    shift 2 ;;
        --length-guard-knee)      LENGTH_GUARD_KNEE="$2";       shift 2 ;;
        --allow-regulator-change) ALLOW_REGULATOR_CHANGE=true;  shift 1 ;;
        --natural-only)           NATURAL_ONLY=true;            shift ;;
        --no-natural-only)        NATURAL_ONLY=false;           shift ;;
        # Absolutised while still in the invocation cwd: the training command runs from
        # $REPO/trl_repo, and this script re-invokes itself inside the allocation.
        --question-boxes)
            QUESTION_BOXES="$2"
            [[ -e "$QUESTION_BOXES" ]] && QUESTION_BOXES="$(cd "$(dirname "$QUESTION_BOXES")" && pwd)/$(basename "$QUESTION_BOXES")"
            shift 2 ;;
        --eval-steps)             EVAL_STEPS="$2";              shift 2 ;;
        --no-eval)                VAL_SETS_DIR="";              shift ;;
        # Third one. Its default is already absolute, so only a relative --val-sets-dir
        # is affected, and it fails late but honestly: the "$VAL_SETS_DIR/$_split not
        # found" check below runs AFTER the cd, so it names a directory that does exist.
        --val-sets-dir)
            VAL_SETS_DIR="$2"
            [[ -e "$VAL_SETS_DIR" ]] && VAL_SETS_DIR="$(cd "$VAL_SETS_DIR" && pwd)"
            shift 2 ;;
        # Accepted and ignored. Benchmarks are never run during training now, but a
        # run that was submitted before that change re-invokes this script with the
        # flag on every requeue, and the catch-all below would forward it to the
        # training script as an unknown argument.
        --auto-bench|--no-auto-bench)                           shift ;;
        --bench-gpus)             BENCH_GPUS="$2";              shift 2 ;;
        --dino-port)              DINO_PORT="$2";               shift 2 ;;
        --vllm-port)              VLLM_PORT="$2";               shift 2 ;;
        --vllm-gpu-mem)           VLLM_GPU_MEM="$2";            shift 2 ;;
        --share-sidecar-gpu)      SHARE_SIDECAR_GPU=true;       shift ;;
        --no-share-sidecar-gpu)   SHARE_SIDECAR_GPU=false;      shift ;;
        --vllm-max-model-len)     VLLM_MAX_MODEL_LEN="$2";      shift 2 ;;
        --vllm-enforce-eager)     VLLM_ENFORCE_EAGER="$2";      shift 2 ;;
        # The training command runs from $REPO/trl_repo, so a relative dataset path
        # given on the command line would be resolved against the wrong directory.
        # Absolutize it here, while we are still in the invocation cwd; anything
        # that is not an existing local path (a Hub id, say) is passed through.
        --dataset_name|--dataset-name)
            _ds="$2"
            [[ -e "$_ds" ]] && _ds="$(cd "$(dirname "$_ds")" && pwd)/$(basename "$_ds")"
            EXTRA_ARGS="$EXTRA_ARGS --dataset_name $_ds"; shift 2 ;;
        *)                        EXTRA_ARGS="$EXTRA_ARGS $1";  shift ;;
    esac
done

# ---------- partition, resolved once DURATION is final ----------
# Deliberately after the argument loop: which partitions can hold this job depends on how
# long it asks for. batch_short (MaxTime=2h) is in the list at DURATION=1 or 2 and out of
# it at 4, and resolving next to the defaults would have decided that against the built-in
# 1 rather than against a --duration on the command line.
PARTITION=${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}

# ---------- saliency method and metric, resolved once ----------
# --saliency-method is the flag; --grad / --glimpse are kept as shorthand and
# REWARD_VARIANT remains the internal name, so nothing downstream had to change.
case "${SALIENCY_METHOD:-}" in
    attention) REWARD_VARIANT=ours ;;
    grad)      REWARD_VARIANT=grad ;;
    glimpse)   REWARD_VARIANT=glimpse ;;
    "")        ;;
    *) echo "ERROR: --saliency-method must be attention|grad|glimpse (got '$SALIENCY_METHOD')" >&2; exit 1 ;;
esac
# The resolved method, spelled the way the flag spells it. The SLURM path re-invokes this
# script with --direct and has to hand the map back explicitly: REWARD_VARIANT is an
# internal variable and does not survive a new process.
case "$REWARD_VARIANT" in
    grad)    SALIENCY_METHOD_R=grad ;;
    glimpse) SALIENCY_METHOD_R=glimpse ;;
    *)       SALIENCY_METHOD_R=attention ;;
esac
# One metric flag now serves three maps that had three different historical defaults, so
# an unset metric resolves per map -- otherwise a bare --grad would silently stop using
# the roll-null it has always used.
if [[ -z "$OVERLAP_METRIC" ]]; then
    case "$REWARD_VARIANT" in
        grad)    OVERLAP_METRIC=logratio ;;
        glimpse) OVERLAP_METRIC=mean_in_v2 ;;
        *)       OVERLAP_METRIC=mean_in ;;
    esac
fi
case "$OVERLAP_METRIC" in
    mean_in|mean_in_v2|auroc|logratio) ;;
    *) echo "ERROR: --overlap-metric must be mean_in|mean_in_v2|auroc|logratio (got '$OVERLAP_METRIC')" >&2; exit 1 ;;
esac
# mean_in divides by the map's own PEAK, so a map that merely flattens scores higher --
# the mechanism behind the wov0.4 hack. It stays available for the attention map because
# every trained run used it, but on a map with no such history it is a trap, not a default.
if [[ "$OVERLAP_METRIC" == "mean_in" && "$REWARD_VARIANT" != "ours" ]]; then
    echo "WARNING: --overlap-metric mean_in on the $REWARD_VARIANT map. It normalises by the" >&2
    echo "         map's peak, which rewards FLATTENING; mean_in_v2 is the same numerator" >&2
    echo "         over the map's mean and has no such hole. Continuing." >&2
fi

# ---------- --maskfree: the two no-box rewards ----------
# WANT_DINO is the single switch every later block reads. It is resolved here, before the
# GPU layout, because "is there a DINO server" changes how many GPUs training gets.
WANT_DINO=true
if [[ -n "$MASKFREE" ]]; then
    case "$MASKFREE" in
        flatness|mass) ;;
        *) echo "ERROR: --maskfree must be flatness|mass (got '$MASKFREE')" >&2; exit 1 ;;
    esac
    if [[ "$REWARD_VARIANT" != "ours" ]]; then
        echo "ERROR: --maskfree $MASKFREE needs the attention map (--saliency-method attention);" >&2
        echo "       got --saliency-method $REWARD_VARIANT. The weights below were measured" >&2
        echo "       on that map; another map needs its own probe run first." >&2
        exit 1
    fi
    if [[ -n "$PLACEBO" ]]; then
        echo "ERROR: --maskfree $MASKFREE and --placebo $PLACEBO both REPLACE the overlap" >&2
        echo "       reward in the same reward_funcs slot. Pick one." >&2
        exit 1
    fi
    # THE POINT OF THE FLAG. No boxes are ever needed, so no server is started and the
    # GPU it would have held goes to training. --maskfree-parity is the one path that
    # still grounds (it re-imposes the reference's scored/unscored set as a boolean gate),
    # and it needs the server back.
    if [ "$MASKFREE_PARITY" != true ]; then
        WANT_DINO=false
    fi
fi

# ---------- --overlap-rect-frac: the overlap reward on a fixed rectangle ----------
# Sets the same WANT_DINO switch for the same reason -- no boxes are ever requested -- but
# the reward is still think_overlap_reward, so unlike --maskfree it keeps --overlap-metric,
# the reward slot and (subject to the warning below) the weight.
if [[ -n "$RECT_FRAC" ]]; then
    if ! awk -v f="$RECT_FRAC" 'BEGIN{exit !(f+0 > 0 && f+0 <= 1)}' </dev/null; then
        echo "ERROR: --overlap-rect-frac must be in (0, 1] (got '$RECT_FRAC')." >&2
        echo "       It is the fraction of the patch grid the rectangle covers." >&2
        exit 1
    fi
    if [[ "$REWARD_VARIANT" != "ours" ]]; then
        echo "ERROR: --overlap-rect-frac needs the attention map (--saliency-method attention);" >&2
        echo "       got --saliency-method $REWARD_VARIANT. The gradient and glimpse rewards" >&2
        echo "       build their own masks from the same boxes; that is a separate change." >&2
        exit 1
    fi
    if [[ -n "$PLACEBO" || -n "$MASKFREE" || -n "$MISMATCH_BANK" ]]; then
        echo "ERROR: --overlap-rect-frac replaces the overlap reward's MASK, while --placebo /" >&2
        echo "       --maskfree / --mismatch-bank replace the reward itself. Pick one." >&2
        exit 1
    fi
    # The mask is the same on every step, so a union cap is all-or-nothing here: above the
    # rectangle's own area it drops every step and the run trains with no overlap term at
    # all, which looks in the logs like a weak reward rather than a broken one.
    if [[ -n "$MAX_UNION_AREA" ]] && \
       awk -v f="$RECT_FRAC" -v c="$MAX_UNION_AREA" 'BEGIN{exit !(c+0 > 0 && f+0 > c+0)}' </dev/null; then
        echo "ERROR: --overlap-rect-frac $RECT_FRAC exceeds --max-union-area $MAX_UNION_AREA." >&2
        echo "       The rectangle is identical on every step, so that cap would drop all of" >&2
        echo "       them and every completion would go unscored. Lower one or drop the cap." >&2
        exit 1
    fi
    # THE POINT OF THE FLAG: no detector, so no sidecar and its GPU goes to training.
    WANT_DINO=false
fi

# ---------- --mismatch-bank: the wrong-question, wrong-picture control ----------
if [[ -n "$MISMATCH_BANK" ]]; then
    if [[ ! -f "$MISMATCH_BANK" ]]; then
        echo "ERROR: --mismatch-bank $MISMATCH_BANK does not exist. Build one first:" >&2
        echo "       bash launch_mismatch_bank.sh --out-dir outputs/mismatch_bank/8k" >&2
        exit 1
    fi
    if [[ "$REWARD_VARIANT" != "ours" ]]; then
        echo "ERROR: --mismatch-bank needs the attention map (--saliency-method attention);" >&2
        echo "       got --saliency-method $REWARD_VARIANT. The weight below was measured on" >&2
        echo "       that map; another map needs its own probe run first." >&2
        exit 1
    fi
    if [[ -n "$PLACEBO" || -n "$MASKFREE" || -n "$RECT_FRAC" ]]; then
        echo "ERROR: --mismatch-bank replaces the overlap reward, and so does" >&2
        echo "       ${PLACEBO:+--placebo $PLACEBO}${MASKFREE:+--maskfree $MASKFREE}${RECT_FRAC:+--overlap-rect-frac $RECT_FRAC} (which replaces its mask). Pick one." >&2
        exit 1
    fi
    # --overlap_question_boxes is the sibling control -- the row's OWN question, grounded
    # offline -- and it lives inside think_overlap_reward, which this replaces. Two box
    # sources, one of which would be silently ignored.
    if [[ -n "${QUESTION_BOXES:-}" ]]; then
        echo "ERROR: --mismatch-bank and --overlap-question-boxes are two different offline" >&2
        echo "       box sources for the same slot: the first replaces the reward, the second" >&2
        echo "       feeds it. Pick one. They are the two rungs of the same experiment --" >&2
        echo "       the row's own question against another row's chain -- so run them as two" >&2
        echo "       runs, not one." >&2
        exit 1
    fi
    # THE POINT OF THE FLAG, exactly as for --maskfree: the boxes were grounded offline,
    # so no DINO server is started and the GPU it would have held goes to training. There
    # is no parity escape hatch here -- a parity gate would have to run the real DINO on
    # the real sentence, which is the dependence this control exists to sever. What that
    # costs is 0.5% of completions (the share the cold start's DINO leaves with nothing
    # gradeable); see docs/mismatch-boxes.md.
    WANT_DINO=false
fi

# ---------- --placebo: the three direction controls ----------
if [[ -n "$PLACEBO" ]]; then
    case "$PLACEBO" in
        roll|random|length) ;;
        *) echo "ERROR: --placebo must be roll|random|length (got '$PLACEBO')" >&2; exit 1 ;;
    esac
    # The placebos are controls FOR THE ATTENTION-OVERLAP REWARD and inherit its
    # scored/unscored set. Against another map they would be compared to a reference
    # that was never run.
    if [[ "$REWARD_VARIANT" != "ours" ]]; then
        echo "ERROR: --placebo $PLACEBO needs the attention map (--saliency-method attention);" >&2
        echo "       got --saliency-method $REWARD_VARIANT." >&2
        exit 1
    fi
    # The roll-null is ALREADY scored against rolled copies of the union, so --placebo
    # roll would be a control of a control; and its scorer draws random placements, so
    # using it as the scored/unscored parity gate would consume that draw and
    # double-count the diagnostics. trl/rewards/placebo_rewards.configure() refuses it too.
    if [[ "$OVERLAP_METRIC" == "logratio" ]]; then
        echo "ERROR: --placebo does not work with --overlap-metric logratio. Use mean_in" >&2
        echo "       (the reference the placebos are controls for), mean_in_v2 or auroc." >&2
        exit 1
    fi
fi

# ---------- --question-boxes: one grounding per question, done before the run ----------
# The other way to reach WANT_DINO=false, and for the same reason as --maskfree: no boxes
# are needed AT TRAINING TIME, so no server is started and the GPU it would have held goes
# to training. Here the boxes still exist -- they were grounded once per dataset row by
# precompute_question_boxes.py -- so unlike --maskfree the reward is still a box-overlap
# reward. See dino_text_sensitivity.py for why one grounding per question is not a worse
# mask than one per step.
#
# THE GPU LAYOUT CHANGES, and that changes TRAIN_N and hence gen_batch. Read the
# --maskfree note under the layout block below before comparing this run to a per-step
# reference; --num-gpus 7 reproduces the reference's gen_batch of 48.
if [[ -n "$QUESTION_BOXES" ]]; then
    if [[ "$REWARD_VARIANT" != "ours" ]]; then
        echo "ERROR: --question-boxes is read by the attention-overlap reward only," >&2
        echo "       but --saliency-method $REWARD_VARIANT puts a different reward in that" >&2
        echo "       slot and grounds per step itself. The run would be identical to one" >&2
        echo "       without the flag." >&2
        exit 1
    fi
    if [[ -n "$PLACEBO" || -n "$MASKFREE" ]]; then
        echo "ERROR: --question-boxes and --${PLACEBO:+placebo }${MASKFREE:+maskfree }both" >&2
        echo "       act on the overlap reward's slot; the placebo/maskfree rewards do their" >&2
        echo "       own grounding and would ignore the cache." >&2
        exit 1
    fi
    # Both replace the per-step union, in the same place. Whichever won, the other arm's
    # name would be on a run that is not it. overlap_rewards._validate_rect refuses the
    # pair too, so this cannot be reached round the launcher either.
    if [[ -n "$RECT_FRAC" ]]; then
        echo "ERROR: --question-boxes and --overlap-rect-frac $RECT_FRAC both REPLACE the" >&2
        echo "       per-step DINO union. Pick one." >&2
        exit 1
    fi
    if [[ ! -f "$QUESTION_BOXES" ]]; then
        echo "ERROR: --question-boxes $QUESTION_BOXES does not exist. Build it first:" >&2
        echo "       bash launch_precompute_question_boxes_job.sh --dataset <corpus>" >&2
        exit 1
    fi
    WANT_DINO=false
fi


if [ "$WANT_DINO" != true ] || [ "$SHARE_SIDECAR_GPU" = true ]; then MIN_GPUS=2; else MIN_GPUS=3; fi
if (( NUM_GPUS < MIN_GPUS )); then
    if [ "$WANT_DINO" != true ]; then
        echo "ERROR: need >=2 GPUs with $([[ -n "$RECT_FRAC" ]] && echo '--overlap-rect-frac' || { [[ -n "$QUESTION_BOXES" ]] && echo '--question-boxes' || { [[ -n "$MISMATCH_BANK" ]] && echo '--mismatch-bank' || echo '--maskfree'; }; }) (1 vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    elif [ "$SHARE_SIDECAR_GPU" = true ]; then
        echo "ERROR: need >=2 GPUs with --share-sidecar-gpu (1 shared DINO+vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    else
        echo "ERROR: need >=3 GPUs (1 DINO + 1 vLLM + >=1 training); got --num-gpus $NUM_GPUS" >&2
    fi
    exit 1
fi
if (( NUM_GPUS > 8 )); then
    echo "ERROR: single-node launcher (max 8 GPUs). localhost DINO/vLLM sidecars cannot serve a second node." >&2
    exit 1
fi

# Sidecar placement. Shared: both servers on GPU 0, training on 1..N-1. Separate (default):
# DINO on 0, vLLM on 1, training on 2..N-1. With --maskfree there IS no DINO, so vLLM
# takes GPU 0 and training gets the rest -- the same layout as --share-sidecar-gpu, but
# because a server was removed rather than moved.
#
# THAT CHANGES TRAIN_N, AND TRAIN_N IS IN THE GENERATION BATCH. gen_batch =
# per_device x TRAIN_N x grad_accum, so 8 GPUs here give 7 training procs and gen_batch
# 56 against the reference's 48 -- a different number of prompts behind each optimizer
# step and a different meaning for the LR schedule. To reproduce the reference exactly,
# pass --num-gpus 7: 1 vLLM + 6 training, gen_batch 48. The banner prints gen_batch;
# check it before deciding the run is comparable.
DINO_GPU=0
if [ "$WANT_DINO" != true ]; then
    VLLM_GPU=0
    TRAIN_N=$(( NUM_GPUS - 1 ))
    TRAIN_GPUS=$(seq -s, 1 $(( NUM_GPUS - 1 )))
elif [ "$SHARE_SIDECAR_GPU" = true ]; then
    VLLM_GPU=0
    TRAIN_N=$(( NUM_GPUS - 1 ))
    TRAIN_GPUS=$(seq -s, 1 $(( NUM_GPUS - 1 )))
else
    VLLM_GPU=1
    TRAIN_N=$(( NUM_GPUS - 2 ))
    TRAIN_GPUS=$(seq -s, 2 $(( NUM_GPUS - 1 )))
fi

# vLLM sizes its KV cache as a fraction of TOTAL GPU memory, but DINO starts first and is
# already resident, so 0.90 can over-subscribe a shared GPU. 0.85 of 185 GB still leaves
# ~28 GB -- ample for DINO's measured 1.6 GB plus fragmentation. An explicit
# --vllm-gpu-mem / VLLM_GPU_MEM always wins over both defaults.
# The 0.85 is a haircut for DINO's resident 1.6 GB, so it applies only when DINO is
# actually sharing the GPU. Under --maskfree, GPU 0 holds vLLM alone and gets the full 0.90.
if [ -z "$VLLM_GPU_MEM" ]; then
    if [ "$SHARE_SIDECAR_GPU" = true ] && [ "$WANT_DINO" = true ]; then
        VLLM_GPU_MEM=0.85
    else
        VLLM_GPU_MEM=0.90
    fi
fi

REFORWARD_SALIENCY=True

# --overlap-metric auroc is a complete, self-sufficient configuration: it carries the
# two settings that do not transfer from mean_in, so the only change needed from a
# previous run is the metric flag itself. An explicit --w-overlap / --mass-floor-tau
# still wins (pass --mass-floor-tau 0 to force the floor off). mean_in is untouched,
# so a bare invocation is still bit-identical to the runs already trained.
if [[ "$OVERLAP_METRIC" == "auroc" && "$REWARD_VARIANT" == "ours" ]]; then
    [[ -z "$MASS_FLOOR_TAU" ]] && MASS_FLOOR_TAU=0.0022
    [[ -z "${W_OVERLAP_SET:-}" ]] && W_OVERLAP=0.11
fi
# mean_in_v2 carries only the weight -- 0.033, measured (see the header). No mass floor
# by default: the tau that fits this corpus is not auroc's 0.0022 (p10 of image_mass on
# set_a is 0.00078, and 0.0022 bites on 30% of steps here), and 0.033 was measured with
# the floor OFF. If you pass --mass-floor-tau anyway, drop the weight to ~0.024.
if [[ "$OVERLAP_METRIC" == "mean_in_v2" && "$REWARD_VARIANT" == "ours" ]]; then
    [[ -z "${W_OVERLAP_SET:-}" ]] && W_OVERLAP=0.033
fi

# --placebo carries its OWN weight, and it overrides the metric's: the point of the
# experiment is that all four runs apply the same tie-breaking PRESSURE and differ only
# in direction, so the weight is set from the placebo's spread, not the metric's.
#
#   w_placebo = 0.4 x sd_within(mean_in) / sd_within(placebo)
#
# sd_within is the pooled WITHIN-GROUP sd -- the advantage subtracts the group mean, so
# a term's spread across prompts never reaches the gradient. Measured with
# `python overlap_metric_spread.py <probe_merged.json>`, which now prints it next to the
# per-sample column the incumbent weights came from, and computes these three rows by
# importing trl/rewards/placebo_rewards.py itself.
#
#   MEASURED on the cold-start policy this run starts from, temperature 1, 8 generations
#   (2026-08-20). w at w_ref=0.4:
#
#     placebo   set_a 40x8 (315 compl)   val_natural 30x8 (231 compl)   default
#     roll      -- (probe kept no maps)  0.320                          0.32
#     random    0.013                    0.010                          0.013
#     length    0.031                    0.031                          0.031
#
#   `length` is the stable one: 0.031 on both corpora. `random` is analytic up to the
#   reference -- U(0,1) has sd 0.2887 and the measured 0.2917/0.2930 confirm the hash is
#   uniform -- so its whole spread is sd_within(mean_in)'s, which is 0.0098 on set_a
#   against 0.0071 on val_natural. `roll` has only the one cold-start measurement; the
#   trained checkpoint mean_in_v2_cp_1700 on set_a puts it at 0.52 instead, so read 0.32
#   as bracketed by [0.32, 0.52]. It sits near the reference's own 0.4 for a structural
#   reason: it is the SAME metric on the SAME map with an equal-area mask, so only the
#   mask's location differs. To re-measure on the corpus you are actually training on:
#
#     bash launch_overlap_probe.sh --n-samples 40 --no-judge \
#         --out-dir outputs/overlap_probe/placebo_spread --dataset <your dataset>
#     python overlap_metric_spread.py outputs/overlap_probe/placebo_spread
#
#   (the probe must keep its maps -- the default -- or the `roll` row cannot be built).
#
# --overlap-rect-frac gets no weight of its own, deliberately: it is the SAME metric on a
# different mask, so the metric's weight is the honest starting point and inventing a
# second number would hide the one thing the arm is meant to isolate. But it is not known
# to transfer either -- the rectangle moves the value spread, and it scores every step
# where a DINO run scores only the groundable ones -- so say so once, here, after the
# metric has picked W_OVERLAP and before the run name is built.
if [[ -n "$RECT_FRAC" && -z "${W_OVERLAP_SET:-}" ]]; then
    echo "NOTE: --overlap-rect-frac $RECT_FRAC keeps w_overlap=$W_OVERLAP, the weight measured" >&2
    echo "      for --overlap-metric $OVERLAP_METRIC on DINO unions. The rectangle changes the" >&2
    echo "      value spread and scores EVERY step, so re-check it with overlap_metric_spread.py" >&2
    echo "      if this run is meant to be pressure-matched to a DINO reference." >&2
fi

# An explicit --w-overlap always wins, as it does for every other metric.
if [[ -n "$PLACEBO" && -z "${W_OVERLAP_SET:-}" ]]; then
    case "$PLACEBO" in
        roll)   W_OVERLAP=0.32 ;;
        random) W_OVERLAP=0.013 ;;
        length) W_OVERLAP=0.031 ;;
    esac
fi

# --maskfree carries its own weight on the same rule and for the same reason: the point
# is that it applies the SAME tie-breaking pressure as mean_in w0.4 and differs only in
# what it scores.
#
#   w_maskfree = 0.4 x sd_within(mean_in) / sd_within(maskfree)
#
#   MEASURED on the cold-start policy this run starts from, val_natural, temperature 1,
#   8 generations (231 completions in 30 groups), with sd_within(mean_in) = 0.00715 on
#   exactly those completions:
#
#     variant     level      sd_within   w at w_ref=0.4   default
#     flatness    0.0513     0.0064      0.447            0.45
#     mass       12.1621     0.4586      0.006            0.006
#
#   Both rows come straight out of `overlap_metric_spread.py`, which computes them by
#   importing trl/rewards/maskfree_rewards.py -- so the number a run is launched with
#   comes from the function the run will use, and re-measuring is one command.
#
#   `flatness` lands near mean_in's own 0.4 for a structural reason: it IS mean_in with
#   the union replaced by the image, so it is the same statistic on a superset of the same
#   patches and its spread can only be close. `mass` is three orders of magnitude away
#   because it is a log, whose sd is a RATIO spread (0.43 in log space is a factor of
#   ~1.5), not a level spread.
#
#   ONE-CORPUS CAVEAT. Both rows come from val_natural, the same probe that put `roll` at
#   0.32 while a trained checkpoint put it at 0.52 -- so read these as +-25% like every
#   other weight here, not as three significant figures. To re-measure on the corpus you
#   train on (no GPU needed beyond the probe itself):
#
#     bash launch_overlap_probe.sh --n-samples 40 --no-judge \
#         --out-dir outputs/overlap_probe/maskfree_spread --dataset <your dataset>
#     python overlap_metric_spread.py outputs/overlap_probe/maskfree_spread
#
# An explicit --w-overlap always wins, as it does for every other metric.
if [[ -n "$MASKFREE" && -z "${W_OVERLAP_SET:-}" ]]; then
    case "$MASKFREE" in
        flatness) W_OVERLAP=0.45 ;;
        mass)     W_OVERLAP=0.006 ;;
    esac
fi

# --mismatch-bank carries its own weight on the same rule again: the control must apply
# the SAME tie-breaking pressure as mean_in w0.4 and differ only in which sentence and
# which picture the boxes were grounded on.
#
#   w_mismatch = 0.4 x sd_within(mean_in) / sd_within(mismatch)
#
#   MEASURED by re-scoring the maps the val_natural and grad_spread probes stored for the
#   cold-start policy (538 chains over 70 images), completion-level, within generation
#   group, 24 donor chains per group:
#
#     real reward (each completion on its own boxes)         0.0115
#     mismatched, all 8 rollouts on ONE donor chain          0.0094   -> w 0.49
#
#   0.82 of the reference's spread survives losing the pairing entirely, which is the
#   headline of the offline evidence and the reason this run is worth its GPUs. The
#   weight is close to mean_in's own 0.4 for the same structural reason --maskfree
#   flatness lands near it: it is the SAME metric on the SAME map, and only the mask
#   differs.
#
#   The number that must NOT be used here is the one a per-completion donor would give:
#   drawing a donor per rollout adds 0.0117 of pure draw noise, which would roughly
#   double the spread and halve the weight -- and the run would be --placebo random.
#
#   ONE-CORPUS CAVEAT, as everywhere in this block: read it as +-25%. To re-measure on
#   the corpus you train on, once a bank exists:
#
#     bash launch_overlap_probe.sh --n-samples 40 --no-judge \
#         --out-dir outputs/overlap_probe/mismatch_spread --dataset <your dataset>
#     python overlap_metric_spread.py outputs/overlap_probe/mismatch_spread
#
# An explicit --w-overlap always wins, as it does for every other metric.
if [[ -n "$MISMATCH_BANK" && -z "${W_OVERLAP_SET:-}" ]]; then
    W_OVERLAP=0.49
fi

# --grad replaces the attention map with the PIXEL GRADIENT of each observe step's own
# tokens and the metric with the roll-null log ratio. It carries NO weight default: the
# spread of log(||g_U||/||g_null||) has not been measured on this corpus, and copying
# 0.033 or 0.11 across would apply an unknown multiple of the intended pressure. Measure
# it first and pass --w-overlap explicitly:
#
#   bash launch_overlap_probe.sh --n-samples 40 --no-judge \
#       --out-dir outputs/overlap_probe/grad_spread --map grad --score logratio
#   python overlap_metric_spread.py outputs/overlap_probe/grad_spread
#
# then take the `logratio` row's sd/sample and set --w-overlap = 0.4 x 0.0086 / sd, which
# is mean_in's wov0.4 pressure (0.0035). The 0.0086 is the ATTENTION map's own
# sd_per_sample and has to be written in by hand: do NOT read the table's own "w that
# matches mean_in" column on a --map grad run, where every row INCLUDING mean_in is
# computed on the gradient map, which nothing was ever trained with. That run also fixes
# --grad-logratio-clip, and the clip has to be read off BOTH tails of logratio_raw rather
# than p99 alone -- the metric runs negative here, so an upper-tail clip is one-sided in
# effect and truncates the side that carries the signal. The mass floor does not apply --
# it is an attention-mass gate, and the gradient reward's analogous magnitude is logged
# as grad/n_image instead.
if [[ "$REWARD_VARIANT" == "grad" ]]; then
    if [[ -z "${W_OVERLAP_SET:-}" ]]; then
        echo "ERROR: --grad needs an explicit --w-overlap." >&2
        echo "  The roll-null log-ratio's spread is not known on this corpus, so no default" >&2
        echo "  can be honest. Measure it first:" >&2
        echo "    bash launch_overlap_probe.sh --n-samples 40 --no-judge \\" >&2
        echo "        --out-dir outputs/overlap_probe/grad_spread --map grad --score logratio" >&2
        echo "    python overlap_metric_spread.py outputs/overlap_probe/grad_spread" >&2
        echo "  then pass --w-overlap = 0.4 * 0.0086 / sd, where sd is the LOGRATIO row's" >&2
        echo "  sd/sample and 0.0086 is the attention map's -- not the table's own mean_in" >&2
        echo "  column, which on a --map grad run is mean_in computed on the gradient map." >&2
        echo "  Set --grad-logratio-clip from both tails of logratio_raw in the same run." >&2
        exit 1
    fi
    [[ -n "$MASS_FLOOR_TAU" ]] && { echo "ERROR: --mass-floor-tau does not apply to --grad." >&2; exit 1; }
fi

# GLIMPSE needs the same explicit --w-overlap. The four spreads ARE measured now (the
# table below), but the weight still cannot be defaulted: it depends on --max-union-area,
# which changes it by up to 1.5x, and mean_in_v2's incumbent 0.033 was calibrated on the
# ATTENTION map, not this one. It also gets a cost warning the other two do not need.
if [[ "$REWARD_VARIANT" == "glimpse" ]]; then
    if [[ -z "${W_OVERLAP_SET:-}" ]]; then
        echo "ERROR: --glimpse needs an explicit --w-overlap." >&2
        echo "  The incumbent's 0.033 was calibrated on the ATTENTION map and does not" >&2
        echo "  transfer. MEASURED on set_a (base cold-start, 40 samples x 8 gens, default" >&2
        echo "  glimpse knobs, 1099 grounded steps / 307 completions, 2026-08-24):" >&2
        echo "" >&2
        echo "      metric        level    sd/sample    w_glimpse    w with --max-union-area 0.5" >&2
        echo "      mean_in      0.1426       0.0268        0.13                          0.088" >&2
        echo "      mean_in_v2   1.1910       0.1407       0.024                          0.013" >&2
        echo "      auroc        0.5523       0.0485       0.071                          0.060" >&2
        echo "      logratio     0.0998       0.1071       0.032                          0.020" >&2
        echo "" >&2
        echo "  THE UNION CAP IS THE FORK, not a rounding difference: --max-union-area 0.5" >&2
        echo "  skips 61% of grounded steps (median union area 0.569) and 32% of the" >&2
        echo "  completions, which moves every weight by up to 1.5x. Take the last column" >&2
        echo "  if this run passes the cap, the w_glimpse column if it does not." >&2
        echo "  w = 0.4 x 0.0086 / sd, anchored on the ATTENTION map's spread -- do NOT" >&2
        echo "  read the mean_in column of that report, which self-anchors to 0.4 on a" >&2
        echo "  glimpse run (the pairing trap in overlap_metric_spread.py)." >&2
        echo "  An earlier draw (2026-08-12, 1077 steps / 308 completions) gave 0.020 and" >&2
        echo "  0.063 for the uncapped mean_in_v2 / auroc; every glimpse run on record was" >&2
        echo "  launched with those, 13-20% under this draw and inside the +-25% the" >&2
        echo "  cross-map anchor carries. Both levels reproduce, so neither needs changing." >&2
        echo "  To re-measure on another corpus (8 GPUs, ~30 min at these knobs):" >&2
        echo "    bash launch_overlap_probe.sh --n-samples 40 --gpus 8 --no-judge \\" >&2
        echo "        --out-dir outputs/overlap_probe/glimpse_spread \\" >&2
        echo "        --map glimpse --trained-adapter none --dataset <set>" >&2
        echo "    python overlap_metric_spread.py outputs/overlap_probe/glimpse_spread/probe_merged.json" >&2
        echo "  That report is always the UNCAPPED one -- it has no --max-union-area flag." >&2
        echo "  For the capped column, filter the steps by box_area_frac in probe_merged.json," >&2
        echo "  which is exactly the union fraction --max-union-area gates on." >&2
        exit 1
    fi
    # Same refusal as --grad, same reason: the mass floor gates on the fraction of an
    # ATTENTION row spent on image tokens, and a GLIMPSE relevance row is not that.
    [[ -n "$MASS_FLOOR_TAU" ]] && { echo "ERROR: --mass-floor-tau does not apply to --glimpse (it gates attention mass; glimpse/n_image is the analogous magnitude)." >&2; exit 1; }
    cat >&2 <<GLIMPSE_WARN
------------------------------------------------------------------------------
 --glimpse is EXPENSIVE and screened NEGATIVE. Both measured, both on record:

   cost   11.3-18.3 s per case at layer_frac 1.0 against the gradient reward's
          0.20-0.25 -- 55-59x, or 100-145 s added to a ~40 s optimizer step.
          --glimpse-layer-frac 0.6 buys 1.64x; --glimpse-token-cap is linear.
   signal on 3,471 steps / 1,157 completions the map IS grounded (auroc level
          0.567, 0.712 on the smallest union decile -- the first map here that
          is), but its correlation with the model being RIGHT is null to
          slightly negative for both metrics, and nothing clears Bonferroni.

 The level decays hard with union area (r = -0.487), so --max-union-area is the
 knob that decides which regime this run is in. Watch glimpse/union_frac next
 to glimpse/score_raw: mean_in_v2's ceiling is n_patches/n_in, so a rising
 score with a rising union is the ceiling moving, not grounding improving.
------------------------------------------------------------------------------
GLIMPSE_WARN
fi

# ---------- naming: every swept HP appears in the model AND wandb name ----------
N_HEADS=$(echo "$OVERLAP_HEADS" | awk -F, '{print NF}')
if [[ "$REWARD_VARIANT" == "grad" ]]; then
    # The attention knobs are not in the name because they do not apply: no layer, no
    # heads, no token reduction. What does apply is in it, so a name states what ran.
    SUFFIX="__wov${W_OVERLAP}_grad${GRAD_TARGET}_${OVERLAP_METRIC}"
    [[ "$OVERLAP_METRIC" == "logratio" ]] && SUFFIX="${SUFFIX}_rn${GRAD_NULL_OFFSETS}_clip${GRAD_LOGRATIO_CLIP}"
elif [[ "$REWARD_VARIANT" == "glimpse" ]]; then
    # No layer, no heads, no token reduction -- those are attention-map knobs. What IS
    # swept here is the metric (the two variants), the target, and the two cost dials,
    # and a cost dial changes the map, so it belongs in the name.
    SUFFIX="__wov${W_OVERLAP}_glimpse${OVERLAP_METRIC}_${GLIMPSE_TARGET}"
    [[ "$GLIMPSE_LAYER_FRAC" != "1.0" ]] && SUFFIX="${SUFFIX}_lf${GLIMPSE_LAYER_FRAC}"
    [[ "$GLIMPSE_TOKEN_CAP" != "0" ]]    && SUFFIX="${SUFFIX}_tc${GLIMPSE_TOKEN_CAP}"
    [[ "$GLIMPSE_DEPTH_TEMP" != "0.2" ]] && SUFFIX="${SUFFIX}_dt${GLIMPSE_DEPTH_TEMP}"
    [[ "$GLIMPSE_TOKEN_WEIGHT" != "full" ]] && SUFFIX="${SUFFIX}_tw${GLIMPSE_TOKEN_WEIGHT}"
    [[ "$OVERLAP_METRIC" == "logratio" ]] && SUFFIX="${SUFFIX}_rn${ROLLNULL_OFFSETS}_clip${ROLLNULL_CLIP}"
else
SUFFIX="__wov${W_OVERLAP}_${N_HEADS}head_tr${TOKEN_REDUCTION}"
# Only non-default metric settings extend the suffix, so existing mean_in run names
# (and the checkpoints already on disk) stay exactly as they are.
[[ "$OVERLAP_METRIC" != "mean_in" ]] && SUFFIX="${SUFFIX}_${OVERLAP_METRIC}"
# The roll-null is random and clipped, so two runs differing only in these
# are different experiments and must not share a checkpoint dir.
if [[ "$OVERLAP_METRIC" == "logratio" ]]; then
    SUFFIX="${SUFFIX}_rn${ROLLNULL_OFFSETS}_clip${ROLLNULL_CLIP}"
fi
fi
# A placebo run is NOT the reward its metric names, so it must not share a checkpoint
# directory or a wandb run name with a real one. The weight already differs, but relying
# on that would break the moment someone passes --w-overlap explicitly.
[[ -n "$PLACEBO" ]] && SUFFIX="${SUFFIX}_placebo${PLACEBO}"
# Same rule for --maskfree: it is not the reward its metric names either, and a
# --maskfree-parity run is a third thing again (same value, DINO-gated scored set).
[[ -n "$MASKFREE" ]] && SUFFIX="${SUFFIX}_maskfree${MASKFREE}"
[[ -n "$MASKFREE" && "$MASKFREE_PARITY" == true ]] && SUFFIX="${SUFFIX}_parity"
# Same rule again: --overlap-rect-frac IS the reward its metric names, but scored against
# a different mask, which is exactly the kind of difference that must not share a
# checkpoint dir with its reference. The fraction is in the name because it is the arm.
[[ -n "$RECT_FRAC" ]] && SUFFIX="${SUFFIX}_rect${RECT_FRAC}"
# Same rule a third time. The seed is in the name because it IS the experiment: two
# mismatch runs at different seeds are different random pairings of the same corpus, and
# they must not land in one checkpoint directory.
[[ -n "$MISMATCH_BANK" ]] && SUFFIX="${SUFFIX}_mismatch${MISMATCH_SEED}"
[[ -n "$MASS_FLOOR_TAU" ]] && SUFFIX="${SUFFIX}_mf${MASS_FLOOR_TAU}"
[[ -n "$MAX_UNION_AREA" ]] && SUFFIX="${SUFFIX}_mu${MAX_UNION_AREA}"
[[ "$MAX_BOX_AREA" == "0" ]] && SUFFIX="${SUFFIX}_nobox"
# The reward differs from a plain run, so the checkpoints and the wandb run must not
# share a name with one.
[[ "$NATURAL_ONLY" == true ]] && SUFFIX="${SUFFIX}_natonly"
# One grounding per question is a different reward, not a cheaper way of computing the
# same one -- it changes which steps are scored. It must never share a checkpoint dir or
# a wandb name with a per-step run.
[[ -n "$QUESTION_BOXES" ]] && SUFFIX="${SUFFIX}_qbox"
# Regulators change the objective, so a regulated run must not share a checkpoint
# directory or a wandb run name with the unregulated control it exists to be compared
# against. Only non-default settings extend the suffix, which keeps every existing run
# name and every checkpoint already on disk untouched. The band and knee are appended only
# when they differ from the calibrated defaults, so the common case reads `_lenguard217`.
[[ "$BETA" != "0" && "$BETA" != "0.0" ]] && SUFFIX="${SUFFIX}_beta${BETA}"
if [[ -n "$LENGTH_GUARD_REF" ]]; then
    SUFFIX="${SUFFIX}_lenguard${LENGTH_GUARD_REF}"
    [[ "$LENGTH_GUARD_WEIGHT"  != "0.20" ]] && SUFFIX="${SUFFIX}w${LENGTH_GUARD_WEIGHT}"
    [[ "$LENGTH_GUARD_BAND_LO" != "0.30" ]] && SUFFIX="${SUFFIX}lo${LENGTH_GUARD_BAND_LO}"
    [[ "$LENGTH_GUARD_BAND_HI" != "3.0"  ]] && SUFFIX="${SUFFIX}hi${LENGTH_GUARD_BAND_HI}"
    [[ "$LENGTH_GUARD_KNEE"    != "1.0"  ]] && SUFFIX="${SUFFIX}kn${LENGTH_GUARD_KNEE}"
fi
# A different adapter is a different experiment, so it must not share a name (or a
# checkpoint dir) with a q+v run. q_proj,v_proj is the historical set and stays unmarked,
# which keeps every existing run name and every checkpoint already on disk untouched.
LORA_SLUG=$(echo "$LORA_TARGETS" | sed 's/_proj//g; s/,//g')
[[ "$LORA_TARGETS" != "q_proj,v_proj" ]] && SUFFIX="${SUFFIX}_lora${LORA_SLUG}"
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
case "$REWARD_VARIANT" in
    grad)    REWARD_SLUG=grad ;;
    glimpse) REWARD_SLUG=glimpse ;;
    *)       REWARD_SLUG=overlap ;;
esac
RUN_NAME="grpo-${MODEL_SLUG}-${REWARD_SLUG}${SUFFIX}"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$REPO/checkpoint/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# ---------- resume gate: the checkpoint remembers the map, the command line must too ----
# --saliency-method is not sticky. Nothing in a checkpoint sets it, so every relaunch --
# a new allocation, an autoresume chunk, a repeat typed from memory -- has to carry it
# again. Leave it off and REWARD_VARIANT falls back to its default `ours`:
# think_overlap_reward takes the same reward_funcs slot at the same --reward_weights
# position, training continues from the checkpoint on a DIFFERENT reward at a weight
# calibrated for another map, the run name does not change, and nothing says so. That is
# how grad runs wov0.017 and wov0.027 spent their last hundreds of steps on the attention
# map, in the same WandB run (WANDB_RESUME=allow), visible only as train/rewards/
# think_grad_reward/mean going flat while think_overlap_reward/mean started.
# A checkpoint does record which reward ran, indirectly: trainer_state.json's log_history
# carries the reward's own key, rewards/<function name>/mean, which GRPOTrainer builds
# from the function's __name__. Read it back out and refuse a launch that disagrees.
_ckpt_reward_fn() {   # the last saliency reward function a checkpoint logged, if any
    local st="$1/trainer_state.json"
    [[ -f "$st" ]] || return 0
    # think_format_reward is in every variant and says nothing about the map. Ordering is
    # log_history's, which is chronological, so the last hit is the most recent step.
    grep -o '"rewards/think_[a-z]*_reward/mean"' "$st" 2>/dev/null \
        | grep -v think_format_reward | tail -1 | tr -d '"' | sed 's|rewards/||; s|/mean||' || true
}
# What THIS command line would put in that slot. A --reward_variant passed through as an
# extra arg lands AFTER the launcher's own copy on the python command line and therefore
# wins, so read it back rather than judging a run by a flag it overrides.
_EFFECTIVE_VARIANT="$REWARD_VARIANT"
[[ "$EXTRA_ARGS" =~ --reward_variant[[:space:]]+([a-z]+) ]] && _EFFECTIVE_VARIANT="${BASH_REMATCH[1]}"
case "$_EFFECTIVE_VARIANT" in
    grad)    WANT_REWARD_FN=think_grad_reward;    _MAP_SHOWN=grad ;;
    glimpse) WANT_REWARD_FN=think_glimpse_reward; _MAP_SHOWN=glimpse ;;
    ours)    if   [[ -n "$PLACEBO"  ]]; then WANT_REWARD_FN=think_placebo_reward
             elif [[ -n "$MASKFREE" ]]; then WANT_REWARD_FN=think_maskfree_reward
             elif [[ -n "$MISMATCH_BANK" ]]; then WANT_REWARD_FN=think_mismatch_reward
             else                            WANT_REWARD_FN=think_overlap_reward; fi
             _MAP_SHOWN=attention ;;
    # 'none' and anything else: no saliency slot, so nothing to check and nothing to name.
    *)       WANT_REWARD_FN="";                   _MAP_SHOWN="$_EFFECTIVE_VARIANT" ;;
esac
_GATE_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
if [[ -n "$_GATE_CKPT" && -n "$WANT_REWARD_FN" ]]; then
    HAVE_REWARD_FN=$(_ckpt_reward_fn "$OUTPUT_DIR/checkpoint-$_GATE_CKPT")
    # An empty read is "cannot tell" (a pre-overlap checkpoint, a --reward_variant none
    # run, a trainer_state.json that never logged a saliency reward), not "disagrees".
    if [[ -n "$HAVE_REWARD_FN" && "$HAVE_REWARD_FN" != "$WANT_REWARD_FN" ]]; then
        if [[ "$ALLOW_MAP_CHANGE" == true ]]; then
            echo "WARNING: resuming checkpoint-$_GATE_CKPT ($HAVE_REWARD_FN) as $WANT_REWARD_FN." >&2
            echo "         --allow-map-change given, continuing. The run's curve now has two" >&2
            echo "         rewards in it and its name states only one." >&2
        else
            echo "ERROR: this launch would resume a checkpoint trained with a DIFFERENT reward." >&2
            echo "  output dir:  $OUTPUT_DIR" >&2
            echo "  resuming:    checkpoint-$_GATE_CKPT, whose last logged saliency reward is" >&2
            echo "               $HAVE_REWARD_FN" >&2
            echo "  this launch: $WANT_REWARD_FN (--saliency-method $_MAP_SHOWN$([[ -n "$PLACEBO" ]] && echo " --placebo $PLACEBO")$([[ -n "$MASKFREE" ]] && echo " --maskfree $MASKFREE")$([[ -n "$MISMATCH_BANK" ]] && echo " --mismatch-bank $MISMATCH_BANK"))" >&2
            echo "" >&2
            echo "  Nothing in a checkpoint sets --saliency-method, so a relaunch that omits it" >&2
            echo "  falls back to the attention map and keeps training, silently, at a weight" >&2
            echo "  chosen for another one. If this is a relaunch, add the flag it is missing:" >&2
            case "$HAVE_REWARD_FN" in
                think_grad_reward)    echo "      --saliency-method grad" >&2 ;;
                think_glimpse_reward) echo "      --saliency-method glimpse" >&2 ;;
                think_placebo_reward) echo "      --saliency-method attention --placebo <roll|random|length>" >&2 ;;
                think_maskfree_reward) echo "      --saliency-method attention --maskfree <flatness|mass>" >&2 ;;
                think_mismatch_reward) echo "      --saliency-method attention --mismatch-bank <bank.json>" >&2 ;;
                *)                    echo "      --saliency-method attention" >&2 ;;
            esac
            echo "  If the map change is deliberate, pass --allow-map-change (and expect one" >&2
            echo "  curve with two rewards in it), or point --output-dir at a new directory." >&2
            exit 1
        fi
    fi
fi

# ---------- resume gate 2: the regulators are not sticky either ------------------------
# Same failure as the map gate above, one level worse: a relaunch that forgets --beta or
# --length-guard does not crash, it silently continues the run with the anchor removed --
# and the checkpoint keeps the name that says the anchor is on. An autoresume chunk is the
# likely place for it. Both regulators leave a readable trace in trainer_state.json:
# beta != 0 makes the trainer log a "kl" key every step (grpo_trainer_qwen3.py:3025-3030),
# and the length guard adds rewards/length_guard_reward/mean under its function's __name__.
# Read them back and refuse a launch that disagrees, in EITHER direction -- adding a
# regulator halfway through a run changes the objective just as silently as dropping one.
_ckpt_has_key() {   # 1 if $2 appears as a JSON key anywhere in the checkpoint's log_history
    local st="$1/trainer_state.json"
    [[ -f "$st" ]] || return 1
    grep -q "\"$2\"" "$st" 2>/dev/null
}
if [[ -n "$_GATE_CKPT" ]]; then
    _CK="$OUTPUT_DIR/checkpoint-$_GATE_CKPT"
    _REG_MSGS=()
    _HAVE_KL=false;  _ckpt_has_key "$_CK" "kl"                            && _HAVE_KL=true
    _HAVE_LG=false;  _ckpt_has_key "$_CK" "rewards/length_guard_reward/mean" && _HAVE_LG=true
    _WANT_KL=false;  [[ "$BETA" != "0" && "$BETA" != "0.0" ]]             && _WANT_KL=true
    _WANT_LG=false;  [[ -n "$LENGTH_GUARD_REF" ]]                         && _WANT_LG=true
    [[ "$_HAVE_KL" == true && "$_WANT_KL" == false ]] && _REG_MSGS+=("checkpoint logged 'kl' (beta != 0) but this launch has --beta $BETA. Add: --beta <the value it used>")
    [[ "$_HAVE_KL" == false && "$_WANT_KL" == true ]] && _REG_MSGS+=("checkpoint logged no 'kl' (it trained at beta=0) but this launch passes --beta $BETA")
    [[ "$_HAVE_LG" == true && "$_WANT_LG" == false ]] && _REG_MSGS+=("checkpoint logged rewards/length_guard_reward but this launch has no --length-guard. Add: --length-guard <the reference length it used>")
    [[ "$_HAVE_LG" == false && "$_WANT_LG" == true ]] && _REG_MSGS+=("checkpoint logged no length guard but this launch passes --length-guard $LENGTH_GUARD_REF")
    if [[ ${#_REG_MSGS[@]} -gt 0 ]]; then
        if [[ "$ALLOW_REGULATOR_CHANGE" == true ]]; then
            echo "WARNING: resuming checkpoint-$_GATE_CKPT with DIFFERENT regulators:" >&2
            for _m in "${_REG_MSGS[@]}"; do echo "         - $_m" >&2; done
            echo "         --allow-regulator-change given, continuing. The run's curve now has" >&2
            echo "         two objectives in it and its name states only one." >&2
        else
            echo "ERROR: this launch would resume a checkpoint trained with DIFFERENT regulators." >&2
            echo "  output dir:  $OUTPUT_DIR" >&2
            echo "  resuming:    checkpoint-$_GATE_CKPT" >&2
            for _m in "${_REG_MSGS[@]}"; do echo "  - $_m" >&2; done
            echo "" >&2
            echo "  Nothing in a checkpoint sets --beta or --length-guard, so a relaunch that" >&2
            echo "  omits one keeps training with the regulator gone and says nothing. If the" >&2
            echo "  change is deliberate, pass --allow-regulator-change, or point --output-dir" >&2
            echo "  at a new directory." >&2
            exit 1
        fi
    fi
fi

REWARD_WEIGHTS="1.0 ${W_OVERLAP} 1.0 1.0"

echo "=========================================================================="
echo "Model:            $MODEL"
echo "GPUs (total $NUM_GPUS):  $([ "$WANT_DINO" = true ] && echo "DINO=cuda:$DINO_GPU  " || echo 'DINO=none  ')vLLM=cuda:$VLLM_GPU  train=cuda:[$TRAIN_GPUS] ($TRAIN_N procs)$([ "$SHARE_SIDECAR_GPU" = true ] && [ "$WANT_DINO" = true ] && echo '  [sidecars SHARED on cuda:0]')"
echo "Generation:       vLLM server  127.0.0.1:$VLLM_PORT  gpu_mem=$VLLM_GPU_MEM  max_len=$VLLM_MAX_MODEL_LEN"
# Four different reasons to have no DINO server, and they must not read alike:
# --maskfree scores no boxes at all, --overlap-rect-frac scores a rectangle instead of
# boxes, --question-boxes scores boxes that were grounded before the run, and
# --mismatch-bank scores ANOTHER ROW'S boxes, grounded before the run. This banner is
# the run's only record of which one it was, so the mismatch arm must not print itself
# as "--maskfree  scores no boxes" (with an empty MASKFREE) the way it used to.
#
# The braces around the tail are load-bearing: `A && echo X || B && echo Y` groups as
# `((A && echo X) || B) && echo Y`, so a true A prints BOTH arms. Every three-way
# reason string below is braced for that reason.
if [[ -n "$QUESTION_BOXES" ]]; then
echo "DINO reward:      NOT STARTED -- boxes were grounded before the run  box_threshold=$BOX_THRESHOLD max_box_area=$([[ "$MAX_BOX_AREA" == "0" ]] && echo 'off (no per-box cap)' || echo "$MAX_BOX_AREA") max_union_area=$([[ -n "$MAX_UNION_AREA" ]] && echo "$MAX_UNION_AREA" || echo 'off')"
echo "Grounding:        ONCE per question, from $QUESTION_BOXES"
elif [ "$WANT_DINO" != true ]; then
echo "DINO reward:      NOT STARTED -- $([[ -n "$RECT_FRAC" ]] && echo "--overlap-rect-frac $RECT_FRAC scores a fixed rectangle" || { [[ -n "$MISMATCH_BANK" ]] && echo "--mismatch-bank: another row's boxes, grounded before the run" || echo "--maskfree $MASKFREE scores no boxes"; })"
else
echo "DINO reward:      127.0.0.1:$DINO_PORT  box_threshold=$BOX_THRESHOLD max_box_area=$([[ "$MAX_BOX_AREA" == "0" ]] && echo 'off (no per-box cap)' || echo "$MAX_BOX_AREA") max_union_area=$([[ -n "$MAX_UNION_AREA" ]] && echo "$MAX_UNION_AREA" || echo 'off')"
echo "Grounding:        once per observe step, on the step text"
fi
echo "Saliency map:     $_MAP_SHOWN -> ${WANT_REWARD_FN:-(no saliency reward)}$([[ -n "${_GATE_CKPT:-}" ]] && echo "   (resuming checkpoint-$_GATE_CKPT)")"
echo "Overlap reward:   layer=$OVERLAP_LAYER heads=[$OVERLAP_HEADS] token_reduction=$TOKEN_REDUCTION w_overlap=$W_OVERLAP"
if [[ -n "$MASKFREE" ]]; then
echo "Metric:           (none -- --maskfree replaces the metric; --overlap-metric is ignored)"
else
echo "Metric:           $OVERLAP_METRIC$([[ -n "$MASS_FLOOR_TAU" ]] && echo " mass_floor_tau=$MASS_FLOOR_TAU" || echo " (no mass floor)")"
fi
if [[ -n "$RECT_FRAC" ]]; then
echo "Mask:             CENTRED RECTANGLE covering $RECT_FRAC of the patch grid -- no Grounding-DINO."
echo "                  Same metric, same reward slot, same weight; only the mask differs from a"
echo "                  DINO run. Every observe step is scored (nothing can be ungroundable), so"
echo "                  the scored set is LARGER than its reference's -- see outputs/centre_box_probe."
fi
if [[ -n "$QUESTION_BOXES" ]]; then
echo "Mask:             the row's QUESTION union, grounded once before the run -- no Grounding-DINO."
echo "                  Same metric, same reward slot, same weight; only where the boxes came from"
echo "                  differs. The scored SET differs too: a row grounds for all of its steps or"
echo "                  for none, where per-step grounding skips only the steps that ground nothing,"
echo "                  and --max-union-area is likewise now per row -- see dino_text_sensitivity.py."
fi
if [[ -n "$MASKFREE" ]]; then
    case "$MASKFREE" in
        flatness) _MF_WHAT="mean(m)/max(m) over the whole grid -- mean_in with the union replaced by the image" ;;
        mass)     _MF_WHAT="log(sum(m)) + anchor -- the probability mass the step's tokens put on the image" ;;
    esac
    echo "MASK-FREE:        $MASKFREE -- $_MF_WHAT"
    echo "                  w=$W_OVERLAP$([[ -n "${W_OVERLAP_SET:-}" ]] && echo ' (explicit --w-overlap)' || echo " (= 0.4 x sd_within(mean_in)/sd_within($MASKFREE), val_natural, cold start)")"
    echo "                  NO boxes, NO Grounding-DINO. Segmentation and the layer-22 re-forward are kept,"
    echo "                  so the maps are the ones think_overlap_reward would have scored."
    if [ "$MASKFREE_PARITY" = true ]; then
    echo "                  --maskfree-parity ON: DINO IS running, used only as the scored/unscored gate."
    else
    echo "                  scored set: every completion with a gradeable observe step (measured identical"
    echo "                  to the DINO-gated set on val_natural: 231/240 either way). --maskfree-parity re-checks."
    fi
fi
# gen_batch != the reference's 48. Printed for EVERY cause, not just --maskfree's, which
# is where this note used to live and where it caught one of the four flags that reach it.
#
# Every flag that removes Grounding-DINO gives its GPU to training, so TRAIN_N goes from
# NUM_GPUS-2 to NUM_GPUS-1 and eight cards produce SEVEN training procs: gen_batch 56
# against 48, i.e. seven prompts behind each optimizer step instead of six, and a
# different meaning for the same LR schedule. That is four flags now (--maskfree,
# --overlap-rect-frac, --question-boxes, --mismatch-bank) plus --share-sidecar-gpu and any
# odd --num-gpus, so the note belongs to TRAIN_N rather than to any one of them.
#
# Two ways back to 48, and which ones exist depends on TRAIN_N. More GPUs always works:
# TRAIN_N is NUM_GPUS minus the sidecars, so asking for (sidecars + 6) restores six procs.
# --grad-accum only works when TRAIN_N divides 48 -- it does at 3 procs (grad_accum 16,
# which is what the 4-GPU recipe in the header uses) and does NOT at 7, where 48/7 is not
# an integer. So the suggestion is computed rather than fixed; a wrong one here is how a
# run ends up looking comparable and not being.
if (( TRAIN_N != 6 )); then
    _WHY=""
    [ "$WANT_DINO" != true ] && _WHY=" (no Grounding-DINO, so its GPU went to training)"
    [ "$SHARE_SIDECAR_GPU" = true ] && [ "$WANT_DINO" = true ] && _WHY=" (--share-sidecar-gpu)"
    _FIX="pass --num-gpus $(( NUM_GPUS - TRAIN_N + 6 )) for 6 procs"
    if (( 48 % (TRAIN_N * PER_DEVICE_BATCH) == 0 )); then
        _FIX="$_FIX, or --grad-accum $(( 48 / (TRAIN_N * PER_DEVICE_BATCH) )) at this proc count"
    fi
    echo "GEN BATCH:        $TRAIN_N training procs -> gen_batch $(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM ))$_WHY."
    echo "                  The reference runs used 48 (1 x 6 x 8), so this run is NOT step-for-step"
    echo "                  comparable with them: $_FIX."
fi
if [[ -n "$PLACEBO" ]]; then
    case "$PLACEBO" in
        roll)   _PLACEBO_WHAT="the same metric on the step's own box union MOVED (same area, same shape, wrong place)" ;;
        random) _PLACEBO_WHAT="a stable hash of the completion text -> U(0,1): variance with no direction" ;;
        length) _PLACEBO_WHAT="-n_completion_tokens/1000: the brevity reward, made explicit" ;;
    esac
    echo "PLACEBO:          $PLACEBO -- $_PLACEBO_WHAT"
    echo "                  w=$W_OVERLAP$([[ -n "${W_OVERLAP_SET:-}" ]] && echo ' (explicit --w-overlap)' || echo " (= 0.4 x sd_within(mean_in)/sd_within($PLACEBO), measured on the cold-start policy)")"
    echo "                  scored on exactly the completions $OVERLAP_METRIC would score: same"
    echo "                  segmentation, same Grounding-DINO call, same union, real metric used as the gate."
fi
if [[ -n "$MISMATCH_BANK" ]]; then
    echo "MISMATCHED BOXES: real DINO unions from a DIFFERENT question about a DIFFERENT picture"
    echo "                  bank=$MISMATCH_BANK  seed=$MISMATCH_SEED (which row is paired with which)"
    echo "                  w=$W_OVERLAP$([[ -n "${W_OVERLAP_SET:-}" ]] && echo ' (explicit --w-overlap)' || echo " (= 0.4 x 0.0115/0.0094: the control keeps 0.82 of the reference's within-group spread)")"
    echo "                  one donor row per PROMPT, shared by all $NUM_GENERATIONS rollouts -- a donor per"
    echo "                  rollout would add 0.0117 of draw noise to a 0.0115 reward, i.e. --placebo random."
    echo "                  step count matched inside that donor when it can be; otherwise its nearest"
    echo "                  length, wrapped. Watch mismatch/exact_len_frac: it falls as the policy's"
    echo "                  chains outgrow the cold start's (which stop at 14 observe steps)."
    echo "                  NO Grounding-DINO anywhere in this run -- that is 16.6 s off a 40.5 s step."
fi
# Regulators. Printed even when off, because "no anchor at all" is the state every run on
# record trained in and the thing this block exists to make visible.
if [[ "$BETA" != "0" && "$BETA" != "0.0" ]]; then
    echo "KL anchor:        beta=$BETA to the base policy (LoRA adapter disabled = the reference:"
    echo "                  no second model, no extra GPU memory, one extra forward per step)."
    echo "                  Read train/kl. On 50k set_a it reached 0.049 over 5050 steps with no hack."
else
    echo "KL anchor:        OFF (beta=0 -> ref_model=None, no KL computed and none logged)"
fi
if [[ -n "$LENGTH_GUARD_REF" ]]; then
    # Cosmetic only -- never let the banner abort a launch under `set -e`.
    read -r _LG_LO _LG_HI _LG_COSTS <<<"$(python3 -c "
import sys; sys.path.insert(0,'$REPO')
from trl.rewards.length_guard_rewards import penalty
R,LO,HI,KN,K = $LENGTH_GUARD_REF,$LENGTH_GUARD_BAND_LO,$LENGTH_GUARD_BAND_HI,$LENGTH_GUARD_KNEE,$LENGTH_GUARD_WEIGHT
c = lambda x: f'{x:.2f}x:{K*penalty(x*R,R,LO,HI,KN):+.3f}'
print(f'{LO*R:.0f} {HI*R:.0f} ' + '  '.join(c(x) for x in (0.10,0.20,LO,HI,4.0)))" 2>/dev/null || echo '? ? ?')" || true
    echo "LENGTH GUARD:     ref=$LENGTH_GUARD_REF tokens, free band ${LENGTH_GUARD_BAND_LO}x..${LENGTH_GUARD_BAND_HI}x = [$_LG_LO, $_LG_HI] tokens"
    echo "                  k=$LENGTH_GUARD_WEIGHT (appended to --reward_weights as a 5th term), knee=$LENGTH_GUARD_KNEE, log-ratio shape"
    echo "                  cost at n/ref =  $_LG_COSTS"
    echo "                  ref must be THIS corpus's base-policy mean length -- read completions/mean_length at step 0,"
    echo "                  or run overlap_metric_spread.py, which prints it and the resulting pressure."
    echo "                  STRONG against length COLLAPSE (no other reward penalises a 13-token completion);"
    echo "                  weak against inflation, which max_completion_length already punishes via accuracy+format."
    echo "                  Read lenguard/frac_penalized: 0.00 means the guard is inert and length is the other rewards' doing."
else
    echo "LENGTH GUARD:     OFF (no --length-guard; reward_funcs and --reward_weights unchanged)"
fi
echo "Overlap rows:     $([ "$NATURAL_ONLY" = true ] && echo 'natural images only (non-natural: format+accuracy+judge)' || echo 'all rows')"
echo "Validation:       $([ -n "$VAL_SETS_DIR" ] && echo "accuracy only, step 0 then every $EVAL_STEPS steps, from $VAL_SETS_DIR" || echo 'off')"
echo "Checkpoints:      save every $SAVE_STEPS, keep every $CKPT_KEEP_EVERY"
echo "Benchmarks:       none during training (score afterwards with watch_bench_evals.sh)"
echo "Batch:            per_device=$PER_DEVICE_BATCH num_generations=$NUM_GENERATIONS grad_accum=$GRAD_ACCUM  (gen_batch=$(( PER_DEVICE_BATCH * TRAIN_N * GRAD_ACCUM )))"
echo "LoRA targets:     ${LORA_TARGETS//,/ }"
echo "T5 step clf:      $OVERLAP_STEPS_DEVICE  ckpt=$OVERLAP_STEPS_CKPT"
echo "Run name:         $RUN_NAME"
echo "Output dir:       $OUTPUT_DIR"
echo "Mode:             $($DIRECT && echo 'direct (no SLURM)' || echo "SLURM ($PARTITION, ${DURATION}h)")"
echo "Judge key:        $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo '(set)' || echo '(MISSING - openai_reward will fail)')"
echo "WandB:            $([[ -n "${WANDB_API_KEY:-}" ]] && echo '(online)' || echo '(offline)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:       $EXTRA_ARGS"
echo "=========================================================================="

# ---------- SLURM path ----------
if ! $DIRECT; then
    if ! command -v submit_job >/dev/null 2>&1; then
        for CI_ROOT in \
            /lustre/fs1/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface \
            /lustre/fsw/portfolios/adlr/projects/adlr_other_infra/release/cluster-interface; do
            for CAND in "$CI_ROOT/latest" $(ls -1dt "$CI_ROOT"/*/ 2>/dev/null); do
                if [ -x "${CAND%/}/submit_job" ]; then export PATH="${CAND%/}:$PATH"; break 2; fi
            done
        done
    fi
    command -v submit_job >/dev/null 2>&1 || {
        echo "ERROR: submit_job not found under cluster-interface paths. Use --direct to run without SLURM." >&2
        exit 1
    }

    LOG_ROOT="$REPO/outputs/logs"
    mkdir -p "$LOG_ROOT"

    submit_job \
        --account "$ACCOUNT" \
        --partition "$PARTITION" \
        --name "$RUN_NAME" \
        --gpu "$NUM_GPUS" \
        --duration "$DURATION" \
        --autoresume_uninstrumented \
        --outfile "$LOG_ROOT/${RUN_NAME}.%j.out" \
        --logroot "$LOG_ROOT" \
        -c "bash -c '
            export CONDA_ENV=$CONDA_ENV;
            export HF_HOME=$HF_HOME;
            export WANDB_API_KEY=${WANDB_API_KEY:-};
            export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/home/uberger/scratch/cache/wandb_data};
            export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/home/uberger/scratch/cache/wandb_cache};
            export VLLM_NO_USAGE_STATS=1;
            export DO_NOT_TRACK=1;
            ${HF_TOKEN:+export HF_TOKEN=$HF_TOKEN;}
            ${NVIDIA_API_KEY:+export NVIDIA_API_KEY=$NVIDIA_API_KEY;}
            ${OPENAI_API_KEY:+export OPENAI_API_KEY=$OPENAI_API_KEY;}
            ${OPENAI_BASE_URL:+export OPENAI_BASE_URL=$OPENAI_BASE_URL;}
            ${JUDGE_MODEL:+export JUDGE_MODEL=$JUDGE_MODEL;}
            bash $SCRIPT_PATH --direct \
                --num-gpus $NUM_GPUS \
                --model $MODEL \
                --output-dir $OUTPUT_DIR \
                --max-completion-length $MAX_COMPLETION_LENGTH \
                --num-generations $NUM_GENERATIONS \
                --grad-accum $GRAD_ACCUM \
                --per-device-batch $PER_DEVICE_BATCH \
                --learning-rate $LEARNING_RATE \
                --w-overlap $W_OVERLAP \
                --token-reduction $TOKEN_REDUCTION \
                --lora-targets $LORA_TARGETS \
                --overlap-heads $OVERLAP_HEADS \
                --overlap-layer $OVERLAP_LAYER \
                --box-threshold $BOX_THRESHOLD \
                --max-box-area $MAX_BOX_AREA \
                ${MAX_UNION_AREA:+--max-union-area $MAX_UNION_AREA} \
                --overlap-metric $OVERLAP_METRIC \
                ${MASS_FLOOR_TAU:+--mass-floor-tau $MASS_FLOOR_TAU} \
                ${PLACEBO:+--placebo $PLACEBO} \
                ${MASKFREE:+--maskfree $MASKFREE} \
                $([ "$MASKFREE_PARITY" = true ] && echo --maskfree-parity) \
                ${RECT_FRAC:+--overlap-rect-frac $RECT_FRAC} \
                ${MISMATCH_BANK:+--mismatch-bank $MISMATCH_BANK --mismatch-seed $MISMATCH_SEED} \
                --saliency-method $SALIENCY_METHOD_R \
                --grad-target $GRAD_TARGET \
                --grad-null-offsets $GRAD_NULL_OFFSETS \
                --grad-logratio-clip $GRAD_LOGRATIO_CLIP \
                --glimpse-target $GLIMPSE_TARGET \
                --glimpse-layer-frac $GLIMPSE_LAYER_FRAC \
                --glimpse-token-cap $GLIMPSE_TOKEN_CAP \
                --glimpse-depth-temp $GLIMPSE_DEPTH_TEMP \
                --glimpse-temp $GLIMPSE_TEMP \
                --glimpse-token-weight $GLIMPSE_TOKEN_WEIGHT \
                --rollnull-offsets $ROLLNULL_OFFSETS \
                --rollnull-clip $ROLLNULL_CLIP \
                --rollnull-seed $ROLLNULL_SEED \
                $([ "$NATURAL_ONLY" = true ] && echo --natural-only) \
                ${QUESTION_BOXES:+--question-boxes $QUESTION_BOXES} \
                $([ "$ALLOW_MAP_CHANGE" = true ] && echo --allow-map-change) \
                --beta $BETA \
                ${LENGTH_GUARD_REF:+--length-guard $LENGTH_GUARD_REF} \
                --length-guard-weight $LENGTH_GUARD_WEIGHT \
                --length-guard-band-lo $LENGTH_GUARD_BAND_LO \
                --length-guard-band-hi $LENGTH_GUARD_BAND_HI \
                --length-guard-knee $LENGTH_GUARD_KNEE \
                $([ "$ALLOW_REGULATOR_CHANGE" = true ] && echo --allow-regulator-change) \
                --dino-port $DINO_PORT \
                --vllm-port $VLLM_PORT \
                --vllm-gpu-mem $VLLM_GPU_MEM \
                --vllm-max-model-len $VLLM_MAX_MODEL_LEN \
                --vllm-enforce-eager $VLLM_ENFORCE_EAGER \
                --eval-steps $EVAL_STEPS \
                --bench-gpus $BENCH_GPUS \
                ${VAL_SETS_DIR:+--val-sets-dir $VAL_SETS_DIR} \
                $([ -z "$VAL_SETS_DIR" ] && echo --no-eval) \
                $([ "$SHARE_SIDECAR_GPU" = true ] && echo --share-sidecar-gpu) \
                $EXTRA_ARGS
        '"
    echo "Submitted $RUN_NAME"
    exit 0
fi

# ---------- direct path ----------
source "$CONDA_SH"
echo "Activating conda env $CONDA_ENV"
# conda activate and package activate.d hooks (e.g. cuda-nvcc's, which expands
# $NVCC_PREPEND_FLAGS with no default) assume nounset is OFF. Our `set -u` makes
# any such unguarded expansion a fatal "unbound variable". Disable nounset for
# the duration of activation only, then restore it.
set +u
conda activate "$CONDA_ENV"
set -u
# Activating across conda installs -- e.g. when this script is launched (bash)
# from a fish shell that already had a different env active -- can leave a stale
# env's bin/ ahead of ours on PATH, so `python` resolves to the WRONG interpreter
# even though CONDA_DEFAULT_ENV/CONDA_PREFIX are correct. Force this env's bin to
# the front and clear bash's command hash so the right python/torchrun are used.
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: 'conda activate $CONDA_ENV' failed (no CONDA_PREFIX)." >&2; exit 1; }
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV')." >&2
    exit 1
fi

source "$REPO/setup_cuda_home.sh"
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: after CUDA_HOME setup, python resolves to '$(command -v python)', expected '$CONDA_PREFIX/bin/python' (env '$CONDA_ENV'). CUDA_HOME='$CUDA_HOME' shadowed it." >&2
    exit 1
fi
bash "$REPO/check_cuda_home.sh" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME
export HF_HUB_OFFLINE=1
export HF_TOKEN=${HF_TOKEN:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
export WANDB_PROJECT=vlm_reasoning
export WANDB_ENTITY=nvr-israel
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
# Keep wandb artifact staging/cache OFF the tiny /home partition (10G quota); it
# otherwise piles up under ~/.local/share/wandb and ~/.cache/wandb and fills /home.
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/home/uberger/scratch/cache/wandb_data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/home/uberger/scratch/cache/wandb_cache}
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
# Don't write vLLM usage stats into ~/.config (also on /home).
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OVERLAP_STEPS_DEVICE
export OVERLAP_STEPS_CKPT
[ -d "$OVERLAP_STEPS_CKPT/encoder" ] || {
    echo "ERROR: steps-classifier ckpt not found at $OVERLAP_STEPS_CKPT (need encoder/ tokenizer/ head.pt). Set OVERLAP_STEPS_CKPT to a valid path." >&2
    exit 1
}
[ -n "${NVIDIA_API_KEY:-}" ] && export NVIDIA_API_KEY
[ -n "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY
[ -n "${OPENAI_BASE_URL:-}" ] && export OPENAI_BASE_URL
[ -n "${JUDGE_MODEL:-}" ] && export JUDGE_MODEL

LOG_DIR="$OUTPUT_DIR/sidecar_logs"
mkdir -p "$LOG_DIR"

# ---------- cleanup: kill sidecars (and their worker children) on any exit ----------
DINO_PID=""
VLLM_PID=""
CLEANUP_PID=""
cleanup() {
    echo "[cleanup] shutting down sidecars ..."
    for pid in "$VLLM_PID" "$DINO_PID" "$CLEANUP_PID"; do
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
    # vLLM spawns a detached EngineCore worker that holds GPU memory and doesn't
    # match the serve cmdline -- kill it explicitly or it orphans GPU 1.
    pkill -TERM -u "$USER" -f "trl.scripts.vllm_serve --model $MODEL" 2>/dev/null || true
    pkill -TERM -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 2
    pkill -KILL -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1" name="$2" timeout_s="$3" pid="$4" waited=0
    echo "[health] waiting for $name at $url (timeout ${timeout_s}s) ..."
    until curl -sf "$url" >/dev/null 2>&1; do
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[health] ERROR: $name process (pid $pid) died before becoming healthy." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        if grep -qE "Engine core initialization failed|EngineCore failed to start|Traceback \(most recent call last\)" "$LOG_DIR/${name}.log" 2>/dev/null; then
            echo "[health] ERROR: $name logged a fatal error (worker died); aborting." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
        sleep 5; waited=$(( waited + 5 ))
        if (( waited >= timeout_s )); then
            echo "[health] ERROR: $name not healthy after ${timeout_s}s." >&2
            tail -40 "$LOG_DIR/${name}.log" >&2 2>/dev/null || true
            exit 1
        fi
    done
    echo "[health] $name is up (after ${waited}s)."
}

# ---------- 1. Grounding-DINO reward server on GPU 0 ----------
# Skipped entirely under --maskfree: those rewards need no boxes, so there is nothing to
# serve. This is where the ~16.6 s of a 40.5 s optimizer step goes away, and it is also
# why the GPU layout above gives training one more process.
if [ "$WANT_DINO" = true ]; then
    echo "[start] Grounding-DINO on cuda:$DINO_GPU -> 127.0.0.1:$DINO_PORT"
    CUDA_VISIBLE_DEVICES=$DINO_GPU DINO_SERVER_BATCH=${DINO_SERVER_BATCH:-8} \
        python "$REPO/serve_grounding_dino.py" --host 127.0.0.1 --port "$DINO_PORT" \
        > "$LOG_DIR/dino.log" 2>&1 &
    DINO_PID=$!
else
    echo "[start] Grounding-DINO SKIPPED ($([[ -n "$RECT_FRAC" ]] && echo "--overlap-rect-frac $RECT_FRAC scores a fixed rectangle" || { [[ -n "$QUESTION_BOXES" ]] && echo "--question-boxes: grounded once per question before the run" || { [[ -n "$MISMATCH_BANK" ]] && echo "--mismatch-bank: another question's boxes, grounded before the run" || echo "--maskfree $MASKFREE needs no boxes"; }; }))"
fi

# ---------- 2. vLLM generation server on GPU 1 ----------
cd "$REPO/trl_repo"
echo "[start] vLLM server on cuda:$VLLM_GPU -> 127.0.0.1:$VLLM_PORT"
which python
VLLM_EAGER_FLAG=""
case "$VLLM_ENFORCE_EAGER" in
    True|true|1) VLLM_EAGER_FLAG="--enforce_eager True" ;;
esac
CUDA_VISIBLE_DEVICES=$VLLM_GPU \
    python -m trl.scripts.vllm_serve \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --gpu_memory_utilization "$VLLM_GPU_MEM" \
        --dtype bfloat16 \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        --enable_prefix_caching True \
        $VLLM_EAGER_FLAG \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# DINO loads fast (~1-2 min); vLLM must load the 8B + capture CUDA graphs (~5-15 min).
[ "$WANT_DINO" = true ] && wait_for_health "http://127.0.0.1:$DINO_PORT/health"  "dino" 600  "$DINO_PID"
wait_for_health "http://127.0.0.1:$VLLM_PORT/health/" "vllm" 1800 "$VLLM_PID"

# ---------- 3. checkpoint housekeeping ----------
_cleanup_checkpoints() {
    local output_dir="$1" prev_latest=""
    while true; do
        sleep 30
        local latest
        latest=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
        if [[ -n "$latest" && "$latest" != "$prev_latest" ]]; then
            prev_latest="$latest"
            ls -d "$output_dir"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | while read -r step; do
                if (( step % CKPT_KEEP_EVERY != 0 )) && [[ "$step" != "$latest" ]]; then
                    echo "[checkpoint cleanup] Removing $output_dir/checkpoint-$step"
                    rm -rf "$output_dir/checkpoint-$step"
                fi
            done
        fi
    done
}
_cleanup_checkpoints "$OUTPUT_DIR" &
CLEANUP_PID=$!

RESUME_FLAG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | tail -1 || true)
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/checkpoint-$LATEST_CKPT"

MASTER_PORT=${MASTER_PORT:-$(shuf -i 29500-65000 -n 1)}

# Omitted entirely when unset, so the dataclass default (None = floor off) applies.
# A value of 0 is still passed through: _mass_gate treats tau<=0 as "off", so an
# explicit --mass-floor-tau 0 disables the floor without falling back to the auroc
# default above.
MASS_FLOOR_FLAG=""
[[ -n "$MASS_FLOOR_TAU" ]] && MASS_FLOOR_FLAG="--mass_floor_tau $MASS_FLOOR_TAU"

# Same shape: omitted when unset, so the dataclass default (None = no union cap) applies
# and an existing run's command line is reproduced byte for byte. _union_mask treats a
# non-positive value as "off", so --max-union-area 0 disables it explicitly.
MAX_UNION_FLAG=""
[[ -n "$MAX_UNION_AREA" ]] && MAX_UNION_FLAG="--max_union_area $MAX_UNION_AREA"

# Omitted when off, so the dataclass default (None = the real overlap reward) applies and
# every existing run's command line is reproduced byte for byte.
PLACEBO_FLAG=""
[[ -n "$PLACEBO" ]] && PLACEBO_FLAG="--placebo $PLACEBO"

# Same shape again. --maskfree_parity is a store_true, so it is present or absent rather
# than true/false: passing it as `--maskfree_parity False` would set it True.
MASKFREE_FLAG=""
[[ -n "$MASKFREE" ]] && MASKFREE_FLAG="--maskfree $MASKFREE"
[[ -n "$MASKFREE" && "$MASKFREE_PARITY" == true ]] && MASKFREE_FLAG="$MASKFREE_FLAG --maskfree_parity"

# Same shape once more: omitted when unset, so the dataclass default (None = score the
# DINO union) applies and every existing run's command line is byte-identical.
RECT_FLAG=""
[[ -n "$RECT_FRAC" ]] && RECT_FLAG="--overlap_rect_frac $RECT_FRAC"
# Same shape again, and the seed goes with it: --mismatch_seed has a dataclass default of
# 0, so emitting it only with the bank keeps every existing run's command line byte-identical.
MISMATCH_FLAG=""
[[ -n "$MISMATCH_BANK" ]] && MISMATCH_FLAG="--mismatch_bank $MISMATCH_BANK --mismatch_seed $MISMATCH_SEED"

# Same shape a third time, and the reason matters more here than for the two above: the
# length guard is OFF unless --length-guard names a reference length, and off must mean
# the dataclass defaults apply and reward_funcs/reward_weights come out byte-identical to
# every run on record. trl_repo/ is shared and re-patched under jobs that are already
# QUEUED, so a run submitted before this change and started after it must be unaffected.
# The four shape knobs are only emitted with the guard, so their defaults live in exactly
# one place per side (the dataclass) rather than being echoed unconditionally.
# Same rule for --beta. GRPOConfig's own default is 0.0 (grpo_config.py:146), which is
# what every run on record trained at, so emitting nothing at BETA=0 reproduces their
# command lines exactly AND is semantically identical -- at beta == 0 the trainer sets
# ref_model = None and computes no KL at all.
BETA_FLAG=""
[[ "$BETA" != "0" && "$BETA" != "0.0" ]] && BETA_FLAG="--beta $BETA"

LENGTH_GUARD_FLAG=""
if [[ -n "$LENGTH_GUARD_REF" ]]; then
    LENGTH_GUARD_FLAG="--length_guard_ref $LENGTH_GUARD_REF"
    LENGTH_GUARD_FLAG="$LENGTH_GUARD_FLAG --length_guard_weight $LENGTH_GUARD_WEIGHT"
    LENGTH_GUARD_FLAG="$LENGTH_GUARD_FLAG --length_guard_band_lo $LENGTH_GUARD_BAND_LO"
    LENGTH_GUARD_FLAG="$LENGTH_GUARD_FLAG --length_guard_band_hi $LENGTH_GUARD_BAND_HI"
    LENGTH_GUARD_FLAG="$LENGTH_GUARD_FLAG --length_guard_knee $LENGTH_GUARD_KNEE"
fi

# Point at the DINO server only when one was started. Passing a URL to a port nothing is
# listening on would turn "no grounding needed" into a connection error on the first step
# if any code path ever reached for boxes -- better to have no address at all, so a stray
# call fails loudly at the flag rather than silently retrying a dead socket.
DINO_API_FLAG=""
[ "$WANT_DINO" = true ] && DINO_API_FLAG="--dino_api_base http://127.0.0.1:$DINO_PORT"

# Same shape again: omitted when unset, so the dataclass default (None = grounding once
# per observe step) applies and an existing run's command line is reproduced byte for byte.
QUESTION_BOXES_FLAG=""
[[ -n "$QUESTION_BOXES" ]] && QUESTION_BOXES_FLAG="--overlap_question_boxes $QUESTION_BOXES"

# Omitted when off, so the dataclass default (False) applies and the command line of an
# existing run is reproduced byte for byte.
NATURAL_ONLY_FLAG=""
GRAD_NATURAL_ONLY_FLAG=""
GLIMPSE_NATURAL_ONLY_FLAG=""
# --natural-only means the same thing in every variant (score photographs only, because
# Grounding-DINO is a photograph detector), but each reward reads its own flag.
if [[ "$NATURAL_ONLY" == true ]]; then
    case "$REWARD_VARIANT" in
        grad)    GRAD_NATURAL_ONLY_FLAG="--grad_natural_only True" ;;
        glimpse) GLIMPSE_NATURAL_ONLY_FLAG="--glimpse_natural_only True" ;;
        *)       NATURAL_ONLY_FLAG="--overlap_natural_only True" ;;
    esac
fi

# Omitted entirely unless --glimpse, so every existing run's command line is reproduced
# byte for byte and the dataclass defaults apply.
GLIMPSE_FLAGS=""
if [[ "$REWARD_VARIANT" == "glimpse" ]]; then
    GLIMPSE_FLAGS="--glimpse_target $GLIMPSE_TARGET \
    --glimpse_layer_frac $GLIMPSE_LAYER_FRAC \
    --glimpse_token_cap $GLIMPSE_TOKEN_CAP \
    --glimpse_temp $GLIMPSE_TEMP \
    --glimpse_depth_temp $GLIMPSE_DEPTH_TEMP \
    --glimpse_token_weight $GLIMPSE_TOKEN_WEIGHT"
fi

# Held-out validation. Omitted entirely when off, so eval_strategy stays "no" and the
# command line of an existing run is unchanged.
EVAL_FLAGS=""
if [[ -n "$VAL_SETS_DIR" ]]; then
    for _split in val_natural val_nonnatural; do
        [[ -d "$VAL_SETS_DIR/$_split" ]] || {
            echo "ERROR: $VAL_SETS_DIR/$_split not found. Build the validation sets first:" >&2
            echo "         python build_grpo_sets.py --build-val --out-dir $VAL_SETS_DIR" >&2
            echo "       or pass --no-eval to train without validation." >&2
            exit 1
        }
    done
    # Accuracy-only validation, driven by a callback rather than the Trainer's
    # evaluation loop, so it costs one batched vLLM call instead of re-running the
    # whole reward pipeline. A step-0 baseline is always taken.
    EVAL_FLAGS="--val_sets_dir $VAL_SETS_DIR --val_eval_steps $EVAL_STEPS"
fi

# ---------- benchmark evals: none during training ----------
# This used to start watch_bench_evals.sh, which then submitted a 1-GPU eval job per
# kept checkpoint for the length of the run. Nothing is started here any more: no
# test benchmark is scored while training is in flight, and no flag turns it back on.
#
# The one thing still done is the part that cannot be reconstructed afterwards --
# recording which model the run started from, so a later pass can score it as step 0.
mkdir -p "$OUTPUT_DIR/bench_eval"
echo "$MODEL" > "$OUTPUT_DIR/bench_eval/base_model.txt"
echo "[bench] benchmarks are not run during training. To score this run's checkpoints"
echo "[bench] afterwards (holds no GPU itself, backfills whatever has piled up):"
echo "[bench]   bash $REPO/watch_bench_evals.sh --run-dir $OUTPUT_DIR \\"
echo "[bench]        --num-gpus $BENCH_GPUS --every $CKPT_KEEP_EVERY \\"
echo "[bench]        --natural-n $BENCH_NATURAL_N --nonnatural-n $BENCH_NONNATURAL_N"

# ---------- 4. GRPO training on GPUs 2..N-1 ----------
echo "[start] training on cuda:[$TRAIN_GPUS] ($TRAIN_N procs)"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$TRAIN_N" \
    --main_process_port "$MASTER_PORT" \
    examples/scripts/grpo_vlm_qwen3.py \
    --model_name_or_path "$MODEL" \
    --attn_implementation sdpa \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --reforward_saliency "$REFORWARD_SALIENCY" \
    --reward_variant "$REWARD_VARIANT" \
    --grad_target "$GRAD_TARGET" \
    --grad_null_offsets "$GRAD_NULL_OFFSETS" \
    --grad_logratio_clip "$GRAD_LOGRATIO_CLIP" \
    $GRAD_NATURAL_ONLY_FLAG \
    $GLIMPSE_FLAGS \
    --overlap_metric "$OVERLAP_METRIC" \
    --rollnull_offsets "$ROLLNULL_OFFSETS" \
    --rollnull_clip "$ROLLNULL_CLIP" \
    --rollnull_seed "$ROLLNULL_SEED" \
    $GLIMPSE_NATURAL_ONLY_FLAG \
    --overlap_layer "$OVERLAP_LAYER" \
    --overlap_heads "$OVERLAP_HEADS" \
    --token_reduction "$TOKEN_REDUCTION" \
    --box_threshold "$BOX_THRESHOLD" \
    --max_box_area "$MAX_BOX_AREA" \
    $MAX_UNION_FLAG \
    --overlap_metric "$OVERLAP_METRIC" \
    $MASS_FLOOR_FLAG \
    $PLACEBO_FLAG \
    $MASKFREE_FLAG \
    $RECT_FLAG \
    $MISMATCH_FLAG \
    $BETA_FLAG \
    $LENGTH_GUARD_FLAG \
    $NATURAL_ONLY_FLAG \
    $QUESTION_BOXES_FLAG \
    $EVAL_FLAGS \
    $DINO_API_FLAG \
    --reward_weights $REWARD_WEIGHTS \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "$VLLM_PORT" \
    --use_peft \
    --lora_target_modules ${LORA_TARGETS//,/ } \
    --log_completions \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_generations "$NUM_GENERATIONS" \
    --report_to wandb \
    --logging_steps 5 \
    --save_steps "$SAVE_STEPS" \
    --num_train_epochs 3 \
    --temperature 1 \
    $RESUME_FLAG \
    $EXTRA_ARGS

echo "Finished $RUN_NAME"
