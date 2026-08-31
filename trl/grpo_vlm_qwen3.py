# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir grpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 2  \

"""

import glob
import json
import os
import time

import torch
from datasets import load_dataset, load_from_disk
from latex2sympy2_extended import NormalizationConfig
from PIL import Image
from math_verify import LatexExtractionConfig, parse, verify
from transformers import TrainerCallback

from trl import (
    GRPOConfig,
    GRPOTrainerQwen3 as GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward, think_saliency_reward, openai_reward


def _with_image_placeholder(messages):
    """Rewrite a text-only conversation into multimodal content, as the trainer does.

    The chat template only emits <|vision_start|><|image_pad|><|vision_end|> when the
    user message content is a list containing an {"type": "image"} entry. Our
    conversations are built as plain strings, so without this the prompt carries no
    placeholder -- and handing vLLM an image with nowhere to bind it does not raise,
    it wedges the worker inside multimodal processing, forever. Measured: identical
    request with images deadlocked past 1800s, without images returned in 7.1s.

    It is silent because vllm_serve calls llm.generate with no try/except, so a
    failure there skips connection.send() and the server blocks in recv() with no
    timeout. That deadlock propagated to the training job through the rank-0 client
    and killed two runs.

    Mirrors GRPOTrainerQwen3._generate_and_score_completions; new dicts rather than
    in-place edits, since these rows are handed back to the dataset.
    """
    converted = []
    for message in messages:
        content, role = message.get("content"), message.get("role")
        if isinstance(content, str):
            if role == "user":
                message = {**message, "content": [{"type": "image"}, {"type": "text", "text": content}]}
            else:
                message = {**message, "content": [{"type": "text", "text": content}]}
        converted.append(message)
    return converted


class ValidationAccuracyCallback(TrainerCallback):
    """Score the held-out sets on answer accuracy alone, as cheaply as possible.

    This deliberately does NOT go through the Trainer's evaluation loop. GRPO's eval
    path reproduces the whole training pipeline -- num_generations completions per
    prompt, Grounding-DINO, the saliency re-forward, the LLM judge, and a log-prob
    forward through the policy -- which cost 21.6 minutes per set (measured), or
    about 90% of training throughput at a 100-step cadence. None of that is needed
    to answer "is the model getting the answers right".

    So: one greedy completion per prompt, in a single batched vLLM call, scored by
    the same accuracy_reward the training rewards use. That is ~250 completions in
    one request rather than 2,016 in 42 requests of six.

    Greedy rather than sampled, and one completion rather than eight, because the
    point is to compare checkpoints: with temperature 0 a change in the curve is a
    change in the model, not in the sampling draw.

    The policy never runs a forward pass here, so validation cannot disturb
    DeepSpeed's ZeRO-3 module trace.

    Two things this got wrong the first time, both of which killed a run:

    Only the main process holds the vLLM client, but every rank must still walk the
    same sequence of collectives. Generating on rank 0 and returning early elsewhere
    let the other five ranks run on into the next step's DeepSpeed parameter gather,
    where they waited 30 minutes for a rank that was busy generating, and the job
    died on an ALLGATHER timeout. So every rank enters, and they synchronise on
    broadcast_object_list -- the same main-generates-then-broadcast shape
    _generate_and_score_completions uses.

    And the request is chunked. Asking for all 256 prompts at once wedged the server
    at "Adding requests: 0/256"; the training path never sends more than 48
    sequences per call (6 prompts x num_generations), so CHUNK_SEQUENCES stays at
    what that path is known to sustain.
    """

    # Sequences per vLLM request. The training path sends 6 prompts x 8 generations
    # = 48 and is known to work; 256 in one request is not. Note this asks the server
    # for the same number of sequences as training but 8x the images (one per
    # sequence, rather than one shared by eight), so if it ever stalls again, drop
    # VAL_CHUNK_SEQUENCES to 6 to match training's image count exactly -- an
    # environment variable so that costs a restart, not a code change.
    CHUNK_SEQUENCES = int(os.environ.get("VAL_CHUNK_SEQUENCES", 48))

    def __init__(self, val_sets, every, accuracy_fn, max_new_tokens):
        self.val_sets = val_sets
        self.every = every
        self.accuracy_fn = accuracy_fn
        self.max_new_tokens = max_new_tokens
        self.trainer = None
        self._warned = False
        self._axis_declared = False

    def on_train_begin(self, args, state, control, **kwargs):
        # A step-0 baseline: without it the first point is at `every` steps and there
        # is nothing to say whether training moved anything.
        #
        # Only at a genuine step 0. On resume the adapter is loaded from a checkpoint
        # but the vLLM server still holds the base weights until the trainer's first
        # sync, so evaluating here would score the base model and file it under the
        # resumed step -- a wrong point, which is worse than a missing one. The run
        # being resumed already recorded its own step-0.
        if state.global_step == 0:
            self._evaluate(state)

    def on_step_end(self, args, state, control, **kwargs):
        if self.every > 0 and state.global_step % self.every == 0:
            self._evaluate(state)

    def _evaluate(self, state):
        trainer = self.trainer
        if trainer is None:
            return
        # Deliberately NOT `if not is_main_process: return` -- see the class docstring.
        # Every rank walks the same chunks and meets the main process at each
        # broadcast, so none of them can wander into the next collective alone.
        accelerator = trainer.accelerator
        is_main = accelerator.is_main_process
        client = getattr(trainer, "vllm_client", None)
        if client is None and is_main and not self._warned:
            self._warned = True
            print("[val] no vLLM client (needs --use_vllm --vllm_mode server); "
                  "skipping validation")

        from accelerate.utils import broadcast_object_list
        from trl.data_utils import maybe_apply_chat_template

        metrics = {}
        for name, dataset in self.val_sets.items():
            started = time.time()
            rows = list(dataset)
            prompts = [
                maybe_apply_chat_template(
                    {"prompt": _with_image_placeholder(r["prompt"])}, trainer.processing_class
                )["prompt"]
                for r in rows
            ]
            images = [r["image"] for r in rows]

            completion_ids, aborted = [], False
            for begin in range(0, len(rows), self.CHUNK_SEQUENCES):
                end = min(begin + self.CHUNK_SEQUENCES, len(rows))
                if is_main and client is not None:
                    chunk = client.generate(
                        prompts=prompts[begin:end],
                        images=images[begin:end],
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                        repetition_penalty=1.0,
                        max_tokens=self.max_new_tokens,
                    )
                else:
                    chunk = None
                # Collective: every rank blocks here until the main process has its
                # chunk, which is what keeps them in lockstep.
                chunk = broadcast_object_list([chunk], from_process=0)[0]
                if chunk is None:  # no client anywhere -- give up, on every rank alike
                    aborted = True
                    break
                completion_ids.extend(chunk)
            if aborted:
                return

            texts = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": t}] for t in texts]
            scores = self.accuracy_fn(completions=completions, solution=[r["solution"] for r in rows])

            # accuracy_reward returns None when the answer could not be parsed at all;
            # counting those as wrong would conflate "got it wrong" with "could not be
            # graded", so they are reported separately instead.
            graded = [s for s in scores if s is not None]
            metrics[f"val/{name}/accuracy"] = sum(graded) / len(graded) if graded else float("nan")
            metrics[f"val/{name}/ungraded"] = (len(scores) - len(graded)) / max(1, len(scores))
            metrics[f"val/{name}/seconds"] = time.time() - started
            print(f"[val] step {state.global_step} {name}: "
                  f"accuracy {metrics[f'val/{name}/accuracy']:.4f} over {len(graded)} rows "
                  f"in {metrics[f'val/{name}/seconds']:.0f}s")

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        # Plot against the training step, not WandB's internal counter. The Trainer
        # sets `define_metric("*", step_metric="train/global_step")`, so without a
        # step metric of its own the validation curve would be drawn on a different
        # x-axis from every other curve in the run and could not be read next to
        # them. Declared once, on the first log.
        if not self._axis_declared:
            wandb.run.define_metric("val/step")
            wandb.run.define_metric("val/*", step_metric="val/step")
            self._axis_declared = True
        wandb.run.log({**metrics, "val/step": state.global_step})


class BenchmarkResultsCallback(TrainerCallback):
    """Log mini-benchmark scores produced by the out-of-process eval job.

    run_bench_eval.sh evaluates checkpoints on a separate allocation and drops one
    flat JSON of scalars per checkpoint into <output_dir>/bench_eval/. This picks
    them up at each logging step and writes them into the live WandB run, so the
    benchmark curves sit alongside the reward curves in one place.

    They are logged against `bench/step`, their own x-axis, rather than the current
    training step: a result arrives whenever its job finishes, which is well after
    the checkpoint it describes, and WandB's global step cannot go backwards.
    Anything still unfinished when training exits is appended afterwards by
    `bench_eval.py --backfill`.

    Results are also scanned one directory deeper, because the eval job files them
    by SAMPLE PROFILE: 100 documents per benchmark keeps the flat
    `bench_eval/step-<N>.json`, and anything else -- 300 on the natural half is now
    the default -- goes to `bench_eval/n300_100/`. A non-recursive glob therefore
    saw nothing at all, and a run whose every checkpoint was scored logged no
    benchmark curve and said nothing about why.

    Each profile keeps its own key namespace (`bench/*`, `bench_n300_100/*`) so
    that no panel can put two sample sizes on one line. The namespace is derived
    from the directory name, which IS the profile name -- see profile_dir() and
    wandb_prefix() in eval_mini/benchmarks.py, whose spelling this has to match.
    It is duplicated here rather than imported because this runs inside the
    training process, where an import error would be a crash in a callback that is
    only supposed to draw a curve.
    """

    def __init__(self, bench_dir):
        self.bench_dir = bench_dir
        self.logged = set()
        self._axes_declared = set()

    def _step_files(self):
        """Every step file under bench_dir, flat and one level of profile deep."""
        return sorted(glob.glob(os.path.join(self.bench_dir, "step-*.json")) +
                      glob.glob(os.path.join(self.bench_dir, "n*_*", "step-*.json")))

    def _prefix(self, path):
        """The WandB key namespace a step file belongs in, from its directory."""
        parent = os.path.basename(os.path.dirname(path))
        return "bench" if parent == os.path.basename(self.bench_dir) else f"bench_{parent}"

    def _wandb_run(self):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.run

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        run = self._wandb_run()
        if run is None:
            return

        for path in self._step_files():
            if path in self.logged:
                continue
            try:
                with open(path) as fh:
                    payload = json.load(fh)
                metrics, step = payload["metrics"], payload["step"]
            except Exception as exc:  # a partial or malformed file must not kill training
                print(f"[bench] skipping {path}: {type(exc).__name__}: {exc}")
                self.logged.add(path)
                continue

            # bench_eval.py writes the keys under `bench/`; re-namespace them to
            # this file's profile so two sample sizes cannot land on one curve.
            prefix = self._prefix(path)
            if prefix != "bench":
                metrics = {f"{prefix}/{k.split('/', 1)[1]}": v for k, v in metrics.items()}
            if prefix not in self._axes_declared:
                run.define_metric(f"{prefix}/step")
                run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
                self._axes_declared.add(prefix)
            run.log({**metrics, f"{prefix}/step": step})
            self.logged.add(path)
            print(f"[bench] logged checkpoint {step} ({len(metrics)} scalars) "
                  f"to WandB as {prefix}/*")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    # Defaults to the paper's corpus, so existing launchers keep working unchanged.
    # --dataset_name points the same pipeline at any other VQA corpus exposing the
    # columns this script consumes: `problem`, `solution` and `image`, plus `bbox`
    # when --reward_variant saliency_r1.
    dataset_name = script_args.dataset_name or "peterant330/saliency-r1-8k"
    # A directory written by Dataset.save_to_disk (what build_grpo_sets.py produces)
    # is not a load_dataset() input: load_dataset would fall back to the generic
    # arrow builder, which ignores dataset_info.json and so hands back `image` as a
    # raw {bytes, path} struct instead of a decoded PIL image, after copying the
    # whole corpus into the HF cache. Detect that layout and load it properly.
    if os.path.isfile(os.path.join(dataset_name, "dataset_info.json")) or os.path.isfile(
        os.path.join(dataset_name, "dataset_dict.json")
    ):
        dataset = load_from_disk(dataset_name)
        # save_to_disk on a DatasetDict keeps the splits; take the requested one.
        if not hasattr(dataset, "train_test_split"):
            dataset = dataset[script_args.dataset_train_split]
    else:
        dataset = load_dataset(
            dataset_name,
            name=script_args.dataset_config,
            split=script_args.dataset_train_split,
        )
    dataset = dataset.train_test_split(test_size=100, seed=42)
    '''
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    '''
    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, "
        "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
    )


    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Cap the long side at 512px, preserving aspect ratio. saliency-r1-8k ships
    # pre-resized within this bound, but larger sources (full Visual-CoT, V*,
    # VisDrone) do not -- dropping oversized samples would silently discard most
    # of such a dataset, so downscale instead of filtering. Boxes in the `bbox`
    # column are normalized to [0, 1], so they survive the resize unchanged.
    MAX_IMAGE_SIDE = 512

    def prepare_image(example):
        image = example["image"]
        width, height = image.size
        if max(width, height) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.BICUBIC,
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(prepare_image)

    train_dataset = dataset["train"]

    ################
    # Validation sets
    ################
    # The held-out sets are separate corpora, not a slice of the training one: their
    # images never appear in set_a or set_b (build_grpo_sets.py --build-val enforces
    # and --verify-val proves it). They are passed as a dict so natural and
    # non-natural imagery are evaluated and logged separately -- the whole point of
    # having two of them is to see the curves diverge.
    #
    # The `dataset.train_test_split` above is left exactly as it was, so `train` is
    # byte-identical to what previous runs trained on.
    #
    # These are NOT handed to the Trainer as eval_dataset. Its evaluation loop runs
    # the full GRPO pipeline per prompt -- num_generations completions, DINO, the
    # saliency re-forward, the judge, a log-prob forward -- which measured 21.6
    # minutes per set, about 90% of training throughput at a 100-step cadence.
    # ValidationAccuracyCallback scores them instead: one greedy completion per
    # prompt in a single batched vLLM call, accuracy only. No trimming to a multiple
    # of the process count is needed either, since it does not shard across ranks.
    def load_val_set(path):
        ds = load_from_disk(path)
        if not hasattr(ds, "train_test_split"):  # a DatasetDict written by save_to_disk
            ds = ds[next(iter(ds.keys()))]
        return ds.map(make_conversation).map(prepare_image)

    val_sets = {}
    if script_args.val_sets_dir:
        for name in ("val_natural", "val_nonnatural"):
            path = os.path.join(script_args.val_sets_dir, name)
            if os.path.isdir(path):
                val_sets[name] = load_val_set(path)
                print(f"[val] {name}: {len(val_sets[name])} rows from {path}")
        if not val_sets:
            raise SystemExit(
                f"--val_sets_dir {script_args.val_sets_dir} holds neither val_natural/ "
                f"nor val_nonnatural/; run build_grpo_sets.py --build-val first."
            )
    # The Trainer's own evaluation stays off; validation is the callback's job.
    eval_dataset = None

    ################
    # Reward Function for Training
    ################
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        import re as _re

        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            # Extract only the answer portion after </think>
            m = _re.search(r"</think>\s*(.*)", content, _re.DOTALL)
            answer_text = m.group(1).strip() if m else content.strip()

            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        answer_text,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {answer_text}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(answer_text.lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards

    ################
    # Reward function selection (flag-selectable: their saliency reward vs ours)
    ################
    # Keep reward_funcs order stable so --reward_weights lines up:
    #   [format, <saliency|overlap>, accuracy, judge]   (saliency_r1 / ours)
    #   [format, accuracy, judge]                        (none)
    # --saliency_method is the flag; --reward_variant is the older spelling kept so
    # existing command lines still run. Normalise here, once, so every branch below and
    # the whole trainer keep working off the single `reward_variant` string they always
    # have -- introducing a second thing to branch on is how the two drift apart.
    _METHOD_TO_VARIANT = {"attention": "ours", "grad": "grad", "glimpse": "glimpse"}
    if script_args.saliency_method is not None:
        script_args.reward_variant = _METHOD_TO_VARIANT[script_args.saliency_method]
    # One metric flag now serves three maps that had three different historical defaults,
    # so an UNSET metric has to resolve per map or every existing --reward_variant grad
    # command line would silently switch from the roll-null to mean_in.
    if script_args.overlap_metric is None:
        script_args.overlap_metric = {"grad": "logratio", "glimpse": "mean_in_v2"}.get(
            script_args.reward_variant, "mean_in")
    # --placebo's whole contract is "unscored on exactly the completions the ATTENTION
    # overlap reward leaves unscored", and that set is defined by think_overlap_reward.
    # Silently applying it to another map would compare against a reference that was
    # never run.
    if script_args.placebo and script_args.reward_variant != "ours":
        raise SystemExit(
            f"--placebo {script_args.placebo} needs --saliency_method attention "
            f"(--reward_variant ours); got reward_variant='{script_args.reward_variant}'. "
            "The placebos are controls for the attention-overlap reward and inherit its "
            "scored/unscored set, which is the one thing that must not differ."
        )
    # --maskfree is a control on the same reference for the same reason, and its weights
    # were measured on the ATTENTION map. `flatness` and `mass` are well defined on the
    # gradient and GLIMPSE maps too, but those have different scales and would need their
    # own calibration, so allowing them silently would apply an unknown multiple of the
    # intended pressure. Lifting this is a one-line change plus one probe run.
    if script_args.maskfree and script_args.reward_variant != "ours":
        raise SystemExit(
            f"--maskfree {script_args.maskfree} needs --saliency_method attention "
            f"(--reward_variant ours); got reward_variant='{script_args.reward_variant}'. "
            "The mask-free rewards are controls for the attention-overlap reward and "
            "their weights were measured on that map."
        )
    # Both take the same slot in reward_funcs, so one of them would silently win.
    if script_args.maskfree and script_args.placebo:
        raise SystemExit(
            f"--maskfree {script_args.maskfree} and --placebo {script_args.placebo} both "
            "REPLACE the overlap reward in the same reward_funcs slot. Pick one."
        )

    if script_args.reward_variant == "ours":
        from trl.rewards.overlap_rewards import configure as configure_overlap
        from trl.rewards.overlap_rewards import think_overlap_reward

        configure_overlap(
            box_threshold=script_args.box_threshold,
            max_box_area=script_args.max_box_area,
            max_union_area=script_args.max_union_area,
            metric=script_args.overlap_metric,
            null_offsets=script_args.rollnull_offsets,
            logratio_clip=script_args.rollnull_clip,
            inframe_rolls=script_args.rollnull_inframe,
            roll_seed=script_args.rollnull_seed,
            mass_floor_tau=script_args.mass_floor_tau,
            dino_api_base=script_args.dino_api_base,
            natural_only=script_args.overlap_natural_only,
        )
        if script_args.placebo:
            # --placebo takes the overlap reward's SLOT, so --reward_weights lines up
            # unchanged and the run differs from its reference in the reward's value and
            # nothing else. It still runs the whole overlap pipeline -- segmentation,
            # Grounding-DINO, the configured metric -- because the metric's score is what
            # decides which completions are scored at all, and that set has to match the
            # reference exactly or the comparison has two variables. Configured AFTER
            # configure_overlap: it reads the resolved metric back out to refuse logratio.
            from trl.rewards.placebo_rewards import configure as configure_placebo
            from trl.rewards.placebo_rewards import think_placebo_reward

            # length_anchor is the completion cap, so --placebo length stays in
            # [0, cap/1000]: an unscored completion must not read as the BEST possible
            # length under the pre-8489767 `nansum` fold that trl_repo still runs. It
            # cancels in the advantage either way -- see placebo_rewards._CFG.
            configure_placebo(kind=script_args.placebo, seed=script_args.rollnull_seed,
                              inframe=script_args.rollnull_inframe,
                              length_anchor=float(training_args.max_completion_length))
            reward_funcs = [think_format_reward, think_placebo_reward, accuracy_reward, openai_reward]
        elif script_args.maskfree:
            # --maskfree takes the same slot, and unlike --placebo it does NOT run the
            # grounding pipeline: no boxes, no union, no Grounding-DINO. configure_overlap
            # above still ran because --overlap_natural_only lives in its config and the
            # optional --maskfree-parity path borrows its helpers; with parity off (the
            # default) none of the DINO knobs are ever read.
            from trl.rewards.maskfree_rewards import configure as configure_maskfree
            from trl.rewards.maskfree_rewards import think_maskfree_reward

            configure_maskfree(kind=script_args.maskfree,
                               parity=script_args.maskfree_parity,
                               mass_anchor=script_args.maskfree_mass_anchor)
            reward_funcs = [think_format_reward, think_maskfree_reward, accuracy_reward, openai_reward]
        else:
            reward_funcs = [think_format_reward, think_overlap_reward, accuracy_reward, openai_reward]
    elif script_args.reward_variant == "grad":
        # Same slot in reward_funcs as the overlap reward, so --reward_weights lines up
        # unchanged. The DINO-side knobs live in overlap_rewards._CFG -- grad_rewards
        # calls its grounding helpers rather than duplicating them -- so both are
        # configured here.
        from trl.rewards.grad_rewards import configure as configure_grad
        from trl.rewards.grad_rewards import think_grad_reward
        from trl.rewards.overlap_rewards import configure as configure_overlap

        configure_overlap(
            # The gradient map is scored by the same four metrics as the other two now.
            # 'logratio' still takes grad_rewards' own path and its own --grad_* knobs --
            # see _score_step there for why that asymmetry is deliberate.
            metric=script_args.overlap_metric,
            box_threshold=script_args.box_threshold,
            max_box_area=script_args.max_box_area,
            max_union_area=script_args.max_union_area,
            null_offsets=script_args.rollnull_offsets,
            logratio_clip=script_args.rollnull_clip,
            inframe_rolls=script_args.rollnull_inframe,
            roll_seed=script_args.rollnull_seed,
            dino_api_base=script_args.dino_api_base,
        )
        configure_grad(
            metric=script_args.overlap_metric,
            null_offsets=script_args.grad_null_offsets,
            logratio_clip=script_args.grad_logratio_clip,
            inframe_rolls=script_args.grad_inframe_rolls,
            dedupe_steps=script_args.grad_dedupe_steps,
            natural_only=script_args.grad_natural_only,
            seed=script_args.grad_seed,
        )
        reward_funcs = [think_format_reward, think_grad_reward, accuracy_reward, openai_reward]
    elif script_args.reward_variant == "glimpse":
        # Same slot in reward_funcs as the overlap reward, so --reward_weights lines up
        # unchanged. The METRIC and the DINO-side knobs live in overlap_rewards._CFG --
        # glimpse_rewards calls its grounding AND scoring helpers rather than duplicating
        # them, which is what makes "mean_in_v2 here" the same number as "mean_in_v2
        # there" -- so both are configured here. --overlap_metric selects the metric for
        # this map exactly as it does for the other two.
        from trl.rewards.glimpse_rewards import configure as configure_glimpse
        from trl.rewards.glimpse_rewards import think_glimpse_reward
        from trl.rewards.overlap_rewards import configure as configure_overlap

        # --mass_floor_tau is deliberately NOT forwarded. It gates on the fraction of an
        # attention ROW spent on image tokens, and its recommended 0.0022 is the 10th
        # percentile of that quantity on the reference model. A GLIMPSE map is a relevance
        # row scaled by 2^-L with an arbitrary constant, so the same tau would be either
        # inert or a constant zero -- a number that means nothing here. The launcher
        # refuses the flag outright for this variant.
        configure_overlap(
            metric=script_args.overlap_metric,
            null_offsets=script_args.rollnull_offsets,
            logratio_clip=script_args.rollnull_clip,
            inframe_rolls=script_args.rollnull_inframe,
            roll_seed=script_args.rollnull_seed,
            box_threshold=script_args.box_threshold,
            max_box_area=script_args.max_box_area,
            max_union_area=script_args.max_union_area,
            dino_api_base=script_args.dino_api_base,
        )
        configure_glimpse(
            dedupe_steps=script_args.glimpse_dedupe_steps,
            natural_only=script_args.glimpse_natural_only,
        )
        reward_funcs = [think_format_reward, think_glimpse_reward, accuracy_reward, openai_reward]
    elif script_args.reward_variant == "none":
        # Drop the saliency/overlap reward entirely: accuracy + judge + format only.
        reward_funcs = [think_format_reward, accuracy_reward, openai_reward]
    else:
        reward_funcs = [think_format_reward, think_saliency_reward, accuracy_reward, openai_reward]

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        reforward_saliency=script_args.reforward_saliency,
        reward_variant=script_args.reward_variant,
        overlap_layer=script_args.overlap_layer,
        overlap_heads=script_args.overlap_heads,
        token_reduction=script_args.token_reduction,
        overlap_natural_only=(script_args.overlap_natural_only
                              or (script_args.reward_variant == "grad"
                                  and script_args.grad_natural_only)
                              or (script_args.reward_variant == "glimpse"
                                  and script_args.glimpse_natural_only)),
        grad_target=script_args.grad_target,
        glimpse_target=script_args.glimpse_target,
        glimpse_layer_frac=script_args.glimpse_layer_frac,
        glimpse_temp=script_args.glimpse_temp,
        glimpse_depth_temp=script_args.glimpse_depth_temp,
        glimpse_token_weight=script_args.glimpse_token_weight,
        glimpse_token_cap=script_args.glimpse_token_cap,
        glimpse_seed=script_args.glimpse_seed,
    )

    # Benchmark scores are produced by a separate job (run_bench_eval.sh) and land
    # in this directory as they finish; the callback forwards them to WandB.
    # Harmless when nothing ever writes there.
    trainer.add_callback(BenchmarkResultsCallback(os.path.join(training_args.output_dir, "bench_eval")))

    if val_sets:
        validation = ValidationAccuracyCallback(
            val_sets=val_sets,
            every=script_args.val_eval_steps,
            accuracy_fn=accuracy_reward,
            max_new_tokens=training_args.max_completion_length,
        )
        # The callback needs the trainer's vLLM client and processor, which only
        # exist once the trainer is built -- hence the back-reference rather than a
        # constructor argument.
        validation.trainer = trainer
        trainer.add_callback(validation)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
