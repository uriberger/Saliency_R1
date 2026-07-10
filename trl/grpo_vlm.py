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

import os
import re
import shutil

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from transformers import TrainerCallback


class SyncSaveStepsCallback(TrainerCallback):
    """Fix save_steps in TrainerState after a resume.

    When resuming, Trainer replaces self.state with the JSON loaded from the
    checkpoint.  That JSON carries the *old* save_steps value, but
    DefaultFlowCallback uses state.save_steps (not args.save_steps) to decide
    when to save.  on_train_begin fires after the JSON is loaded, so we can
    patch the value here to match the current --save_steps arg.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = args.save_steps


class TieredCheckpointCallback(TrainerCallback):
    """Keep every `milestone_steps` checkpoint permanently; for all other
    `frequent_steps` checkpoints, keep only the most recent one.

    Example: frequent_steps=10, milestone_steps=200
      step 620 saved  -> last_frequent=620
      step 630 saved  -> delete checkpoint-620, last_frequent=630
      step 200 saved  -> delete last_frequent (190), last_frequent=None (permanent)
      step 210 saved  -> last_frequent=210 (nothing to delete, 200 was permanent)
    """

    def __init__(self, output_dir, frequent_steps=10, milestone_steps=200):
        self.output_dir = output_dir
        self.frequent_steps = frequent_steps
        self.milestone_steps = milestone_steps
        # On init (including after resume), find the most recent frequent
        # (non-milestone) checkpoint so we can delete it on the next save.
        self._last_frequent_step = self._scan_last_frequent()

    def _scan_last_frequent(self):
        if not os.path.isdir(self.output_dir):
            return None
        candidates = []
        for name in os.listdir(self.output_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m:
                step = int(m.group(1))
                if step % self.frequent_steps == 0 and step % self.milestone_steps != 0:
                    candidates.append(step)
        return max(candidates) if candidates else None

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        step = state.global_step
        if self._last_frequent_step is not None:
            ckpt = os.path.join(self.output_dir, f"checkpoint-{self._last_frequent_step}")
            if os.path.isdir(ckpt):
                shutil.rmtree(ckpt)
                print(f"[TieredCheckpoint] deleted checkpoint-{self._last_frequent_step}")
        if step % self.milestone_steps != 0:
            self._last_frequent_step = step
        else:
            self._last_frequent_step = None


from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward, think_saliency_reward, openai_reward


def maybe_wandb_rewind(trainer, training_args):
    """When resuming from a checkpoint, rewind the existing wandb run back to the
    checkpoint's step so the reloaded curve overwrites the doomed pre-crash tail —
    yielding one clean run "as if it never crashed" instead of a fork or a run
    with a visible seam.

    Works by pre-initializing wandb with `resume_from` before the Trainer's
    WandbCallback runs; the callback then reuses the existing run (it only calls
    wandb.init when wandb.run is None). No-op unless we're resuming, wandb
    reporting is on/online, we're the main process, and WANDB_RUN_ID is set.
    """
    # Opt-in only: wandb's resume_from (rewind) is a private-preview feature and
    # returns HTTP 400 unless enabled on the account. Skip it by default and rely
    # on WANDB_RESUME=allow (set in the launch script) to continue the same run.
    # Set WANDB_REWIND=1 to attempt the clean truncated rewind once you have access.
    if os.environ.get("WANDB_REWIND", "").lower() not in ("1", "true", "yes"):
        return
    if not training_args.resume_from_checkpoint:
        return
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        return
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        return
    if not trainer.is_world_process_zero():
        return
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        return

    # Resolve the step of the checkpoint we're resuming from (latest checkpoint-<N>
    # in output_dir, unless an explicit checkpoint path was given).
    resume = training_args.resume_from_checkpoint
    ckpt_dir = resume if isinstance(resume, str) and os.path.isdir(resume) else None
    if ckpt_dir is None:
        candidates = []
        for name in os.listdir(training_args.output_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m and os.path.isdir(os.path.join(training_args.output_dir, name)):
                candidates.append((int(m.group(1)), name))
        if not candidates:
            return
        _, name = max(candidates)
        ckpt_dir = os.path.join(training_args.output_dir, name)
    m = re.search(r"checkpoint-(\d+)", os.path.basename(ckpt_dir))
    if not m:
        return
    step = int(m.group(1))

    try:
        import wandb
    except ImportError:
        return

    # resume_from (rewind) and resume are mutually exclusive; drop the env resume
    # flag we set in the launch script so wandb.init doesn't reject the call.
    os.environ.pop("WANDB_RESUME", None)
    try:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            resume_from=f"{run_id}?_step={step}",
        )
        print(f"[wandb-rewind] rewound run {run_id} to step {step}; Trainer will reuse this run.")
    except Exception as e:
        # Older wandb without rewind support, or a bad id -> fall back to plain
        # resume so we at least continue the same run (with a seam).
        os.environ["WANDB_RESUME"] = "allow"
        print(f"[wandb-rewind] rewind unavailable ({e}); falling back to WANDB_RESUME=allow.")


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
    dataset = load_dataset("peterant330/saliency-r1-8k", split="train")
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

    # Filter have big images
    def filter_big_images(example):
        image = example["image"]
        return image.size[0] <= 512 and image.size[1] <= 512

    dataset = dataset.filter(filter_big_images)

    def convert_to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    ################
    # Reward Function for Training
    ################
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """

        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            # Extract answer portion after </think>; fall back to full content
            m = re.search(r"</think>\s*(.*?)\s*$", content, re.DOTALL)
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
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[accuracy_reward, think_format_reward, think_saliency_reward, openai_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer.add_callback(SyncSaveStepsCallback())
    trainer.add_callback(TieredCheckpointCallback(training_args.output_dir))

    maybe_wandb_rewind(trainer, training_args)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
