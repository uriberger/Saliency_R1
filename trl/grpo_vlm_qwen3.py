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

import torch
from datasets import load_dataset, load_from_disk
from latex2sympy2_extended import NormalizationConfig
from PIL import Image
from math_verify import LatexExtractionConfig, parse, verify

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
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

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
    if script_args.reward_variant == "ours":
        from trl.rewards.overlap_rewards import configure as configure_overlap
        from trl.rewards.overlap_rewards import think_overlap_reward

        configure_overlap(
            box_threshold=script_args.box_threshold,
            max_box_area=script_args.max_box_area,
            metric=script_args.overlap_metric,
            mass_floor_tau=script_args.mass_floor_tau,
            dino_api_base=script_args.dino_api_base,
            natural_only=script_args.overlap_natural_only,
        )
        reward_funcs = [think_format_reward, think_overlap_reward, accuracy_reward, openai_reward]
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
        overlap_natural_only=script_args.overlap_natural_only,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
