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

import copy
import inspect
import os
import re
import textwrap
import time
import warnings

import numpy as np
from collections import defaultdict, deque
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available, is_vllm_available
from ..models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
# The per-family seam. Every place below that used to read `image_grid_thw` or assume a
# 2x2 spatial merge now asks this instead -- see `docs/omni-training-harness.md`. It is a
# copy of the repo-root `vlm_family.py` (patch_trl_nemotron.sh keeps them in step), which
# is also what the measuring side imports, so the trainer and the probes cannot disagree
# about where a patch is.
from .vlm_family import family_for
from .utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import cv2


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def repeat_v(hidden_states, n_rep):
    batch,  num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads,n_rep, slen, head_dim)
    return hidden_states.reshape(batch,num_key_value_heads*n_rep, slen, head_dim)

class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def impute_unscored_rewards(rewards_per_func: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Replace every NaN in `rewards_per_func` with its own GROUP's mean of that func.

    `rewards_per_func` is `(n_groups * num_generations, n_funcs)`, already gathered across
    processes and ordered so that consecutive `num_generations` rows form one group -- the
    same layout `rewards.view(-1, num_generations)` relies on when it normalises the
    advantage.

    A NaN means "this reward func did not score this completion", not "it scored badly":
    no groundable observe step, every step over `--max_union_area`, or
    `--overlap_natural_only` masking the row. Imputing the group mean makes such a
    completion's deviation on that dimension exactly 0, so an unmeasured reward drops out
    of its advantage instead of being scored 0 -- which is a real, and for a metric whose
    chance level is not 0 a very large, reward. See the call site for the measured sizes.

    Returns a NEW tensor. The caller keeps `rewards_per_func` un-imputed so the logged
    `rewards/<func>/mean` and `/std` stay nanmean/nanstd over the scored completions only.

    A group where no completion at all was scored imputes 0.0, which is equally neutral:
    every row then shares one value and the func contributes nothing to that group's
    spread whatever that value is.
    """
    # reshape, not view: `rewards_per_func` arrives from a gather and need not be contiguous.
    x = rewards_per_func.reshape(-1, num_generations, rewards_per_func.size(-1))
    scored = ~torch.isnan(x)
    n = scored.sum(dim=1, keepdim=True)                                       # (G, 1, F)
    total = torch.where(scored, x, torch.zeros_like(x)).sum(dim=1, keepdim=True)
    group_mean = total / n.clamp(min=1).to(total.dtype)
    group_mean = torch.where(n > 0, group_mean, torch.zeros_like(group_mean))
    return torch.where(scored, x, group_mean.expand_as(x)).reshape(rewards_per_func.shape)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_sequence_dict(seq_dict: dict[str, Optional[Sequence]]) -> dict[str, Optional[Sequence]]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Optional[Sequence]) -> Optional[Sequence]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_pixel_values_by_grid(batch: dict[str, torch.Tensor]) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
    """
    Splits `batch["pixel_values"]` into a list of tensors based on the product of each row in
    `batch["image_grid_thw"]`, while keeping other entries unchanged.
    """
    if "image_grid_thw" not in batch or "pixel_values" not in batch:
        return batch

    lengths = batch["image_grid_thw"].prod(dim=1).tolist()  # [batch_size]
    pixel_values = batch["pixel_values"]  # [total, feature_dim]

    if sum(lengths) != pixel_values.size(0):
        raise ValueError(f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}")

    split_values = list(torch.split(batch["pixel_values"], lengths, dim=0))
    return {**batch, "pixel_values": split_values}


def unsplit_pixel_values_by_grid(batch: dict[str, Union[torch.Tensor, list[torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]`
    back into a single tensor along the first dimension.
    """
    pixel_values = batch.get("pixel_values")

    if isinstance(pixel_values, list):
        merged = torch.cat(pixel_values, dim=0)
        return {**batch, "pixel_values": merged}
    else:
        return batch


def truncate_with_protected_tokens(
    ids: torch.Tensor, mask: torch.Tensor, target_length: int, protected_tokens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncate tensors to target length while preserving protected tokens.

    Args:
        ids (`torch.Tensor`):
            Input tensor of token IDs, shape (batch_size, sequence_length).
        mask (`torch.Tensor`):
            Input tensor of attention masks, shape (batch_size, sequence_length).
        target_length (`int`):
            Desired length of the output sequences.
        protected_tokens (`list[int]`):
            List of token IDs that should be preserved in the output.
    """
    protected_set = set(protected_tokens)

    def process_sequence(ids, mask):
        # Create boolean masks
        is_protected = torch.tensor([x.item() in protected_set for x in ids])
        is_non_protected = ~is_protected

        # Count tokens
        num_protected = is_protected.sum().item()
        num_non_protected_needed = target_length - num_protected

        if num_non_protected_needed < 0:
            raise ValueError(
                f"target_length ({target_length}) is too small for the protected tokens ({num_protected} tokens). "
                f"Please increase target length to at least {num_protected} or disable truncation."
            )

        # Select which non-protected tokens to keep (rightmost ones)
        non_protected_indices = torch.where(is_non_protected)[0]
        keep_non_protected = torch.zeros_like(is_non_protected)
        if num_non_protected_needed > 0:
            keep_indices = non_protected_indices[-num_non_protected_needed:]
            keep_non_protected[keep_indices] = True

        # Final mask: protected OR selected non-protected
        keep_mask = is_protected | keep_non_protected

        return ids[keep_mask], mask[keep_mask]

    # Process each sequence in the batch
    truncated_seq = []
    truncated_mask = []

    for i in range(ids.shape[0]):
        new_ids, new_mask = process_sequence(ids[i], mask[i])
        truncated_seq.append(new_ids)
        truncated_mask.append(new_mask)

    return torch.stack(truncated_seq), torch.stack(truncated_mask)


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        reforward_saliency: bool = True,
        reward_variant: str = "saliency_r1",
        overlap_layer: int = 22,
        overlap_heads=(28, 31),
        token_reduction: str = "mean",
        overlap_natural_only: bool = False,
        grad_target: str = "clogit",
        glimpse_target: str = "clogit",
        glimpse_layer_frac: float = 1.0,
        glimpse_temp: float = 0.5,
        glimpse_depth_temp: float = 0.2,
        glimpse_token_weight: str = "full",
        glimpse_token_cap: int = 0,
        glimpse_seed: int = 0,
    ):
        self.reforward_saliency = reforward_saliency
        # --- attention-overlap reward config (reward_variant="ours") ---
        self.reward_variant = reward_variant
        # "ours", "grad" and "glimpse" differ only in what the per-case re-forward
        # extracts -- raw attention at one layer, the pixel gradient of the step's own
        # tokens, or GLIMPSE's gradient-weighted attention. All three segment observe
        # steps, ground them with DINO and score per step, so every observe-step-shaped
        # decision below applies to all of them.
        self._per_step_reward = reward_variant in ("ours", "grad", "glimpse")
        self.grad_target = grad_target
        self.glimpse_target = glimpse_target
        self.glimpse_layer_frac = float(glimpse_layer_frac)
        self.glimpse_temp = float(glimpse_temp)
        self.glimpse_depth_temp = float(glimpse_depth_temp)
        self.glimpse_token_weight = glimpse_token_weight
        self.glimpse_token_cap = int(glimpse_token_cap or 0)
        self.glimpse_seed = int(glimpse_seed)
        # Score the overlap reward on natural (photographic) rows only; non-natural rows
        # fall back to format + accuracy + judge. See think_overlap_reward's docstring.
        self.overlap_natural_only = bool(overlap_natural_only) and self._per_step_reward
        if overlap_natural_only and not self.overlap_natural_only:
            warnings.warn(
                f"overlap_natural_only=True is ignored with reward_variant='{reward_variant}': "
                "there is no overlap reward to mask."
            )
        if self.overlap_natural_only:
            # IterableDataset may expose no column_names at all; only validate when we
            # actually get a plain column list (a Dataset), and let the reward raise later
            # otherwise.
            _cols = getattr(train_dataset, "column_names", None)
            if isinstance(_cols, (list, tuple)) and "natural" not in _cols:
                raise KeyError(
                    "overlap_natural_only=True requires a boolean 'natural' column in the "
                    f"train dataset, but its columns are {sorted(_cols)}. Use a corpus built "
                    "by build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
                )
        self.overlap_layer = int(overlap_layer)
        if isinstance(overlap_heads, str):
            overlap_heads = [int(h) for h in overlap_heads.split(",") if h.strip() != ""]
        self.overlap_heads = list(overlap_heads)
        self.token_reduction = token_reduction
        self._overlap_clf = None  # lazily loaded FLAN-T5 steps classifier
        if self._per_step_reward and not self.reforward_saliency:
            # Both per-step extractions need a teacher-forced pass over the whole
            # prompt+completion -- "ours" for full per-token attention, "grad" because a
            # gradient has to be taken through one. Only the re-forward path provides it.
            self.reforward_saliency = True

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            #
            # `trust_remote_code=True` is a no-op for a natively supported checkpoint and
            # the only way to read the config of one that ships its own modelling code.
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if getattr(config, "auto_map", None):
                # A remote-code checkpoint -- every Nemotron VLM here. `getattr(
                # transformers, config.architectures[0])` below raises AttributeError on
                # one, and it would not be enough anyway: these need the attention
                # implementation pinned on every SUB-config before construction, a
                # vendored `rmsnorm_fn` to import at all, and four more repairs applied
                # after. `nemotron_loader` is that set, shared with the measuring side so
                # the two cannot drift. `place=False` because accelerate moves the model
                # and `.eval()` before `get_peft_model` is one more thing to undo.
                from .nemotron_loader import load_model as _load_remote_code

                _dtype = model_init_kwargs.get("torch_dtype") or torch.bfloat16
                if not isinstance(_dtype, torch.dtype):
                    _dtype = torch.bfloat16
                _proc, model = _load_remote_code(
                    model_id, device=None, quant=model_init_kwargs.get("quantization_config"),
                    dtype=_dtype, place=False)
                if processing_class is None:
                    processing_class = _proc
            else:
                architecture = getattr(transformers, config.architectures[0])
                model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # THE FAMILY, resolved here because everything below needs it and because two of
        # the three things it decides happen before the PEFT wrap. Bound off the bare
        # model: `family_for` reads `config.model_type`, and the processor is not resolved
        # until further down (it is re-bound there, once, when it is).
        self.family = family_for(model=model)
        # Which decoder layers HAVE an attention matrix. None on a dense decoder (every
        # layer does); the real list on a hybrid, where 46 of 52 layers are state-space or
        # MoE blocks. Read here, off the bare model, because `_report_lora_landing` asserts
        # against it BEFORE the decoder navigation further down would have set it.
        self._attention_layers = self.family.attention_layers(model)

        # Gradient checkpointing, and the ORDER is the whole content of this block.
        #
        # A Nemotron needs it enabled on the LANGUAGE MODEL before the wrap, because
        # `NemotronHPreTrainedModel` never declares `supports_gradient_checkpointing` and
        # transformers' own switch therefore refuses on a model whose blocks ARE
        # `GradientCheckpointingLayer`s. Doing it here rather than in
        # `_enable_gradient_checkpointing` also keeps that method's reach away from a
        # wrapper whose behaviour under recomputation nobody has checked.
        #
        # Then `get_peft_model` sees checkpointing already on and re-installs the
        # `enable_input_require_grads` hook that `after_peft_wrap` has to take back off --
        # which is why that call is AFTER the wrap and not before. See its docstring: the
        # hook makes the embedding output require grad and this wrapper scatters the
        # picture into those embeddings in place.
        _family_ckpt = None
        if args.gradient_checkpointing:
            _family_ckpt = self.family.enable_gradient_checkpointing(model)
            if _family_ckpt:
                print(f"[grad-ckpt] recompute ON for {_family_ckpt} decoder blocks "
                      f"({self.family.name})", flush=True)

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            # Scope the launcher's bare `q_proj,k_proj,v_proj` to where they are meant to
            # land. A no-op on Qwen3-VL; on an Omni it is what keeps 144 of 180 adapters
            # off a 24-layer AUDIO tower that an image-only batch never runs. See
            # `NemotronVL.lora_target_modules`.
            if getattr(peft_config, "target_modules", None):
                _scoped = self.family.lora_target_modules(peft_config.target_modules)
                if _scoped != peft_config.target_modules:
                    print(f"[lora] targets scoped for {self.family.name}: "
                          f"{peft_config.target_modules} -> {_scoped!r}", flush=True)
                    peft_config.target_modules = _scoped
            model = get_peft_model(model, peft_config)
            self._report_lora_landing(model)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing and _family_ckpt is None:
            model = self._enable_gradient_checkpointing(model, args)
        elif args.gradient_checkpointing:
            # The family already turned it on; only `use_cache` is still ours to set.
            model.config.use_cache = False
        self.is_gradient_checkpointing = args.gradient_checkpointing

        if args.gradient_checkpointing:
            _dropped = self.family.after_peft_wrap(model)
            if _dropped:
                print(f"[grad-ckpt] removed enable_input_require_grads hooks from "
                      f"{_dropped} modules", flush=True)
            _still_on = sum(1 for m in model.modules()
                            if getattr(m, "gradient_checkpointing", False))
            if _family_ckpt and _still_on == 0:
                raise RuntimeError("the peft wrap lost the gradient-checkpointing switch")

        if _family_ckpt is not None:
            # `Trainer.train()` opens with
            #     if args.gradient_checkpointing:
            #         self.model.gradient_checkpointing_enable(...)
            # on the OUTER wrapper -- whose `supports_gradient_checkpointing` is False, so
            # it raises "<Model> does not support gradient checkpointing" on a model whose
            # blocks are already recomputing. Declaring support on the wrapper would only
            # trade that for the other failure: the call re-installs the
            # `enable_input_require_grads` hook that `after_peft_wrap` just removed, and
            # the first forward then dies scattering the picture into a leaf that requires
            # grad.
            #
            # Recompute is ON, on every block of the decoder, and asserted just above. So
            # what is left to do is tell the Trainer it has nothing to do. DDP is
            # unaffected: `find_unused_parameters` reads `model.is_gradient_checkpointing`
            # first, which walks the modules and still answers True.
            args.gradient_checkpointing = False
            self.is_gradient_checkpointing = True
            print(f"[grad-ckpt] {_still_on} blocks recomputing; Trainer.train() told not "
                  "to re-enable it on the wrapper", flush=True)

        # FA2 cannot return attention weights, so reforward_saliency is required with it.
        if getattr(model.config, "_attn_implementation", None) == "flash_attention_2" and not self.reforward_saliency:
            import warnings
            warnings.warn(
                "flash_attention_2 does not support output_attentions=True during generate; "
                "forcing reforward_saliency=True.",
                UserWarning,
            )
            self.reforward_saliency = True

        # Qwen3-VL (transformers 5.13): `language_model` lives inside the Qwen3VLModel
        # (`raw_model.model.language_model`). After get_peft_model(), `model.model` resolves
        # via PEFT's __getattr__ to `base_model.model` (Qwen3VLForConditionalGeneration), which
        # has no `.language_model`. Unwrap one level first, then ask the family -- a
        # Nemotron hangs `language_model` off the wrapper itself and puts the stack one
        # level further down again, and guessing lands on `None.layers`.
        _raw = model.base_model.model if is_peft_model(model) else model
        lang_model = self.family.decoder(_raw)
        self.NUM_LAYER = len(lang_model.layers)
        # Cache a direct module reference so the training loop can reach language_model
        # submodules without repeating this PEFT-aware unwrapping on every step.
        self._qwen3_lang_model = lang_model
        # NUM_GROUP and DIMS are read ONLY by the original Saliency-R1 value-propagation
        # readout, which multiplies every layer's attention by that layer's value states.
        # A hybrid has no such object at 46 of its 52 positions -- `self_attn` does not
        # exist on a Mamba or MoE block -- so this is guarded rather than computed, and
        # `supports_saliency_r1()` is what refuses the readout itself further down.
        if self.family.supports_saliency_r1():
            _k = lang_model.layers[0].self_attn.k_proj
            self.NUM_GROUP = _k.in_features // _k.out_features
            self.DIMS = model.lm_head.in_features
        else:
            self.NUM_GROUP = self.DIMS = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        # Re-bind the family now that the processor exists: `bind` is where a family that
        # resolves its image token off the TOKENIZER (Nemotron reads
        # `config.img_context_token_id`, or falls back to converting `<image>`) gets the
        # chance to, and where the hybrid's decoder attention class is read off the text
        # config rather than guessed.
        self.family.bind(model=_raw, processor=processing_class, config=_raw.config)

        self.image_token = getattr(processing_class, "image_token", None)
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(model.config, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(model.config, "vision_end_token_id", None)
        # The family is the fallback, not the override: on Qwen3-VL the processor answers
        # all four and these are no-ops, which is what keeps an existing run identical.
        # On a Nemotron the processor answers none of them -- its image token is
        # `<image>` (id 18) named in the CONFIG, and its `<img>` / `</img>` delimiters are
        # tokenizer lookups -- and without this the prompt-truncation guard would protect
        # nothing and the saliency read would mask on token id None.
        if self.image_token_id is None:
            self.image_token_id = self.family.image_token_id
        if self.image_token is None and self.image_token_id is not None:
            self.image_token = tokenizer.decode([self.image_token_id])
        if self.vision_start_token_id is None and self.family.vision_start_ids:
            self.vision_start_token_id = self.family.vision_start_ids[0]
        if self.vision_end_token_id is None and self.family.vision_end_ids:
            self.vision_end_token_id = self.family.vision_end_ids[0]
        if self.image_token_id is None:
            raise ValueError(
                f"no image token id for family {self.family.name!r}: the saliency read "
                "masks the prompt's image columns with it, and `== None` would select no "
                "columns and report an empty map as a result")

        # A hybrid has an attention matrix at only a few of its layers, and
        # --overlap_layer is an index into ALL of them. Pointing it at a Mamba or MoE
        # layer attaches the capture hook to nothing, and the run then trains on a reward
        # that is silently zero everywhere.
        if self._per_step_reward and self._attention_layers is not None:
            if self.overlap_layer not in self._attention_layers:
                raise ValueError(
                    f"--overlap_layer {self.overlap_layer} is not an attention layer of "
                    f"this model. {self.family.name} has attention at "
                    f"{self._attention_layers} and state-space / MLP blocks everywhere "
                    "else, so there is no attention matrix to read at that index.")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # warnings_issued was removed in transformers 5.x; guard for compatibility.
        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._zero3_lookup_warned = False
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "image": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                )
            elif self.vllm_mode not in ("server", "colocate"):
                # In server mode only the main process sets up the client (the condition
                # above is `server and is_main_process`); non-main ranks legitimately fall
                # through here and must NOT raise. Only a genuinely invalid mode is an error.
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.use_transformers_paged:
                generation_kwargs["max_batch_tokens"] = 512
                generation_kwargs["num_blocks"] = 1024
                generation_kwargs["block_size"] = 128
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _refuse_qwen3_only(self, what, why):
        """Stop a readout that is defined only on Qwen3-VL's geometry. Never returns.

        Three of the maps this trainer can build are not portable and saying so is the
        point. The overlap map is a slice of one attention matrix over the image-token
        COLUMNS, which any family has; these three reach further in:

          * `grad` differentiates w.r.t. `pixel_values` and folds the result back onto
            patches using Qwen3-VL's `patch_size` x `temporal_patch_size` packing;
          * `glimpse` propagates gradient-weighted attention across EVERY layer;
          * the original Saliency-R1 readout multiplies each layer's attention by that
            layer's value states and pushes it through `o_proj`.

        On a Nemotron-H hybrid the last two have no object at 46 of 52 layers, and the
        first has a different pixel packing. Each would produce a number rather than an
        error, which is the failure worth refusing.
        """
        raise NotImplementedError(
            f"{what} is not implemented for family {self.family.name!r}: {why}. "
            "Use --saliency-method attention, whose map is a slice of one attention "
            "matrix and is defined wherever there is an attention layer.")

    def _mm_forward_kwargs(self, mm_source, lo, hi, seq_len=None):
        """The multimodal kwargs for samples [lo, hi), as `forward()` wants them.

        The family decides which processor outputs those are and how to cut them: Qwen3-VL
        stacks a batch's patches into ONE flat `pixel_values` and needs `image_grid_thw`
        to find a sample's slice of it, and the Omni emits one row per picture and needs
        nothing. Geometry-only keys (`imgs_sizes`) are never in here -- the model rejects
        them -- and the batch keeps them separately for `token_grid`.
        """
        mm = dict(self.family.forward_defaults)
        if not mm_source:
            return mm
        mm.update(self.family.mm_slice(mm_source, lo, hi))
        # mm_token_type_ids comes from prompt_inputs only; pad with zeros (text type) to
        # cover completion tokens.
        t = mm.get("mm_token_type_ids")
        if t is not None and seq_len is not None and t.size(1) < seq_len:
            mm["mm_token_type_ids"] = torch.cat(
                [t, torch.zeros(t.size(0), seq_len - t.size(1), dtype=t.dtype,
                                device=t.device)],
                dim=1,
            )
        return mm

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        mm_source=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_inputs.update(
            self._mm_forward_kwargs(mm_source, 0, input_ids.size(0), input_ids.size(1))
        )

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(
        self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float, accelerator=None
    ) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold and
                `False` otherwise.
        """
        non_pad_entropies = entropies[mask.bool()].float()
        if non_pad_entropies.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)
        all_non_pad_entropies = self.accelerator.gather(non_pad_entropies)
        # Filter out any empty tensors that might result from processes with no valid tokens
        entropy_threshold = torch.quantile(all_non_pad_entropies, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        mm_source=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            stop = start + batch_size
            input_ids_batch = input_ids[start:stop]
            attention_mask_batch = attention_mask[start:stop]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            model_inputs.update(
                self._mm_forward_kwargs(mm_source, start, stop, input_ids.size(1))
            )

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    def _report_lora_landing(self, model):
        """Where the adapters actually went, and a refusal if it is not where they belong.

        Asserted rather than trusted because the failure is silent: a LoRA bolted onto a
        tower the batch never runs produces no gradient, no error and a perfectly normal
        loss curve. `docs/omni-training-blockers.md` records the run that did it.
        """
        trained = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trained:
            raise RuntimeError("no trainable parameters after the peft wrap")
        hit = sorted({n.split(".lora_")[0].split(".")[-1] for n in trained})
        layers = sorted({int(n.split("layers.")[1].split(".")[0])
                         for n in trained if "layers." in n})
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[lora] {len(trained)} tensors, {n_train/1e6:.2f}M of {n_total/1e9:.2f}B "
              f"({n_train/n_total*100:.4f}%) on {hit}", flush=True)
        print(f"[lora] decoder layers: {layers}", flush=True)
        want = self._attention_layers
        if want is not None and layers != sorted(want):
            raise RuntimeError(
                f"the LoRA landed on decoder layers {layers}, not this model's attention "
                f"layers {sorted(want)}. q/k/v_proj exist only inside attention, so "
                "anything else means the match went somewhere unintended -- an audio "
                "tower, most likely, which an image-only batch never runs.")

    def _lora_touched_prefixes(self):
        """Module paths whose weights a merged LoRA actually changes. -> set or None.

        WHY THIS EXISTS. `_move_model_to_vllm` pushes EVERY named parameter to the
        generation server, one HTTP round trip and one NCCL broadcast each. On Qwen3-VL-8B
        that is ~700 tensors and costs about a second. The Omni is a 33B mixture of
        experts: 7,349 tensors, of which 5,934 are expert weights, and at a few
        milliseconds of round trip apiece the sync alone would be tens of seconds on a
        step whose whole budget is ~47 s.

        Under LoRA the base is FROZEN, and the server loaded that same base from the same
        checkpoint at startup. So after `merge_adapter()` the only weights that differ
        from the server's copy are the ones a LoRA sits on -- 18 tensors here, the q/k/v
        projections of the six attention layers -- plus anything in `modules_to_save`,
        which is trained outright. Everything else is a re-send of bytes the server
        already has.

        Returns None when that argument does not hold (no PEFT model, or a base parameter
        that still requires grad), in which case the caller pushes everything. `None` is
        also what `SR1_VLLM_SYNC_ALL=1` forces, to take the optimisation back out without
        editing anything.
        """
        if os.environ.get("SR1_VLLM_SYNC_ALL") == "1":
            return None
        if os.environ.get("SR1_VLLM_SYNC_LORA_ONLY") != "1":
            return None          # off unless a launcher asks: existing runs stay identical
        model = self.model
        if not is_peft_model(model):
            return None
        prefixes = set()
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") or "modules_to_save" in name:
                prefixes.add(self._fix_param_name_to_vllm(
                    name.removeprefix("base_model.model.").replace(".base_layer", ""),
                    extra_prefixes=["modules_to_save.default."]))
        if not prefixes:
            return None
        # A trainable weight OUTSIDE the adapter means the base is not frozen after all
        # and the server's copy of it is going stale. Refuse the shortcut rather than
        # training against a generator that silently drifts.
        for name, param in model.named_parameters():
            if not param.requires_grad or model.prefix in name:
                continue
            clean = self._fix_param_name_to_vllm(
                name.removeprefix("base_model.model.").replace(".base_layer", ""))
            if not any(clean.startswith(p) for p in prefixes):
                warnings.warn(
                    f"SR1_VLLM_SYNC_LORA_ONLY is set but {clean!r} is trainable and sits "
                    "outside every adapter, so the base is not frozen. Pushing every "
                    "parameter instead.")
                return None
        return prefixes

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    touched = self._lora_touched_prefixes()
                    n_sent = 0
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
                        # Skip the frozen base: the server loaded it from the same
                        # checkpoint and nothing has written to it. See
                        # `_lora_touched_prefixes`.
                        if touched is not None and not any(name.startswith(p) for p in touched):
                            continue
                        n_sent += 1

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                if touched is not None and self.state.global_step == 0 and \
                        self.accelerator.is_main_process:
                    print(f"[vllm-sync] pushing {n_sent} adapted tensors per step, not "
                          f"{sum(1 for _ in self.model.named_parameters())} "
                          "(SR1_VLLM_SYNC_LORA_ONLY=1; the frozen base is already on the "
                          "server)", flush=True)
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list, attn_map, valid_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # Only the saliency reward consumes ground-truth boxes. The "ours" (attention
        # overlap) and "none" variants derive their signal from the attention map and
        # Grounding-DINO alone, so datasets without box annotations are usable there.
        if "bbox" in inputs[0]:
            bbox_list = [i["bbox"] for i in inputs]
        elif self.reward_variant == "saliency_r1":
            raise KeyError(
                "reward_variant='saliency_r1' requires a 'bbox' column in the dataset, "
                "but none was found. Use --reward_variant ours|none to train on data "
                "without box annotations."
            )
        else:
            bbox_list = [None] * len(inputs)
        question_list = [i['problem'] for i in inputs]

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=question_list, completions=completions, completion_ids=completion_ids_list,
                        saliency_map=attn_map, valid_list=valid_list, bbox_list=bbox_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    # ------------------------------------------------------------------
    # Attention-overlap reward (reward_variant="ours")
    # ------------------------------------------------------------------
    def _get_overlap_classifier(self):
        """Lazily load the FLAN-T5 observe-step classifier (once per process)."""
        if self._overlap_clf is None:
            from .overlap_steps import OverlapStepsClassifier
            # Run the tiny FLAN-T5-base on GPU by default: on CPU it was hundreds of
            # serial encoder forwards per step (the dominant overlap-reward cost).
            # ~0.5 GB on the training GPU; override with OVERLAP_STEPS_DEVICE=cpu.
            dev = os.environ.get("OVERLAP_STEPS_DEVICE", "cuda")
            # Load with DeepSpeed ZeRO-3 hidden from transformers: when zero3 is enabled,
            # from_pretrained applies zero.Init and PARTITIONS this auxiliary model's params
            # (embed_tokens.weight -> non-2-D), which then fails its own forward with
            # "RuntimeError: 'weight' must be 2-D" (it is not a DeepSpeed-managed module).
            # Temporarily clear the global HfDeepSpeedConfig so the classifier is built
            # unpartitioned on any device, then restore it.
            try:
                import transformers.integrations.deepspeed as _ds_int
                _saved_ref = _ds_int._hf_deepspeed_config_weak_ref
                _ds_int._hf_deepspeed_config_weak_ref = None
            except Exception:
                _ds_int = None
                _saved_ref = None
            try:
                self._overlap_clf = OverlapStepsClassifier.load(device=dev)
            finally:
                if _ds_int is not None:
                    _ds_int._hf_deepspeed_config_weak_ref = _saved_ref
        return self._overlap_clf

    @staticmethod
    def _row_is_natural(inputs, case_id):
        """`natural` column of one input row (--overlap_natural_only).

        Raises rather than defaulting: a missing column would otherwise silently mask
        the overlap reward on every row and quietly turn the run into --reward_variant
        none with extra steps.
        """
        row = inputs[case_id]
        if not isinstance(row, dict) or "natural" not in row:
            raise KeyError(
                "overlap_natural_only=True requires a boolean 'natural' column in the "
                "dataset, but the batch row has none. Use a corpus built by "
                "build_grpo_sets.py (cold_data/grpo_sets/*), or drop the flag."
            )
        return bool(row["natural"])

    @profiling_decorator
    def _compute_overlap_step_maps(
        self, inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
        prompt_ids, prompt_length, completion_ids, output_text,
        think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
    ):
        """Per completion -> list of {"map": (grid_h, grid_w) float32, "text": str} for
        each grounded-able observe step. Raw attention at self.overlap_layer, mean over
        self.overlap_heads, ReLU, token-reduced (self.token_reduction) over the step's
        tokens. Segmentation via sentence-split + FLAN-T5 observe classifier. The reward
        fn (think_overlap_reward) does the DINO grounding + mean_in metric.
        """
        from .overlap_steps import segment_observe_steps

        # Bisection switch: skip the entire overlap re-forward (and the T5/DINO work it
        # feeds) to test whether this path is what corrupts the later training forward.
        # Overlap reward then sees no maps -> contributes 0, but training should survive.
        if os.environ.get("DISABLE_OVERLAP_FORWARD") == "1":
            return [[] for _ in range(len(images))]

        clf = self._get_overlap_classifier()
        L = self.overlap_layer
        heads = self.overlap_heads
        tr = self.token_reduction

        fam = self.family

        results = [[] for _ in range(len(images))]

        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as _unwrapped,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # --- Layer-L attention capture (single-layer fast path) ------------------
            # output_attentions=True would run EAGER attention for ALL 36 layers and
            # materialize every layer's [heads, seq, seq] map, then use only layer L.
            # Instead we keep the base attn impl for the forward (flash/sdpa, no weights)
            # and recompute ONLY layer L's weights via a forward hook that re-runs that
            # one attention module in eager mode. It reuses the module's own q/k/v proj,
            # q_norm/k_norm and rotary, so the weights are numerically identical to the
            # all-layer path; we just supply an explicit causal+padding mask because the
            # fast attention paths may hand the layer a None mask. Net cost per case:
            # ~1 fast forward + 1 layer of eager attention (was: 1 all-eager forward).
            _cap = {"attn": None}
            _mask_holder = {"m": None}
            _reentry = {"in": False}
            _mdtype = next(_unwrapped.parameters()).dtype
            _min_val = torch.finfo(_mdtype).min

            _attn_mod = None
            for _m in _unwrapped.modules():
                if type(_m).__name__ in fam.attn_classes and getattr(_m, "layer_idx", None) == L:
                    _attn_mod = _m
                    break

            # A family loaded EAGER throughout already has the softmax weights in the
            # module's own output, so the capture is a plain read and the re-entrant
            # re-run below is unnecessary work on a layer that is not free. A Nemotron is
            # that case: `nemotron_loader` pins eager because the wrapper declares no SDPA
            # support, and `NemotronHAttention.forward` returns (output, weights). If the
            # weights come back None anyway the hook falls through to the re-run, so this
            # is a fast path and not a second implementation.
            _native = fam.attention_weights_are_returned

            def _capture_hook(module, args, kwargs, output):
                # Re-run this single attention module in eager mode to recover its
                # softmax weights (the base flash/sdpa forward returns None for them).
                if _reentry["in"]:
                    return
                if _native:
                    _w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
                    if _w is not None:
                        _cap["attn"] = _w
                        return
                _reentry["in"] = True
                _kw = dict(kwargs)
                _kw["attention_mask"] = _mask_holder["m"]
                _kw["past_key_values"] = None  # avoid double-updating the KV cache
                _kw["use_cache"] = False
                _prev_impl = module.config._attn_implementation
                module.config._attn_implementation = "eager"
                try:
                    _cap["attn"] = module(*args, **_kw)[1]
                finally:
                    module.config._attn_implementation = _prev_impl
                    _reentry["in"] = False

            _hook_handle = (
                _attn_mod.register_forward_hook(_capture_hook, with_kwargs=True)
                if _attn_mod is not None else None
            )

            # Fallback: unexpected model layout -> old all-layer eager path.
            _saved_attn_impl = _unwrapped.config._attn_implementation
            if _hook_handle is None and _saved_attn_impl == "flash_attention_2":
                _unwrapped.config._attn_implementation = "sdpa"

            # OVERLAP_PROFILE=1 -> split this method into forward(capture) vs T5-segment
            # time and confirm the layer-L hook is active. One line/step on main proc.
            import time as _time
            _prof = os.environ.get("OVERLAP_PROFILE") == "1"
            _t_fwd = _t_seg = 0.0
            _n_fwd = 0

            for case_id in range(len(images)):
                ts, te = think_start[case_id], think_end[case_id]
                # Skip malformed / empty think spans (reward -> masked/neutral).
                if not invalid[case_id] or te <= ts:
                    continue
                # --overlap_natural_only: this row's overlap reward is masked anyway, so
                # its T5 segmentation + DINO grounding are pure waste. The skip happens
                # AFTER the capture forward below -- see the note there for why the
                # forward itself cannot be skipped.
                _skip_row = self.overlap_natural_only and not self._row_is_natural(inputs, case_id)

                _case_inputs = {
                    "input_ids": prompt_completion_ids[case_id:case_id + 1],
                    "attention_mask": attention_mask[case_id:case_id + 1],
                }
                _case_inputs.update(self._mm_forward_kwargs(
                    prompt_inputs, case_id, case_id + 1,
                    seq_len=prompt_completion_ids.size(1)))

                if _prof:
                    torch.cuda.synchronize(device); _ts = _time.perf_counter()
                if _hook_handle is not None and not _native:
                    # Build the additive causal+padding mask the eager re-run needs
                    # (0 where attended, finfo.min where masked): the fast forward may
                    # hand layer L a None mask.
                    _am2d = _case_inputs["attention_mask"]  # [1, seq]
                    _seq = _am2d.shape[-1]
                    _masked = torch.triu(
                        torch.ones(_seq, _seq, dtype=torch.bool, device=device), diagonal=1
                    ) | (_am2d[0] == 0)[None, :]
                    _add = torch.zeros(_seq, _seq, dtype=_mdtype, device=device)
                    _add.masked_fill_(_masked, _min_val)
                    _mask_holder["m"] = _add[None, None]

                    _cap["attn"] = None
                    _unwrapped(**_case_inputs)  # triggers _capture_hook at layer L
                    _attn_L = _cap["attn"]
                    del _add, _masked
                elif _hook_handle is not None:
                    # The eager family: the module's own output already carries the
                    # weights, and the mask the base forward built for it is the right
                    # one. Nothing to rebuild and nothing to re-run.
                    _cap["attn"] = None
                    _unwrapped(**_case_inputs)
                    _attn_L = _cap["attn"]
                    if _attn_L is None:
                        raise RuntimeError(
                            f"{fam.name} declares its attention weights are returned, but "
                            f"layer {L} handed back None. The model is not running an "
                            "eager attention implementation -- check "
                            "config._attn_implementation on the LANGUAGE model's config, "
                            "which is a sub-config here and is set separately.")
                else:
                    _fwd = _unwrapped(**_case_inputs, output_attentions=True, output_hidden_states=False)
                    _attn_L = _fwd.attentions[L]
                    del _fwd

                # --overlap_natural_only: bail out here, not before the forward above.
                # `_unwrapped` is the very module DeepSpeed ZeRO-3 hangs its offload
                # hooks on, so each capture forward fires _start_of_forward_hook ->
                # param_coordinator.reset_step() (an allgather) and appends a full pass
                # to the coordinator's __submodule_order. ZeRO-3 asserts that trace is
                # byte-identical across ranks; ranks do not see the same
                # natural/non-natural mix, so skipping the forward makes the traces
                # diverge and the next training forward dies in reset_step with
                # "disagreement between rank0 and rankN" (then a 30-min NCCL timeout).
                # Everything below -- the CPU copy, T5 segmentation, and the DINO
                # grounding it feeds -- is rank-local and safe to skip.
                if _skip_row:
                    del _attn_L
                    continue

                _image_mask = prompt_ids[case_id] == self.image_token_id
                # [1, heads, think_len, n_patches] : observe-token query rows -> image-patch key cols
                raw = _attn_L[
                    :, heads,
                    prompt_length + ts:prompt_length + te + 1,
                    :prompt_length,
                ][:, :, :, _image_mask]
                # [n_heads_sel, think_len, n_patches]; ReLU is a no-op on softmax weights but kept per spec
                per_tok = torch.relu(raw)[0].float().cpu().numpy()
                del _attn_L, raw
                if _prof:
                    torch.cuda.synchronize(device); _t_fwd += _time.perf_counter() - _ts; _n_fwd += 1

                # The token grid, from whatever field THIS family reports it in --
                # Qwen3-VL's `image_grid_thw` over a 2x2 merge, the Omni's `imgs_sizes`
                # over a 32px token. Asking rather than computing is the whole point of
                # the seam: the Omni has no `image_grid_thw` and its grid is a different
                # shape for every picture.
                gh, gw = fam.token_grid(prompt_inputs, case_id)

                question = inputs[case_id].get("problem", "") if isinstance(inputs[case_id], dict) else ""
                if _prof:
                    _tg = _time.perf_counter()
                steps = segment_observe_steps(
                    output_text[case_id], think_start_idx[case_id], think_end_idx[case_id],
                    out, case_id, ts, te, question, clf,
                )
                if _prof:
                    _t_seg += _time.perf_counter() - _tg

                # WHY THIS PRINT EXISTS. Everything downstream of here is silent about
                # its own failures: a completion whose format does not parse is skipped,
                # a chain with no observe step yields no maps, and a map whose length
                # disagrees with the grid is dropped by `continue`. All three end as
                # `think_overlap_reward = nan` and nothing says which. On a new model
                # that is the first thing to know, so the first call reports it once.
                if not getattr(self, "_overlap_shape_logged", False) and \
                        self.accelerator.is_main_process:
                    self._overlap_shape_logged = True
                    print(f"[overlap] first scored case: {int(sum(bool(v) for v in invalid))}"
                          f"/{len(invalid)} completions format-valid, think tokens "
                          f"{ts}-{te}, {len(steps)} observe steps, map {per_tok.shape[-1]} "
                          f"patches against grid {gh}x{gw}={gh * gw}"
                          f"{'  <-- MISMATCH, every map will be dropped' if per_tok.shape[-1] != gh * gw else ''}",
                          flush=True)

                step_maps = []
                for step_text, tok_a, tok_b in steps:
                    la = tok_a - ts
                    lb = tok_b - ts
                    if lb <= la:
                        continue
                    seg = per_tok[:, la:lb, :]  # [n_heads, span_len, n_patches]
                    # token-reduce per head over the step's tokens, THEN mean over heads
                    # (order matters for max/min; matches the offline attn_tr_* reference).
                    if tr == "max":
                        red = seg.max(axis=1)
                    elif tr == "min":
                        red = seg.min(axis=1)
                    else:
                        red = seg.mean(axis=1)
                    m = np.maximum(red.mean(axis=0), 0.0)  # [n_patches]
                    if m.size != gh * gw:
                        continue
                    step_maps.append({"map": m.reshape(gh, gw).astype(np.float32), "text": step_text})
                results[case_id] = step_maps

            if _hook_handle is not None:
                _hook_handle.remove()
            elif _saved_attn_impl == "flash_attention_2":
                _unwrapped.config._attn_implementation = _saved_attn_impl

            if _prof and self.accelerator.is_main_process:
                import sys as _sys
                print(
                    f"[overlap-profile] layer{L}-hook={'active' if _hook_handle is not None else 'FALLBACK'} "
                    f"cases={_n_fwd} fwd(capture)={_t_fwd:.1f}s t5-segment={_t_seg:.1f}s "
                    f"t5-dev={next(self._get_overlap_classifier().parameters()).device}",
                    file=_sys.stderr, flush=True,
                )

        return results

    @profiling_decorator
    def _compute_grad_step_maps(
        self, inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
        prompt_ids, prompt_length, completion_ids, output_text,
        think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
    ):
        """Per completion -> list of {"map": (gh, gw) float32, "text": str} per observe
        step, where the map is the PIXEL GRADIENT of that step's own tokens:

            G_j = || d/d(pixels of image token j)  mean_{n in S} centered_logit(t_n) ||

        Same output contract as `_compute_overlap_step_maps`, so the segmentation, the
        DINO grounding and every probe that reads these maps are unchanged; only the map
        differs. `trl/grad_maps.py` holds the definition and the reasoning; the reward is
        `trl.rewards.grad_rewards.think_grad_reward`.

        Cheaper than the attention path in the model: no eager re-run of a layer and no
        [1, heads, seq, seq] tensor (gradients work under sdpa/FA2), against one extra
        backward that costs about one forward because no parameter gradients are wanted.

        THE ZeRO-3 HAZARD, which is why the structure below looks over-careful. This runs
        a BACKWARD through the very module DeepSpeed hangs its hooks on. Three things keep
        it out of the training state, and all three are load-bearing:

          * `frozen_params` clears `requires_grad` on every weight, so autograd prunes
            every weight-gradient node -- nothing accumulates into `.grad`, no gradient
            hook has anything to reduce;
          * exactly ONE model forward runs per case on EVERY rank, whatever that case
            contains. ZeRO-3 asserts its recorded module order is byte-identical across
            ranks, and ranks do not see the same mix of malformed spans or natural rows,
            so a skipped forward becomes "disagreement between rank0 and rankN" in the
            next training step and then a 30-minute NCCL timeout. A case with nothing to
            score gets a one-token dummy span and has its result discarded;
          * the ZeRO-3 trace is invalidated on both sides, as `evaluate` already does for
            the same reason: this pass is a module order training never sees.

        `DISABLE_GRAD_FORWARD=1` bisects the whole thing out (the reward then sees no maps
        and contributes nothing) if a run starts dying in `reset_step`.
        """
        from .grad_maps import frozen_params, step_grad_maps
        from .overlap_steps import segment_observe_steps

        if os.environ.get("DISABLE_GRAD_FORWARD") == "1":
            return [[] for _ in range(len(images))]

        clf = self._get_overlap_classifier()
        ip = getattr(self.processing_class, "image_processor", None)
        ps = int(getattr(ip, "patch_size", 16))
        tps = int(getattr(ip, "temporal_patch_size", 2))

        if not self.family.supports_saliency_r1():
            self._refuse_qwen3_only(
                "reward_variant='grad'",
                "it folds a pixel gradient back onto patches with Qwen3-VL's "
                "patch_size x temporal_patch_size packing, which this processor does not "
                "use")
        thw = prompt_inputs.get("image_grid_thw")  # [batch, 3]
        if thw is None:
            raise RuntimeError(
                "reward_variant='grad' differentiates w.r.t. pixel_values, but this batch "
                "carries no image_grid_thw. It needs an image corpus."
            )
        patch_offsets = [0]
        for _i in range(thw.shape[0]):
            patch_offsets.append(patch_offsets[-1] + int(thw[_i].prod().item()))

        results = [[] for _ in range(len(images))]
        self._invalidate_zero3_trace("before the gradient re-forward")

        import time as _time
        _prof = os.environ.get("OVERLAP_PROFILE") == "1"
        _t_fwd = _t_seg = 0.0
        _n_spans = 0

        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as _unwrapped,
            frozen_params(_unwrapped),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            for case_id in range(len(images)):
                ts, te = think_start[case_id], think_end[case_id]
                # Unlike the attention path, the spans have to be known BEFORE the
                # forward: they select the rows that reach lm_head. Segmentation is
                # rank-local (FLAN-T5, not the policy), so doing it first changes nothing
                # about the ZeRO-3 trace -- but the forward below must still happen for
                # every case, which is what `discard` is for.
                discard = not invalid[case_id] or te <= ts
                if not discard and self.overlap_natural_only:
                    discard = not self._row_is_natural(inputs, case_id)

                steps = []
                if not discard:
                    question = inputs[case_id].get("problem", "") if isinstance(inputs[case_id], dict) else ""
                    if _prof:
                        _tg = _time.perf_counter()
                    steps = segment_observe_steps(
                        output_text[case_id], think_start_idx[case_id], think_end_idx[case_id],
                        out, case_id, ts, te, question, clf,
                    )
                    if _prof:
                        _t_seg += _time.perf_counter() - _tg

                spans, texts = [], []
                for step_text, tok_a, tok_b in steps:
                    a, b = prompt_length + tok_a, prompt_length + tok_b
                    if b <= a or a <= 0 or b > prompt_completion_ids.shape[1]:
                        continue
                    spans.append((a, b))
                    texts.append(step_text)
                if not spans:
                    # Nothing to score, but the forward is not optional: a one-token dummy
                    # keeps this rank's module order identical to every other rank's.
                    spans = [(prompt_completion_ids.shape[1] - 1, prompt_completion_ids.shape[1])]
                    texts = []
                    discard = True

                _case_inputs = {
                    "input_ids": prompt_completion_ids[case_id:case_id + 1],
                    "attention_mask": attention_mask[case_id:case_id + 1],
                    "pixel_values": prompt_inputs["pixel_values"][
                        patch_offsets[case_id]:patch_offsets[case_id + 1]
                    ],
                    "image_grid_thw": thw[case_id:case_id + 1],
                }
                if prompt_inputs.get("mm_token_type_ids") is not None:
                    _compl_zeros = torch.zeros(1, completion_ids.size(1), dtype=torch.long, device=device)
                    _case_inputs["mm_token_type_ids"] = torch.cat(
                        [prompt_inputs["mm_token_type_ids"][case_id:case_id + 1], _compl_zeros], dim=1
                    )

                if _prof:
                    torch.cuda.synchronize(device); _tsf = _time.perf_counter()
                maps = step_grad_maps(
                    _unwrapped, _case_inputs, spans, thw[case_id].tolist(), ps, tps,
                    target=self.grad_target,
                )
                if _prof:
                    torch.cuda.synchronize(device); _t_fwd += _time.perf_counter() - _tsf
                    _n_spans += len(spans)

                if discard:
                    del maps
                    continue
                results[case_id] = [
                    {"map": maps[k].astype(np.float32), "text": texts[k]}
                    for k in range(len(texts))
                ]

        self._invalidate_zero3_trace("after the gradient re-forward")

        if _prof and self.accelerator.is_main_process:
            import sys as _sys
            print(
                f"[grad-profile] target={self.grad_target} cases={len(images)} "
                f"spans={_n_spans} fwd+bwd={_t_fwd:.1f}s t5-segment={_t_seg:.1f}s",
                file=_sys.stderr, flush=True,
            )
        return results

    @profiling_decorator
    def _compute_glimpse_step_maps(
        self, inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
        prompt_ids, prompt_length, completion_ids, output_text,
        think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
    ):
        """Per completion -> list of {"map": (gh, gw) float32, "text": str} per observe
        step, where the map is the GLIMPSE map of that step's own tokens (docs map 6):
        gradient-weighted attention, propagated with adaptive layer weights and
        aggregated over the step by a confidence x prompt-alignment weight.

        Same output contract as `_compute_grad_step_maps`, so the segmentation, the DINO
        grounding and every probe that reads these maps are unchanged; only the map
        differs. `trl/glimpse_maps.py` holds the definition; the reward is
        `trl.rewards.glimpse_rewards.think_glimpse_reward`.

        THE COST, because it is the fact that governs whether this variant is usable.
        Measured on an H100-80GB against the gradient map on identical cases
        (glimpse_cost_probe.py): 11.3-18.3 s per case at `layer_frac=1.0` against the
        gradient map's 0.20-0.25, i.e. 55-59x, or 100-145 s added to one rank's optimizer
        step against a step that is currently ~40 s in total. `--glimpse_layer_frac 0.6`
        buys 1.64x and `--glimpse_token_cap` is linear. Peak memory is NOT the problem:
        19.7 GiB against the gradient path's 20.1 on the same cases.

        THE ZeRO-3 HAZARD is the gradient path's, plus one. The same three safeguards are
        load-bearing here -- `frozen_params` so no weight gradient is ever accumulated,
        exactly ONE model forward per case on EVERY rank whatever that case contains, and
        the trace invalidated on both sides. What is new is that GLIMPSE also re-runs
        individual decoder layers in eager, once per propagated layer per target token, so
        the number of MODULE forwards varies across ranks and not just the number of
        backwards. That is safe for the same reason the variable backward count is:
        `unwrap_model_for_generation(..., gather_deepspeed3_params=True)` has already
        gathered every parameter, so no per-module fetch is issued against the recorded
        trace during this pass, and the trace is re-recorded afterwards either way.

        `DISABLE_GLIMPSE_FORWARD=1` bisects the whole thing out (the reward then sees no
        maps and contributes nothing) if a run starts dying in `reset_step`.
        """
        # The reward modules are imported ABSOLUTELY and the map modules relatively, and
        # the asymmetry is not a style slip: patch_trl_qwen3.sh copies this file and the
        # map modules into trl_repo/trl/TRAINER/, but the reward modules into
        # trl_repo/trl/REWARDS/. So `.glimpse_maps` resolves here and in trl_repo alike,
        # while `.rewards.glimpse_rewards` resolves only in this tree and raises
        # ModuleNotFoundError: trl.trainer.rewards in the one that actually executes.
        # test_import_layout_cpu.py holds the line.
        from trl.rewards.glimpse_rewards import record_map_info

        from .glimpse_maps import step_glimpse_maps
        from .grad_maps import frozen_params
        from .overlap_steps import segment_observe_steps

        if os.environ.get("DISABLE_GLIMPSE_FORWARD") == "1":
            return [[] for _ in range(len(images))]

        clf = self._get_overlap_classifier()
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)

        if not self.family.supports_saliency_r1():
            self._refuse_qwen3_only(
                "reward_variant='glimpse'",
                "it propagates gradient-weighted attention across every decoder layer, "
                "and a hybrid has no attention matrix at most of its layers")
        thw = prompt_inputs.get("image_grid_thw")  # [batch, 3]
        if thw is None:
            raise RuntimeError(
                "reward_variant='glimpse' reads the map off the image-token columns, but "
                "this batch carries no image_grid_thw. It needs an image corpus."
            )
        patch_offsets = [0]
        for _i in range(thw.shape[0]):
            patch_offsets.append(patch_offsets[-1] + int(thw[_i].prod().item()))

        results = [[] for _ in range(len(images))]
        self._invalidate_zero3_trace("before the glimpse re-forward")

        import time as _time
        _prof = os.environ.get("OVERLAP_PROFILE") == "1"
        _t_fwd = _t_seg = 0.0
        _n_spans = _n_tokens = 0

        with (
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as _unwrapped,
            frozen_params(_unwrapped),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            for case_id in range(len(images)):
                ts, te = think_start[case_id], think_end[case_id]
                # As in the gradient path, the spans have to be known BEFORE the forward:
                # they select the rows that reach lm_head. Segmentation is rank-local
                # (FLAN-T5, not the policy), so doing it first changes nothing about the
                # ZeRO-3 trace -- but the forward below must still happen for every case,
                # which is what `discard` is for.
                discard = not invalid[case_id] or te <= ts
                if not discard and self.overlap_natural_only:
                    discard = not self._row_is_natural(inputs, case_id)

                question = (inputs[case_id].get("problem", "")
                            if isinstance(inputs[case_id], dict) else "")
                steps = []
                if not discard:
                    if _prof:
                        _tg = _time.perf_counter()
                    steps = segment_observe_steps(
                        output_text[case_id], think_start_idx[case_id], think_end_idx[case_id],
                        out, case_id, ts, te, question, clf,
                    )
                    if _prof:
                        _t_seg += _time.perf_counter() - _tg

                spans, texts = [], []
                for step_text, tok_a, tok_b in steps:
                    a, b = prompt_length + tok_a, prompt_length + tok_b
                    if b <= a or a <= 0 or b > prompt_completion_ids.shape[1]:
                        continue
                    spans.append((a, b))
                    texts.append(step_text)
                if not spans:
                    # Nothing to score, but the forward is not optional: a one-token dummy
                    # keeps this rank's module order identical to every other rank's.
                    spans = [(prompt_completion_ids.shape[1] - 1, prompt_completion_ids.shape[1])]
                    texts = []
                    discard = True

                _case_inputs = {
                    "input_ids": prompt_completion_ids[case_id:case_id + 1],
                    "attention_mask": attention_mask[case_id:case_id + 1],
                    "pixel_values": prompt_inputs["pixel_values"][
                        patch_offsets[case_id]:patch_offsets[case_id + 1]
                    ],
                    "image_grid_thw": thw[case_id:case_id + 1],
                }
                if prompt_inputs.get("mm_token_type_ids") is not None:
                    _compl_zeros = torch.zeros(1, completion_ids.size(1), dtype=torch.long, device=device)
                    _case_inputs["mm_token_type_ids"] = torch.cat(
                        [prompt_inputs["mm_token_type_ids"][case_id:case_id + 1], _compl_zeros], dim=1
                    )

                if _prof:
                    torch.cuda.synchronize(device); _tsf = _time.perf_counter()
                maps, info = step_glimpse_maps(
                    _unwrapped, _case_inputs, spans, thw[case_id].tolist(),
                    question=question, tokenizer=tokenizer, prompt_len=prompt_length,
                    target=self.glimpse_target, layer_frac=self.glimpse_layer_frac,
                    temp=self.glimpse_temp, depth_temp=self.glimpse_depth_temp,
                    token_weight=self.glimpse_token_weight,
                    token_cap=self.glimpse_token_cap, seed=self.glimpse_seed,
                    image_token_id=self.image_token_id or 151655,
                )
                if _prof:
                    torch.cuda.synchronize(device); _t_fwd += _time.perf_counter() - _tsf
                    _n_spans += len(spans)
                    _n_tokens += int(info.get("n_target_tokens") or 0)

                if discard:
                    del maps
                    continue
                info["n_steps_built"] = len(texts)
                record_map_info(info)
                results[case_id] = [
                    {"map": maps[k].astype(np.float32), "text": texts[k]}
                    for k in range(len(texts))
                ]

        self._invalidate_zero3_trace("after the glimpse re-forward")

        if _prof and self.accelerator.is_main_process:
            import sys as _sys
            print(
                f"[glimpse-profile] target={self.glimpse_target} "
                f"frac={self.glimpse_layer_frac} cap={self.glimpse_token_cap} "
                f"cases={len(images)} spans={_n_spans} tokens={_n_tokens} "
                f"fwd+bwd={_t_fwd:.1f}s t5-segment={_t_seg:.1f}s",
                file=_sys.stderr, flush=True,
            )
        return results

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        self._lap()  # start the step's stopwatch; see SR1_LAP above _mem_report

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            # How THIS processor wants the pictures. Qwen3-VL's batches them one list per
            # sample; the Omni's walks a flat list, replacing each `<image>` in the text
            # rows in order, and a nested list is not iterable the way it expects.
            kwargs = {"images": self.family.batch_image_arg(images)}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                # The USER turn has to become structured: that is the only
                                # way to say "a picture goes here" to a chat template.
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            # THE SYSTEM TURN IS LEFT A PLAIN STRING, and that is a fix.
                            #
                            # It used to be wrapped the same way. Qwen3-VL's template
                            # understands that shape for the system role; the Omni's does
                            # NOT, and Jinja stringifies the list, so every Omni run ever
                            # launched sent its model this as the system prompt:
                            #
                            #   <|im_start|>system
                            #   [{'type': 'text', 'text': 'A conversation between user ...'}]
                            #
                            # Read off the live wov0.4 run's own logged completions table,
                            # 2026-10-08. The instructions were still in there, wrapped in a
                            # Python repr, which is not what any Qwen3-VL run ever sent --
                            # so the two arms differed in their prompts as well as in their
                            # cold start.
                            #
                            # A plain string is the universal form and renders BYTE-IDENTICAL
                            # on Qwen3-VL, so this is a no-op there and a repair here.
                            # test_system_prompt_render_cpu.py pins both.

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # Anything `forward()` requires that the processor does not emit. A no-op for
        # every family but Nemotron, whose forward opens with `image_flags.squeeze(-1)`
        # on a key its own processor never produces.
        prompt_inputs = self.family.after_processor(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
            # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original chat template in case it originally
            # applies it. Otherwise, it assumes that the chat template uses only vision_start_token_id to indicate images
            # (e.g. Gemma 3) and removes all image_token instances and vision_end_token_id as well, leaving only
            # the vision_start_token_id (e.g. <start_of_image>).
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                # Search for the image token in the chat template
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    # The family folds the run back to the CHAT TEMPLATE's spelling, which
                    # is what the generation server expands from. For Qwen3-VL that is the
                    # `re.sub` this replaces, verbatim; a processor that also wraps the run
                    # in `<img>`/`</img>` has to shed those or the server re-wraps it.
                    prompts_text = [
                        self.family.collapse_image_run(text, self.image_token)
                        for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, we remove all instances of it + vision_end_token_id
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # DOES THE PROMPT ALREADY OPEN THE REASONING BLOCK?
        #
        # Qwen3-VL's chat template ends the generation prompt at `<|im_start|>assistant\n`
        # and the policy writes `<think>` itself, so a completion carries both tags. The
        # Omni's ends at `<|im_start|>assistant\n<think>\n` -- the assistant turn starts
        # INSIDE the block, and the completion carries only the closing tag.
        #
        # Nothing downstream knows that. `judge_format` requires exactly one of each, so
        # every completion of a perfectly well-behaved model reads as malformed: format
        # 0.000, and then the overlap reward NaN on all of them, because the per-step maps
        # are only computed for completions whose format is valid. That is the whole
        # failure, and it looks like a broken reward rather than a template.
        #
        # Read off the ACTUAL prompt rather than declared per family, because it is a fact
        # about the chat template and a template can change under a checkpoint. The
        # anchored `$` is what keeps the system prompt's own `<think></think>` out of it.
        self._prompt_opens_think = bool(
            prompts_text and re.search(r"<think>\s*$", prompts_text[0]))

        self._lap("prep_prompts")  # chat template + processor (images) + prompt truncation

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = [
                        {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
                        for prompt, image in zip(all_prompts_text, all_images)
                    ]
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            '''
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
                
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            '''

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                if self.is_gradient_checkpointing:
                    unwrapped_model.base_model.gradient_checkpointing_disable()
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                _gen_kwargs = dict(
                    generation_config=self.generation_config,
                    temperature=1.0,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                if not self.reforward_saliency:
                    _gen_kwargs["output_attentions"] = True
                outputs = unwrapped_model.generate(**prompt_inputs, **_gen_kwargs)
                if self.is_gradient_checkpointing:
                    unwrapped_model.base_model.gradient_checkpointing_enable()

            prompt_completion_ids = outputs.sequences
            if not self.reforward_saliency:
                attentions = outputs.attentions
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # prompt_length is assigned only inside the HF-generate branch; the live vLLM and
        # transformers_paged paths don't set it. Set it here for ALL paths so the saliency
        # re-forward (which splits prompt vs completion by prompt_length) doesn't hit an
        # UnboundLocalError. prompt_ids is the un-sliced prompt in vLLM/paged, and in the HF
        # branch it was already sliced to prompt_length, so this is consistent either way.
        prompt_length = prompt_ids.size(1)

        # The whole generation block, MINUS the `vLLM.generate` call the profiler already
        # times: the weight sync, the gather_object of prompts and PIL images across six
        # ranks, and the scatter of the results back.
        self._lap("generate_block")

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    mm_source=prompt_inputs,
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        mm_source=prompt_inputs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            mm_source=prompt_inputs,
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        output_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        out = self.processing_class.tokenizer(output_text)


        pattern = r"^<think>\s*([^\s].*?)\s*</think>\s*([^\s].*?)\s*$"
        completion_contents = [completion[0]["content"] for completion in completions]

        # When the prompt opened the block, the completion is judged as the continuation
        # it is: the opening tag is real, it just lives in the prompt. Prepending it here
        # rather than loosening the pattern keeps ONE definition of the format, and keeps
        # a model that writes a second `<think>` failing, which it should.
        _opener = "<think>\n" if self._prompt_opens_think else ""

        def judge_format(pattern, response):
            response = _opener + response
            return re.match(pattern, response, re.DOTALL | re.MULTILINE) is not None and \
                response.count('<think>') == 1 and response.count('</think>') == 1
        invalid = [judge_format(pattern, content) for content in completion_contents]

        if self._prompt_opens_think:
            # The reasoning starts at the completion's first non-space character, because
            # everything before it is in the prompt.
            think_start_idx = [re.search(r"\s*(\S)", i, re.DOTALL | re.MULTILINE)
                               for i in output_text]
        else:
            think_start_idx = [re.search(r"<think>\s*(\S\S*)", i, re.DOTALL | re.MULTILINE) for i in output_text]
        think_end_idx = [re.search(r"(\S)\s*</think>", i, re.DOTALL | re.MULTILINE) for i in output_text]
        answer_start_idx = [re.search(r"</think>\s*(\S\S*)", i, re.DOTALL | re.MULTILINE) for i in output_text]
        answer_end_idx = [re.search(r"(\S)\s*<\|im_end\|>", i, re.DOTALL | re.MULTILINE) for i in output_text]

        think_start_idx = [i.start(1) if i else -1 for i in think_start_idx]
        think_end_idx = [i.start(1) if i else -1 for i in think_end_idx]
        answer_start_idx = [i.start(1) if i else -1 for i in answer_start_idx]
        answer_end_idx = [i.start(1) if i else -1 for i in answer_end_idx]

        think_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(think_start_idx)]
        think_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(think_end_idx)]
        answer_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(answer_start_idx)]
        answer_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(answer_end_idx)]

        _max_completion_idx = (len(attentions) - 1) if not self.reforward_saliency else (completion_ids.size(1) - 1)
        think_end = [i if j else 0 for i, j in zip(think_end, invalid)]
        think_start = [min(i, z) if j else 0 for i, j, z in zip(think_start, invalid, think_end)]
        answer_end = [min(i, _max_completion_idx) if j else 1 for i, j in zip(answer_end, invalid)]
        answer_start = [min(z, i) if j else 1 for i, j, z in zip(answer_start, invalid, answer_end)]


        '''
        pattern = r'.*<think>\s*([^\s].*?)\s*</think>.*<answer>\s*([^\s].*?)\s*</answer>.*'
        completion_contents = [completion[0]["content"] for completion in completions]

        def judge_format(pattern, response):
            return re.match(pattern, response, re.DOTALL | re.MULTILINE) is not None and \
                response.count('<think>') == 1 and response.count('</think>') == 1 and \
                response.count('<answer>') == 1 and response.count('</answer>') == 1

        invalid = [judge_format(pattern, content) for content in completion_contents]
        attn_batch = []
        #debug = [k for i, j, z, k in zip(think_start_idx, think_end_idx, invalid, output_text) if z and (i<0 or j<0)]
        #print(debug)
        
        think_start_idx = [re.search(r"<think>\s*(\S\S*)", i, re.DOTALL | re.MULTILINE) for i in output_text]
        think_end_idx = [re.search(r"(\S)\s*</think>", i, re.DOTALL | re.MULTILINE) for i in output_text]
        answer_start_idx = [re.search(r"<answer>\s*(\S\S*)", i, re.DOTALL | re.MULTILINE) for i in output_text]
        answer_end_idx = [re.search(r"(\S)\s*</answer>", i, re.DOTALL | re.MULTILINE) for i in output_text]

        think_start_idx = [i.start(1) if i else -1 for i in think_start_idx]
        think_end_idx = [i.start(1) if i else -1 for i in think_end_idx]
        answer_start_idx = [i.start(1) if i else -1 for i in answer_start_idx]
        answer_end_idx = [i.start(1) if i else -1 for i in answer_end_idx]


        think_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(think_start_idx)]
        think_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(think_end_idx)]
        answer_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(answer_start_idx)]
        answer_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(answer_end_idx)]
        
        think_end = [i if j else 0 for i, j in zip(think_end, invalid)]
        think_start = [min(i, z) if j else 0 for i, j, z in zip(think_start, invalid, think_end)]
        answer_end = [min(i, len(attentions) - 1) if j else 1 for i, j in zip(answer_end, invalid)]
        answer_start = [min(z, i) if j else 1 for i, j, z in zip(answer_start, invalid, answer_end)]
        '''
        # Decode, EOS-mask, old/ref log-probs, and the think/answer span search.
        self._lap("post_generate")

        attn_batch = []

        if self.reward_variant == "none":
            # No saliency/overlap reward: skip all attention extraction. attn_batch stays
            # [] -- the remaining reward funcs (accuracy/judge/format) ignore saliency_map.
            pass
        elif self.reward_variant == "ours":
            # --- Attention-overlap reward: raw per-head observe->patch attention at a
            # single layer, per observe step, instead of the whole-completion rollout. ---
            attn_batch = self._compute_overlap_step_maps(
                inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
                prompt_ids, prompt_length, completion_ids, output_text,
                think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
            )
        elif self.reward_variant == "grad":
            # --- Roll-null gradient reward: the pixel gradient of each observe step's
            # own tokens, instead of attention. Same per-step map contract. ---
            attn_batch = self._compute_grad_step_maps(
                inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
                prompt_ids, prompt_length, completion_ids, output_text,
                think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
            )
        elif self.reward_variant == "glimpse":
            # --- GLIMPSE grounding reward: gradient-weighted attention propagated with
            # adaptive layer weights, per observe step. Same per-step map contract, and
            # 55-59x the gradient path's cost -- see _compute_glimpse_step_maps. ---
            attn_batch = self._compute_glimpse_step_maps(
                inputs, images, prompt_inputs, prompt_completion_ids, attention_mask,
                prompt_ids, prompt_length, completion_ids, output_text,
                think_start_idx, think_end_idx, think_start, think_end, invalid, out, device,
            )
        elif self.reforward_saliency:
            # --- Re-forward path: generate without output_attentions, then do a cheap
            # per-case forward pass to extract attention slices for saliency. ---
            if not self.family.supports_saliency_r1():
                self._refuse_qwen3_only(
                    "the original Saliency-R1 readout (--reward_variant saliency_r1)",
                    "it multiplies every layer's attention by that layer's value states "
                    "and pushes the result through `o_proj`, and a hybrid decoder has no "
                    "`self_attn` at most of its layers")
            thw = prompt_inputs.get("image_grid_thw")  # [batch, 3]
            patch_offsets = [0]
            if thw is not None:
                for _i in range(thw.shape[0]):
                    patch_offsets.append(patch_offsets[-1] + int(thw[_i].prod().item()))

            # Phase 1: one full-sequence re-forward per case; extract small attention
            # slices and immediately free the large attention tensor.
            all_think_attns = []  # list[list[Tensor]]  per case, per layer
            all_token_attns = []
            all_image_masks = []

            with (
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as _unwrapped,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # FA2 can't return attention weights; temporarily use SDPA for the
                # re-forward passes (dispatch is resolved at forward time from config).
                _saved_attn_impl = _unwrapped.config._attn_implementation
                if _saved_attn_impl == "flash_attention_2":
                    _unwrapped.config._attn_implementation = "sdpa"
                for case_id in range(len(images)):
                    _case_inputs = {
                        "input_ids": prompt_completion_ids[case_id:case_id + 1],
                        "attention_mask": attention_mask[case_id:case_id + 1],
                    }
                    if thw is not None:
                        _case_inputs["pixel_values"] = prompt_inputs["pixel_values"][
                            patch_offsets[case_id]:patch_offsets[case_id + 1]
                        ]
                        _case_inputs["image_grid_thw"] = thw[case_id:case_id + 1]
                    if prompt_inputs.get("mm_token_type_ids") is not None:
                        _compl_zeros = torch.zeros(
                            1, completion_ids.size(1), dtype=torch.long, device=device
                        )
                        _case_inputs["mm_token_type_ids"] = torch.cat(
                            [prompt_inputs["mm_token_type_ids"][case_id:case_id + 1], _compl_zeros], dim=1
                        )

                    _fwd = _unwrapped(**_case_inputs, output_attentions=True, output_hidden_states=False)

                    _image_mask = prompt_ids[case_id] == 151655
                    _think_per_layer, _token_per_layer = [], []
                    for _l in range(self.NUM_LAYER):
                        _attn = _fwd.attentions[_l]  # [1, heads, full_len, full_len]
                        _think_per_layer.append(_attn[
                            :, :,
                            prompt_length + answer_start[case_id]:prompt_length + answer_end[case_id] + 1,
                            prompt_length + think_start[case_id]:prompt_length + think_end[case_id] + 1,
                        ].clone())
                        _token_per_layer.append(_attn[
                            :, :,
                            prompt_length + think_start[case_id]:prompt_length + think_end[case_id] + 1,
                            :prompt_length,
                        ][:, :, :, _image_mask].clone())
                    del _fwd
                    all_think_attns.append(_think_per_layer)
                    all_token_attns.append(_token_per_layer)
                    all_image_masks.append(_image_mask)
                if _saved_attn_impl == "flash_attention_2":
                    _unwrapped.config._attn_implementation = _saved_attn_impl

            # Phase 2: saliency computation using extracted slices (same math as
            # original, adapted for the [1, heads, n_ans, think_len] slice shape).
            with torch.no_grad():
                for case_id in range(len(images)):
                    _image_mask = all_image_masks[case_id]
                    logits = 0
                    for layers in range(self.NUM_LAYER):
                        value_states = outputs.past_key_values.layers[layers].values.clone().detach()[[case_id]]
                        value_states = repeat_v(value_states, self.NUM_GROUP)
                        value_states = value_states[:, :, :prompt_length, :]
                        value_states = value_states[:, :, _image_mask, :]

                        think_attn = all_think_attns[case_id][layers]   # [1, heads, n_ans, think_len]
                        token_attn = all_token_attns[case_id][layers]   # [1, heads, think_len, n_img]
                        token_attn = token_attn[:, :, :think_attn.shape[-1], :]
                        num_answer = think_attn.shape[2]
                        # permute to [n_ans, heads, n_img, 1] — same shape as original agg_attn
                        agg_attn = (think_attn @ token_attn).permute(2, 1, 3, 0)
                        sv = (agg_attn * value_states).transpose(1, 2).reshape(num_answer, -1, self.DIMS)
                        logits += self._qwen3_lang_model.layers[layers].self_attn.o_proj(sv)
                    logits = self._qwen3_lang_model.norm(logits) * logits.norm(dim=-1, keepdim=True)
                    hidden_norm = torch.cat([outputs.hidden_states[answer_token][-1][[case_id]] for answer_token in
                               range(answer_start[case_id], answer_end[case_id] + 1)], dim=0).norm(dim=-1, keepdim=True)
                    logits = logits / hidden_norm
                    out = self.model.lm_head(logits)
                    indices = [completion_ids[case_id][answer_token] for answer_token in
                               range(answer_start[case_id], answer_end[case_id] + 1)]
                    out = out[torch.arange(out.size(0)), :, torch.tensor(indices)].sum(dim=0).reshape(
                        prompt_inputs.get("image_grid_thw")[case_id, 1] // 2,
                        prompt_inputs.get("image_grid_thw")[case_id, 2] // 2)
                    saliency = torch.relu(out).detach().cpu().float().numpy()
                    saliency = cv2.resize(saliency, images[case_id].size)
                    attn_batch.append(saliency)

        else:
            # --- Original path: attentions stored during generate (output_attentions=True). ---
            if not self.family.supports_saliency_r1():
                self._refuse_qwen3_only(
                    "the original Saliency-R1 readout (--reward_variant saliency_r1)",
                    "it multiplies every layer's attention by that layer's value states "
                    "and pushes the result through `o_proj`, and a hybrid decoder has no "
                    "`self_attn` at most of its layers")
            with torch.no_grad():
                for case_id in range(len(images)):
                    logits = 0
                    for layers in range(self.NUM_LAYER):
                        value_states = outputs.past_key_values.layers[layers].values.clone().detach()[[case_id]]
                        value_states = repeat_v(value_states, self.NUM_GROUP)
                        value_states = value_states[:, :, :prompt_length, :]
                        value_states = value_states[:, :, prompt_ids[case_id] == 151655, :]
                        think_attn = torch.cat([attentions[answer_token][layers]
                                                    [[case_id], :, -1:,
                                                prompt_length + think_start[case_id]:prompt_length + think_end[case_id] + 1]
                                                for answer_token in range(answer_start[case_id], answer_end[case_id] + 1)],
                                               dim=0)
                        token_attn = torch.cat([attentions[i][layers][[case_id]][:, :, -1:,
                                                prompt_completion_ids[case_id, :i + prompt_length] == 151655]
                                                for i in range(think_start[case_id], think_end[case_id] + 1)], dim=2)
                        token_attn = token_attn[:, :, :think_attn.shape[-1], :]
                        agg_attn = (think_attn @ token_attn).transpose(2, 3)
                        sv = (agg_attn * value_states).transpose(1, 2).reshape(len(think_attn), -1, self.DIMS)
                        logits += self._qwen3_lang_model.layers[layers].self_attn.o_proj(sv)
                    logits = self._qwen3_lang_model.norm(logits) * logits.norm(dim=-1, keepdim=True)
                    hidden_norm = torch.cat([outputs.hidden_states[answer_token][-1][[case_id]] for answer_token in
                               range(answer_start[case_id], answer_end[case_id] + 1)], dim=0).norm(dim=-1, keepdim=True)
                    logits = logits / hidden_norm
                    out = self.model.lm_head(logits)
                    indices = [completion_ids[case_id][answer_token] for answer_token in
                               range(answer_start[case_id], answer_end[case_id] + 1)]
                    out = out[torch.arange(out.size(0)), :, torch.tensor(indices)].sum(dim=0).reshape(
                        prompt_inputs.get("image_grid_thw")[case_id, 1] // 2,
                        prompt_inputs.get("image_grid_thw")[case_id, 2] // 2)
                    saliency = torch.relu(out).detach().cpu().float().numpy()
                    saliency = cv2.resize(saliency, images[case_id].size)
                    attn_batch.append(saliency)

        torch.cuda.empty_cache()

        # THIS RANK's saliency capture, then the wait for the slowest one. The split
        # matters: the capture is eight teacher-forced forwards over completions whose
        # lengths differ by 10x between ranks, so the step costs the MAXIMUM and the
        # profiler only ever saw the main process's own share.
        self._lap("saliency_block")
        self._lap_barrier("wait_ranks_after_saliency")

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list, attn_batch, invalid)
        self._lap("rewards_block")

        # Apply weights to each reward function's output and sum.
        #
        # A reward func returns None -> NaN when it did not APPLY to a completion, not
        # when the completion scored badly: every observe step ungroundable, every step
        # dropped by --max_union_area, or --overlap_natural_only masking the row. The
        # `nansum` this replaces read that NaN as 0, which is only neutral for a metric
        # whose chance level happens to BE 0. Measured on the four 8k runs (2026-08-18):
        #
        #   mean_in_v2 (level ~1.27, chance 1.0)  masked completion scored 0 -> mean
        #                                         advantage -1.86 in answer-tied groups
        #   roll-null  (level ~-0.28, chance 0)   masked completion scored 0 -> mean
        #                                         advantage +1.05, i.e. masking PAID
        #
        # and the magnitude is not w's to control: scale_rewards renormalises each group
        # by its own std, so in a group whose answer-side rewards tie -- 25-44% of them,
        # rising as accuracy saturates -- an unmeasured saliency reward was the ENTIRE
        # advantage. That also makes the cap self-defeating on a roll-null run, which is
        # why this is fixed before --max_union_area is turned on rather than after.
        #
        # Impute the GROUP's mean over the completions the func did score. The masked
        # completion then deviates from the group mean by exactly 0 on that dimension:
        # it neither gains nor loses advantage from a reward that was never measured on
        # it, while its groupmates still score against each other. This is the "masked ->
        # neutral in the GRPO advantage" that overlap_rewards / grad_rewards /
        # glimpse_rewards already promise in their docstrings.
        rewards = (impute_unscored_rewards(rewards_per_func, self.num_generations)
                   * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        # WITHIN-GROUP std of each reward term. The /std above is across all rollouts in
        # the batch, and that is NOT the quantity that reaches the gradient: the advantage
        # is `reward - group_mean`, so a term's spread ACROSS prompts cancels and only its
        # spread WITHIN one prompt's generation group survives. Two reward terms with the
        # same /std can therefore apply very different pressure at the same weight, which
        # is exactly what the auxiliary weights (w 0.4 / 0.11 / 0.033 / 0.020) are set from
        # -- and until now the number they are set from could only be measured offline
        # from a probe run or reconstructed from the wandb completions table.
        #
        # Group-mean-centre, then pool: sqrt( sum_g sum_i (x_gi - mean_g)^2 / sum_g (n_g - 1) ),
        # over the completions the func actually scored (NaN = not applied, and a group
        # with fewer than 2 scored completions contributes nothing). No new collective --
        # rewards_per_func is already gathered, so every rank computes the same number.
        _rpf = rewards_per_func.view(-1, self.num_generations, rewards_per_func.size(1))
        _n_scored = (~torch.isnan(_rpf)).sum(dim=1)                       # [groups, funcs]
        _dev2 = (_rpf - torch.nanmean(_rpf, dim=1, keepdim=True)) ** 2    # NaN where unscored
        _ss = torch.nansum(_dev2, dim=1).sum(dim=0)                       # [funcs]
        _dof = (_n_scored - 1).clamp(min=0).sum(dim=0)                    # [funcs]
        for i, reward_func_name in enumerate(self.reward_func_names):
            if _dof[i] > 0:
                self._metrics[mode][f"rewards/{reward_func_name}/within_group_std"].append(
                    (_ss[i] / _dof[i]).sqrt().item()
                )
        if self.overlap_natural_only:
            # Share of rollouts the overlap reward was actually scored on. The
            # rewards/think_overlap_reward/mean above is a nanmean, so it already
            # averages over these rows only -- this says how many they were.
            _nat = torch.tensor(
                [float(self._row_is_natural(inputs, i)) for i in range(len(inputs))],
                dtype=torch.float32, device=device,
            )
            self._metrics[mode]["overlap/natural_frac"].append(gather(_nat).mean().item())
        # --length-guard by-products. Drained OUTSIDE every reward_variant branch, because
        # the guard is an additional term that applies under all of them (including
        # 'none'). `frac_penalized` is the one to read: it says what share of completions
        # the guard is actually touching, which is the open question about the term --
        # set_c's MEAN length at its worst step sat inside any band wide enough to permit
        # the healthy shortening every good run does, so the guard reaches that failure
        # only through the tail, if at all. 0.00 means it is inert and the run's length
        # behaviour is entirely the other rewards' doing. is_active() is a rank-uniform CLI
        # decision, so branching the collectives below on it is safe -- same argument as
        # the placebo and mask-free blocks.
        from trl.rewards.length_guard_rewards import is_active as _lenguard_active
        from trl.rewards.length_guard_rewards import pop_diagnostics as pop_lenguard_diagnostics

        if _lenguard_active():
            for _k, _v in pop_lenguard_diagnostics().items():
                _t = torch.tensor([_v], dtype=torch.float32, device=device)
                _g = gather(_t)
                _g = _g[~torch.isnan(_g)]
                if _g.numel():
                    self._metrics[mode][f"lenguard/{_k}"].append(_g.mean().item())
        if self.reward_variant in ("ours", "grad", "glimpse"):
            # Roll-null by-products, when --overlap_metric is 'logratio'. For 'grad'
            # that metric takes grad_rewards' own path instead, so these stay NaN
            # there and the grad/* block above carries them -- drained either way,
            # because the number of collectives must not depend on the metric.
            # All NaN (and so dropped) for the other metrics, but drained unconditionally:
            # the gather below is a collective, so the NUMBER of them must not depend on
            # which metric a rank was configured with. `toroidal_frac` is the one to read
            # -- it says the in-frame control pool was too small and the null wrapped
            # across the image border, which changes what the score means.
            from trl.rewards.overlap_rewards import pop_diagnostics as pop_roll_diagnostics

            _pfx = {"glimpse": "glimpse", "grad": "grad"}.get(self.reward_variant, "overlap")
            for _k, _v in pop_roll_diagnostics().items():
                _t = torch.tensor([_v], dtype=torch.float32, device=device)
                _g = gather(_t)
                _g = _g[~torch.isnan(_g)]
                if _g.numel():
                    self._metrics[mode][f"{_pfx}/roll_{_k}"].append(_g.mean().item())
        if self.reward_variant == "ours":
            # --placebo by-products. `roll_toroidal_frac` is the one to read: it says the
            # union's bounding box already filled the grid, so the control wrapped across
            # the image border and no longer has the union's SHAPE -- which is the only
            # property that makes `roll` a control rather than a second copy of `random`.
            # All NaN for --placebo random|length (no roll), and drained only when a
            # placebo is installed, which is a rank-uniform CLI decision -- so the number
            # of collectives below is the same on every rank, which is what matters.
            from trl.rewards.placebo_rewards import is_active as _placebo_active
            from trl.rewards.placebo_rewards import pop_diagnostics as pop_placebo_diagnostics

            if _placebo_active():
                for _k, _v in pop_placebo_diagnostics().items():
                    _t = torch.tensor([_v], dtype=torch.float32, device=device)
                    _g = gather(_t)
                    _g = _g[~torch.isnan(_g)]
                    if _g.numel():
                        self._metrics[mode][f"placebo/{_k}"].append(_g.mean().item())

            # --maskfree by-products. Both variants record BOTH `flatness` and `mass`, not
            # just the one being scored: the hypothesis under test is that mean_in raised
            # image mass through a flatness reward, so a --maskfree flatness run whose mass
            # rises is the result, and it is invisible if only the scored quantity is
            # logged. Same rank-uniform argument as the placebo block above.
            from trl.rewards.maskfree_rewards import is_active as _maskfree_active
            from trl.rewards.maskfree_rewards import pop_diagnostics as pop_maskfree_diagnostics

            if _maskfree_active():
                for _k, _v in pop_maskfree_diagnostics().items():
                    _t = torch.tensor([_v], dtype=torch.float32, device=device)
                    _g = gather(_t)
                    _g = _g[~torch.isnan(_g)]
                    if _g.numel():
                        self._metrics[mode][f"maskfree/{_k}"].append(_g.mean().item())

            # --mismatch_bank by-products. `exact_len_frac` is the one to read: the donor
            # bank is built from the cold-start model, whose chains stop at 14 observe
            # steps, and a drifting policy walks past that (the trained checkpoints in the
            # same probes reach 85). It falling is the run leaving the lengths the bank can
            # match exactly, which is a description of the drift and not a failure -- the
            # nearest length from the SAME donor row costs 0.21x the reward's within-group
            # spread. Same rank-uniform argument as the two blocks above.
            from trl.rewards.mismatch_rewards import is_active as _mismatch_active
            from trl.rewards.mismatch_rewards import pop_diagnostics as pop_mismatch_diagnostics

            if _mismatch_active():
                for _k, _v in pop_mismatch_diagnostics().items():
                    _t = torch.tensor([_v], dtype=torch.float32, device=device)
                    _g = gather(_t)
                    _g = _g[~torch.isnan(_g)]
                    if _g.numel():
                        self._metrics[mode][f"mismatch/{_k}"].append(_g.mean().item())

            # Where the MASK came from, under --overlap_rect_placement and
            # --overlap_chain_boxes. `ring_frac` is the one to read: the grid's one-patch
            # border is the sink `mean_in` divides by (it holds ~half the attention mass
            # and most of the peaks), so a mask that reaches it is scoring the sink
            # against itself. Both interior placements contract to 0.000, and the number
            # drifting off zero means the arm is no longer the control it is named for.
            # `chain_ungrounded_frac` says how often one-call-per-completion grounding
            # lost a whole completion where per-step grounding would have lost one step.
            # Same rank-uniform argument as the three blocks above.
            from trl.rewards.overlap_rewards import mask_diag_active as _mask_diag_active
            from trl.rewards.overlap_rewards import pop_mask_diagnostics

            if _mask_diag_active():
                for _k, _v in pop_mask_diagnostics().items():
                    _t = torch.tensor([_v], dtype=torch.float32, device=device)
                    _g = gather(_t)
                    _g = _g[~torch.isnan(_g)]
                    if _g.numel():
                        self._metrics[mode][f"mask/{_k}"].append(_g.mean().item())
        if self.reward_variant == "grad":
            # The roll-null closes box size and confidence; it does not close the centre
            # hack (`ecc` rising, and correlating with the score), the reward going hollow
            # (`n_image` collapsing while the score rises), or the step-set hacks
            # (`dup_frac`, `n_steps`, `grounded_frac`). These are how those become
            # visible; see trl/rewards/grad_rewards.py. Rank-local means, gathered here.
            # pop_diagnostics returns a FIXED key set (NaN where this rank saw nothing),
            # because `gather` below is a collective: a rank-dependent key set would mean
            # a rank-dependent number of collectives, which hangs rather than fails.
            from trl.rewards.grad_rewards import pop_diagnostics

            for _k, _v in pop_diagnostics().items():
                _t = torch.tensor([_v], dtype=torch.float32, device=device)
                _g = gather(_t)
                _g = _g[~torch.isnan(_g)]
                # Uniform across ranks: the decision is taken on the GATHERED tensor.
                if _g.numel():
                    self._metrics[mode][f"grad/{_k}"].append(_g.mean().item())
        if self.reward_variant == "glimpse":
            # `union_frac` and `ceiling` are the ones to read first, not `score_raw`:
            # mean_in_v2's ceiling IS n_patches/n_in, so a union that grows raises the
            # score with no change in the map, and the screen measured the map's own
            # grounding decaying with union area (r = -0.487). The rest are the same
            # hack monitors the gradient reward carries. Fixed key set, same NCCL reason.
            from trl.rewards.glimpse_rewards import pop_diagnostics as pop_glimpse_diagnostics

            for _k, _v in pop_glimpse_diagnostics().items():
                _t = torch.tensor([_v], dtype=torch.float32, device=device)
                _g = gather(_t)
                _g = _g[~torch.isnan(_g)]
                if _g.numel():
                    self._metrics[mode][f"glimpse/{_k}"].append(_g.mean().item())
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        # Overall (weighted-sum) reward mean/std across ALL rollouts, in the same
        # rewards/* namespace and same "across rollouts" semantics as the per-function
        # stats above. Note: this /std differs from `reward_std`, which is the mean
        # within-group std used for GRPO advantage normalization.
        self._metrics[mode]["rewards/overall/mean"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["rewards/overall/std"].append(nanstd(rewards).item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # The advantage normalisation and ~15 scalar `gather`s. Split from the object
        # gathers below because the two are nothing alike: a scalar all-gather is a
        # rendezvous and some bytes, and `gather_object` PICKLES. Job 7103770 measured the
        # pair together at 38.8 s a step -- 17% of a 222.5 s step -- and which half that
        # is decides whether the fix is free.
        self._lap("epilogue_metrics")

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        self._lap("epilogue_log_text")

        # `self._logs` is a deque(maxlen=generation_batch_size) that is READ only under
        # --log_completions, so without the flag this gather feeds a buffer nobody opens.
        # Correctness, not a speed-up: set_a's pictures are capped at 512 px on the long
        # side and pickle to a median 0.50 MB, so the whole step moves ~24 MB across six
        # ranks -- milliseconds on NVLink. The 38.8 s this span was measured at is far more
        # likely the 24 `.item()` calls above it, each of which is a host-device sync.
        if has_images and self.log_completions:
            self._logs["image"].extend(gather_object(images))

        self._lap("epilogue_log_images")

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        # Carry the family's multimodal inputs forward to the loss pass, plus its
        # GEOMETRY keys -- `imgs_sizes` is not a forward kwarg and `token_grid` reads it,
        # so dropping it here would leave the loss pass unable to say how big the picture
        # was. Every one of these is row-aligned with the batch except the ones the family
        # declares packed, which is what lets `shuffle_sequence_dict` reorder them.
        for key in (*self.family.mm_inputs, *self.family.geometry_inputs):
            if key in prompt_inputs:
                output[key] = prompt_inputs[key]
        # Advantage normalisation, the metric gathers, and -- under --log_completions --
        # a `gather_object` of every prompt, completion and PIL image in the batch.
        self._lap("epilogue")
        self._lap_flush()
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            mm_source=inputs,
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    # ---- SR1_LAP: the part of the step that no profiler covers -------------------------
    #
    # TRL's `profiling_context` wraps whole METHODS, so everything it can see is a method:
    # `vLLM.generate`, `_compute_overlap_step_maps`, `_calculate_rewards`, `compute_loss`.
    # On the Omni those add to ~102 s inside a `_prepare_inputs` that measures 169 s, and
    # the missing 67 s is all inline code in the middle of `_generate_and_score_completions`
    # -- the processor call on eight native-resolution pictures, `gather_object` on the
    # images, the decode-and-pad, the logging gathers -- none of which is a method and none
    # of which can be wrapped without re-indenting branches in a 970-line function.
    #
    # So this is a stopwatch and not a context manager: `_lap("name")` closes the span that
    # began at the previous mark and names it. Single-line inserts cannot mis-indent an
    # `if`, and adding one more probe is one more line rather than a diff that moves code.
    #
    # Everything here is off unless SR1_LAP=1. `_lap_barrier` is the exception worth
    # stating: it calls `wait_for_everyone()`, which does not exist in the uninstrumented
    # run. It does not ADD time -- the ranks already meet at the next collective -- it MOVES
    # it, out of whatever span happens to follow and into a line that says "waiting for the
    # slowest rank". That distinction is the whole question for the saliency capture, whose
    # per-step cost on the main process ranges 11-270 s across runs.

    def _lap(self, name=None):
        """Mark the end of a span. `name=None` just starts the clock."""
        if os.environ.get("SR1_LAP") != "1":
            return
        now = time.perf_counter()
        prev = getattr(self, "_lap_t", None)
        if prev is not None and name:
            acc = getattr(self, "_lap_acc", None)
            if acc is None:
                acc = self._lap_acc = {}
            acc[name] = acc.get(name, 0.0) + (now - prev)
        self._lap_t = now

    def _lap_barrier(self, name):
        """Close a span at a rendezvous, so straggler time is attributed to itself."""
        if os.environ.get("SR1_LAP") != "1":
            return
        self.accelerator.wait_for_everyone()
        self._lap(name)

    def _lap_flush(self):
        """Log one step's spans and reset. wandb, and the log when SR1_LAP_STDOUT=1."""
        if os.environ.get("SR1_LAP") != "1":
            return
        acc = getattr(self, "_lap_acc", None)
        if not acc:
            return
        self._lap_acc = {}
        self._lap_t = None
        if self.accelerator.is_main_process:
            if "wandb" in self.args.report_to:
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log({f"lap/{k}": v for k, v in acc.items()})
                except Exception:
                    pass
            if os.environ.get("SR1_LAP_STDOUT") == "1":
                step = getattr(getattr(self, "state", None), "global_step", -1)
                parts = "  ".join(f"{k} {v:.1f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1]))
                print(f"[lap] step {step:>4}  {parts}", flush=True)

    def _mem_report(self, where, tokens=None):
        """One line of CUDA accounting per micro-step.

        SR1_MEM_REPORT=N prints it for the first N calls and then goes quiet. It exists
        because "73.84 GB in use and it wants 5.50 more" says nothing about WHICH 73.84:
        the weights are 62 GB, the measured single-process benchmark peaked at 73.3 GB
        INCLUDING that request, and the gap between those two numbers is the whole
        question. `reserved - allocated` separates "the allocator is holding it" from
        "something owns it", which is what decides whether the fix is a knob or a design.

        FREE IS THE COLUMN THAT WAS MISSING, and it is the only one of the four that can
        explain the failure this was written for. The three torch counters describe torch's
        own pool; `cuda.mem_get_info()` asks the DRIVER. Everything that allocates outside
        the caching allocator -- the CUDA context, cuBLAS workspaces, and NCCL's
        communicator and channel buffers -- is invisible to the first three and comes
        straight off the fourth. The 2026-09-28 resumes died in
        `ncclUnhandledCudaError: Cuda failure 2 'out of memory'` raised from DDP's
        allreduce inside `loss.backward()`, NOT in `torch.OutOfMemoryError`, with torch's
        own peak at 68.0 GB of 79.2. A report that prints only torch's numbers says "11 GB
        spare" about a card that had none left to give NCCL.

        Three env knobs, all off by default so a Qwen3-VL run is untouched:

            SR1_MEM_REPORT=N          first N calls (unchanged)
            SR1_MEM_REPORT_EVERY=K    and then every Kth call, for the whole run -- this is
                                      what lets peak be plotted against step number
            SR1_MEM_REPORT_RANKS=all  every rank, not just the main process. Peak is
                                      per-DEVICE and the ranks do not carry equal work, so
                                      the main process's number is a lower bound on the
                                      one that decides whether the step fits.

        `tokens` is the micro-step's actual forwarded sequence length, so the working set
        can be regressed on it rather than on the batch's `completions/mean_length`.
        """
        n = int(os.environ.get("SR1_MEM_REPORT", "0"))
        every = int(os.environ.get("SR1_MEM_REPORT_EVERY", "0"))
        if not n and not every:
            return
        calls = getattr(self, "_mem_reports", 0)
        self._mem_reports = calls + 1
        due = (calls < n) or (every and calls % every == 0)
        if not due:
            return
        if os.environ.get("SR1_MEM_REPORT_RANKS", "") != "all" and not self.accelerator.is_main_process:
            return
        g = 2 ** 30
        free, total = torch.cuda.mem_get_info()
        rank = self.accelerator.process_index
        step = getattr(getattr(self, "state", None), "global_step", -1)
        print(f"[mem] r{rank} step {step:>4} {where:<22} "
              f"allocated {torch.cuda.memory_allocated()/g:5.1f}  "
              f"reserved {torch.cuda.memory_reserved()/g:5.1f}  "
              f"peak {torch.cuda.max_memory_allocated()/g:5.1f}  "
              f"free {free/g:5.1f}  (of {total/g:.1f} GB)"
              + ("" if tokens is None else f"  tokens {tokens}"), flush=True)

    def _compute_loss(self, model, inputs):
        # BEFORE the empty_cache below, deliberately: this is the only point in the loop
        # that sees what the PREVIOUS micro-step's backward left behind. `empty_cache()`
        # returns the allocator's pool to the driver, so "before the forward" always reads
        # tidy and can never show the post-backward reserved that NCCL had to allocate
        # around.
        self._mem_report("entering compute_loss")
        if os.environ.get("SR1_EMPTY_CACHE_PER_MICROSTEP") == "1":
            # RELEASE THE CACHE BEFORE THE BIGGEST ALLOCATION OF THE STEP, and the numbers
            # are why. On the Omni a training rank sits at
            #
            #     allocated 62.4 GB   reserved 71.9 GB   peak 74.1 GB   of 79.2
            #
            # before the forward -- so 9.2 GB is held by the ALLOCATOR and owned by
            # nothing, left over from the transients of generation and the saliency
            # capture. The backward then asks for one ~5.5 GB block (an MoE recompute; 128
            # experts) and fails with ~5.3 GB free, on a card whose real occupancy is 62.
            # Fragmentation, not a shortfall, and this is what it is for.
            #
            # Only the forward's own activations survive it -- 0.3 GB, because every one of
            # the 52 blocks is recomputing -- so nothing needed is thrown away.
            #
            # OFF by default: it is a sync per micro-step, and a run with room does not
            # need it.
            torch.cuda.empty_cache()
        # ("before the forward" is reported further down, once the padding trim has fixed
        # the sequence length, so the line can carry the token count it is explaining.)
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        if os.environ.get("SR1_TRIM_COMPLETION_PADDING") == "1":
            # DON'T FORWARD PADDING THE LOSS ALREADY MASKS.
            #
            # `completion_ids` is padded to the longest completion in the whole GENERATION
            # batch -- 1,024 whenever any one rollout hits the cap -- and a micro-batch is
            # ONE sequence, whose own completion averages ~300 tokens. Every position past
            # its EOS has `completion_mask == 0`, contributes nothing to the loss, and is
            # forwarded and back-propagated anyway. On a 33B mixture of experts that is not
            # a rounding error: the backward's transients scale with tokens, and the single
            # allocation that decides whether a step fits is ~5.5 GB at 1,397 positions.
            #
            # Exact, not an approximation: the dropped columns are masked everywhere they
            # appear (`per_token_logps * completion_mask`, the entropy mask, the length
            # normaliser), and the advantage is per SEQUENCE. Off by default, because the
            # shapes change and a matmul of a different shape is not bit-identical.
            _keep = int(completion_mask.sum(dim=1).max().item())
            if 0 < _keep < completion_ids.size(1):
                completion_ids = completion_ids[:, :_keep]
                completion_mask = completion_mask[:, :_keep]
                for _k in ("old_per_token_logps", "ref_per_token_logps"):
                    if inputs.get(_k) is not None:
                        inputs[_k] = inputs[_k][:, :_keep]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # The sequence this micro-step actually forwards, AFTER the padding trim above --
        # the working set is linear in it, and it is not `completions/mean_length`, which
        # is a mean over the whole generation batch and misses the one rollout at the cap
        # that decides whether the card holds.
        self._mem_report("before the forward", tokens=input_ids.size(1))

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            mm_source=inputs,
        )

        self._mem_report("after the forward", tokens=input_ids.size(1))

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        if os.environ.get("SR1_EMPTY_CACHE_PER_MICROSTEP") == "1":
            # AND AGAIN HERE, which is the release that matters. `training_step` calls
            # `accelerator.backward(loss)` on the very next line, and the measurement
            # across this boundary is
            #
            #     before the forward   allocated 61.6   reserved 61.6
            #     after  the forward   allocated 61.9   reserved 67.8
            #
            # so the forward leaves 5.9 GB cached and owned by nothing -- and the
            # allocation the backward then fails on is 5.5 GB. Releasing it here is
            # handing the backward almost exactly the block it is about to ask for.
            #
            # Safe: `empty_cache` frees only blocks nothing references, and everything
            # autograd saved is referenced by the graph `loss` still holds.
            torch.cuda.empty_cache()
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _zero3_param_coordinator(self):
        """DeepSpeed ZeRO-3's parameter coordinator, or None if that is not what we run under."""
        for engine in (getattr(self, "deepspeed", None), getattr(self, "model_wrapped", None)):
            offload = getattr(getattr(engine, "optimizer", None), "parameter_offload", None)
            if offload is None:
                continue
            try:
                return offload.get_param_coordinator()
            except Exception:  # a DeepSpeed version whose accessor differs
                continue

        # Nothing found. Harmless when ZeRO-3 is not in use -- and a silent
        # no-op that lets the run die at the first eval when it is, so say so.
        if getattr(self, "is_deepspeed_enabled", False) and not self._zero3_lookup_warned:
            self._zero3_lookup_warned = True
            warnings.warn(
                "DeepSpeed is enabled but its ZeRO-3 parameter coordinator could not be "
                "located, so the module trace cannot be invalidated around evaluation. "
                "If this run is ZeRO-3, expect the first eval to fail with 'tracing error "
                "at step 0'; disable evaluation (--no-eval) or fix the lookup in "
                "_zero3_param_coordinator."
            )
        return None

    def _invalidate_zero3_trace(self, why):
        """Make ZeRO-3 re-record its module trace instead of replaying a stale one.

        ZeRO-3 records the exact order of module executions during one fwd+bwd and
        then prefetches parameters against it. Evaluation runs a *different* order --
        no backward, and `_generate_and_score_completions` fires its own saliency
        re-forwards -- so once training's trace is COMPLETE, the first eval forward
        dies in fetch_sub_module with "tracing error at step 0: expected ...
        layers.34.self_attn.q_proj ... but got ... embed_tokens". DeepSpeed's own
        trace_prologue only auto-invalidates on a *module* mismatch, and here the
        modules line up while the parameter queue does not, so it never fires.

        Invalidating on both sides of eval costs one re-record each way and leaves
        training replaying a trace that describes training.

        This must run on every rank: reset_step() calls assert_ints_same_as_other_ranks
        on the recorded order, so a coordinator invalidated on some ranks and not
        others trades this crash for a cross-rank disagreement and an NCCL timeout.
        `evaluate` is called on all ranks, which is why this lives here.
        """
        coordinator = self._zero3_param_coordinator()
        if coordinator is None or coordinator.is_invalid_trace():
            return
        try:
            coordinator._invalidate_trace()
        except Exception as exc:
            warnings.warn(f"could not invalidate the ZeRO-3 trace {why}: {exc}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Remember which eval dataset is being scored, so `log` can name its metrics.

        With a dict of eval datasets, `Trainer.evaluate` recurses once per dataset with
        `metric_key_prefix="eval_<name>"`, and each pass calls `log`. The reward metrics
        that GRPO accumulates in `self._metrics["eval"]` carry no dataset name of their
        own, so without this every dataset would write the same `eval_rewards/...` keys
        at the same step and only the last one would survive -- two validation sets
        would silently collapse into one curve.
        """
        previous, self._eval_metric_prefix = getattr(self, "_eval_metric_prefix", "eval"), metric_key_prefix
        self._invalidate_zero3_trace("before evaluating")
        try:
            return super().evaluate(
                eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
            )
        finally:
            self._eval_metric_prefix = previous
            self._invalidate_zero3_trace("after evaluating")

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_" (or "eval_<dataset>_" when several eval sets are used). Match that prefix so the
        # accumulated reward metrics land beside the losses of the dataset they were computed on.
        if mode == "eval":
            prefix = getattr(self, "_eval_metric_prefix", "eval")
            metrics = {f"{prefix}_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["image"]:
                    table["image"] = []
                    for img in self._logs["image"]:
                        if img is not None:
                            # Convert images to wandb Image objects for proper visualization
                            table["image"].append(wandb.Image(img))
                        else:
                            table["image"].append(None)

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

# Alias so this module can be imported as GRPOTrainerQwen3 alongside the original GRPOTrainer.
GRPOTrainerQwen3 = GRPOTrainer
