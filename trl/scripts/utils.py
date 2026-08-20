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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    val_sets_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory holding the held-out validation sets written by "
            "`build_grpo_sets.py --build-val` (val_natural/ and val_nonnatural/). Whichever "
            "of the two exist are scored separately, so natural and non-natural imagery get "
            "their own curves. Their images are disjoint from set_a and set_b, so this does "
            "not touch the training corpus. Scored on answer accuracy only, by one greedy "
            "completion per prompt through vLLM -- not by the Trainer's evaluation loop, "
            "which would re-run the whole reward pipeline."
        },
    )
    val_eval_steps: int = field(
        default=100,
        metadata={
            "help": "How often (in optimizer steps) to score the validation sets. A step-0 "
            "baseline is always taken. 0 disables periodic scoring, leaving only step 0."
        },
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode."},
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )
    reforward_saliency: bool = field(
        default=True,
        metadata={
            "help": "Whether to compute saliency via a separate re-forward pass instead of capturing attention "
            "weights during generate(). Required when max_completion_length > 1024 to avoid OOM."
        },
    )
    # ---- attention-overlap reward (reward_variant="ours"; see grpo-reward-port-plan) ----
    saliency_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "WHICH SALIENCY MAP the per-observe-step reward scores. This is the "
            "flag to use; --reward_variant is the older spelling and is kept only so "
            "existing command lines keep working. "
            "'attention' = raw observe->patch attention at --overlap_layer, mean of "
            "--overlap_heads (docs/saliency-maps.md map 1); 'grad' = the PIXEL GRADIENT "
            "of the step's own tokens (map 5b); 'glimpse' = GLIMPSE's gradient-weighted "
            "attention (map 6, ~55x the cost of grad -- see trl/glimpse_maps.py). "
            "All three produce the same per-step map contract and are scored by "
            "--overlap_metric, so the map and the metric are now independent choices. "
            "Maps to --reward_variant ours|grad|glimpse; set both and this one wins. "
            "Leave unset (with --reward_variant saliency_r1|none) for the two variants "
            "that are not per-observe-step rewards at all.",
            "choices": ["attention", "grad", "glimpse"],
        },
    )
    reward_variant: str = field(
        default="saliency_r1",
        metadata={
            "help": "Which reward + attention-extraction mode to use. 'saliency_r1' = the paper's "
            "whole-completion rollout saliency reward. 'ours' = raw per-head observe->patch "
            "attention overlap reward (layer 22, 2-head mean, per observe step, DINO-grounded). "
            "'none' = accuracy + judge + format only (no saliency/overlap reward; skips the "
            "attention re-forward pass entirely). 'grad' = the roll-null gradient reward: "
            "same per-observe-step, DINO-grounded shape as 'ours', but the map is the "
            "PIXEL GRADIENT of the step's own tokens and the score is "
            "log(||g_U|| / ||g_rolled||) -- see trl/rewards/grad_rewards.py. "
            "'glimpse' = the GLIMPSE grounding reward: same per-observe-step, "
            "DINO-grounded shape, but the map is GLIMPSE's gradient-weighted attention "
            "(docs/saliency-maps.md map 6) and the score is --overlap_metric. It costs "
            "55-59x the 'grad' variant per case (100-145 s added to a ~40 s optimizer "
            "step) and its screened correlation with correctness is null to slightly "
            "negative -- read trl/rewards/glimpse_rewards.py before using it.",
            "choices": ["saliency_r1", "ours", "none", "grad", "glimpse"],
        },
    )
    # ---- roll-null gradient reward (reward_variant="grad") ----
    grad_target: str = field(
        default="clogit",
        metadata={
            "help": "reward_variant='grad': the scalar differentiated per generated token. "
            "'clogit' (default) = the raw logit minus the vocabulary mean: it does not "
            "saturate as the model grows confident (d log P/dz -> 0 as p -> 1, which would "
            "make the reward pay for uncertain steps) and it drops the common-mode channel "
            "shared by every vocabulary item, which is otherwise the SAME map for every "
            "step and so a one-shot lift for all of them. 'logit' keeps that channel; "
            "'logprob' saturates. Both are for probes.",
            "choices": ["clogit", "logit", "logprob"],
        },
    )
    grad_null_offsets: int = field(
        default=16,
        metadata={
            "help": "reward_variant='grad': how many translated copies of the box union "
            "form the null. Their SQUARED norms are pooled before the log, so one control "
            "landing on a dead region cannot dominate. Pure numpy on a ~16x16 map: free."
        },
    )
    grad_logratio_clip: float = field(
        default=1.0,
        metadata={
            "help": "reward_variant='grad': clip |log(||g_U||/||g_null||)| to this. A ratio "
            "has a heavy tail, and with scale_rewards=True one outlier completion takes "
            "most of its group's normalised advantage. Set it from the measured spread "
            "(overlap_metric_spread.py) rather than trusting the default."
        },
    )
    grad_inframe_rolls: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='grad': draw the control placements so the translated "
            "union stays inside the grid, instead of wrapping toroidally across the image "
            "border. Falls back to toroidal (counted in grad/toroidal_frac) when a "
            "near-full-frame union leaves too few in-frame positions."
        },
    )
    grad_dedupe_steps: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='grad': drop repeated observe-step texts before the "
            "mean over steps. The score is a mean, so re-quoting one easily-grounded "
            "sentence pulls it up and dilutes the hard perception steps -- measured going "
            "0.00 -> 0.19 in the wov0.4 run. grad/dup_frac is logged either way."
        },
    )
    grad_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='grad': score only rows whose 'natural' column is True. "
            "Same rationale as --overlap_natural_only: Grounding-DINO is a photograph "
            "detector, so on charts/documents the box union -- and the whole score -- is "
            "noise."
        },
    )
    grad_seed: int = field(
        default=0,
        metadata={"help": "reward_variant='grad': seed for the control placements."},
    )
    # ---- GLIMPSE grounding reward (reward_variant="glimpse") ----
    glimpse_target: str = field(
        default="clogit",
        metadata={
            "help": "reward_variant='glimpse': the scalar differentiated per generated "
            "token to get dz/dA. Means exactly what --grad_target means, and for the same "
            "reasons: 'clogit' does not saturate as the model grows confident and drops "
            "the common-mode channel shared by every vocabulary item.",
            "choices": ["clogit", "logit", "logprob"],
        },
    )
    glimpse_layer_frac: float = field(
        default=1.0,
        metadata={
            "help": "reward_variant='glimpse': fraction of the decoder stack propagated, "
            "taken off the TOP. THE FIRST COST DIAL: 0.6 measured 1.64x cheaper (34-36x "
            "the gradient reward instead of 55-59x) and the paper's own ablation loses "
            "nothing there -- but it is a METHOD change, not a memory one, and the map "
            "that was screened is 1.0."
        },
    )
    glimpse_token_cap: int = field(
        default=0,
        metadata={
            "help": "reward_variant='glimpse': score at most this many target tokens per "
            "observe step, drawn uniformly at random without replacement (never the first "
            "k -- these maps carry a reading-order prior). THE SECOND COST DIAL, and the "
            "strongest one: cost is exactly linear in it, so a cap of 6 against a median "
            "~20 tokens per step is ~3.5x. Eq 18 renormalises beta inside the step, so a "
            "random subset estimates the same weighted mean -- but what the cap does to "
            "the SCORE has not been measured at scale, which is why 0 (every token) is "
            "the default."
        },
    )
    glimpse_temp: float = field(
        default=0.5,
        metadata={
            "help": "reward_variant='glimpse': lambda in eq 6, the head-fusion softmax "
            "temperature. The paper's value."
        },
    )
    glimpse_depth_temp: float = field(
        default=0.2,
        metadata={
            "help": "reward_variant='glimpse': lambda_d in eq 9, the exponential depth "
            "prior. 0.2 is the paper's text; it was tuned on a 64-layer backbone where it "
            "spans 7.8% of the depth, and 0.36 is what matches that SHAPE on this "
            "36-layer model. The ablation calls this the single most important component."
        },
    )
    glimpse_token_weight: str = field(
        default="full",
        metadata={
            "help": "reward_variant='glimpse': eq 18's beta_t. 'full' = p_t x prompt "
            "alignment (the paper); the other three reproduce its token-saliency "
            "ablation. Eq 17 crosses the modalities on purpose -- a token earns its say "
            "in where the model looked by being about the QUESTION, since weighting it by "
            "its own visual alignment would be circular.",
            "choices": ["full", "confidence", "prompt", "uniform"],
        },
    )
    glimpse_dedupe_steps: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='glimpse': drop repeated observe-step texts before "
            "the mean over steps. Same hack and same rationale as --grad_dedupe_steps; "
            "glimpse/dup_frac is logged either way."
        },
    )
    glimpse_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='glimpse': score only rows whose 'natural' column is "
            "True. Same rationale as --overlap_natural_only: Grounding-DINO is a "
            "photograph detector, so on charts/documents the box union -- and the whole "
            "score -- is noise."
        },
    )
    glimpse_seed: int = field(
        default=0,
        metadata={
            "help": "reward_variant='glimpse': seed for the --glimpse_token_cap draw. "
            "Irrelevant when the cap is 0."
        },
    )
    # ---- the roll-null, when it is used as a METRIC (--overlap_metric logratio /
    # metric). reward_variant='grad' keeps its own --grad_* copies of
    # these, because there the roll-null IS the reward rather than one metric of four,
    # and separating them keeps every existing grad run reproducible byte for byte.
    rollnull_offsets: int = field(
        default=16,
        metadata={
            "help": "--overlap_metric logratio: how many translated copies of the box union form "
            "the null. Their SQUARED masses are pooled BEFORE the log, so one control "
            "landing on a dead region cannot dominate. Pure numpy on a ~16x16 map: free."
        },
    )
    rollnull_clip: float = field(
        default=1.0,
        metadata={
            "help": "--overlap_metric logratio: clip |log(N(U)/N_0)| to this. A ratio has a heavy "
            "tail, and with scale_rewards=True one outlier completion takes most of its "
            "group's normalised advantage. 1.0 == a ratio of e."
        },
    )
    rollnull_inframe: bool = field(
        default=True,
        metadata={
            "help": "--overlap_metric logratio: draw the control placements so the translated "
            "union stays INSIDE the grid rather than wrapping toroidally across the image "
            "border. Falls back to toroidal when a near-full-frame union leaves too few "
            "in-frame positions -- watch <variant>/toroidal_frac, because that fallback "
            "changes what the null means."
        },
    )
    rollnull_seed: int = field(
        default=0,
        metadata={"help": "--overlap_metric logratio: seed for the control placements."},
    )
    token_reduction: str = field(
        default="mean",
        metadata={
            "help": "reward_variant='ours': reduce per-token saliency maps within an observe step "
            "(mean|max). Sweep dimension — appears in the model/wandb name as trmean/trmax.",
            "choices": ["mean", "max", "min"],
        },
    )
    overlap_layer: int = field(
        default=22,
        metadata={"help": "reward_variant='ours': transformer layer to read raw attention from."},
    )
    overlap_heads: str = field(
        default="28,31",
        metadata={
            "help": "reward_variant='ours': comma-separated head indices at overlap_layer to mean "
            "together (default the fixed 2-head (22,28)+(22,31) option)."
        },
    )
    overlap_metric: Optional[str] = field(
        default=None,
        metadata={
            "help": "How to score a step's map against its DINO box union, for EVERY "
            "--saliency_method (attention, grad and glimpse alike). UNSET means the "
            "historical default OF THAT MAP -- mean_in for attention, logratio for grad, "
            "mean_in_v2 for glimpse -- so an existing command line keeps its behaviour "
            "now that one flag serves all three. "
            "'mean_in' (default, the incumbent) = mean of the MAX-normalized map inside the "
            "box; it divides by the map's own peak, so a map that merely FLATTENS scores "
            "higher (measured: 32x more movement under flattening than under real "
            "grounding). 'mean_in_v2' = the same mean over the box divided by the mean over "
            "the whole map instead of by its peak: chance is 1.0, rescale-invariant, and "
            "unlike auroc it still sees magnitudes. Measured on the cold-start policy it runs "
            "median 0.74 / p99 1.36 with 12x mean_in's per-sample spread, so its w_overlap is "
            "0.033 (= mean_in's wov0.4 pressure), applied by the launchers. 'auroc' = "
            "P(in-box patch outranks out-box patch), which depends "
            "only on patch order and is therefore exactly invariant to that flattening, and "
            "predicts correctness more stably (mean |r| 0.238 vs 0.181, sd 0.028 vs 0.089). "
            "'logratio' = the ROLL-NULL: log of the map's mass inside the union over its "
            "mass in the SAME union translated to random in-frame offsets. Chance is "
            "exactly 0 and the control has the union's own shape and area by "
            "construction, so this is the only one of the four that closes the box-size "
            "confound rather than merely bounding it -- at the cost of being random "
            "(it draws control placements) and of squaring the mass, which weights peaks "
            "more than a plain sum. Its knobs are --rollnull_*. "
            "Sweep dimension — appears in the model/wandb name.",
            "choices": ["mean_in", "mean_in_v2", "auroc", "logratio"],
        },
    )
    mass_floor_tau: Optional[float] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': if set, multiply each step's score by "
            "min(1, image_mass/tau), where image_mass is the fraction of the attention row "
            "spent on image tokens. Closes the one hole a rank-based metric cannot see — a "
            "model withdrawing attention from the image while keeping a good ranking. Also "
            "raises the correctness correlation (0.227 -> 0.238) because image_mass is "
            "itself predictive. Recommended 0.0022 = the 10th percentile of the reference "
            "model's image_mass. Keep near p10: much above p25 it stops being a floor and "
            "'raise image attention uniformly' becomes its own exploitable direction. "
            "Sweep dimension — appears in the model/wandb name.",
        },
    )
    box_threshold: float = field(
        default=0.10,
        metadata={"help": "reward_variant='ours': Grounding-DINO confidence threshold for per-step boxes."},
    )
    max_box_area: float = field(
        default=0.5,
        metadata={
            "help": "reward_variant='ours': drop INDIVIDUAL DINO boxes whose area fraction exceeds this "
            "cap. Set to 0 to disable the per-box cap entirely (keep every box above --box_threshold). "
            "This bounds no. of pixels per box, not the union — see --max_union_area."
        },
    )
    max_union_area: Optional[float] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': skip (do not score) any observe step whose rasterised box "
            "UNION covers more than this fraction of the image, e.g. 0.4. The step is masked exactly "
            "like an ungroundable one — SKIPPED, not scored 0 — so it drops out of the per-completion "
            "mean. None/0 (default) disables the cap, leaving only the existing 100%-coverage "
            "degenerate guard. Needed because --max_box_area is per-box: N disjoint boxes each under "
            "the cap can still cover the whole image. Sweep dimension — appears in the model/wandb name."
        },
    )
    dino_api_base: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': base URL of a served batched Grounding-DINO endpoint. "
            "If unset, DINO runs locally on each training process's device."
        },
    )
    placebo: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': REPLACE the attention-overlap reward with a "
            "control that has its within-group spread but not its grounding, to find out "
            "whether the reward's DIRECTION matters at all. 'roll' = the configured "
            "--overlap_metric on the same map, scored against the step's own box union "
            "MOVED to a deterministic wrong place (same area, same shape); 'random' = a "
            "stable hash of the completion text -> U(0,1), pure variance with no "
            "direction; 'length' = -n_completion_tokens/1000, i.e. the brevity reward the "
            "overlap term is suspected of being in disguise (within a group it correlates "
            "-0.04 to -0.11 with completion length and ~0.00 with accuracy). Each takes "
            "the overlap reward's slot in reward_funcs, so --reward_weights is unchanged, "
            "and each must be given the weight that matches mean_in w0.4's WITHIN-GROUP "
            "sd -- the launcher resolves it. Every placebo returns unscored on exactly "
            "the completions the real reward would leave unscored (it runs the same "
            "segmentation, the same Grounding-DINO call and the same metric, and uses the "
            "real score only as a gate), so the comparison has one variable and not two. "
            "Not available with --overlap_metric logratio. See "
            "docs/next-reward-experiments.md and trl/rewards/placebo_rewards.py.",
            "choices": ["roll", "random", "length"],
        },
    )
    overlap_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='ours': apply the overlap reward ONLY to rows whose "
            "'natural' column is True; non-natural rows (charts, documents, diagrams) are "
            "scored by format + accuracy + judge alone. Grounding-DINO is trained on "
            "photographs, so its boxes -- and hence the overlap score -- are noise on "
            "non-natural imagery. Requires a boolean 'natural' column (build_grpo_sets.py "
            "emits one). Off by default, so mixed-corpus runs stay reproducible. Sweep "
            "dimension — appears in the model/wandb name as natonly."
        },
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]` or `None`, *optional*, defaults to `None`):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser


    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"


    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Optional[Iterable[str]] = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"
