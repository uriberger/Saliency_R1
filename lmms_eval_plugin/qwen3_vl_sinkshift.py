"""Qwen3-VL with the Saliency-R1 attention edit applied while it answers.

`qwen3_vl` with one thing added: before evaluation starts, `sink_shift.install()` from
the saliency_r1 repo swaps the text decoder's attention implementation for one that moves
attention weight off the image border and into the middle of the picture. Nothing else
differs -- prompts, generation kwargs, decoding and scoring are inherited unchanged, so a
score from this model sits on the same axis as a score from `qwen3_vl`.

    bash scripts/slurm/launch_lmms_eval_job.sh --model CKPT --model-type qwen3_vl_sinkshift ...

configured entirely through the environment, because the launcher builds `--model_args`
itself and has no passthrough:

    SINK_SHIFT_ALPHA    0 to 1. DEFAULT 0, which is exactly the identity -- selecting
                        this model type without setting it evaluates the stock model.
    SINK_SHIFT_ARM      centre | core | outward | flat | reverse | text   (default centre)
    SINK_SHIFT_LAYERS   "22", "0-35", or "all"                            (default 22)
    SINK_SHIFT_HEADS    "28,31" or "all"                                  (default 28,31)
    SINK_SHIFT_ROWS     after_image | generated | all                     (default after_image)
    SINK_SHIFT_RECT_FRAC                                                  (default 0.565)
    SINK_SHIFT_REPO     where sink_shift.py lives

Every one can also be passed as a model_arg (`ss_alpha=0.5`) where a caller has a way to
set them; the model_arg wins.

WHY THIS IS A SEPARATE FILE. lmms-eval imports only the model actually requested, so a
job running `--model qwen3_vl` never opens this one. Registering it costs a single added
key in `lmms_eval/models/__init__.py`, which cannot change what any other key resolves to.
That is the whole reason it is not a flag on `qwen3_vl`: no concurrent evaluation should
be able to notice that this exists.

BATCH SIZE 1 ONLY. The edit locates the picture from the prompt's own token ids; left
padding in a wider batch moves every image column, so a batch is refused rather than
silently edited in the wrong place. The Saliency-R1 bench harness already runs at 1.
"""

import os
import sys

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL

DEFAULT_REPO = "/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _parse_layers(spec):
    """"all" -> None (every layer); "22" or "0-21,30" -> a list."""
    if spec is None or str(spec).strip().lower() in ("all", "none", ""):
        return None
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_heads(spec):
    if spec is None or str(spec).strip().lower() in ("all", "none", ""):
        return None
    return [int(x) for x in str(spec).split(",") if x.strip()]


@register_model("qwen3_vl_sinkshift")
class Qwen3_VL_SinkShift(Qwen3_VL):
    is_simple = False

    def __init__(self, ss_arm=None, ss_alpha=None, ss_layers=None, ss_heads=None,
                 ss_rows=None, ss_rect_frac=None, ss_repo=None, **kwargs):
        super().__init__(**kwargs)

        arm = ss_arm or _env("SINK_SHIFT_ARM", "centre")
        alpha = float(ss_alpha if ss_alpha is not None else _env("SINK_SHIFT_ALPHA", 0.0))
        layers = _parse_layers(ss_layers if ss_layers is not None
                               else _env("SINK_SHIFT_LAYERS", "22"))
        heads = _parse_heads(ss_heads if ss_heads is not None
                             else _env("SINK_SHIFT_HEADS", "28,31"))
        rows = ss_rows or _env("SINK_SHIFT_ROWS", "after_image")
        frac = float(ss_rect_frac if ss_rect_frac is not None
                     else _env("SINK_SHIFT_RECT_FRAC", 0.565))
        repo = ss_repo or _env("SINK_SHIFT_REPO", DEFAULT_REPO)

        self.sink_shift = None
        if alpha <= 0:
            # The identity. Say so loudly: a whole benchmark run of the stock model under
            # an arm's name is the single most expensive mistake available here.
            eval_logger.warning(
                "qwen3_vl_sinkshift: SINK_SHIFT_ALPHA is 0, so NO EDIT IS APPLIED and "
                "this run is the stock model. Set SINK_SHIFT_ALPHA to enable it.")
            return

        if int(self.batch_size) != 1:
            raise ValueError(
                f"qwen3_vl_sinkshift needs batch_size=1, got {self.batch_size}: the edit "
                "locates the picture from the prompt's token ids, and left padding in a "
                "wider batch moves every image column.")

        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import sink_shift
        except ImportError as exc:
            raise ImportError(
                f"cannot import sink_shift from {repo}; set SINK_SHIFT_REPO") from exc

        self.sink_shift = sink_shift.install(
            self._model, arm=arm, alpha=alpha, layers=layers, heads=heads,
            rows=rows, rect_frac=frac)
        eval_logger.warning(
            f"qwen3_vl_sinkshift ACTIVE: arm={arm} alpha={alpha} "
            f"layers={'all' if layers is None else layers} "
            f"heads={'all' if heads is None else heads} rows={rows} rect_frac={frac}")

    def generate_until(self, requests):
        out = super().generate_until(requests)
        if self.sink_shift is not None:
            d = self.sink_shift.diagnostics()
            # The landing check, in the eval log. A run whose border share did not fall
            # to (1-alpha) of itself did not apply the edit, and its score is a score of
            # the stock model however the directory is named.
            eval_logger.warning(
                f"qwen3_vl_sinkshift landed: border share "
                f"{d['frame_share_before']:.4f} -> {d['frame_share_after']:.4f} "
                f"(want {(1 - self.sink_shift.alpha) * d['frame_share_before']:.4f}), "
                f"picture holds {d['image_mass']:.5f} of a row, this edit moved "
                f"{d['row_mass_moved']:.5f} of it, over {d['rows_edited']} rows on "
                f"layers {d['layers_touched']}")
        return out
