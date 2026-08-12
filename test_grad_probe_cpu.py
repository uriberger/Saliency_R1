#!/usr/bin/env python
"""The probe's two saliency paths must forward exactly the same tensors.

`overlap_probe.py` bridges `generate()`'s completion representation into two map
functions -- `capture_layer_attention` (attention) and `grad_step_maps` (gradient).
They were written independently against the same convention and only the attention
one had ever been run, so the gradient adapter assumed `comp_ids` was a tensor and
dropped `mm_token_type_ids`. `test_grad_maps_cpu.py` missed it because it drives
`trl/grad_maps.step_grad_maps` directly with a toy that already takes tensors: it
covers the gradient math, not either caller's calling convention.

These tests cover that seam. They exercise `teacher_forced_case` with the types
`generate()` actually returns, and assert both adapters still route through it.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
torch.set_num_threads(1)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OP = _load("_t_overlap_probe", "overlap_probe.py")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# generate() returns `row.tolist()`, so a completion is a plain list[int].
# ---------------------------------------------------------------------------
PROMPT_LEN, N_COMP, N_PATCH, D_PIX = 9, 6, 24, 24


def prompt_inputs(mm: bool = True, image: bool = True):
    ids = torch.arange(100, 100 + PROMPT_LEN)[None]
    out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if image:
        out["pixel_values"] = torch.randn(N_PATCH, D_PIX)
        out["image_grid_thw"] = torch.tensor([[1, 4, 6]])
    if mm:
        # 1 marks image tokens in the prompt; the completion is all text.
        mt = torch.zeros(1, PROMPT_LEN, dtype=torch.long)
        mt[0, 2:6] = 1
        out["mm_token_type_ids"] = mt
    return out


COMP_IDS = list(range(200, 200 + N_COMP))       # list[int], exactly as generate() gives it


def test_accepts_generate_output_types():
    """The reported crash: `'list' object has no attribute 'to'`."""
    pi = prompt_inputs()
    case = OP.teacher_forced_case(pi, COMP_IDS, "cpu")
    ids = case["input_ids"]
    check("a list[int] completion is accepted, as generate() returns it",
          isinstance(ids, torch.Tensor), f"got {type(ids).__name__}")
    check("ids are [1, prompt + completion]",
          tuple(ids.shape) == (1, PROMPT_LEN + N_COMP), f"{tuple(ids.shape)}")
    check("ids are the prompt followed by the completion, in order",
          ids[0, :PROMPT_LEN].tolist() == pi["input_ids"][0].tolist()
          and ids[0, PROMPT_LEN:].tolist() == COMP_IDS)
    check("attention_mask matches ids and is all ones",
          case["attention_mask"].shape == ids.shape and bool((case["attention_mask"] == 1).all()))


def test_mm_token_type_ids_extended_over_the_completion():
    """The second bug: a dropped key makes the forward differ from the trained one."""
    pi = prompt_inputs()
    case = OP.teacher_forced_case(pi, COMP_IDS, "cpu")
    check("mm_token_type_ids is carried through", "mm_token_type_ids" in case)
    mt = case["mm_token_type_ids"]
    check("mm_token_type_ids spans prompt + completion",
          tuple(mt.shape) == (1, PROMPT_LEN + N_COMP), f"{tuple(mt.shape)}")
    check("the prompt part is unchanged",
          mt[0, :PROMPT_LEN].tolist() == pi["mm_token_type_ids"][0].tolist())
    check("the completion part is zeros -- text, not image",
          bool((mt[0, PROMPT_LEN:] == 0).all()))
    check("dtype stays long, as the model's embedding lookup needs",
          mt.dtype == torch.long, str(mt.dtype))


def test_optional_keys_are_omitted_not_faked():
    case = OP.teacher_forced_case(prompt_inputs(mm=False), COMP_IDS, "cpu")
    check("no mm_token_type_ids in, none out", "mm_token_type_ids" not in case)
    case = OP.teacher_forced_case(prompt_inputs(image=False, mm=False), COMP_IDS, "cpu")
    check("a text-only prompt forwards no image entries",
          "pixel_values" not in case and "image_grid_thw" not in case)


def test_image_entries_are_passed_through_untouched():
    pi = prompt_inputs()
    case = OP.teacher_forced_case(pi, COMP_IDS, "cpu")
    check("pixel_values is the processor's tensor, not a copy or a reshape",
          case["pixel_values"] is pi["pixel_values"])
    check("image_grid_thw is passed through",
          case["image_grid_thw"] is pi["image_grid_thw"])


def test_both_adapters_route_through_the_helper():
    """The drift guard: neither adapter may rebuild the forward on its own again."""
    for fn in (OP.grad_step_maps, OP.capture_layer_attention):
        src = inspect.getsource(fn)
        check(f"{fn.__name__} builds its forward with teacher_forced_case",
              "teacher_forced_case(" in src)
        check(f"{fn.__name__} does not re-derive the sequence itself",
              "torch.tensor([comp_ids]" not in src)


# ---------------------------------------------------------------------------
def main():
    for t in (
        test_accepts_generate_output_types,
        test_mm_token_type_ids_extended_over_the_completion,
        test_optional_keys_are_omitted_not_faked,
        test_image_entries_are_passed_through_untouched,
        test_both_adapters_route_through_the_helper,
    ):
        print(f"\n{t.__name__}")
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
