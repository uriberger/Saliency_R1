#!/usr/bin/env python
"""CPU half of P2's three assertions -- no GPU, no job, no checkpoint weights.

    conda activate deepeyes; and python test_deepeyes_rope_cpu.py

The GPU smoke run (launch_deepeyes_smoke_job.sh) is the real article, but everything
here is decidable from the processor and the tokenizer alone, and finding it on a login
node in 40 seconds beats finding it 20 minutes into an 8-GPU allocation. What CPU
cannot answer is whether FSDP, vLLM and the agent loop hold together; that is the job.

Nine checks, in the order they would bite:

  1  the Qwen2VLImageProcessor gate is TRUE on Qwen3-VL         <- why the bug is quiet
  2  agent_loop.py now selects qwen3_vl.get_rope_index
  3  rl_dataset.py already did (the pattern edit [1] copies)
  4  qwen3_vl and qwen2_vl get_rope_index AGREE on still images <- the fix is a no-op...
  5  ...and DISAGREE as soon as a video grid appears            <- ...but only for stills
  6  position_ids from a real tool turn are (4, seq)
  7  the tripwire raises when a Qwen3VLProcessor hits qwen2_vl
  8  the forward assertion rejects 2-D position_ids
  9  response_mask zeros cover the tool turn's whole
     <|vision_start|><|image_pad|>...<|vision_end|> span

Check 9 reproduces `tool_agent_loop._handle_processing_tools_state`'s token accounting
rather than importing it, because importing it drags in ray and a server manager. The
arithmetic it reproduces is three lines long and is quoted at the call site.
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# THE STOCK INSTRUCT MODEL, not our cold start, and that is the point: P2 asks whether
# the STACK runs, so every local modification removed from the question is one fewer
# thing a failure could be about.
#
# It is also the only one that loads here today. The cold start
# checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged was written by
# transformers 5.7.0 and serialises `extra_special_tokens` as a LIST; 4.57.6 -- which is
# what the `deepeyes` env pins, because it is what ships Qwen3-VL -- calls `.keys()` on
# it and dies with `AttributeError: 'list' object has no attribute 'keys'`.
# fix_coldstart_tokenizer.py exists for exactly this and is idempotent, but it rewrites
# a checkpoint SHARED with every other session, so P9 should run it deliberately rather
# than have P2 do it as a side effect. See the report.
MODEL = os.environ.get("DEEPEYES_SMOKE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

_results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw) or ""
                _results.append((name, True, str(detail)))
                print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
                return True
            except Exception as e:
                _results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}  -- {type(e).__name__}: {e}")
                traceback.print_exc()
                return False

        return run

    return deco


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    print(f"model: {MODEL}")
    processor = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    print(f"processor      : {type(processor).__name__}")
    print(f"image_processor: {type(processor.image_processor).__name__}")
    print()

    # ---------------------------------------------------------------- 1
    @check("the Qwen2VLImageProcessor gate is TRUE on Qwen3-VL")
    def c1():
        name = type(processor.image_processor).__name__
        assert "Qwen2VLImageProcessor" in name, (
            f"image processor is {name}; if Qwen3-VL ever stops reusing "
            "Qwen2VLImageProcessor then every one of verl's gates flips to False and "
            "position ids fall through to 1-D -- which is DeepEyes' parallel_env.py:282 "
            "bug exactly. Re-read the branch before changing this assertion."
        )
        assert "Qwen3VLProcessor" in type(processor).__name__
        return f"{name} / {type(processor).__name__}"

    # ---------------------------------------------------------------- 2,3
    @check("agent_loop.py selects qwen3_vl.get_rope_index")
    def c2():
        import verl.experimental.agent_loop.agent_loop as al

        src = open(al.__file__).read()
        i = src.find('"Qwen2VLImageProcessor" in self.processor.image_processor')
        assert i > 0, "the gate moved; re-read _postprocess"
        window = src[i : i + 700]
        assert "Qwen3VLProcessor" in window, (
            "agent_loop.py still imports qwen2_vl.get_rope_index unconditionally. "
            "Run: bash patch_deepeyes_fork.sh"
        )
        assert "from verl.models.transformers.qwen3_vl import get_rope_index" in window
        return os.path.dirname(al.__file__)

    @check("rl_dataset.py already branched (the pattern the fix copies)")
    def c3():
        import verl.utils.dataset.rl_dataset as rd

        src = open(rd.__file__).read()
        assert "from verl.models.transformers.qwen3_vl import get_rope_index" in src

    # ----------------------------------------------- build a real tool turn
    # Two images: the original figure, and a crop of it -- which is what one
    # ImageZoomInTool call produces. Sizes are deliberately different so a grid mix-up
    # cannot hide behind equal shapes.
    img0 = Image.new("RGB", (476, 364), (90, 120, 200))
    img1 = img0.crop((100, 60, 300, 230))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Find the labelled angle. Zoom in first."},
            ],
        },
        {
            "role": "assistant",
            "content": "I will zoom in.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "image_zoom_in_tool",
                        "arguments": {"bbox_2d": [100, 60, 300, 230], "label": "angle"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Zoomed in on the image to the region [100, 60, 300, 230]."},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(text=[text], images=[img0, img1], return_tensors="pt")
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]
    image_grid_thw = model_inputs["image_grid_thw"]
    print(f"  tool-turn sequence: {input_ids.numel()} tokens, image_grid_thw={image_grid_thw.tolist()}")

    from verl.models.transformers.qwen2_vl import get_rope_index as rope2
    from verl.models.transformers.qwen3_vl import get_rope_index as rope3

    # ---------------------------------------------------------------- 4
    @check("qwen3_vl and qwen2_vl get_rope_index agree on still images")
    def c4():
        os.environ["DEEPEYES_ASSERT_MROPE"] = "0"  # we are calling the wrong one on purpose
        try:
            a = rope3(processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
            b = rope2(processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        finally:
            os.environ["DEEPEYES_ASSERT_MROPE"] = "1"
        assert a.shape == b.shape == (3, input_ids.numel()), (a.shape, b.shape)
        assert torch.equal(a, b), (
            "they disagree on a still-image tool turn. That makes the agent_loop fix a "
            "LIVE bug rather than a latent one, and every multi-turn rollout taken "
            "before it was wrong. Report this before running anything."
        )
        return f"identical over {a.shape[1]} positions, max={int(a.max())}"

    # ---------------------------------------------------------------- 5
    @check("they DISAGREE once a video grid appears (so the fix is not cosmetic)")
    def c5():
        # Smallest sequence that exercises the branch: one 2-frame video.
        vt = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vs = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        grid = torch.tensor([[2, 4, 4]])  # t=2, 2x2 merged patches per frame
        n = 2 * (4 // 2) * (4 // 2)
        ids = torch.tensor([vs] + [vt] * n + [ve] + [9] * 5)
        am = torch.ones_like(ids)
        os.environ["DEEPEYES_ASSERT_MROPE"] = "0"
        try:
            a = rope3(processor, input_ids=ids, video_grid_thw=grid.clone(), attention_mask=am)
            b = rope2(processor, input_ids=ids, video_grid_thw=grid.clone(), attention_mask=am)
        finally:
            os.environ["DEEPEYES_ASSERT_MROPE"] = "1"
        assert not torch.equal(a, b), (
            "no divergence on video either -- then the two implementations may have "
            "converged upstream and this check should be deleted rather than silenced."
        )
        return f"t-row max {int(a[0].max())} (qwen3) vs {int(b[0].max())} (qwen2)"

    # ---------------------------------------------------------------- 6
    @check("a real tool turn yields (4, seq) position_ids")
    def c6():
        vision = rope3(processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        valid = attention_mask.bool()
        textrow = torch.ones((1, input_ids.numel()), dtype=torch.long)
        textrow[0, valid] = torch.arange(int(valid.sum()))
        pos = torch.cat((textrow, vision), dim=0)
        assert pos.shape == (4, input_ids.numel()), pos.shape
        # The text row must be a plain 0..n-1 ramp and the vision rows must NOT be --
        # if all four rows were identical we would be looking at the 1-D fall-through
        # dressed up as 4 rows.
        assert torch.equal(pos[0], torch.arange(input_ids.numel()))
        assert not torch.equal(pos[1], pos[2]), "t and h rows identical; MRoPE did not fire"
        return f"shape {tuple(pos.shape)}, text max {int(pos[0].max())}, vision max {int(pos[1:].max())}"

    # ---------------------------------------------------------------- 7
    @check("the tripwire raises when Qwen3VLProcessor hits qwen2_vl")
    def c7():
        os.environ["DEEPEYES_ASSERT_MROPE"] = "1"
        try:
            rope2(processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        except RuntimeError as e:
            assert "DEEPEYES-MROPE" in str(e), e
            return "raised as designed"
        raise AssertionError(
            "qwen2_vl.get_rope_index accepted a Qwen3VLProcessor without complaint. "
            "Run: bash patch_deepeyes_fork.sh"
        )

    # ---------------------------------------------------------------- 8
    @check("the forward assertion rejects 2-D position_ids")
    def c8():
        from verl.utils.deepeyes_mrope_guard import assert_mrope_position_ids

        assert_mrope_position_ids(torch.zeros(4, 1, 32, dtype=torch.long), "selftest")  # must not raise
        for bad, why in [
            (torch.zeros(2, 32, dtype=torch.long), "1-D LM ids"),
            (torch.zeros(3, 1, 32, dtype=torch.long), "3 rows, text row missing"),
            (None, "None"),
        ]:
            try:
                assert_mrope_position_ids(bad, "selftest")
            except RuntimeError:
                continue
            raise AssertionError(f"accepted {why}")
        return "accepts (4,*,*); rejects 2-D, 3-row and None"

    # ---------------------------------------------------------------- 9
    @check("response_mask zeros cover the tool turn's vision span")
    def c9():
        tok = processor.tokenizer
        vs = tok.convert_tokens_to_ids("<|vision_start|>")
        ve = tok.convert_tokens_to_ids("<|vision_end|>")
        pad = tok.convert_tokens_to_ids("<|image_pad|>")

        # tool_agent_loop._handle_processing_tools_state, lines quoted:
        #     model_inputs = self.processor(text=[raw_tool_response], images=current_images, ...)
        #     response_ids  = model_inputs.pop("input_ids").squeeze(0).tolist()
        #     agent_data.prompt_ids   += response_ids
        #     agent_data.response_mask += [0] * len(response_ids)
        # The tool turn is rendered on its OWN, with only the new image, and every one
        # of its tokens is masked. So reproduce exactly that.
        prompt_text = processor.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=False)
        prompt_ids = processor(text=[prompt_text], images=[img0], return_tensors="pt")["input_ids"][0].tolist()

        tool_text = processor.apply_chat_template(messages[2:], add_generation_prompt=True, tokenize=False)
        tool_ids = processor(text=[tool_text], images=[img1], return_tensors="pt")["input_ids"][0].tolist()

        full = prompt_ids + tool_ids
        # The assistant turn in this fixture stands in for a generated turn (mask 1);
        # everything the tool emitted is mask 0.
        mask = [1] * len(prompt_ids) + [0] * len(tool_ids)

        zero_frac = 1.0 - sum(mask) / len(mask)
        assert zero_frac > 0.0, "nothing masked -- the observation would train the policy"

        # Every image_pad of the SECOND image must sit in the masked region, and the
        # vision_start/vision_end that bracket it with them.
        starts = [i for i, t in enumerate(full) if t == vs]
        ends = [i for i, t in enumerate(full) if t == ve]
        assert len(starts) == len(ends) == 2, f"expected 2 vision spans, got {len(starts)}/{len(ends)}"
        lo, hi = starts[1], ends[1]
        assert all(m == 0 for m in mask[lo : hi + 1]), "the tool image's vision span is not fully masked"
        n_pad_in_span = sum(1 for t in full[lo : hi + 1] if t == pad)
        assert n_pad_in_span > 0
        # ...and the FIRST image, which came from the prompt, must not be masked here.
        assert all(m == 1 for m in mask[starts[0] : ends[0] + 1])
        return (
            f"{zero_frac:.3f} of the response masked; tool vision span "
            f"[{lo},{hi}] = {hi - lo + 1} tokens incl. {n_pad_in_span} image_pad, all zero"
        )

    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9):
        fn()

    print()
    bad = [n for n, ok, _ in _results if not ok]
    print(f"{len(_results) - len(bad)}/{len(_results)} passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
