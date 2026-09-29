#!/usr/bin/env python
"""The PROBES' geometry seam: Qwen3-VL unchanged, and the Omni actually different.

`test_vlm_geometry_cpu.py` does this for the trainer. This is the same job one level
sideways, for the two probes that had to stop being Qwen3-VL-only so the Omni's saliency
head pair could be selected rather than inherited:

  * `head_correlation_probe.py` -- the scan. It hardcoded `Qwen3VLTextAttention`, the
    image token 151655, `image_grid_thw`, and `getattr(transformers,
    config.architectures[0])` through `overlap_probe.load_model`.
  * `intervene_probe.py --stage prepare` -- what BUILDS the cases the scan reads. Same
    four, plus the one that is not geometry at all: the Omni's chat template opens
    `<think>` in the prompt, so every completion of a well-behaved model reads as
    malformed and the whole corpus is dropped with nothing in the log saying "template".

Qwen3-VL has published numbers on both (`outputs/head_corr/coldstart_setA`), so each
replaced expression is restated here and checked against the family's answer.

CPU only, no model, no weights: every one of these is a pure function of a batch dict, a
config, or a string.

    python test_probe_family_cpu.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import vlm_family as VF  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# fakes: enough processor to run `Family.build_inputs`, and nothing more
# ---------------------------------------------------------------------------
class Batch(dict):
    """A processor output. `.to(device)` is the only method build_inputs calls."""

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    """Records what `Family.build_inputs` asked it for.

    `apply_chat_template` returns a marker carrying the messages, so the test can check
    the PROMPT the probe would have built without owning a copy of any chat template.
    """

    def __init__(self, token_ids=None, out=None):
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda n: (token_ids or {}).get(n, -1))
        self.seen_msgs = None
        self.seen_call = None
        self._out = out or {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        self.seen_msgs = msgs
        return f"PROMPT({len(msgs)} msgs)"

    def __call__(self, **kw):
        self.seen_call = kw
        return Batch(self._out)


def qwen_family(proc=None):
    fam = VF._REGISTRY["qwen3_vl"]()
    return fam.bind(processor=proc, config=SimpleNamespace(model_type="qwen3_vl"))


def omni_family(proc=None):
    """The Omni, bound to the config fields `NemotronVL.bind` actually reads."""
    cfg = SimpleNamespace(
        model_type="NemotronH_Nano_Omni_Reasoning_V3",
        img_context_token_id=18,                       # NOT 151655
        text_config=SimpleNamespace(model_type="nemotron_h"),
    )
    fam = VF._REGISTRY["NemotronH_Nano_Omni_Reasoning_V3"]()
    return fam.bind(processor=proc, config=cfg)


# ---------------------------------------------------------------------------
def test_attention_modules():
    """was: `type(m).__name__ == "Qwen3VLTextAttention"` in AllHeadCapture.__init__"""
    print("\nWhich modules the scan installs on")
    q = qwen_family()
    check("qwen3_vl still names exactly Qwen3VLTextAttention",
          tuple(q.attn_classes) == ("Qwen3VLTextAttention",), str(q.attn_classes))
    check("and the old `==` test is the new `in` test",
          "Qwen3VLTextAttention" in q.attn_classes)

    o = omni_family()
    check("the Omni names NemotronHAttention, resolved off its TEXT config",
          tuple(o.attn_classes) == ("NemotronHAttention",), str(o.attn_classes))
    check("a Qwen3-VL-shaped search would find nothing there",
          "Qwen3VLTextAttention" not in o.attn_classes)

    # was: the hook re-ran its own module in eager to recover the softmax weights.
    check("Qwen3-VL still has to re-run the module in eager",
          q.attention_weights_are_returned is False)
    check("the Omni's attention hands its weights back, so it does not",
          o.attention_weights_are_returned is True)


def test_image_token():
    """was: `IMAGE_TOKEN_ID = 151655`, used to find the picture's columns"""
    print("\nWhere the picture is in the prompt")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t_probe", REPO / "overlap_probe.py")
    # Only the constant is wanted, and importing overlap_probe pulls the whole reward
    # stack, so the literal is read out of the source instead.
    src = (REPO / "overlap_probe.py").read_text()
    want = int(re.search(r"^IMAGE_TOKEN_ID = (\d+)", src, re.M).group(1))
    check("overlap_probe's constant is still 151655", want == 151655, str(want))
    check("qwen3_vl's image_token_id IS that constant",
          qwen_family().image_token_id == want)
    check("the Omni's is 18, from config.img_context_token_id",
          omni_family().image_token_id == 18, str(omni_family().image_token_id))
    assert spec is not None          # keeps the unused-import check honest


def test_prompt_construction():
    """was: PROBE.build_prompt(processor, q) + processor(text=[text], images=[[image]],
             return_tensors="pt", padding=True, padding_side="left",
             add_special_tokens=False).to(device)
    """
    print("\nThe prompt, and how the pictures are handed over")
    sys_prompt = "SYS"
    proc = FakeProcessor()
    fam = qwen_family(proc)
    fam.system_prompt = sys_prompt
    fam.build_inputs(proc, ["IMG"], "q?", "cpu")

    # build_prompt's messages, verbatim from overlap_probe.build_prompt
    want_msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "q?"}]},
    ]
    check("the messages are build_prompt's, system prompt included",
          proc.seen_msgs == want_msgs, str(proc.seen_msgs))
    call = proc.seen_call
    check("images are nested one list per sample, as the old call spelled it",
          call["images"] == [["IMG"]], str(call["images"]))
    check("the four processor flags are unchanged",
          call["return_tensors"] == "pt" and call["padding"] is True
          and call["padding_side"] == "left" and call["add_special_tokens"] is False)

    proc2 = FakeProcessor()
    om = omni_family(proc2)
    om.system_prompt = sys_prompt
    om.build_inputs(proc2, ["IMG"], "q?", "cpu")
    check("the Omni gets the same messages", proc2.seen_msgs == want_msgs)
    check("but a FLAT image list -- its processor walks the list replacing <image> in "
          "order and cannot iterate a nested one",
          proc2.seen_call["images"] == ["IMG"], str(proc2.seen_call["images"]))


def test_teacher_forced_forward():
    """was, in head_correlation_probe.scan_case:
           fwd = {"input_ids": ids, "attention_mask": ones_like(ids)}
           if "pixel_values" in inputs:
               fwd["pixel_values"]   = inputs["pixel_values"]
               fwd["image_grid_thw"] = inputs["image_grid_thw"]
           if inputs.get("mm_token_type_ids") is not None:
               pad = zeros(1, seq - prompt_len)
               fwd["mm_token_type_ids"] = cat([inputs["mm_token_type_ids"], pad], 1)
           out = model(**fwd, use_cache=True)
    """
    print("\nThe one measured forward: prompt ++ chain")
    prompt_len, n_comp = 7, 4
    inputs = {
        "input_ids": torch.arange(prompt_len).view(1, -1),
        "pixel_values": torch.zeros(12, 3),
        "image_grid_thw": torch.tensor([[1, 4, 6]]),
        "mm_token_type_ids": torch.ones(1, prompt_len, dtype=torch.long),
    }
    chain = [101, 102, 103, 104]
    got = qwen_family().teacher_forced_case(inputs, chain, "cpu")

    want_ids = torch.cat([inputs["input_ids"], torch.tensor([chain])], dim=1)
    want_mm = torch.cat([inputs["mm_token_type_ids"],
                         torch.zeros(1, n_comp, dtype=torch.long)], dim=1)
    check("the carried keys are the same four (+ input_ids/attention_mask)",
          set(got) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                       "mm_token_type_ids"}, str(sorted(got)))
    check("input_ids is prompt ++ chain", torch.equal(got["input_ids"], want_ids))
    check("attention_mask is ones over all of it",
          torch.equal(got["attention_mask"], torch.ones_like(want_ids)))
    check("mm_token_type_ids is extended over the completion with zeros",
          torch.equal(got["mm_token_type_ids"], want_mm))
    check("pixel_values and the grid are carried verbatim",
          torch.equal(got["pixel_values"], inputs["pixel_values"])
          and torch.equal(got["image_grid_thw"], inputs["image_grid_thw"]))
    # was: model(**fwd, use_cache=True). forward_defaults must not add anything here, or
    # the Qwen3-VL pass would differ from the one every published number was measured on.
    check("Qwen3-VL adds no forward defaults", qwen_family().forward_defaults == {})

    om = omni_family()
    oin = {"input_ids": torch.arange(prompt_len).view(1, -1),
           "pixel_values": torch.zeros(1, 3, 4, 4),
           "image_flags": torch.ones(1, 1, dtype=torch.long),
           "imgs_sizes": torch.tensor([[416, 672]])}
    ogot = om.teacher_forced_case(oin, chain, "cpu")
    check("the Omni carries image_flags and NOT image_grid_thw",
          set(ogot) == {"input_ids", "attention_mask", "pixel_values", "image_flags"},
          str(sorted(ogot)))
    check("no geometry key reaches the forward",
          not (set(ogot) & set(om.geometry_inputs)))
    check("use_cache is forced off, so no hybrid cache is built and discarded",
          om.forward_defaults.get("use_cache") is False)
    check("and logits_to_keep 1, so the lm_head skips a 131k-vocab pass nothing reads",
          om.forward_defaults.get("logits_to_keep") == 1)


def test_grid():
    """was: gh = inputs["image_grid_thw"][0, 1] // 2 ; gw = ...[0, 2] // 2"""
    print("\nThe patch grid the step masks are drawn on")
    thw = torch.tensor([[1, 8, 12]])
    batch = {"image_grid_thw": thw}
    check("qwen3_vl: token_grid == (h//2, w//2)",
          qwen_family().token_grid(batch, 0)
          == (int(thw[0, 1]) // 2, int(thw[0, 2]) // 2),
          str(qwen_family().token_grid(batch, 0)))

    om = omni_family()
    sizes = torch.tensor([[416, 672]])
    check("the Omni derives it from imgs_sizes over a 32px token",
          om.token_grid({"imgs_sizes": sizes}, 0) == (416 // 32, 672 // 32),
          str(om.token_grid({"imgs_sizes": sizes}, 0)))
    try:
        om.token_grid({}, 0)
        ok = False
    except RuntimeError:
        ok = True
    check("and refuses to guess one when the key is absent", ok)


def test_generate_inputs():
    """was: model.generate(**inputs, ...) -- every processor output, straight through"""
    print("\nWhat generate() is allowed to see")
    om = omni_family()
    full = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 4, 4),
            "image_flags": torch.ones(1, 1, dtype=torch.long),
            "num_patches": torch.ones(1, dtype=torch.long),
            "num_tokens": torch.ones(1, dtype=torch.long),
            "imgs_sizes": torch.tensor([[416, 672]])}
    fwd = om.model_inputs(full)
    check("model_inputs drops what forward() raises TypeError on",
          set(fwd) == {"input_ids", "pixel_values", "image_flags"}, str(sorted(fwd)))
    gen = om.generate_inputs(fwd)
    check("generate_inputs drops image_flags on top -- the wrapper eats it itself",
          set(gen) == {"input_ids", "pixel_values"}, str(sorted(gen)))
    check("imgs_sizes is still on the FULL dict, which is what token_grid reads",
          "imgs_sizes" in full)

    q = qwen_family()
    same = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(2, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    check("Qwen3-VL's generate sees exactly what it always did",
          q.generate_inputs(q.model_inputs(same)) is same)


def test_prompt_opens_think():
    """The trap that is not geometry: the Omni's template opens the reasoning block."""
    print("\nDoes the prompt already open <think>?")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t_iv", REPO / "intervene_probe.py")
    IV = importlib.util.module_from_spec(spec)
    sys.modules["_t_iv"] = IV
    spec.loader.exec_module(IV)

    class Tok:
        def __init__(self, text):
            self.text = text

        def decode(self, ids, **kw):
            return self.text

    sysp = IV.PROBE.SYSTEM_PROMPT
    qwen_prompt = (f"<|im_start|>system\n{sysp}<|im_end|>\n<|im_start|>user\n"
                   "<|vision_start|><|image_pad|><|vision_end|>q?<|im_end|>\n"
                   "<|im_start|>assistant\n")
    omni_prompt = (f"<|im_start|>system\n{sysp}<|im_end|>\n<|im_start|>user\n"
                   "<img><image></img>\nq?<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n")
    check("Qwen3-VL's generation prompt does not",
          IV.prompt_opens_think(Tok(qwen_prompt), None) is False)
    check("the Omni's does", IV.prompt_opens_think(Tok(omni_prompt), None) is True)
    check("the SYSTEM prompt's own <think></think> is not mistaken for it -- the anchor "
          "is what makes that true",
          "<think>" in sysp and IV.prompt_opens_think(Tok(qwen_prompt), None) is False)

    # The format gate, with and without the opener. A well-behaved Omni completion carries
    # only the closing tag; without the opener every one of them reads as malformed and
    # `prepare` drops the entire corpus as bad_format.
    omni_completion = "\nThe image shows a red car.\n</think>\nB"
    qwen_completion = "<think>\nThe image shows a red car.\n</think>\nB"
    jf = IV.PROBE.judge_format
    check("a real Omni completion fails the bare gate", jf(omni_completion) is False)
    check("and passes it with the opener prepended",
          jf("<think>\n" + omni_completion) is True)
    check("a Qwen3-VL completion passes without one", jf(qwen_completion) is True)
    check("a model that writes a SECOND <think> still fails, opener or not",
          jf("<think>\n" + qwen_completion) is False)

    # And the span rule that goes with it: with no `<think>` in the text there is nothing
    # for the old regex to anchor on, so the reasoning starts at the first non-space char.
    ms_old = re.search(r"<think>\s*(\S\S*)", omni_completion, re.DOTALL | re.MULTILINE)
    ms_new = re.search(r"\s*(\S)", omni_completion, re.DOTALL | re.MULTILINE)
    check("the old think-start regex finds nothing on an Omni completion",
          ms_old is None)
    check("the opener rule starts at the first non-space character",
          ms_new is not None and omni_completion[ms_new.start(1)] == "T",
          repr(omni_completion[ms_new.start(1):ms_new.start(1) + 4]))
    # On a Qwen3-VL completion the two rules must agree, or the ported branch would move
    # the published corpus.
    a = re.search(r"<think>\s*(\S\S*)", qwen_completion, re.DOTALL | re.MULTILINE)
    check("on a Qwen3-VL completion the old rule is still the one that runs",
          a is not None and qwen_completion[a.start(1)] == "T")


def test_loader_dispatch():
    """was: PROBE.load_model(...) -- getattr(transformers, config.architectures[0])"""
    print("\nWhich loader a checkpoint gets")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t_nl", REPO / "nemotron_loader.py")
    NL = importlib.util.module_from_spec(spec)
    sys.modules["_t_nl"] = NL
    spec.loader.exec_module(NL)

    seen = {}

    def native(path, adapter, device, attn_impl):
        seen.update(path=path, adapter=adapter, device=device, attn_impl=attn_impl)
        return ("proc", "model")

    NL.is_remote_code = lambda p: (False, SimpleNamespace(model_type="qwen3_vl"))
    got = NL.load_any("/ckpt", "adapter", "cuda:0", "sdpa", native)
    check("a native checkpoint goes to the native loader, arguments untouched",
          got == ("proc", "model")
          and seen == dict(path="/ckpt", adapter="adapter", device="cuda:0",
                           attn_impl="sdpa"), str(seen))

    NL.is_remote_code = lambda p: (True, SimpleNamespace(model_type="omni"))
    NL.load_model = lambda path, device, attn_impl="eager", **kw: ("p2", (path, attn_impl))
    got = NL.load_any("/omni", "", "cuda:1", "sdpa", native)
    check("a remote-code one is pinned to eager, whatever --attn-impl said",
          got[1] == ("/omni", "eager"), str(got[1]))
    try:
        NL.load_any("/omni", "some-adapter", "cuda:1", "sdpa", native)
        ok = False
    except SystemExit:
        ok = True
    check("and refuses an --adapter rather than silently ignoring it", ok)


def test_answer_grading():
    """The label the head ranking is built on, and how much of it is the grader.

    Every string here is one the Nemotron-Omni actually wrote in the smoke run, with the
    gold it was scored against. The trainer's `accuracy_reward` gave five of these zero.
    """
    print("\nGrading a verbose answer against a short gold")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t_ag", REPO / "answer_grading.py")
    AG = importlib.util.module_from_spec(spec)
    sys.modules["_t_ag"] = AG
    spec.loader.exec_module(AG)

    #  (what the model wrote, gold, is it right)
    real = [
        ("(C) water supply", "C", True),
        ("(C) working", "C", True),
        ("\\boxed{A}", "A", True),
        ("\\boxed{D}", "A", False),
        ("This is my answer. \nD", "D", True),
        ("This is my answer. singing", "singing", True),
        ("A", "A", True),
        ("Yes", "Yes", True),
        ("This is my answer.", "horses", False),
        ("Thus, the answer is (D).", "C", False),
    ]
    for text, gold, want in real:
        got = bool(AG.grade_completion(text, gold)["soft"])
        check(f"{text[:26]!r} vs {gold!r} -> {'right' if want else 'wrong'}",
              got == want, f"got {got}")

    # The trainer's own rule, restated, so the GAP is what this file asserts and not just
    # the new grader's behaviour.
    def trainer_rule(text, gold):
        return text.strip().lower() == gold.strip().lower()

    missed = [t for t, g, w in real if w and not trainer_rule(t, g)]
    check("the trainer's exact-match rule misses 5 of the 7 correct ones",
          len(missed) == 5, str(len(missed)))
    check("and the article 'A' in prose is still not an answer",
          AG.mcq_letter("A large bus is parked outside.") is None,
          str(AG.mcq_letter("A large bus is parked outside.")))
    check("a boxed letter that disagrees with the gold stays wrong",
          AG.grade_completion("\\boxed{D}", "A")["soft"] is False)


def test_label_audit():
    """`report` must print the two accuracies without changing any of them."""
    print("\nThe label audit")
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location("_t_hc", REPO / "head_correlation_probe.py")
    HC = importlib.util.module_from_spec(spec)
    sys.modules["_t_hc"] = HC
    spec.loader.exec_module(HC)

    with tempfile.TemporaryDirectory() as td:
        cases = Path(td) / "cases"
        cases.mkdir()
        (cases / "shard00.json").write_text(json.dumps({"config": {}, "dropped": {}, "cases": [
            {"row_index": 4, "gold": "C", "answer_text": "(C) water supply"},
            {"row_index": 7, "gold": "A", "answer_text": "A"},
            {"row_index": 9, "gold": "C", "answer_text": "Thus, the answer is (D)."},
        ]}))
        # two steps per completion, the label constant within one -- the scan's shape
        row = np.array([4, 4, 7, 7, 9, 9])
        cor = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])      # the trainer's grades
        soft, note = HC.label_audit(td, row, cor)

    check("row 4 flips: (C) water supply IS C",
          soft is not None and list(soft[:2]) == [1.0, 1.0], str(None if soft is None else soft))
    check("row 7 was already right and stays right", list(soft[2:4]) == [1.0, 1.0])
    check("row 9 answered D against gold C and stays wrong", list(soft[4:]) == [0.0, 0.0])
    check("both accuracies are printed", "0.333" in note and "0.667" in note, note.strip())
    check("and the trainer's label is named as the primary",
          "primary number" in note)
    check("the audit returns a NEW array -- `cor` is not mutated",
          list(cor) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    with tempfile.TemporaryDirectory() as td:
        cases = Path(td) / "cases"
        cases.mkdir()
        (cases / "shard00.json").write_text(json.dumps({"config": {}, "dropped": {}, "cases": [
            {"row_index": 4, "gold": "C"},           # prepared before answer_text existed
        ]}))
        soft, note = HC.label_audit(td, np.array([4]), np.array([0.0]))
    check("cases with no stored answer say so rather than grading nothing",
          soft is None and "answer_text" in note)


def main():
    test_attention_modules()
    test_image_token()
    test_prompt_construction()
    test_teacher_forced_forward()
    test_grid()
    test_generate_inputs()
    test_prompt_opens_think()
    test_loader_dispatch()
    test_answer_grading()
    test_label_audit()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
