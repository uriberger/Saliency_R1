#!/usr/bin/env python
"""Print what a model actually writes on the sink-location corpus, verbatim.

    python dump_completions.py --model M --out-dir DIR [--n 6]

`format_ok` in the observe-step scan is a pass/fail on a regex, and a pass/fail is not
readable: a 0% can mean the model ignored the format, or that it emitted the tags twice,
or that the prompt already opened the think block so the completion only ever carries the
closing tag. Those are three different facts about the run and the regex cannot tell them
apart. This prints the completion, so the number in the report is one somebody checked.

Deliberately not a stage of the probe: it generates without measuring anything, and a
scan stage that writes no statistics is a stage somebody will run by mistake.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out-dir", required=True, help="a scan dir, for its corpus")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--system-prompt", default="auto")
    ap.add_argument("--chars", type=int, default=1200)
    ap.add_argument("--like-scan", action="store_true",
                    help="generate through an installed SinkScan, as the scan does")
    ap.add_argument("--segment", action="store_true",
                    help="also run observe_spans and print its diagnostics")
    args = ap.parse_args()

    import torch
    P = _load("_dc_probe", "sink_location_probe.py")
    from PIL import Image

    rows = P.read_manifest(args.out_dir, "")[: args.n]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = P.PROBE.load_model(args.model, args.adapter, device, "sdpa")
    fam = P.load_family(model, processor, args.system_prompt)
    print(f"model={args.model}\nfamily={fam.name}  system prompt="
          f"{'(none)' if not fam.system_prompt else repr(fam.system_prompt[:60]) + '...'}")

    # The prompt matters as much as the completion: a chat template that opens <think>
    # itself is the single likeliest reason a completion carries only the closing tag.
    inputs = P.build_inputs(fam, processor, [Image.open(rows[0]["path"]).convert("RGB")],
                            rows[0]["question"], device)
    prompt_text = processor.tokenizer.decode(inputs["input_ids"][0],
                                             skip_special_tokens=False)
    print("\n" + "=" * 78 + "\nPROMPT (tail)\n" + "=" * 78)
    print(repr(prompt_text[-400:]))

    # Generate through the SAME object the scan uses -- hooks installed, paused during
    # generation -- because "the dump and the scan disagree" is otherwise unresolvable:
    # a plain generate is a different code path and proves nothing about the scan.
    scan = P.SL.install(model, family=fam) if args.like_scan else _Dummy(fam)
    clf = P.STEPS.OverlapStepsClassifier.load(None, device=device) if args.segment else None
    for r in rows:
        im = Image.open(r["path"]).convert("RGB")
        gen = P.generate_then_teacher_force(model, processor, [im], r["question"],
                                            device, scan, args.max_new_tokens)
        if gen is None:
            print(f"\n--- {r['key']}: nothing generated")
            continue
        _inputs, _plen, comp = gen
        text = processor.tokenizer.decode(comp, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
        body = re.sub(r"<\|im_end\|>\s*$", "", text).strip()
        print("\n" + "=" * 78)
        print(f"{r['key']}  ({r['type']})  {len(comp)} tokens"
              f"   <think>x{body.count('<think>')}  </think>x{body.count('</think>')}"
              f"   format_ok={bool(re.match(P.PROBE.FORMAT_PATTERN, body, re.DOTALL | re.MULTILINE) and body.count('<think>') == 1 and body.count('</think>') == 1)}")
        print("=" * 78)
        if clf is not None:
            spans, diag = P.observe_spans(Proc(processor), comp, r["question"], clf)
            print(f"observe_spans -> diag={diag}  spans={spans}")
        print(text[: args.chars] + ("…" if len(text) > args.chars else ""))
    return 0


class Proc:
    """observe_spans only reaches through `.tokenizer`."""
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer


class _Dummy:
    """generate_then_teacher_force only reads `.family` and toggles `.paused`."""
    def __init__(self, fam):
        self.family, self.paused = fam, False


if __name__ == "__main__":
    sys.exit(main())
