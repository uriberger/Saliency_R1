#!/usr/bin/env python
"""Mark the four trained tags NON-special, so LASER's decode path can see them.

    python unspecial_tags.py <model-or-checkpoint-dir> [...]

THE INCOMPATIBILITY THIS RESOLVES

ReVisual-R1's cold-start config adds `<think>,</think>,<answer>,</answer>` via
`new_special_tokens` + `resize_vocab`, i.e. as **special** tokens. LASER's trainer then
reads every rollout back with

    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

in `dp_actor.py` (both the format gate on the attention rewards and, via the `dapo`
manager, `compute_score_vanilla`). `skip_special_tokens=True` **deletes special tokens**.
So a model that emits a perfectly formed

    <think> ... </think><answer> \\boxed{42} </answer>

is scored on

    ' ... \\boxed{42} '

which has zero tags, fails `format_reward` and `_format_reward_think`, and therefore zeroes
BOTH attention rewards on every rollout. As published the two halves cannot both hold.

Measured on our epoch-5 checkpoint before this fix: format 0.000, think_gate 0.000,
accuracy 0.000 across all four prompt variants -- while the generations themselves were
coherent and ended in `\\boxed{}`. The model was right and the readout was lying.

WHAT THIS CHANGES, AND WHAT IT DOES NOT

Only the `special` flag on those four entries in `tokenizer.json`'s `added_tokens`. Token
ids, the learned embeddings, and the model weights are all untouched -- the tokens still
encode to one id each, so nothing the SFT taught is lost. What changes is that
`skip_special_tokens=True` stops deleting them, which is the behaviour LASER's code
assumes.

The alternative is retraining with LLaMA-Factory's `add_tokens` (non-special) instead of
`add_special_tokens`. Same end state, three and a half more GPU-hours, and a deviation
from ReVisual-R1's config either way -- the deviation is forced, the only choice is where
to pay for it.

Idempotent; backs up `tokenizer.json` to `tokenizer.json.special.orig` on first run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def unspecial(d: Path) -> str:
    p = d / "tokenizer.json"
    if not p.is_file():
        return "no tokenizer.json"
    tj = json.loads(p.read_text())
    added = tj.get("added_tokens", [])
    hits = [a for a in added if a.get("content") in TAGS]
    if len(hits) != len(TAGS):
        return f"REFUSED: found {len(hits)} of {len(TAGS)} tags; not the expected checkpoint"
    if all(a.get("special") is False for a in hits):
        return "already non-special"
    if not (d / "tokenizer.json.special.orig").exists():
        shutil.copy2(p, d / "tokenizer.json.special.orig")
    for a in hits:
        a["special"] = False
    p.write_text(json.dumps(tj, ensure_ascii=False))
    return f"flipped {len(hits)} tags to special=False (ids {[a['id'] for a in hits]})"


def verify(d: Path) -> str:
    """Round-trip through the real tokenizer -- the only check that means anything."""
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(str(d))
    s = "<think>\nr\n</think>\n<answer>\n\\boxed{42}\n</answer>"
    ids = t(s, add_special_tokens=False).input_ids
    kept = t.decode(ids, skip_special_tokens=True)
    n_tok = [len(t(x, add_special_tokens=False).input_ids) for x in TAGS]
    ok = all(tag in kept for tag in TAGS)
    return (f"decode(skip_special_tokens=True) keeps all four tags: {ok}; "
            f"still one token each: {n_tok}")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for p in argv:
        d = Path(p)
        print(f"{d.name:<40} {unspecial(d)}")
        try:
            print(f"{'':<40} {verify(d)}")
        except Exception as exc:
            print(f"{'':<40} verify failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
