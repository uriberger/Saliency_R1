#!/usr/bin/env python
"""The system prompt must reach the model as text, on every family.

`_generate_and_score_completions` rewrites a conversational prompt's turns before handing
them to the chat template. The USER turn has to become structured content -- that is the
only way to say "a picture goes here". The SYSTEM turn does not, and wrapping it was a bug:

    Qwen3-VL   <|im_start|>system\\nA conversation between user and assistant.
    Omni       <|im_start|>system\\n[{'type': 'text', 'text': 'A conversation between ...'}]

Qwen3-VL's template understands a content LIST for the system role. The Omni's does not,
so Jinja falls back to stringifying the Python object, and the model reads a repr as its
instructions. Every Omni GRPO run ever launched did this; it was found on 2026-10-08 by
reading the live wov0.4 run's own logged completions table, 1,221 steps in. Every Qwen3-VL
run was clean, so the two arms being compared differed in their prompts as well as in
their cold start.

This pins the repaired behaviour on BOTH families, because `grpo_trainer_qwen3.py` is
shared: `patch_trl_qwen3.sh` installs it under the Qwen3-VL runs and
`patch_trl_nemotron.sh` under the Omni ones. A change that helps one and breaks the other
would be invisible until a run's format reward collapsed.

CPU only -- it loads processors and tokenizers, never weights. Needs the two models in the
HF cache; skips with a clear message if they are not there.
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYS = ("A conversation between user and assistant. The user asks a question, and the "
       "assistant solves it.")
QUESTION = "What is on the luggage?"

MODELS = {
    "Qwen3-VL": "Qwen/Qwen3-VL-8B-Instruct",
    "Omni": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
}

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def render(tok, system_content):
    """Exactly the shape `_generate_and_score_completions` hands the template."""
    msgs = [
        {"role": "system", "content": system_content},
        # The user turn is structured on every family -- that part is not the bug.
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    try:
        from transformers import AutoProcessor
    except Exception as e:
        print(f"SKIP: transformers unavailable ({type(e).__name__})")
        return
    ran = 0
    for label, model in MODELS.items():
        try:
            proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"skip  {label}: not in the HF cache ({type(e).__name__})")
            continue
        tok = getattr(proc, "tokenizer", proc)
        ran += 1
        plain = render(tok, SYS)                                   # the repaired form
        wrapped = render(tok, [{"type": "text", "text": SYS}])     # the old, buggy form

        # 1. The thing that actually matters: the instructions arrive as text.
        check(f"{label}: system prompt renders as plain text",
              SYS in plain and "{'type'" not in plain, repr(plain[:140]))
        # 2. And the repr never appears. This is the regression that was shipped.
        check(f"{label}: no Python repr leaks into the prompt",
              "'type':" not in plain and "{'text'" not in plain, repr(plain[:140]))
        # 3. The user turn still carries its image marker -- the fix must not reach it.
        check(f"{label}: the user turn still announces an image",
              "image" in plain.lower().split("<|im_start|>user")[-1][:60] or "<image>" in plain,
              repr(plain[-200:]))

        # 4. Family-specific: state what the old form did, so the asymmetry is on record
        #    rather than being rediscovered.
        if label == "Qwen3-VL":
            check("Qwen3-VL: the fix is a NO-OP (its template handled both)",
                  plain == wrapped, "they differ, which this test did not expect")
        else:
            check("Omni: the old form really did wrap the prompt in a repr",
                  "{'type'" in wrapped, repr(wrapped[:140]))
            check("Omni: and the fix removes it", "{'type'" not in plain, repr(plain[:140]))

    if ran == 0:
        print("SKIP: neither model is in the HF cache; nothing checked.")
        return
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print(f"all checks passed ({ran} families)")


if __name__ == "__main__":
    main()
