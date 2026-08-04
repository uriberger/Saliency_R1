#!/usr/bin/env python
"""Find which parameter makes a TRL vllm_serve /generate/ request hang forever.

Validation hung a training run twice. The server received the request, printed
"Adding requests: 0/48", produced nothing, and never replied -- while the training
request immediately before it completed 48 sequences in 7 seconds. No error was
logged anywhere, and vllm_serve's worker loop explains why:

    method = getattr(llm, method_name)
    result = method(*args, **kwargs)      # no try/except
    if command["type"] == "call":
        connection.send(result)           # skipped if the line above raises

Any exception inside llm.generate() therefore skips the reply, the server blocks
in connection.recv() with no timeout, and the caller waits forever. A crash and a
deadlock look identical from outside, which is why this needs bisecting rather
than reading.

Every request that run ever served successfully used n=8 and temperature=1.0. The
validation call was the first with n=1 and temperature=0.0, so those are the prime
suspects, ahead of the prompt and image counts.

Each case runs against a FRESH server, because one hang wedges the worker and
every request after it would hang too -- which would look like every variable
mattering. The HTTP timeout is what keeps the probe itself from hanging.

Usage (inside a GPU allocation, with a server already serving):
    python probe_vllm_hang.py --url http://127.0.0.1:8000 --case NAME
    python probe_vllm_hang.py --list
"""

import argparse
import base64
import io
import json
import sys
import time

import requests

# The control -- the exact shape training uses -- HUNG, which exonerates n,
# temperature and prompt count: none of them differ in that case. So the first
# question is no longer "which parameter", it is "is this a deadlock or just very
# slow, and is it the image path at all".
#
#   text_only      no images. If this passes, the hang is multimodal.
#   control_slow   the control again with a 30-minute timeout. If it completes,
#                  nothing is deadlocked and we are looking at pathological
#                  slowness in image preprocessing on this node.
#
# For reference, the training server logged the identical lazy-processor warning
# and finished the same request in 3 seconds, then ran at 104 it/s.
CASES = {
    "text_only":              dict(n_prompts=6,  n=8, temperature=1.0, images=False),
    "control_slow":           dict(n_prompts=6,  n=8, temperature=1.0),
    "control_training_shape": dict(n_prompts=6,  n=8, temperature=1.0),
    "n1_temp1":               dict(n_prompts=6,  n=1, temperature=1.0),
    "n8_temp0":               dict(n_prompts=6,  n=8, temperature=0.0),
    "n1_temp0":               dict(n_prompts=6,  n=1, temperature=0.0),
    "many_prompts_n8_temp1":  dict(n_prompts=48, n=8, temperature=1.0),
    "validation_shape":       dict(n_prompts=48, n=1, temperature=0.0),
}


def encode_image(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_inputs(n_prompts, val_dir, model):
    """Prompts and images exactly as ValidationAccuracyCallback builds them."""
    from datasets import load_from_disk
    from transformers import AutoProcessor
    from trl.data_utils import maybe_apply_chat_template

    processor = AutoProcessor.from_pretrained(model)
    rows = list(load_from_disk(val_dir).select(range(n_prompts)))
    SYSTEM = (
        "A conversation between user and assistant. The user asks a question, and the "
        "assistant solves it. The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. The reasoning process and "
        "answer are enclosed within <think></think> tags, "
        "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
    )
    prompts, images = [], []
    for row in rows:
        example = {"prompt": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": row["problem"]}]}
        prompts.append(maybe_apply_chat_template(example, processor)["prompt"])
        images.append(encode_image(row["image"]))
    return prompts, images


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--case", help="which case to run (see --list)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--val-dir", default="cold_data/grpo_sets/val_natural")
    p.add_argument("--model", required=False)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--out", help="append the verdict to this JSONL file")
    args = p.parse_args()

    if args.list:
        for name, spec in CASES.items():
            print(f"  {name:24s} {spec}")
        return
    if not args.case or args.case not in CASES:
        raise SystemExit(f"--case must be one of: {', '.join(CASES)}")
    if not args.model:
        raise SystemExit("--model is required")

    spec = CASES[args.case]
    prompts, images = build_inputs(spec["n_prompts"], args.val_dir, args.model)
    if spec.get("images", True) is False:
        images = None
    payload = {
        "prompts": prompts,
        "images": images,
        "n": spec["n"],
        "temperature": spec["temperature"],
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": args.max_tokens,
        "generation_kwargs": {},
    }

    print(f"[probe] {args.case}: {spec}  ({len(prompts)} prompts, "
          f"{spec['n_prompts'] * spec['n']} sequences, "
          f"{'no images' if images is None else str(len(images)) + ' images'}, "
          f"timeout {args.timeout}s)", flush=True)
    started = time.time()
    try:
        response = requests.post(f"{args.url}/generate/", json=payload, timeout=args.timeout)
        elapsed = time.time() - started
        completions = len(response.json().get("completion_ids", []))
        verdict = "OK"
        detail = f"HTTP {response.status_code}, {completions} completions in {elapsed:.1f}s"
    except requests.exceptions.Timeout:
        elapsed, verdict = time.time() - started, "HANG"
        detail = f"no response in {args.timeout}s -- the worker never replied"
    except Exception as exc:
        elapsed, verdict = time.time() - started, "ERROR"
        detail = f"{type(exc).__name__}: {str(exc)[:200]}"

    print(f"[probe] {args.case}: {verdict} -- {detail}", flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"case": args.case, **spec, "verdict": verdict,
                                 "detail": detail, "seconds": round(elapsed, 1)}) + "\n")
    sys.exit(0 if verdict == "OK" else 1)


if __name__ == "__main__":
    main()
