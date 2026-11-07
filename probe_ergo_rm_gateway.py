#!/usr/bin/env python
"""Does the NVIDIA gateway judge accept images? (ERGO r_region feasibility probe)

ERGO's r_region shows a *frozen* reward model only the crop and the question --
never the original image -- and scores 1 if it answers correctly. The paper uses a
local Qwen2.5-VL-72B (~145 GB bf16, a 4-GPU sidecar). Before paying that, check
whether the gateway that already serves `trl/rewards/openai_rewards.py` can do the
job for zero GPUs.

Two things have to be true and neither is documented:
  1. the gateway accepts `image_url` parts in the chat message at all;
  2. a ~1 MP crop fits inside whatever payload cap sits in front of it.

Usage (fish):
    set -x NVIDIA_API_KEY (cat ~/.config/nvidia/api_key)
    python probe_ergo_rm_gateway.py

The data is one real V* bench row: image, ground-truth box, question, options.

VERDICT (2026-11-07, V* direct_attributes/sa_4690, gateway inference-api.nvidia.com)
------------------------------------------------------------------------------------
Outcome 1: the gateway works. Zero GPUs. `azure/openai/gpt-4o-mini` -- the judge's
existing default model, on the existing key -- accepts `image_url` data-URI parts
and answers the V* question correctly from the 232x236 crop alone.

  * payload cap: none that bites. 32 MP / 9.4 MB PNG accepted. 1 MP is already
    past the point where the server-side resize stops caring.
  * the reward is a signal, not a constant. Gold crop -> "(A)". Same-size crop of
    the wrong corner, of the far corner, and a flat grey rectangle -> "I can't
    determine the material of the glove from the image." The RM declines rather
    than guessing, so a crop that misses scores 0 on its own.
  * throughput: 128 calls in 10-16 s at 32 workers, 0 errors, 128/128 correct,
    median latency 1.4-1.7 s -- with the crop sent at *native* resolution as JPEG
    (14 kB). The same 128 calls with a 1 MP PNG upscale take 32-80 s. Send the
    crop as-is; do not upscale.
  * cost: 8,556 prompt tokens per native-crop call (gpt-4o-mini bills images at a
    large multiplier), 25,557 at 1 MP. At 128 calls/step x ~4k steps that is
    ~4.4e9 input tokens, i.e. ~$650 of gateway spend at list price (~$2k if you
    upscale). Worth knowing before the run, not a blocker.
  * no Qwen2.5-VL-72B on the gateway (257 models, none Qwen-VL). `azure/openai/gpt-4o`
    is 403 for this key; gpt-4o-mini, gpt-4.1-mini, the gpt-5 family and the Gemini
    flash family are all present if a stronger RM is wanted.

Two things the integration must handle:
  1. a refusal has no option letter -- parse "no letter" as reward 0, not as None.
  2. the RM is gpt-4o-mini, not the paper's Qwen2.5-VL-72B. That is a deviation and
     belongs in the paper's caption.
"""

import argparse
import base64
import io
import json
import os
import pathlib
import sys
import time

from PIL import Image

# --------------------------------------------------------------------------
# Config -- same defaults as trl/rewards/openai_rewards.py so a pass here means
# the existing judge path works unchanged.
# --------------------------------------------------------------------------
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com")
DEFAULT_MODELS = [
    "azure/openai/gpt-4o-mini",   # the judge's current default
    "azure/openai/gpt-4o",
]
KEY_FILE = pathlib.Path.home() / ".config" / "nvidia" / "api_key"

VSTAR_REPO = "craigwu/vstar_bench"
VSTAR_SAMPLE = "sa_4690"          # direct_attributes: "What is the material of the glove?"


def load_key():
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"no key: set NVIDIA_API_KEY or put one in {KEY_FILE}")
    return key


def load_vstar_sample():
    """One real V* row: full image, [x,y,w,h] box, question, options (options[0] is GT)."""
    from huggingface_hub import hf_hub_download

    img_p = hf_hub_download(VSTAR_REPO, f"direct_attributes/{VSTAR_SAMPLE}.jpg", repo_type="dataset")
    ann_p = hf_hub_download(VSTAR_REPO, f"direct_attributes/{VSTAR_SAMPLE}.json", repo_type="dataset")
    ann = json.load(open(ann_p))
    return Image.open(img_p).convert("RGB"), ann


def ergo_crop(img, bbox_xywh, pad=0.25):
    """ERGO crops the region the policy pointed at, with a little context padding."""
    x, y, w, h = bbox_xywh
    px, py = w * pad, h * pad
    box = (max(0, x - px), max(0, y - py), min(img.width, x + w + px), min(img.height, y + h + py))
    return img.crop([int(v) for v in box])


def to_megapixels(img, mp):
    """Rescale so the image is ~mp megapixels -- the payload-size axis of the probe."""
    import math

    scale = math.sqrt((mp * 1e6) / (img.width * img.height))
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def data_uri(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw = buf.getvalue()
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(raw).decode(), len(raw)


def ask(client, model, img, question, options, timeout=120):
    """The r_region call: crop + question only. The original image is never sent."""
    uri, nbytes = data_uri(img)
    opts = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"{question}\n{opts}\n"
                    "Answer with the option's letter from the given choices directly."
                )},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }],
        timeout=timeout,
    )
    return r.choices[0].message.content, nbytes, time.time() - t0


def controls(client, model, img, ann):
    """Is r_region a signal or a constant?

    ERGO pays the crop only if the RM *fails* when the crop misses. Three negative
    crops; if the RM still says (A) on all of them the reward is degenerate and the
    whole mechanism collapses to a constant.
    """
    question, options = ann["question"], ann["options"]
    x, y, w, h = ann["bbox"][0]
    W, H = img.width, img.height
    cases = {
        "gold crop (box + 25% pad)": ergo_crop(img, ann["bbox"][0]),
        "wrong corner, same size": img.crop((0, 0, int(w * 1.5), int(h * 1.5))),
        "far corner, same size": img.crop((W - int(w * 1.5), H - int(h * 1.5), W, H)),
        "grey rectangle": Image.new("RGB", (int(w * 1.5), int(h * 1.5)), (128, 128, 128)),
        "whole image (DeepEyes-style)": img,
    }
    print("\n-- controls: does the crop decide the answer? --")
    for name, c in cases.items():
        try:
            reply, nbytes, dt = ask(client, model, to_megapixels(c, 1.0), question, options)
            print(f"  {name:30s} -> {reply!r}")
        except Exception as e:
            print(f"  {name:30s} -> FAIL {type(e).__name__}: {str(e)[:120]}")


def throughput(client, model, img, ann, n=128, workers=32):
    """ERGO needs one RM call per rollout: gen_batch 48 x 8 rollouts ~ 128-384 a step."""
    from concurrent.futures import ThreadPoolExecutor

    crop = to_megapixels(ergo_crop(img, ann["bbox"][0]), 1.0)
    question, options = ann["question"], ann["options"]

    def one(_):
        try:
            return ask(client, model, crop, question, options)[0]
        except Exception as e:
            return f"ERR {type(e).__name__}"

    print(f"\n-- throughput: {n} calls, {workers} workers --")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, range(n)))
    dt = time.time() - t0
    errs = sum(1 for o in out if str(o).startswith("ERR"))
    print(f"  {n} calls in {dt:.1f}s  ({dt / n * 1000:.0f} ms/call amortised)  errors={errs}")
    if errs:
        from collections import Counter
        print("  ", Counter(o for o in out if str(o).startswith("ERR")).most_common(5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--mps", nargs="*", type=float, default=[0.25, 1.0, 4.0],
                    help="megapixel sizes to try, to find the payload cap")
    ap.add_argument("--controls", action="store_true", help="negative crops: is the reward a signal?")
    ap.add_argument("--throughput", type=int, default=0, help="N concurrent calls to time")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    import openai

    client = openai.OpenAI(api_key=load_key(), base_url=BASE_URL)
    img, ann = load_vstar_sample()
    question, options = ann["question"], ann["options"]
    crop = ergo_crop(img, ann["bbox"][0])

    print(f"gateway        {BASE_URL}")
    print(f"V* sample      {VSTAR_SAMPLE}  full={img.width}x{img.height}  "
          f"crop={crop.width}x{crop.height}  box={ann['bbox'][0]}")
    print(f"question       {question}")
    print(f"gold answer    {options[0]}   (ERGO scores 1 iff the RM picks A-equivalent)")
    print()

    if args.controls:
        controls(client, args.models[0], img, ann)
    if args.throughput:
        throughput(client, args.models[0], img, ann, n=args.throughput, workers=args.workers)
    if args.controls or args.throughput:
        return

    for model in args.models:
        for mp in args.mps:
            scaled = to_megapixels(crop, mp)
            label = f"{model:26s} {mp:>5.2f} MP ({scaled.width}x{scaled.height})"
            try:
                reply, nbytes, dt = ask(client, model, scaled, question, options)
                print(f"OK   {label}  png={nbytes/1e6:5.2f} MB  {dt:5.1f}s  -> {reply!r}")
            except Exception as e:
                print(f"FAIL {label}  {type(e).__name__}: {str(e)[:220]}")

    # Control: text-only on the same gateway, to tell "images rejected" apart from
    # "key/model/endpoint broken".
    try:
        r = client.chat.completions.create(
            model=args.models[0], temperature=0, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word OK."}], timeout=60)
        print(f"\ncontrol (text-only, {args.models[0]}) -> {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"\ncontrol (text-only) FAILED too: {type(e).__name__}: {str(e)[:220]}")


if __name__ == "__main__":
    main()
