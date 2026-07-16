"""Batched Grounding-DINO HTTP endpoint for the attention-overlap GRPO reward.

Serve this on a GPU OUTSIDE the 8-GPU training allocation (spare card / MIG slice) so
the ZeRO-3 policy stays at 8 GPUs (see grpo-reward-port-plan memory). The training
processes hit it via trl.rewards.overlap_rewards._dino_boxes_served, batching every
(observe-step image, step-text) pair in the rollout into one request per reward call.

Wire contract (must match overlap_rewards._dino_boxes_served):

    POST /ground
      {"images": [<base64 PNG>, ...], "texts": [...],
       "box_threshold": float, "text_threshold": float}
    -> {"boxes": [[[x1, y1, x2, y2], ...],   # per input item, RELATIVE [0,1] coords
                  ...]}

    GET /health -> {"status": "ok", "model": ..., "device": ...}

Box selection + relative-coord conversion are byte-for-byte the same as the local path
(_dino_boxes_local) so served and local runs are interchangeable. Area filtering is done
client-side (max_box_area), so it is intentionally NOT applied here.

Run:
    bash serve_grounding_dino.sh                 # port 8100, GPU 0 of this node
    bash serve_grounding_dino.sh --port 8100 --gpu 0
Then launch training with:
    --dino-api-base http://<this-node>:8100
"""

from __future__ import annotations

import argparse
import base64
import io
import os

import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

GROUNDING_DINO_HF_ID = "IDEA-Research/grounding-dino-base"
# Cap the per-forward batch so a large rollout can't OOM the DINO card; the endpoint
# still accepts an arbitrarily long request and chunks it internally.
SERVER_BATCH = int(os.environ.get("DINO_SERVER_BATCH", "32"))

app = FastAPI(title="grounding-dino-overlap-reward")
_STATE: dict = {"proc": None, "model": None, "device": None}


class GroundRequest(BaseModel):
    images: list[str]           # base64-encoded PNG bytes
    texts: list[str]
    box_threshold: float = 0.10
    text_threshold: float = 0.10


class GroundResponse(BaseModel):
    boxes: list[list[list[float]]]


def _load():
    if _STATE["model"] is None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = os.environ.get("DINO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[serve_grounding_dino] loading {GROUNDING_DINO_HF_ID} on {device} ...", flush=True)
        proc = AutoProcessor.from_pretrained(GROUNDING_DINO_HF_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_HF_ID).to(device).eval()
        _STATE.update(proc=proc, model=model, device=device)
        print("[serve_grounding_dino] ready.", flush=True)
    return _STATE["proc"], _STATE["model"], _STATE["device"]


def _decode(b64: str) -> Image.Image:
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    return im.convert("RGB") if im.mode != "RGB" else im


@torch.no_grad()
def _ground_batch(images, texts, box_threshold, text_threshold):
    proc, model, device = _load()
    prompts = [(t.strip() + "." if not t.strip().endswith(".") else t.strip()) for t in texts]
    out_boxes: list[list[list[float]]] = [None] * len(images)  # type: ignore[list-item]
    for start in range(0, len(images), SERVER_BATCH):
        imgs = images[start:start + SERVER_BATCH]
        txts = prompts[start:start + SERVER_BATCH]
        inputs = proc(
            images=imgs, text=txts, return_tensors="pt",
            padding=True, truncation=True, max_length=256,
        ).to(device)
        outputs = model(**inputs)
        target_sizes = [(im.size[1], im.size[0]) for im in imgs]  # (h, w)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=target_sizes,
        )
        for j, res in enumerate(results):
            w, h = imgs[j].size
            boxes = []
            for box in res["boxes"].tolist():
                x1, y1, x2, y2 = box
                boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            out_boxes[start + j] = boxes
    return out_boxes


@app.get("/health")
def health():
    return {"status": "ok", "model": GROUNDING_DINO_HF_ID, "device": _STATE["device"]}


@app.post("/ground", response_model=GroundResponse)
def ground(req: GroundRequest):
    if len(req.images) != len(req.texts):
        return {"boxes": []}
    if not req.images:
        return {"boxes": []}
    images = [_decode(b) for b in req.images]
    boxes = _ground_batch(images, req.texts, req.box_threshold, req.text_threshold)
    return {"boxes": boxes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    import uvicorn

    _load()  # eager-load so /health is meaningful immediately and the first request is fast
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
