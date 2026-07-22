"""Observe-step segmentation for the attention-overlap GRPO reward (reward_variant="ours").

Ported from the vlm_reasoning repo (steps_classifier/ + grpo/reward.py). Two pieces:

  1. OverlapStepsClassifier — the FLAN-T5-base POD step classifier (T5 encoder +
     mean-pooling + MLP head), inference-only. Labels a reasoning-step fragment as
     plan / observe / deduce / none. Checkpoint layout is the one produced by
     steps_classifier/train_classifier.py:StepClassifier.save().

  2. segment_observe_steps() — splits the freeform <think>…</think> text into
     sentences, classifies each, and returns the token spans (in the trainer's
     `out` tokenisation space) of the sentences labelled "observe".

Saliency-R1 completions are freeform <think>…</think> (no <observe>/<step> tags),
so segmentation is sentence-split + classifier (mirrors extract_steps() fallback in
steps_classifier/generate_data.py). The classifier was trained across prompt formats;
whether it segments this output format well is validated separately (see the
grpo-reward-port-plan memory, TODO 2) — this module does not block on that.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn


@contextlib.contextmanager
def _no_deepspeed_zero3_init():
    """Temporarily hide HF's global ZeRO-3 config from ``from_pretrained``.

    Under DeepSpeed ZeRO-3 (accelerate ``zero3_init_flag: true``) transformers
    registers a process-global HfDeepSpeedConfig; every subsequent
    ``from_pretrained`` — including this auxiliary T5 encoder — then gets wrapped
    in ``deepspeed.zero.Init`` and has its parameters partitioned into 1-D shards.
    A sharded ``embed_tokens.weight`` is no longer 2-D, so the encoder's embedding
    lookup raises ``RuntimeError: 'weight' must be 2-D`` at inference. This
    classifier is a small, frozen, single-device model that must be fully
    materialised, so we null the weakref for the duration of the load and restore
    it afterwards (no-op if transformers has no deepspeed integration).
    """
    try:
        import transformers.integrations.deepspeed as _ds
    except Exception:
        yield
        return
    saved = getattr(_ds, "_hf_deepspeed_config_weak_ref", None)
    _ds._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        _ds._hf_deepspeed_config_weak_ref = saved

# POD taxonomy + catch-all. Order must match steps_classifier training (LABELS).
LABELS = ("plan", "observe", "deduce", "none")
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_STEP_SEP = "[STEP]"
_CHAIN_SEP = "[CHAIN]"

# Default checkpoint: the trained best classifier in the vlm_reasoning repo (same
# lustre filesystem, stable absolute path). Override with OVERLAP_STEPS_CKPT.
_DEFAULT_CKPT = (
    "/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/"
    "vlm_reasoning/steps_classifier/checkpoints/best"
)


def _build_input(step_text: str, chain: str, question: str, include_chain: bool) -> str:
    """Reproduce steps_classifier.train_classifier._build_input exactly."""
    parts = [_STEP_SEP, step_text.strip()]
    if include_chain:
        ctx = f"Question: {question.strip()} Chain: {chain.strip()}" if question.strip() else chain.strip()
        parts += [_CHAIN_SEP, ctx]
    return " ".join(parts)


class OverlapStepsClassifier(nn.Module):
    """T5 encoder with attention-weighted mean pooling and an MLP head (inference-only)."""

    def __init__(self, encoder, num_labels: int, include_chain: bool):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.1),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d // 2, num_labels),
        )
        self.include_chain = include_chain
        self._tokenizer = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (enc_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)

    @classmethod
    def load(cls, path: str | None = None, device: str | None = None, num_labels: int = 4):
        from transformers import AutoTokenizer, T5EncoderModel

        ckpt = Path(path or os.environ.get("OVERLAP_STEPS_CKPT", _DEFAULT_CKPT))
        cfg = json.loads((ckpt / "cfg.json").read_text()) if (ckpt / "cfg.json").exists() else {}
        include_chain = bool(cfg.get("include_chain", True))

        # apex may replace T5LayerNorm with a FusedLayerNorm whose CUDA extension can
        # be broken; fall back to a plain-PyTorch RMS-norm (mirrors StepClassifier.load).
        import importlib
        import transformers.models.t5.modeling_t5 as _t5_mod
        _apex_broken = False
        try:
            importlib.import_module("fused_layer_norm_cuda")
        except ImportError:
            _apex_broken = True
        _orig_t5_ln = _t5_mod.T5LayerNorm
        if _apex_broken:
            class _FallbackT5LN(torch.nn.Module):
                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                    self.variance_epsilon = eps

                def forward(self, x):
                    v = x.float().pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(v + self.variance_epsilon)
                    if self.weight.dtype in (torch.float16, torch.bfloat16):
                        x = x.to(self.weight.dtype)
                    return self.weight * x
            _t5_mod.T5LayerNorm = _FallbackT5LN
        # Load fully materialised: never let DeepSpeed ZeRO-3 partition this
        # auxiliary encoder (would 1-D-shard embed_tokens.weight -> "'weight'
        # must be 2-D" in the embedding lookup).
        with _no_deepspeed_zero3_init():
            encoder = T5EncoderModel.from_pretrained(ckpt / "encoder")
        if _apex_broken:
            _t5_mod.T5LayerNorm = _orig_t5_ln

        obj = cls(encoder, num_labels=num_labels, include_chain=include_chain)
        obj.head.load_state_dict(torch.load(ckpt / "head.pt", map_location="cpu"))
        obj._tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.eval().to(device)
        return obj

    @torch.no_grad()
    def predict(self, step_text: str, chain: str, question: str) -> str:
        inp = _build_input(step_text, chain, question, self.include_chain)
        enc = self._tokenizer(inp, return_tensors="pt", truncation=True, max_length=512)
        device = next(self.parameters()).device
        logits = self(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        return ID2LABEL[int(logits.argmax(dim=-1).item())]


# ---------------------------------------------------------------------------
# Sentence splitting + observe-step segmentation
# ---------------------------------------------------------------------------

_SENT_SEP = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences_with_spans(text: str, base_offset: int = 0, min_len: int = 10):
    """Split into sentences, returning (sentence_text, char_start, char_end) tuples.

    char_start/char_end are absolute positions (base_offset + local index) of the
    stripped sentence within the original string. Mirrors
    steps_classifier.generate_data._split_sentences (same regex, same min length).
    """
    spans = []
    pos = 0
    for m in _SENT_SEP.finditer(text):
        seg = text[pos:m.start()]
        stripped = seg.strip()
        if len(stripped) >= min_len:
            lstrip = len(seg) - len(seg.lstrip())
            s = pos + lstrip
            spans.append((stripped, base_offset + s, base_offset + s + len(stripped)))
        pos = m.end()
    seg = text[pos:]
    stripped = seg.strip()
    if len(stripped) >= min_len:
        lstrip = len(seg) - len(seg.lstrip())
        s = pos + lstrip
        spans.append((stripped, base_offset + s, base_offset + s + len(stripped)))
    return spans


def _char_to_tok(out, case_id: int, char_idx: int, total_chars: int):
    """out.char_to_token, walking forward over whitespace gaps that map to None."""
    for c in range(char_idx, min(char_idx + 8, total_chars)):
        t = out.char_to_token(case_id, c)
        if t is not None:
            return t
    return None


def segment_observe_steps(
    output_text: str,
    think_start_char: int,
    think_end_char: int,
    out,
    case_id: int,
    tok_lo: int,
    tok_hi: int,
    question: str,
    classifier: "OverlapStepsClassifier",
):
    """Return observe-step token spans [(step_text, tok_a, tok_b), ...].

    Token spans are half-open in the trainer's `out` tokenisation space, clamped to
    [tok_lo, tok_hi + 1] (the think-token range whose attention rows were extracted).
    Only sentences the classifier labels "observe" are returned. Steps whose span is
    empty after clamping are dropped.
    """
    if think_start_char < 0 or think_end_char < think_start_char:
        return []
    think_text = output_text[think_start_char:think_end_char + 1]
    sentences = split_sentences_with_spans(think_text, base_offset=think_start_char)
    if not sentences:
        return []

    total_chars = len(output_text)
    result = []
    for sent_text, cs, ce in sentences:
        if classifier.predict(sent_text, think_text, question) != "observe":
            continue
        tok_a = _char_to_tok(out, case_id, cs, total_chars)
        tok_b_incl = out.char_to_token(case_id, ce - 1)
        if tok_b_incl is None:
            tok_b_incl = _char_to_tok(out, case_id, ce - 1, total_chars)
        if tok_a is None or tok_b_incl is None:
            continue
        tok_a = max(tok_a, tok_lo)
        tok_b = min(tok_b_incl + 1, tok_hi + 1)
        if tok_b > tok_a:
            result.append((sent_text, tok_a, tok_b))
    return result
