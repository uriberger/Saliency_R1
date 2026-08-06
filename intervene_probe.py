#!/usr/bin/env python
"""Causal intervention probe: does forcing an observe step's attention ONTO the
objects that step names change the answer?

Everything measured so far is correlational. Heads (22,28)/(22,31) were picked
because their `mean_in` CORRELATES with correctness, and three GRPO runs then raised
that correlate by 25-60% while benchmark accuracy fell. A correlate is not a cause: a
head can be a read-only indicator of a computation happening elsewhere, in which case
shaping it moves the metric and nothing else. This probe tests the causal claim
directly, with no training.

For each sample we take ONE greedy chain, segment its observe steps exactly as the
reward does, ground each step's text with Grounding-DINO, then re-run the forward with
the layer-L attention of that step's query rows replaced by the ideal map -- all the
image mass, inside the step's box -- and read off what happens to the answer.

THE INTERVENTION is a mixture, parameterised by alpha, over the IMAGE keys only:

    A'[k] = (1-a)*A[k] + a*m*T[k]     k in image keys      (T sums to 1 over them)
    A'[k] = A[k]                      k in text/sink keys  (untouched)
    m     = sum of A over the image keys

so the row still sums to 1, `image_mass` is exactly preserved at every alpha, and no
text or sink column is touched. a=0 is a no-op; a=1 is "all the image mass in the
box"; 0<a<1 keeps every weight strictly positive, i.e. on the manifold softmax can
actually produce -- which matters, because a null at a=1 alone could just mean layers
L+1.. were handed an activation they have never seen. Targets T:

    box    uniform over the step's DINO union            the supervision target
    roll   uniform over that union rolled to a random    matched-area, wrong-place
           offset, same area                             control -- THE key control
    shape  the step's own in-box weights, renormalised   all mass in box WITHOUT
                                                         forcing flatness inside it
    image  uniform over the whole image                  controls for "any
                                                         redistribution moves logits"
    perm   the row's image weights randomly permuted     does this head matter AT ALL

READOUTS, per (sample, layer, head-set, variant), paired against an alpha=0 baseline
run through the identical code path at the same layer:

    logp_gold   sum log P(gold answer tokens | prompt, chain)   primary: continuous
                and paired, so far more powerful than accuracy at these sample sizes
    first_top1  argmax at the first answer position             accuracy proxy

Read the box-vs-roll DIFFERENCE, not box alone. Only the gap between them is
location-specific; box alone also moves under "any large perturbation".

STAGES
  prepare   generate chains, segment, ground with DINO -> cases/shard*.json. Once.
  selftest  alpha=0 must reproduce the un-hooked forward, and alpha=1 must move it.
  run       the grid. Append-only JSONL, resumable at (case, layer, head, variant).
  report    per-(layer, head) causal table with paired bootstrap CIs.
  monitor   aggregate every shard's heartbeat into one ETA.

Stage 0 of the plan is `--head-mode layer` (all heads of a layer at once): it bounds
the search, because if forcing every head at layer L does nothing, no single head
there will. Stage 1 is `--head-mode each` on the layers Stage 0 flags.

    bash launch_intervene_probe.sh --stage prepare --n-samples 1000
    bash launch_intervene_probe.sh --stage run --layers 0-35 --head-mode layer
    bash launch_intervene_probe.sh --stage run --layers 12,18,22,26 --head-mode each
    python intervene_probe.py --stage report --out-dir <dir>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path, falling back to the central tree.

    Worktrees only symlink the paths in .worktree-links; gitignored corpora such as
    cold_data/grpo_sets/ exist in neither the worktree nor as a link.
    """
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    """Import a leaf module by path, bypassing the `trl` package __init__."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_iv_overlap_probe", "overlap_probe.py")
OSTEPS = PROBE.OSTEPS
OREW = PROBE.OREW
IMAGE_TOKEN_ID = PROBE.IMAGE_TOKEN_ID
CONDITIONS = ("box", "roll", "shape", "image", "perm")


def text_config(model):
    c = model.config
    return getattr(c, "text_config", c)


# ---------------------------------------------------------------------------
# progress: one heartbeat file per shard, so a single --monitor process can print
# a global ETA for a whole node (or two) with no IPC.
# ---------------------------------------------------------------------------
def fmt_dt(s):
    if s is None or not math.isfinite(s):
        return "?"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """Rolling-rate progress with an ETA, to stdout and to a heartbeat JSON.

    The rate is an EWMA rather than a simple average: the first units of a shard
    include the model load and are not representative of the steady state.
    """

    HEARTBEAT_SECS = 15.0

    def __init__(self, path: Path, total: int, label: str, log_every: int = 25,
                 already_done: int = 0):
        self.path, self.total, self.label = path, int(total), label
        self.log_every = max(1, int(log_every))
        self.done, self.resumed, self.rate = 0, int(already_done), None
        self.t0 = self.tlast = self.twrite = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        print(f"[{label}] starting: {self.resumed} already done of {self.total}", flush=True)

    @property
    def completed(self):
        return self.resumed + self.done

    def tick(self, n: int = 1):
        now = time.time()
        dt = (now - self.tlast) / max(1, n)
        self.tlast = now
        self.rate = dt if self.rate is None else 0.9 * self.rate + 0.1 * dt
        self.done += n
        # Count-based cadence alone makes a slow stage look dead: `prepare` runs at
        # ~7 s/sample, so log_every=25 is a heartbeat every three minutes and the
        # monitor reported 0/8 for an entire smoke run that was working fine. Write
        # on elapsed time as well, so the monitor's picture is never more than
        # HEARTBEAT_SECS stale whatever the per-unit cost of the stage.
        if (self.done % self.log_every == 0 or self.completed >= self.total
                or now - self.twrite >= self.HEARTBEAT_SECS):
            self.twrite = now
            self._write()
            print(self.line(), flush=True)

    def eta_seconds(self):
        if not self.rate or self.completed >= self.total:
            return 0.0
        return (self.total - self.completed) * self.rate

    def line(self):
        pct = 100.0 * self.completed / max(1, self.total)
        rate = (1.0 / self.rate) if self.rate else 0.0
        return (f"[{self.label}] {self.completed}/{self.total} ({pct:5.1f}%)  "
                f"{rate:.2f} it/s  elapsed {fmt_dt(time.time() - self.t0)}  "
                f"ETA {fmt_dt(self.eta_seconds())}")

    def _write(self):
        payload = {"label": self.label, "total": self.total, "completed": self.completed,
                   "rate": self.rate, "eta": self.eta_seconds(), "updated": time.time(),
                   "pid": os.getpid()}
        try:                                  # a heartbeat is never worth killing a run
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.path)
        except OSError:
            pass

    def close(self):
        self._write()
        print(self.line(), flush=True)


def monitor(out_dir: Path, interval: float, once: bool = False, stage: str = ""):
    """Aggregate every shard heartbeat into one progress line + ETA.

    `once` prints a single line and returns; the launcher uses it after `wait` so the
    final state is shown, which the polling loop otherwise misses (it gets killed the
    moment the shards exit).
    """
    beats = out_dir / "progress"
    if not once:
        print(f"[monitor] watching {beats} (ctrl-C to stop)", flush=True)
    while True:
        entries = []
        # Heartbeats are named <stage><shard>.json. Without the filter the monitor
        # sums a finished `prepare` into a running `run` and reports 100% at once.
        pattern = f"{stage}*.json" if stage else "*.json"
        for f in sorted(beats.glob(pattern)) if beats.is_dir() else []:
            try:
                entries.append(json.loads(f.read_text()))
            except Exception:
                continue                       # mid-write or torn: skip this round
        if not entries:
            print("[monitor] no heartbeats yet", flush=True)
            if once:
                return
            time.sleep(interval)
            continue
        now = time.time()
        tot = sum(e["total"] for e in entries)
        comp = sum(e["completed"] for e in entries)
        alive = [e for e in entries if now - e.get("updated", 0) <= 600]
        rates = [1.0 / e["rate"] for e in alive if e.get("rate")]
        # Shards run in parallel, so wall-clock ETA is the SLOWEST shard's, not the sum.
        etas = [(e["total"] - e["completed"]) * e["rate"] for e in alive if e.get("rate")]
        print(f"[monitor] {comp}/{tot} ({100.0 * comp / max(1, tot):5.1f}%)  "
              f"{sum(rates):.1f} it/s total  "
              f"{len(alive)}/{len(entries)} shards alive  "
              f"ETA {fmt_dt(max(etas) if etas else None)}", flush=True)
        if once:
            return
        if comp >= tot:
            print("[monitor] all shards complete", flush=True)
            return
        if not alive:
            # Every heartbeat has gone stale with work outstanding: the shards died.
            # Return rather than spin forever, so the launcher's foreground monitor
            # never wedges an interactive shell.
            print("[monitor] no shard has reported in 10 min and work remains -- "
                  "check the shard logs; re-running the same command resumes",
                  flush=True)
            return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# stage: prepare
# ---------------------------------------------------------------------------
def b64u8(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def unb64u8(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(shape)


@torch.no_grad()
def greedy_chain(processor, model, image, question, max_new_tokens, device):
    text = PROBE.build_prompt(processor, question)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=processor.tokenizer.pad_token_id)
    ids = out[0, prompt_len:].tolist()
    eos = processor.tokenizer.eos_token_id
    if eos in ids:
        ids = ids[: ids.index(eos) + 1]
    return inputs, prompt_len, ids


def prepare_shard(args, device):
    out = Path(args.out_dir)
    dest = out / "cases" / f"shard{args.shard:02d}.json"
    if dest.exists() and not args.overwrite:
        print(f"[prepare] {dest} exists -- nothing to do (--overwrite to redo)")
        return
    exclude = None
    if args.exclude_cases_dir:
        prev, _cfg, _fp = load_cases(Path(args.exclude_cases_dir))
        exclude = {c["row_index"] for c in prev}
        print(f"[prepare] excluding {len(exclude)} rows already used by "
              f"{args.exclude_cases_dir}")
    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag=f"_iv{args.shard}", split=args.split,
                              exclude=exclude)
    rows = rows[args.shard::args.num_shards]

    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=device)
    tok = processor.tokenizer
    OREW.configure(box_threshold=args.box_threshold, max_box_area=args.max_box_area,
                   dino_device=device, dino_batch_size=args.dino_batch_size)

    prog = Progress(out / "progress" / f"prepare{args.shard:02d}.json", len(rows),
                    f"prepare{args.shard}", args.log_every)
    cases, dropped = [], defaultdict(int)
    for row in rows:
        try:
            case = build_case(args, tok, processor, model, clf, row, device)
        except Exception as e:                  # one bad sample must not kill a shard
            print(f"[prepare] row {row['row_index']} failed: {type(e).__name__}: {e}",
                  flush=True)
            dropped["exception"] += 1
            prog.tick()
            continue
        if isinstance(case, str):
            dropped[case] += 1
        else:
            cases.append(case)
        prog.tick()
    prog.close()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": vars(args), "dropped": dict(dropped),
                                "cases": cases}))
    print(f"[prepare] shard {args.shard}: kept {len(cases)}, dropped {dict(dropped)}")
    print(f"[prepare] -> {dest}")


def build_case(args, tok, processor, model, clf, row, device):
    """One case, or a string naming the reason it was dropped."""
    inputs, prompt_len, comp_ids = greedy_chain(
        processor, model, row["image"], row["question"], args.max_new_tokens, device)
    if not PROBE.judge_format(tok.decode(comp_ids, skip_special_tokens=True)):
        return "bad_format"
    text = tok.decode(comp_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # Think span in the re-tokenised `out` space -- the space the reward's step spans
    # live in, and (the decode/encode round trip being idempotent here) the same
    # indices as the generated ids whose attention rows we will modify.
    enc = tok(text)
    ms = re.search(r"<think>\s*(\S\S*)", text, re.DOTALL | re.MULTILINE)
    me = re.search(r"(\S)\s*</think>", text, re.DOTALL | re.MULTILINE)
    if not ms or not me:
        return "no_think_span"
    ts_char, te_char = ms.start(1), me.start(1)
    ts, te = enc.char_to_token(0, ts_char), enc.char_to_token(0, te_char)
    if ts is None or te is None or te <= ts:
        return "bad_think_tokens"
    steps = OSTEPS.segment_observe_steps(text, ts_char, te_char, enc, 0, ts, te,
                                         row["question"], clf)
    if not steps:
        return "no_observe_steps"

    # Everything after `</think>` is replaced by the gold answer, so the readout is
    # log P(gold | prompt, chain) over a chain identical across every variant.
    #
    # Two things this has to get exactly right, both of which were wrong before:
    #
    #  1. Where the chain ENDS. char_to_token(cut) resolves the character AFTER
    #     `</think>`, i.e. the first token of the model's OWN answer, so taking
    #     cut and adding one left that token in the chain and the gold answer was
    #     appended to a model that had already answered. Ask for the LAST character
    #     of `</think>` instead.
    #  2. The SEPARATOR. The model writes "</think> answer" with a space; tokenising
    #     "\n" + gold both scored a token the model would never emit and put the real
    #     answer tokens off-distribution behind it. Take the separator from the
    #     model's own continuation, and score only the tokens that cover `gold` --
    #     located by offset mapping, because a tokeniser may merge the separator into
    #     the first answer token (" B" is one token, not two).
    cut = text.find("</think>")
    cut = cut + len("</think>") if cut >= 0 else len(text)
    end_tok = enc.char_to_token(0, cut - 1)          # token holding `>` of `</think>`
    chain_len = (end_tok + 1) if end_tok is not None else len(comp_ids)
    chain_len = min(max(chain_len, te + 1), len(comp_ids))
    chain_ids = comp_ids[:chain_len]
    gold = str(row["gt_answer"]).strip()
    if not gold:
        return "empty_gold"
    sep = re.match(r"\s*", text[cut:]).group(0)
    enc_a = tok(sep + gold, add_special_tokens=False, return_offsets_mapping=True)
    gold_ids = enc_a["input_ids"]
    offsets = enc_a["offset_mapping"]
    if not gold_ids:
        return "empty_gold"
    # first token whose span reaches past the separator into `gold` itself
    score_from = next((i for i, (a, b) in enumerate(offsets) if b > len(sep)), None)
    if score_from is None:
        return "empty_gold"

    gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
    gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
    boxes = OREW._dino_boxes([row["image"]] * len(steps), [s[0] for s in steps])
    kept = []
    for (stext, a, b), bx in zip(steps, boxes):
        if b <= a or b > len(chain_ids):
            continue
        mask = OREW._union_mask(bx or [], gh, gw)
        if mask is None:
            continue                # ungroundable or degenerate: nothing to force onto
        kept.append({"text": stext, "tok_a": int(a), "tok_b": int(b),
                     "union_frac": float(mask.mean()),
                     "mask_q": b64u8(mask.astype(np.uint8))})
    if not kept:
        return "no_grounded_steps"
    return {"row_index": row["row_index"], "dataset": row.get("dataset"),
            "question": row["question"], "gold": gold, "sep": sep,
            "chain_text": text[:cut], "chain_ids": chain_ids, "gold_ids": gold_ids,
            "score_from": int(score_from),
            "grid": [gh, gw], "steps": kept}


# ---------------------------------------------------------------------------
# stage: run -- the intervention
# ---------------------------------------------------------------------------
def repeat_v(hidden_states, n_rep):
    """GQA key/value expansion, verbatim from grpo_trainer_qwen3.repeat_v."""
    b, n_kv, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(
        b, n_kv * n_rep, s, d)


def causal_mask(seq, dtype, device):
    m = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
    add = torch.zeros(seq, seq, dtype=dtype, device=device)
    add.masked_fill_(m, torch.finfo(dtype).min)
    return add[None, None]


def step_target(mask_flat, w, kind, rng, gh, gw, device):
    """Target distribution over the image patches, [n_img] or [H,R,n_img]; sums to 1.

    `w` is [H, R, n_img], the rows' current image attention -- `shape` and `perm`
    are built from it, the others ignore it. Operating on this slice alone is what
    guarantees no text or sink column can ever be touched.
    """
    n = mask_flat.numel()
    if kind == "box":
        k = float(mask_flat.sum())
        return (mask_flat / k) if k else None
    if kind == "roll":
        m2 = mask_flat.view(gh, gw)
        m2 = torch.roll(m2, (int(rng.integers(0, gh)), int(rng.integers(0, gw))), (0, 1))
        k = float(m2.sum())
        return (m2.reshape(-1) / k) if k else None
    if kind == "image":
        return torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    if kind == "shape":
        t = w * mask_flat
        s = t.sum(-1, keepdim=True)
        k = float(mask_flat.sum())
        if k == 0:
            return None
        # rows with no mass inside the box have nothing to preserve -> fall back to box
        return torch.where(s > 0, t / s.clamp_min(1e-30), mask_flat / k)
    if kind == "perm":
        s = w.sum(-1, keepdim=True)
        perm = torch.tensor(rng.permutation(n), device=device, dtype=torch.long)
        return torch.where(s > 0, w[..., perm] / s.clamp_min(1e-30),
                           torch.full_like(w, 1.0 / n))
    raise ValueError(kind)


class Intervener:
    """Forward hook on layer L that rewrites selected heads' image attention.

    Re-runs that one attention module in eager mode -- the trick the trainer's
    `_compute_overlap_step_maps` uses to recover softmax weights flash/sdpa
    discards -- builds the modified weights in the same pass, rebuilds the module's
    output from them, and returns it in place of the original. One forward per
    variant: the targets that need the row's own weights (`shape`, `perm`) are
    constructed inside the hook, where those weights are already in hand.

    Only the observe-step query rows are rebuilt. The baseline is run through this
    same path at alpha=0, so the eager-vs-sdpa numerical difference on rebuilt rows
    is common to baseline and intervention and cancels in the paired readout.
    """

    def __init__(self, attn_mod, dtype):
        self.mod, self.dtype = attn_mod, dtype
        self.spec = None
        self.mask = None
        self._reentry = False
        self.n_rows = 0
        # When set, every forward records how far the rebuilt rows are from the
        # module's OWN eager output for those rows. At alpha=0 the two must agree to
        # bf16 rounding; that is the only check that isolates the rebuild, because it
        # is taken before anything propagates through layers L+1.. and the LM head.
        self.audit = False
        self.audit_stats = None
        self.handle = attn_mod.register_forward_hook(self._hook, with_kwargs=True)

    def close(self):
        self.handle.remove()

    def _hook(self, module, args, kwargs, output):
        if self._reentry:
            return None
        self._reentry = True
        kw = dict(kwargs)
        kw["attention_mask"] = self.mask
        kw["past_key_values"] = None            # never double-update the KV cache
        kw["use_cache"] = False
        prev = module.config._attn_implementation
        module.config._attn_implementation = "eager"
        try:
            _out, attn = module(*args, **kw)
        finally:
            module.config._attn_implementation = prev
            self._reentry = False
        if self.spec is None:
            return None
        sp = self.spec
        a = attn.clone()
        img = sp["img_cols"]
        all_rows = []
        for st in sp["steps"]:
            rows = st["rows"]
            w = a[0][:, rows][:, :, img].float()                    # [H_all, R, n_img]
            w_sel = w[sp["heads"]]                                  # [H, R, n_img]
            t = step_target(st["mask"], w_sel, sp["kind"], sp["rng"],
                            sp["gh"], sp["gw"], a.device)
            if t is None:
                continue
            m = w_sel.sum(-1, keepdim=True)
            new = (1.0 - sp["alpha"]) * w_sel + sp["alpha"] * m * t
            a[0, sp["heads"][:, None, None], rows[None, :, None], img[None, None, :]] = \
                new.to(a.dtype)
            all_rows.append(rows)
        if not all_rows:
            return None
        rows = torch.cat(all_rows)
        self.n_rows = int(rows.numel())
        hidden = args[0] if args else kwargs["hidden_states"]
        b, s, _ = hidden.shape
        v = module.v_proj(hidden).view(b, s, -1, module.head_dim).transpose(1, 2)
        v = repeat_v(v, module.num_key_value_groups)
        ctx = torch.matmul(a[:, :, rows, :], v)                     # [1, H_all, R, hd]
        ctx = ctx.transpose(1, 2).reshape(1, rows.numel(), -1)
        new_rows = module.o_proj(ctx)
        if self.audit:
            ref = _out[:, rows, :].float()
            err = (new_rows.float() - ref).abs()
            self.audit_stats = {
                "max_abs_err": float(err.max()),
                "mean_abs_err": float(err.mean()),
                "max_abs_ref": float(ref.abs().max()),
                "rel": float(err.max() / ref.abs().max().clamp_min(1e-12)),
                "n_rows": int(rows.numel()),
            }
        out0 = (output[0] if isinstance(output, tuple) else output).clone()
        out0[:, rows, :] = new_rows.to(out0.dtype)
        return (out0,) + tuple(output[1:]) if isinstance(output, tuple) else out0


@torch.no_grad()
def score_case(model, processor, iv, case, image, device, heads, alpha, kind, seed):
    """One teacher-forced forward with the intervention applied."""
    text = PROBE.build_prompt(processor, case["question"])
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    chain, gold = case["chain_ids"], case["gold_ids"]
    ids = torch.tensor([inputs["input_ids"][0].tolist() + chain + gold], device=device)
    seq = ids.shape[1]
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    gh, gw = case["grid"]
    if img_cols.numel() != gh * gw:
        return None                     # grid and image tokens disagree: skip, not guess

    steps = []
    for st in case["steps"]:
        a, b = prompt_len + st["tok_a"], prompt_len + st["tok_b"]
        if b > prompt_len + len(chain) or b <= a:
            continue
        steps.append({
            "rows": torch.arange(a, b, device=device),
            "mask": torch.tensor(unb64u8(st["mask_q"], (gh, gw)).astype(np.float32),
                                 device=device).reshape(-1),
        })
    if not steps:
        return None

    iv.mask = causal_mask(seq, iv.dtype, device)
    iv.spec = {"steps": steps, "img_cols": img_cols, "gh": gh, "gw": gw,
               "heads": torch.tensor(heads, device=device, dtype=torch.long),
               "alpha": float(alpha), "kind": kind,
               "rng": np.random.default_rng(seed)}
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, seq - prompt_len, dtype=torch.long, device=device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    try:
        out = model(**fwd)
    finally:
        iv.spec = None
        iv.mask = None

    return answer_readout(out.logits[0].float(), prompt_len + len(chain), case,
                          extra={"n_rows": iv.n_rows, "audit": iv.audit_stats})


@torch.no_grad()
def score_case_nohook(model, processor, case, image, device):
    """The un-hooked forward -- selftest only, to check the alpha=0 rebuild."""
    text = PROBE.build_prompt(processor, case["question"])
    inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                       padding=True, padding_side="left", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    chain, gold = case["chain_ids"], case["gold_ids"]
    ids = torch.tensor([inputs["input_ids"][0].tolist() + chain + gold], device=device)
    fwd = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in inputs:
        fwd["pixel_values"] = inputs["pixel_values"]
        fwd["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("mm_token_type_ids") is not None:
        pad = torch.zeros(1, ids.shape[1] - prompt_len, dtype=torch.long, device=device)
        fwd["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], pad], dim=1)
    return answer_readout(model(**fwd).logits[0].float(),
                          prompt_len + len(chain), case)


def load_cases(out: Path, shard: int = 0, num_shards: int = 1, max_cases: int = 0):
    """Cases from every shard file, plus the sampling config that produced them.

    `run` and `selftest` need the IMAGES, which the case files do not store -- they
    re-derive them with load_samples() and look each case up by row_index. That draw
    depends on (dataset, split, n_samples, seed), so taking those from the CLI lets a
    mismatched --n-samples silently find nothing: every case is skipped, `run` writes
    an empty results file while reporting 100% complete, and `selftest` runs zero
    checks and still prints PASS. Read them from the cases file instead, which is the
    only source that cannot disagree with the cases themselves.
    """
    files = sorted((out / "cases").glob("shard*.json"))
    if not files:
        raise SystemExit(f"no cases in {out / 'cases'} -- run --stage prepare first")
    cases, cfg = [], None
    for f in files:
        d = json.loads(f.read_text())
        cases.extend(d["cases"])
        c = {k: d["config"][k] for k in ("dataset", "split", "n_samples", "seed")}
        if cfg is not None and c != cfg:
            raise SystemExit(f"{f.name} was prepared with {c}, earlier shards with "
                             f"{cfg} -- the case files disagree, re-run prepare")
        cfg = c
    cases.sort(key=lambda c: c["row_index"])
    # Fingerprint the WHOLE corpus, before --max-cases and before sharding. Hashing
    # the shard's working slice instead made a calibration run (--max-cases 40) and
    # the full run disagree on identical cases, and the guard rejected the resume it
    # was written to protect.
    fp = hashlib.sha256(json.dumps(
        [[c["row_index"], c["chain_ids"], c["gold_ids"], c.get("score_from", 0),
          [[st["tok_a"], st["tok_b"], st["mask_q"]] for st in c["steps"]]]
         for c in cases], sort_keys=True).encode()).hexdigest()[:16]
    if max_cases:
        cases = cases[:max_cases]
    return cases[shard::num_shards], cfg, fp


def load_case_images(cfg, tag, row_indices=None):
    """Images for the cases, looked up by row_index.

    This used to re-run load_samples() with the config's (n_samples, seed) and hope
    the draw came out the same, which made every consumer depend on a random draw it
    had no reason to care about -- and silently found nothing when it did not match.
    `row_index` is already an index into the (post-carve) dataset, so index it
    directly. That also makes a disjoint confirmation draw a non-event: any sample
    resolves, whatever seed or size produced it.
    """
    from datasets import load_dataset, load_from_disk

    path = cfg["dataset"]
    if os.path.isfile(os.path.join(path, "state.json")):
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):
            ds = ds["train"]
    else:
        ds = load_dataset(path, split="train")
    if cfg["split"] != "all":
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"probe_split_iv{tag}"
        tmp.mkdir(parents=True, exist_ok=True)
        parts = ds.train_test_split(
            test_size=100, seed=42,
            train_indices_cache_file_name=str(tmp / "train.arrow"),
            test_indices_cache_file_name=str(tmp / "test.arrow"))
        ds = parts["train"] if cfg["split"] == "train" else parts["test"]
    out = {}
    for i in sorted(set(row_indices)) if row_indices is not None else range(len(ds)):
        if i < 0 or i >= len(ds):
            continue
        r = ds[int(i)]
        out[int(i)] = {"row_index": int(i), "dataset": r.get("dataset"),
                       "question": r["problem"], "gt_answer": r.get("solution"),
                       "image": PROBE.prepare_image(r["image"])}
    return out


def answer_readout(logits, answer_pos, case, extra=None):
    """log P(gold) and the top-1 at the first token that is actually part of `gold`.

    `score_from` skips the separator the model itself emitted between `</think>` and
    the answer: scoring it would mostly measure how confidently the model predicts a
    space, and comparing top-1 against it made `first_correct` identically 0 for
    every case in the first Stage-0 slice.
    """
    ids = case["gold_ids"]
    k = int(case.get("score_from", 0))
    scored = ids[k:]
    start = answer_pos + k                      # position of the first scored token
    lp = torch.log_softmax(logits[start - 1: start - 1 + len(scored)], dim=-1)
    tgt = torch.tensor(scored, device=logits.device)
    top1 = int(logits[start - 1].argmax())
    out = {"logp_gold": float(lp.gather(1, tgt[:, None]).sum()),
           "n_gold": len(scored),
           "top1_id": top1,
           "first_correct": int(top1 == scored[0])}
    if extra:
        out.update(extra)
    return out


def parse_layers(spec, n_layers):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in sorted(set(out)) if 0 <= l < n_layers]


def build_variants(args, n_heads):
    if args.head_mode == "layer":
        head_sets = [("all", list(range(n_heads)))]
    elif args.head_mode == "each":
        head_sets = [(str(h), [h]) for h in range(n_heads)]
    else:
        hs = [int(h) for h in args.head_mode.split(",")]
        head_sets = [(",".join(map(str, hs)), hs)]
    alphas = [float(a) for a in str(args.alphas).split(",")]
    kinds = [k.strip() for k in str(args.conditions).split(",") if k.strip()]
    bad = [k for k in kinds if k not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown --conditions {bad}; choose from {list(CONDITIONS)}")
    variants = []
    for hname, heads in head_sets:
        for kind in kinds:
            # only box/roll are dose-dependent; the controls are all-or-nothing
            for a in (alphas if kind in ("box", "roll") else [1.0]):
                variants.append({"heads": heads, "hname": hname, "kind": kind,
                                 "alpha": a, "name": f"{kind}_a{a:g}"})
    return variants


def run_shard(args, device):
    out = Path(args.out_dir)
    cases, cfg, fp = load_cases(out, args.shard, args.num_shards, args.max_cases)
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    tc = text_config(model)
    layers = parse_layers(args.layers, tc.num_hidden_layers)
    n_heads = tc.num_attention_heads
    variants = build_variants(args, n_heads)
    if not layers:
        raise SystemExit(f"--layers {args.layers} selected nothing (model has "
                         f"{tc.num_hidden_layers} layers)")

    res_path = out / "results" / f"shard{args.shard:02d}.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume is keyed on (row_index, layer, head, variant), and row_index is stable
    # across a re-prepare with the same seed -- so results computed against an OLDER
    # corpus would be silently reused if the cases were rebuilt. Fingerprint the
    # cases and refuse rather than mixing two corpora in one report.
    fp_path = out / "results" / f"fingerprint{args.shard:02d}.txt"
    if fp_path.exists() and fp_path.read_text().strip() != fp:
        raise SystemExit(
            f"results/shard{args.shard:02d}.jsonl was computed against a different "
            f"corpus (fingerprint {fp_path.read_text().strip()} != {fp}). The cases "
            "have been re-prepared since. Delete results/ or use a fresh --out-dir; "
            "resuming would mix two corpora in one report.")
    fp_path.write_text(fp)
    done = set()
    if res_path.exists():
        with res_path.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue         # a torn final line from a killed run: ignore it
                done.add((d["row_index"], d["layer"], d["hname"], d["variant"]))

    rows = load_case_images(cfg, f"_ivr{args.shard}",
                            [c["row_index"] for c in cases])
    missing = [c["row_index"] for c in cases if c["row_index"] not in rows]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(cases)} cases have no image (e.g. row {missing[0]}). "
            f"The cases were prepared with {cfg}; that draw no longer reproduces. "
            "Re-run --stage prepare rather than scoring a subset silently.")

    # +1 per (case, layer) for the alpha=0 baseline that pairs against every variant
    total = len(cases) * len(layers) * (1 + len(variants))
    print(f"[run] shard {args.shard}: {len(cases)} cases x {len(layers)} layers x "
          f"({len(variants)} variants + 1 baseline) = {total} forwards", flush=True)
    # `done` holds every key in this shard's results file, including grids from
    # earlier runs sharing the out-dir. Counting all of them against THIS grid's total
    # printed "22040/1740 (1266.7%)" once and, worse, an ETA of 0 with most of the work
    # outstanding -- twice it made a healthy run unreadable. Intersect with what this
    # invocation actually plans to do.
    planned = {(c["row_index"], L, hn, nm)
               for c in cases for L in layers
               for hn, nm in ([("-", "base")]
                              + [(v["hname"], v["name"]) for v in variants])}
    already = len(done & planned)
    if len(done) != already:
        print(f"[run] shard {args.shard}: {len(done) - already} records in the results "
              f"file belong to other grids and are not counted as progress", flush=True)
    prog = Progress(out / "progress" / f"run{args.shard:02d}.json", total,
                    f"run{args.shard}", args.log_every, already_done=already)

    fh = res_path.open("a")
    try:
        for ci, case in enumerate(cases):
            image = rows[case["row_index"]]["image"]
            seed = 1000003 * case["row_index"] + args.seed
            meta = {"row_index": case["row_index"],
                    "n_steps": len(case["steps"]),
                    "union": float(np.mean([s["union_frac"] for s in case["steps"]])),
                    "dataset": case.get("dataset")}
            all_heads = list(range(n_heads))
            for layer in layers:
                attn_mod = PROBE.find_attn_module(model, layer)
                if attn_mod is None:
                    raise SystemExit(f"no Qwen3VLTextAttention with layer_idx={layer}")
                iv = Intervener(attn_mod, next(model.parameters()).dtype)
                try:
                    todo = [({"heads": all_heads, "hname": "-", "kind": "box",
                              "alpha": 0.0, "name": "base"})] + variants
                    for v in todo:
                        key = (case["row_index"], layer, v["hname"], v["name"])
                        if key in done:
                            continue
                        try:
                            r = score_case(model, processor, iv, case, image, device,
                                           v["heads"], v["alpha"], v["kind"], seed)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"[run] OOM row {case['row_index']} L{layer} "
                                  f"{v['hname']} {v['name']}; skipped", flush=True)
                            r = None
                        if r is not None:
                            fh.write(json.dumps({**r, **meta, "layer": layer,
                                                 "hname": v["hname"],
                                                 "variant": v["name"]}) + "\n")
                        prog.tick()
                    fh.flush()
                finally:
                    iv.close()
            if ci % 20 == 0:
                torch.cuda.empty_cache()
    finally:
        fh.close()
        prog.close()
    print(f"[run] shard {args.shard} done -> {res_path}")


# ---------------------------------------------------------------------------
# stage: selftest
# ---------------------------------------------------------------------------
def selftest(args, device):
    """Gate: the rebuilt attention rows must equal the module's own eager output.

    Rebuilding a module's output from its attention weights means getting v_proj, the
    GQA expansion and o_proj exactly right. A mistake corrupts every number the probe
    produces and raises nothing, so this runs before the grid.

    The check is taken INSIDE the hook, comparing the rebuilt rows against the
    module's own eager output for the same rows. An earlier version compared
    end-to-end `log P(gold)` against an un-hooked forward instead, and failed at
    0.06-0.43 nats -- because the hook rebuilds its rows in EAGER attention while the
    un-hooked forward runs on sdpa, and in bf16 that difference propagates through
    layers L+1.. and the LM head. That gap says nothing about the rebuild, and it
    cancels in the real measurement (the baseline is hooked at alpha=0 over the same
    rows, so every variant carries the identical offset). It is reported below as
    context, never as a gate.
    """
    out = Path(args.out_dir)
    cases, cfg, _fp = load_cases(out)
    rows = load_case_images(cfg, "_ivs", [c["row_index"] for c in cases])
    cases = [c for c in cases if c["row_index"] in rows]
    if not cases:
        raise SystemExit(
            f"no case image could be resolved. The cases were prepared with {cfg}; "
            "that draw no longer reproduces, so the selftest would check nothing and "
            "still pass. Re-run --stage prepare.")
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    tc = text_config(model)
    layer = parse_layers(args.layers, tc.num_hidden_layers)[0]
    heads = list(range(tc.num_attention_heads))
    iv = Intervener(PROBE.find_attn_module(model, layer),
                    next(model.parameters()).dtype)
    ok, n = True, 0
    print(f"[selftest] layer {layer}, attn_impl={args.attn_impl}, "
          f"tol rel<{args.selftest_tol:g}")
    try:
        iv.audit = True
        for case in cases[:args.selftest_cases]:
            row = rows[case["row_index"]]
            a0 = score_case(model, processor, iv, case, row["image"], device,
                            heads, 0.0, "box", 0)
            if a0 is None or a0["audit"] is None:
                continue
            au = a0["audit"]
            ref = score_case_nohook(model, processor, case, row["image"], device)
            ctx = abs(a0["logp_gold"] - ref["logp_gold"])
            print(f"[selftest] row {case['row_index']:>6}: rows={au['n_rows']:>4}  "
                  f"REBUILD rel_err={au['rel']:.2e} "
                  f"(max {au['max_abs_err']:.2e} of {au['max_abs_ref']:.2e})   "
                  f"[context: end-to-end |d logp| vs un-hooked = {ctx:.2e}, "
                  f"top1 {'==' if a0['top1_id'] == ref['top1_id'] else '!='}]")
            if au["rel"] > args.selftest_tol:
                ok = False
            n += 1
        # The plan must actually reach the model: alpha=1 has to move the answer, and
        # it has to move it relative to the alpha=0 rebuild -- the same comparison the
        # experiment makes -- not relative to the un-hooked forward.
        iv.audit = False
        for case in cases[:1]:
            row = rows[case["row_index"]]
            b = score_case(model, processor, iv, case, row["image"], device,
                           heads, 0.0, "box", 0)
            a1 = score_case(model, processor, iv, case, row["image"], device,
                            heads, 1.0, "box", 0)
            d = a1["logp_gold"] - b["logp_gold"]
            print(f"[selftest] alpha=1 vs alpha=0, whole layer {layer}: "
                  f"d logp = {d:+.4f} (must be nonzero)")
            if abs(d) < 1e-6:
                ok = False
        # alpha=0 twice must be bit-identical: the pipeline has to be deterministic,
        # or a paired delta measures the sampler as much as the intervention.
        for case in cases[:1]:
            row = rows[case["row_index"]]
            r1 = score_case(model, processor, iv, case, row["image"], device,
                            heads, 0.0, "box", 0)
            r2 = score_case(model, processor, iv, case, row["image"], device,
                            heads, 0.0, "box", 0)
            d = abs(r1["logp_gold"] - r2["logp_gold"])
            print(f"[selftest] determinism: repeat of alpha=0 differs by {d:.2e}")
            if d > 1e-9:
                ok = False
    finally:
        iv.close()
    if n == 0:
        print("[selftest] FAIL -- zero checks ran; a gate that checks nothing is not "
              "a pass")
        return 1
    print(f"[selftest] PASS ({n} cases)" if ok else "[selftest] FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def gaps_by_head(out: Path):
    """-> {(layer, hname, alpha_tag): np.array of per-case (box - roll)}.

    Paired by row within a layer/head, which is what makes the eager-vs-sdpa offset
    and the bf16 matmul rounding cancel: both sides run the identical rebuild over
    the identical rows and differ only in the attention values.
    """
    per = defaultdict(dict)
    for f in sorted((out / "results").glob("shard*.jsonl")):
        with f.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r["variant"] == "base":
                    continue
                per[(r["layer"], r["hname"], r["row_index"])][r["variant"]] = r["logp_gold"]
    g = defaultdict(list)
    for (layer, hname, _row), v in per.items():
        for name in v:
            if not name.startswith("box_"):
                continue
            a = name.split("_", 1)[1]
            roll = "roll_" + a
            if roll in v:
                g[(layer, hname, a)].append(v[name] - v[roll])
    return {k: np.asarray(x) for k, x in g.items()}


def confirm_report(args):
    """Rank heads on --out-dir, report their effect on --confirm-dir.

    With 9 layers x 32 heads the best of 288 cells looks impressive whether or not
    any head has an effect; the confirmation set is what separates the two. It is
    single-use -- look at it, adjust, look again, and it is no longer held out.
    """
    sel = gaps_by_head(Path(args.out_dir))
    con = gaps_by_head(Path(args.confirm_dir))
    if not con:
        raise SystemExit(f"no results under {args.confirm_dir}/results")
    rng = np.random.default_rng(0)

    def ci(x):
        b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)])
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    ranked = sorted(sel.items(), key=lambda kv: -abs(kv[1].mean()))
    print(f"\nHELD-OUT CONFIRMATION   select: {args.out_dir}\n"
          f"                        confirm: {args.confirm_dir}")
    print(f"{'layer':>5} {'head':>6} {'a':>5} {'n_sel':>6} {'box-roll(sel)':>14} "
          f"{'n_con':>6} {'box-roll(CONFIRM)':>18} {'95% CI':>20} {'sign':>5}")
    kept = 0
    for (layer, hname, a), x in ranked[: args.top_heads]:
        y = con.get((layer, hname, a))
        if y is None or len(y) < 20:
            print(f"{layer:>5} {hname:>6} {a:>5} {len(x):>6} {x.mean():>+14.4f} "
                  f"{'-':>6} {'not in confirm dir':>18}")
            continue
        lo, hi = ci(y)
        same = np.sign(x.mean()) == np.sign(y.mean())
        kept += bool(same and (lo > 0 or hi < 0))
        print(f"{layer:>5} {hname:>6} {a:>5} {len(x):>6} {x.mean():>+14.4f} "
              f"{len(y):>6} {y.mean():>+18.4f} [{lo:>+.4f},{hi:>+.4f}] "
              f"{'yes' if same else 'NO':>5}")
    print(f"\n{kept}/{min(args.top_heads, len(ranked))} of the top-ranked heads keep "
          f"their sign AND exclude 0 on the confirmation set.")
    print("A head that does neither was selection noise; the selection-set number for "
          "it is not a measurement.")


def report(args):
    if args.confirm_dir:
        return confirm_report(args)
    out = Path(args.out_dir)
    recs = []
    for f in sorted((out / "results").glob("shard*.jsonl")):
        with f.open() as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not recs:
        raise SystemExit(f"no results under {out / 'results'}")
    base = {(r["row_index"], r["layer"]): r for r in recs if r["variant"] == "base"}
    by = defaultdict(list)
    for r in recs:
        if r["variant"] == "base":
            continue
        b = base.get((r["row_index"], r["layer"]))
        if b is None:
            continue
        by[(r["layer"], r["hname"], r["variant"])].append(
            (r["logp_gold"] - b["logp_gold"],
             r["first_correct"] - b["first_correct"], r["union"]))

    rng = np.random.default_rng(0)
    hdr = (f"{'layer':>5} {'head':>6} {'variant':<10} {'n':>6} {'d-logp':>9} "
           f"{'95% CI':>21} {'d-first%':>9} {'d-logp|tight':>13} {'n_tight':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for (layer, hname, variant), vals in sorted(by.items()):
        d = np.array([v[0] for v in vals])
        f = np.array([v[1] for v in vals])
        u = np.array([v[2] for v in vals])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        tight = d[u < args.tight_union]
        dt = float(tight.mean()) if len(tight) >= 20 else float("nan")
        rows.append(dict(layer=layer, head=hname, variant=variant, n=len(d),
                         dlogp=float(d.mean()), lo=lo, hi=hi, dfirst=float(f.mean()),
                         dlogp_tight=dt, n_tight=int(len(tight))))
        ci = "[{:+.4f},{:+.4f}]".format(lo, hi)
        print(f"{layer:>5} {hname:>6} {variant:<10} {len(d):>6} {d.mean():>+9.4f} "
              f"{ci:>21} {100 * f.mean():>+8.2f}% {dt:>+13.4f} {len(tight):>8}")

    # box - roll at matched alpha is the location-specific effect
    print("\nlocation-specific effect (box - roll, paired by row):")
    per = defaultdict(dict)
    for r in recs:
        if r["variant"] == "base":
            continue
        per[(r["layer"], r["hname"], r["row_index"])][r["variant"]] = r["logp_gold"]
    gaps = defaultdict(list)
    for (layer, hname, _row), v in per.items():
        for name in list(v):
            if name.startswith("box_"):
                roll = "roll_" + name.split("_", 1)[1]
                if roll in v:
                    gaps[(layer, hname, name.split("_", 1)[1])].append(v[name] - v[roll])
    print(f"{'layer':>5} {'head':>6} {'alpha':>7} {'n':>6} {'box-roll':>10} {'95% CI':>21}")
    for (layer, hname, a), vals in sorted(gaps.items()):
        d = np.array(vals)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        ci = "[{:+.4f},{:+.4f}]".format(np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        print(f"{layer:>5} {hname:>6} {a:>7} {len(d):>6} {d.mean():>+10.4f} {ci:>21}")

    (out / "report.json").write_text(json.dumps(rows, indent=1))
    print(f"\n-> {out / 'report.json'}")
    print("Read box-vs-roll, not box alone: only the gap is location-specific.\n"
          "A head whose `perm` row is ~0 does not affect the output at all, and "
          "nothing done to it can matter.")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="run",
                   choices=["prepare", "run", "report", "selftest", "monitor"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/set_a")))
    p.add_argument("--split", default="train", choices=["train", "holdout", "all"])
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--steps-ckpt", default=os.environ.get(
        "OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    p.add_argument("--box-threshold", type=float, default=0.10)
    p.add_argument("--max-box-area", type=float, default=0.5)
    p.add_argument("--dino-batch-size", type=int, default=8)
    p.add_argument("--layers", default="22", help="'22', '0-35', '12,18,22,26'")
    p.add_argument("--head-mode", default="layer",
                   help="'layer' (all heads at once, Stage 0), 'each' (one variant per "
                        "head, Stage 1), or an explicit set like '28,31'")
    p.add_argument("--conditions", default="box,roll",
                   help="comma list of: " + ", ".join(CONDITIONS))
    p.add_argument("--alphas", default="1.0",
                   help="mixture strengths for box/roll, e.g. '0.25,0.5,0.75,1.0'")
    p.add_argument("--tight-union", type=float, default=0.35,
                   help="report a separate delta over cases whose mean union is below "
                        "this -- forcing attention into a union that already covers "
                        "most of the image is barely an intervention")
    p.add_argument("--selftest-tol", type=float, default=1e-2,
                   help="max RELATIVE error of the rebuilt attention rows against the "
                        "module's own eager output; bf16 carries ~3 decimal digits")
    p.add_argument("--selftest-cases", type=int, default=3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--monitor-interval", type=float, default=30.0)
    p.add_argument("--monitor-stage", default="",
                   help="--stage monitor: only aggregate heartbeats of this stage "
                        "(prepare|run); otherwise a finished stage is summed in")
    p.add_argument("--once", action="store_true",
                   help="--stage monitor: print one aggregate line and exit")
    p.add_argument("--exclude-cases-dir", default="",
                   help="--stage prepare: draw rows DISJOINT from this out-dir's "
                        "cases, for a confirmation set that shares nothing with the "
                        "selection set")
    p.add_argument("--top-heads", type=int, default=20,
                   help="--stage report with --confirm-dir: how many of the "
                        "selection-ranked heads to carry to the confirmation set")
    p.add_argument("--confirm-dir", default="",
                   help="--stage report: rank heads on --out-dir, then report their "
                        "box-roll on this second out-dir (held-out confirmation)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "monitor":
        return monitor(Path(args.out_dir), args.monitor_interval, once=args.once,
                       stage=args.monitor_stage)
    if args.stage == "report":
        return report(args)
    if args.stage == "prepare":
        return prepare_shard(args, args.device)
    if args.stage == "selftest":
        return selftest(args, args.device)
    return run_shard(args, args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
