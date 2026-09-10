#!/usr/bin/env python
"""Does Qwen3-VL-8B show the pathologies LASER corrects? The go/no-go.

`docs/laser-go-no-go.md` is the design and holds the pre-registered thresholds;
`laser.py` is the measurement. This is the harness.

    python laser_probe.py --stage selftest --out-dir DIR --model M
    python laser_probe.py --stage collect  --out-dir DIR --model M --shard i --num-shards n
    python laser_probe.py --stage report   --out-dir DIR
    python laser_probe.py --stage monitor  --out-dir DIR

STAGES

  selftest  Gates every run. Six checks, and the second is the one that matters:
              - the collector edits nothing, so the model's greedy tokens and its logits
                are IDENTICAL to the unpatched run, not merely close. It delegates the
                forward to stock SDPA and takes its own softmax beside it, so anything
                other than equality is a bug rather than a tolerance question.
              - THE SLICE. `A[t, j]` is checked against transformers' own eager attention
                with `output_attentions=True` on the same forward. Every number in the
                report is a function of that array, and the Q/K recomputation that
                produces it is the only piece of arithmetic here that could be confidently
                wrong. Upstream needs 266 lines of mirrored module internals to do this;
                a registered attention implementation is handed post-RoPE, post-QK-norm
                `query` and `key` directly, and this check is what says so.
              - the picture is located where `sink_shift` independently locates it.
              - `sink_mask` uses torch's unbiased sd, which numpy does not default to.
              - the reward transcription reproduces upstream's torch arithmetic.
              - C1: `a_bos` is identical across a prompt's rollouts. It is a prompt-level
                statistic and attention is causal, so it CANNOT depend on the response --
                if it does, the prompt/response boundary is wrong and nothing downstream
                means anything.

  collect   `--n-prompts` prompts x `--n-rollouts` sampled rollouts at the trainer's own
            settings, then one teacher-forced forward per rollout. Sharded BY PROMPT, so a
            group's rollouts never split across shards and the within-group statistics are
            computable from any single shard's output. Append-only JSONL for resume, bulk
            arrays in sidecar npz parts.

  report    The collected counts, the budget, Finding 1, `R_vis`'s headroom, Finding 2,
            the border cross-check, then the verdict scored mechanically against
            docs/laser-go-no-go.md §4.

WHY IT SAMPLES INSTEAD OF DECODING GREEDILY. LASER's rewards are consumed by GRPO, which
learns from the spread WITHIN a group of rollouts on one prompt. A greedy pass would give
the level of each reward and say nothing about whether the optimiser can see it, and
docs/overlap-reward-hack-set-a.md is the record of what a small within-group spread does
when it is paired with a weight that assumes a large one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_module("_laser_overlap_probe", "overlap_probe.py")
IV = _load_module("_laser_intervene", "intervene_probe.py")
sys.path.insert(0, str(REPO))
import laser as LZ  # noqa: E402

#: run_grpo.sh, so a group here is the group the trainer would have formed.
TRAINER_ROLLOUTS = 8
TRAINER_TEMPERATURE = 1.0
TRAINER_MAX_COMPLETION = 512

#: The pair the overlap reward trained, carried so every head-resolved table can say what
#: LASER's layer-mean hides -- docs/sink-location-by-image-type.md §17.3 measured that
#: these two heads are more border-biased than the model's average head.
TRAINED_LAYER, TRAINED_HEADS = 22, (28, 31)

DEFAULT_DATASETS = ("cold_data/grpo_sets/val_natural", "cold_data/grpo_sets/val_nonnatural")

#: docs/laser-go-no-go.md §4. Written before the run; the report is a lookup against them.
THRESHOLDS = {
    "T1_decay_rho": -0.20,
    "T2_early_share": 0.50,
    "T3_vis_sd": 0.010,
    "T4_length_r": 0.30,
    "T5_sink_ratio": 1.50,
    "T6_supp_sd": 0.010,
    "T7_cv": 0.50,
    "T8_jaccard": 0.50,
}


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def load_prompts(args):
    """The prompt pool, drawn evenly from every dataset named. -> list of rows.

    Sorted by a stable key rather than left in draw order, so `--shard` cuts the same
    prompts into the same shards on a resume with a different shard count.
    """
    paths = [p if Path(p).is_absolute() else str(REPO / p) for p in args.dataset]
    per = max(1, args.n_prompts // max(1, len(paths)))
    rows = []
    for i, p in enumerate(paths):
        got = PROBE.load_samples(p, per, args.seed + i, cache_tag=f"_laser{i}",
                                 split="all")
        for r in got:
            r["set"] = Path(p).name
            r["key"] = f"{Path(p).name}-{r['row_index']:06d}"
        rows += got
    rows.sort(key=lambda r: r["key"])
    return rows[: args.n_prompts]


# ---------------------------------------------------------------------------
# one prompt: G rollouts, then G teacher-forced forwards
# ---------------------------------------------------------------------------
def score_prompt(model, processor, scan, row, device, args):
    """-> list of per-rollout dicts, one per sampled completion.

    Generation runs with the collector PAUSED, so it goes through the fused kernel at full
    speed; only the teacher-forced pass is measured. That pass is
    `overlap_probe.teacher_forced_case`, the same construction `grpo_trainer_qwen3.py`
    uses for its own reward, so `A[t, j]` is what a LASER reward inside our trainer would
    read and not an approximation of it.
    """
    import torch

    scan.paused = True
    try:
        inputs, prompt_len, seqs = PROBE.generate(
            processor, model, row["image"], row["question"], args.n_rollouts,
            args.max_new_tokens, args.temperature, device)
    finally:
        scan.paused = False

    texts = [processor.tokenizer.decode(ids, skip_special_tokens=True)
             for ids, _trunc in seqs]
    accs = PROBE.accuracy_reward([[{"content": t}] for t in texts],
                                 [row["gt_answer"]] * len(texts))
    out = []
    for g, ((ids, truncated), text, acc) in enumerate(zip(seqs, texts, accs)):
        if len(ids) < 2:
            continue
        case = PROBE.teacher_forced_case(inputs, ids, device)
        scan.arm(prompt_len)
        with torch.no_grad():
            model(**case, use_cache=False)
        res = scan.result()
        if res is None:
            continue
        got = LZ.score_rollout(res["A"], res["a_bos"], len(ids),
                               early_weighted=not args.flat_stability)
        gh, gw = res["grid"]
        fmt = float(PROBE.judge_format(text))
        acc = 0.0 if acc is None else float(acc)
        out.append({
            "rollout": g,
            "n_response": int(len(ids)),
            "truncated": bool(truncated),
            "grid": [gh, gw],
            "acc": acc,
            "format": fmt,
            "r_vis": got["r_vis"],
            "r_vis_flat": got["r_vis_flat"],
            "r_supp": got["r_supp"],
            "total": LZ.total_reward(acc, fmt, got["r_vis"], got["r_supp"]),
            "n_query": got["n_query"],
            "short": got["short"],
            "n_windows": got["n_windows"],
            "arg_window": got["arg_window"],
            "n_sinks": got["n_sinks"],
            "sink_frac": got["sink_frac"],
            "mean_alpha": got["mean_alpha"],
            "image_mass": got["image_mass"],
            "sink_ratio_mean": float(np.mean(got["sink_ratio"]))
                               if got["sink_ratio"].size else float("nan"),
            "decay_rho": LZ.decay_spearman(got["alpha"]),
            "sink_cv": LZ.column_cv(res["A"][LZ.response_query_slice(len(ids))],
                                    got["sinks"]),
            "ring": LZ.ring_agreement(got["sinks"], gh, gw),
            "_arrays": {
                "A": res["A"].astype(np.float16),
                "a_bos": np.asarray(res["a_bos"], dtype=np.float16),
                "sinks": got["sinks"].astype(np.int32),
                "alpha": got["alpha"].astype(np.float32),
            },
        })
    return out


# ---------------------------------------------------------------------------
# storage -- JSONL for resume, npz parts for the bulk
# ---------------------------------------------------------------------------
class Sink:
    """Append-only results, in the shape `sink_location_probe.Sink` uses.

    Resume is by unit key out of the JSONL; the arrays are flushed in parts so a killed
    shard loses at most one part rather than a shard's worth of GPU time. Every field
    carries its own per-unit shapes and indices because `A` is ragged -- a rollout's
    response length is whatever the model wrote.
    """

    INT_FIELDS = ("sinks",)

    def __init__(self, out_dir, stage, shard, flush_every=64):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage, self.shard = stage, shard
        self.jsonl = self.dir / f"{stage}_shard{shard}.jsonl"
        self.flush_every = int(flush_every)
        self.buf, self.part = [], 0
        while (self.dir / f"{stage}_shard{shard}_part{self.part}.npz").exists():
            self.part += 1
        self.fh = open(self.jsonl, "a")

    def done_prompts(self):
        if not self.jsonl.exists():
            return set()
        keys = set()
        for line in self.jsonl.read_text().splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue          # a torn last line from a killed job is not a result
        return keys

    def write(self, unit, meta, arrays):
        if arrays:
            self.buf.append((unit, arrays))
        self.fh.write(json.dumps(dict(meta, unit=unit)) + "\n")
        self.fh.flush()
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        packed = {"units": np.asarray([u for u, _ in self.buf])}
        for name in sorted({k for _u, a in self.buf for k in a}):
            have = [(i, a[name]) for i, (_u, a) in enumerate(self.buf)
                    if a.get(name) is not None]
            if not have:
                continue
            dtype = np.int32 if name in self.INT_FIELDS else np.float16
            packed[name] = np.concatenate([np.asarray(v).reshape(-1)
                                           for _i, v in have]).astype(dtype)
            packed[name + "__shapes"] = np.asarray([np.asarray(v).shape for _i, v in have],
                                                   dtype=np.int64)
            packed[name + "__idx"] = np.asarray([i for i, _v in have], dtype=np.int64)
        np.savez_compressed(
            self.dir / f"{self.stage}_shard{self.shard}_part{self.part}.npz", **packed)
        self.part += 1
        self.buf = []

    def close(self):
        self.flush()
        self.fh.close()


def read_stage(out_dir, stage="collect", want_arrays=True):
    d = Path(out_dir)
    meta = []
    for p in sorted(d.glob(f"{stage}_shard*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                meta.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    arrays = {}
    if want_arrays:
        for p in sorted(d.glob(f"{stage}_shard*_part*.npz")):
            with np.load(p, allow_pickle=False) as z:
                units = [str(u) for u in z["units"]]
                for name in z.files:
                    if name == "units" or name.endswith(("__shapes", "__idx")):
                        continue
                    flat, off = z[name], 0
                    for i, sh in zip(z[name + "__idx"], z[name + "__shapes"]):
                        n = int(np.prod(sh))
                        arrays.setdefault(units[int(i)], {})[name] = \
                            flat[off:off + n].reshape(sh)
                        off += n
    return meta, arrays


# ---------------------------------------------------------------------------
# stage: collect
# ---------------------------------------------------------------------------
def stage_collect(args):
    import torch

    rows = load_prompts(args)
    mine = rows[args.shard::args.num_shards]
    sink = Sink(args.out_dir, "collect", args.shard, args.flush_every)
    seen = sink.done_prompts()
    todo = [r for r in mine if r["key"] not in seen]
    prog = IV.Progress(Path(args.out_dir) / "progress" / f"collect{args.shard}.json",
                       len(mine), f"collect/{args.shard}",
                       already_done=len(mine) - len(todo))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    scan = LZ.install(model, want_cells=False)
    try:
        for r in todo:
            got = score_prompt(model, processor, scan, r, device, args)
            if not got:
                print(f"[collect] {r['key']}: no scorable rollout", flush=True)
                sink.write(r["key"], {"key": r["key"], "set": r["set"],
                                      "dataset": r.get("dataset"), "rollouts": []}, {})
                prog.tick()
                continue
            arrays = {}
            for g in got:
                for name, a in g.pop("_arrays").items():
                    arrays[f"{name}{g['rollout']}"] = a
            sink.write(r["key"], {
                "key": r["key"], "set": r["set"], "dataset": r.get("dataset"),
                "question_id": r.get("question_id"), "row_index": r["row_index"],
                "rollouts": got,
            }, arrays)
            prog.tick()
    finally:
        sink.close()
        scan.uninstall()
    prog.close()
    return 0


# ---------------------------------------------------------------------------
# stage: selftest
# ---------------------------------------------------------------------------
def stage_selftest(args):
    import torch

    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    rows = load_prompts(args)[: args.selftest_rows]
    if not rows:
        raise SystemExit("no prompts; check --dataset")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, args.adapter, device, "sdpa")
    print(f"\nselftest  model={args.model}  {len(rows)} prompts", flush=True)

    row = rows[0]
    text = PROBE.build_prompt(processor, row["question"])
    inputs = processor(text=[text], images=[[row["image"]]], return_tensors="pt",
                       padding=True, padding_side="left",
                       add_special_tokens=False).to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.selftest_tokens,
                             do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
    comp = gen[0][prompt_len:].tolist()
    case = PROBE.teacher_forced_case(inputs, comp, device)

    def logits_of():
        with torch.no_grad():
            return model(**case, use_cache=False).logits[0, -1].float().cpu()

    base_logits = logits_of()
    with torch.no_grad():
        base_ids = model.generate(**inputs, max_new_tokens=args.selftest_tokens,
                                  do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id)[0].tolist()

    scan = LZ.install(model, want_cells=True)
    try:
        # 1. the collector edits nothing -- and here that means EXACTLY nothing ---------
        # `sink_location`'s equivalent check has to allow a tolerance, because its scan
        # replaces the fused kernel with an explicit fp32 softmax. This one delegates the
        # forward to the same `_sdpa` the model would have used and computes its slice
        # beside it, so the only admissible answer is equality.
        scan.arm(prompt_len)
        scan_logits = logits_of()
        scan.paused = True
        try:
            with torch.no_grad():
                scan_ids = model.generate(
                    **inputs, max_new_tokens=args.selftest_tokens, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id)[0].tolist()
        finally:
            scan.paused = False
        d = float((scan_logits - base_logits).abs().max())
        check("the collector leaves the forward bit-identical", d == 0.0, f"max|dlogit| {d:.3e}")
        check("the collector picks the same greedy tokens", scan_ids == base_ids,
              f"{sum(a == b for a, b in zip(scan_ids, base_ids))}/{len(base_ids)} equal")

        # 2. THE SLICE, against transformers' own eager attention ---------------------
        scan.arm(prompt_len)
        with torch.no_grad():
            model(**case, use_cache=False)
        res = scan.result()
        check("a picture was located and a response block collected",
              res is not None and res["A"].size > 0,
              "" if res is None else f"A {res['A'].shape}, grid {res['grid']}")
        if res is not None:
            ref, ref_bos = _eager_reference(model, case, scan.img_cols, prompt_len)
            n = min(ref.shape[0], res["A"].shape[0])
            got = res["A"][:n].astype(np.float64)
            want = ref[:n]
            err = float(np.abs(got - want).max())
            rel = err / max(float(np.abs(want).max()), 1e-12)
            check("A[t, j] reproduces transformers' eager attention", rel < 5e-2,
                  f"max abs {err:.2e}, relative to peak {rel:.2e}, over {n} steps")
            b_err = float(np.abs(res["a_bos"].astype(np.float64) - ref_bos).max())
            b_rel = b_err / max(float(np.abs(ref_bos).max()), 1e-12)
            check("a_bos reproduces it at the sink-defining query", b_rel < 5e-2,
                  f"max abs {b_err:.2e}, relative to peak {b_rel:.2e}")
            # The set, not just the array: the sink rule is a 2-sigma cut, so two column
            # vectors can agree to 1e-4 and still disagree about which patches are in S.
            # S is what R_supp acts on, so S is what has to agree.
            s_got, s_want = LZ.sink_mask(res["a_bos"]), LZ.sink_mask(ref_bos)
            check("the SINK SET drawn from it is the same set",
                  np.array_equal(s_got, s_want),
                  f"|S| {int(s_got.sum())} vs {int(s_want.sum())} of "
                  f"{s_got.size} patches")

            # 3. the picture is where sink_shift independently says it is --------------
            import sink_shift as SS
            ss = SS.SinkShift(model, arm="centre", alpha=0.0)
            ss._locate_images(case["input_ids"], case["image_grid_thw"])
            check("the image columns agree with sink_shift's locator",
                  ss.img_cols is not None
                  and torch.equal(ss.img_cols.sort().values, scan.img_cols.sort().values),
                  f"{int(scan.img_cols.numel())} tokens, grid {res['grid']}")
            gh, gw = res["grid"]
            check("the grid accounts for every image token",
                  gh * gw == int(scan.img_cols.numel()), f"{gh}x{gw}")

        # 4. sink_mask uses torch's unbiased sd ---------------------------------------
        rng = np.random.default_rng(0)
        v = rng.random(160)
        t_thr = float(torch.tensor(v).mean() + 2 * torch.tensor(v).std())
        n_thr = float(v.mean() + 2 * v.std(ddof=1))
        check("sink_mask's threshold is torch's, not numpy's default",
              abs(t_thr - n_thr) < 1e-12
              and np.array_equal(LZ.sink_mask(v), v > t_thr),
              f"torch {t_thr:.6f} vs ddof=1 {n_thr:.6f}, ddof=0 "
              f"{float(v.mean() + 2 * v.std()):.6f}")

        # 5. the reward transcription reproduces upstream's torch arithmetic ------------
        a = torch.tensor(rng.random(137), dtype=torch.float64)
        check("R_vis matches the torch original", _vis_matches(a),
              f"{LZ.visual_grounding_reward(a.numpy()):.6f}")
        check("R_vis early-weighted matches the torch original",
              _vis_matches(a, early=True),
              f"{LZ.visual_grounding_reward(a.numpy(), early_weighted=True):.6f}")
        A = torch.tensor(rng.random((41, 160)), dtype=torch.float64)
        s = LZ.sink_mask(rng.random(160) ** 4)
        check("R_supp matches the torch original", _supp_matches(A, s),
              f"{LZ.sink_suppression_reward(A.numpy(), s):.6f}  |S|={int(s.sum())}")

        # 6. C1: a_bos cannot depend on the response ----------------------------------
        # It is the attention paid by the LAST PROMPT TOKEN, and attention is causal, so a
        # different completion must leave it untouched to the last bit. If it does not,
        # `prompt_len` is wrong and every sink set in the report is drawn from the wrong
        # query.
        # Sampled with `num_return_sequences`, which is how `collect` draws its group and
        # which expands the batch to G before the first forward. Running it here is also
        # the check that the collector's batch guard does not fire on the generation it is
        # asleep for -- the failure mode that killed the first run of this selftest.
        bos_list = []
        scan.paused = True
        try:
            with torch.no_grad():
                alt = model.generate(**inputs, max_new_tokens=args.selftest_tokens,
                                     do_sample=True, temperature=1.5, top_p=1.0, top_k=0,
                                     num_return_sequences=3,
                                     pad_token_id=processor.tokenizer.pad_token_id)
        finally:
            scan.paused = False
        check("a paused collector lets a batched `generate` through",
              alt.shape[0] == 3, f"{alt.shape[0]} rollouts sampled at once")
        for r_i in range(alt.shape[0]):
            ids = alt[r_i][prompt_len:].tolist()
            scan.arm(prompt_len)
            with torch.no_grad():
                model(**PROBE.teacher_forced_case(inputs, ids, device), use_cache=False)
            bos_list.append(scan.result()["a_bos"])
        same = all(np.array_equal(bos_list[0], b) for b in bos_list[1:])
        check("C1: a_bos is identical across a prompt's rollouts", same,
              f"{len(bos_list)} rollouts, max delta "
              f"{max(float(np.abs(bos_list[0] - b).max()) for b in bos_list[1:]):.3e}")
    finally:
        scan.uninstall()

    print(f"\n  {'SELFTEST PASS' if ok else 'SELFTEST FAIL'}")
    return 0 if ok else 1


def _eager_reference(model, case, img_cols, prompt_len, max_rows=24):
    """`A[t, j]` and `a_bos[j]` from `output_attentions=True`. -> ([rows, m], [m]) float64.

    The independent implementation, and the whole justification for the collector's
    design: this materialises `[1, H, T, T]` at EVERY layer at once, which is precisely
    the cost the Q/K path avoids. It is affordable here only because the selftest keeps
    the completion to a couple of dozen tokens -- which is also why both readouts come out
    of ONE forward rather than two.

    `torch.stack` over the tuple of per-layer tensors doubles the peak, so the layers are
    reduced one at a time instead.
    """
    import torch

    prev = _get_impl(model)
    _set_impl(model, "eager")
    try:
        with torch.no_grad():
            out = model(**case, use_cache=False, output_attentions=True)
        first = out.attentions[0]
        T = first.shape[-1]
        rows = torch.arange(prompt_len, min(T, prompt_len + max_rows),
                            device=first.device)
        body = bos = None
        for a in out.attentions:                          # each [1, H, T, T]
            hm = a[0].float().mean(dim=0)                 # head-mean -> [T, T]
            b = hm[rows][:, img_cols]
            z = hm[prompt_len - 1][img_cols]
            body = b if body is None else body + b
            bos = z if bos is None else bos + z
        n = float(len(out.attentions))
        return ((body / n).cpu().numpy().astype(np.float64),
                (bos / n).cpu().numpy().astype(np.float64))
    finally:
        _set_impl(model, prev)


def _get_impl(model):
    cfg = getattr(model.config, "text_config", None) or model.config
    return cfg._attn_implementation


def _set_impl(model, impl):
    cfg = getattr(model.config, "text_config", None) or model.config
    cfg._attn_implementation = impl
    for m in model.modules():
        if type(m).__name__ == "Qwen3VLTextAttention":
            m.config._attn_implementation = impl


def _vis_matches(alpha, early=False, tol=1e-9):
    """`laser.visual_grounding_reward` against a torch transcription of the original.

    Written out with `unfold` rather than reusing `laser.windows_of`, so the check is
    against upstream's own operation and not against this module's reading of it.
    """
    import torch

    t = alpha.shape[0]
    if t <= LZ.WINDOW_SIZE:
        return LZ.visual_grounding_reward(alpha.numpy()) == 0.0
    w = alpha.unfold(0, LZ.WINDOW_SIZE, LZ.WINDOW_SIZE // 2).mean(dim=-1)
    ratios = w / (w.max().detach() + 1e-12)
    per = torch.exp(-LZ.SENSITIVITY * (1.0 - ratios)) - LZ.STABILITY_PENALTY
    if early:
        n = w.shape[0]
        idx = torch.arange(n, dtype=per.dtype)
        raw = torch.exp(-LZ.DECAY_RATE * (idx / float(max(n - 1, 1))))
        per = per * (raw * (n / (raw.sum() + 1e-12)))
    want = float((per.sum() if not early else per.sum()) * LZ.STABILITY_SCALE)
    got = LZ.visual_grounding_reward(alpha.numpy(), early_weighted=early)
    return abs(got - want) < tol


def _supp_matches(A, sinks, tol=1e-9):
    import torch

    s = torch.tensor(np.asarray(sinks, dtype=bool))
    if s.sum() == 0:
        return LZ.sink_suppression_reward(A.numpy(), sinks) == 0.0
    ratio = A[:, s].mean(dim=-1) / (A.mean(dim=-1) + 1e-10)
    want = float(torch.exp(-LZ.SUPP_BETA * torch.clamp(ratio - LZ.TAU, min=0.0)).mean()
                 * LZ.SUPP_SCALE)
    return abs(LZ.sink_suppression_reward(A.numpy(), sinks) - want) < tol


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_mean(x, n_boot=10000, seed=20260910):
    """Mean with a bootstrap CI over the unit of analysis. NaNs are dropped, not zeroed."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)), int(x.size))


def boot_median(x, n_boot=10000, seed=20260910):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    return (float(np.median(x)), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)), int(x.size))


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def flatten(meta):
    """-> list of per-rollout dicts, each carrying its prompt's key as `group`."""
    out = []
    for m in meta:
        for r in m.get("rollouts", []):
            out.append(dict(r, group=m["key"], set=m.get("set"),
                            dataset=m.get("dataset")))
    return out


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def report_counts(rr, args):
    print("\n" + "=" * 78)
    print("1. WHAT WAS COLLECTED")
    groups = sorted({r["group"] for r in rr})
    n_short = sum(1 for r in rr if r["short"])
    lens = [r["n_response"] for r in rr]
    correct = [r for r in rr if r["acc"] > 0 and r["format"] > 0]
    print(f"    {len(groups)} prompts x up to {args.expect_rollouts} rollouts = "
          f"{len(rr)} scored rollouts")
    print(f"    accuracy      {np.mean([r['acc'] for r in rr]):.3f}      "
          f"format {np.mean([r['format'] for r in rr]):.3f}      "
          f"gated (acc AND format) {len(correct) / max(1, len(rr)):.3f}")
    print(f"    response length   median {int(np.median(lens))}   "
          f"p10 {int(np.percentile(lens, 10))}   p90 {int(np.percentile(lens, 90))}   "
          f"truncated {np.mean([r['truncated'] for r in rr]):.3f}")
    print(f"\n    SHORT ROLLOUTS (n_query <= {LZ.WINDOW_SIZE}, R_vis == 0 by "
          f"short-circuit): {n_short}/{len(rr)} = {n_short / max(1, len(rr)):.3f}")
    print(f"    windows per rollout: median "
          f"{int(np.median([r['n_windows'] for r in rr]))}")
    print("\n    The window is 10 with stride 5 -- the CALL SITE's value. The wiki page's")
    print("    20/10 would have doubled the share above and is not what the code runs.")
    return groups, correct


def report_budget(rr):
    print("\n" + "=" * 78)
    print("2. THE BUDGET -- how much of a generated token's attention row is on the picture")
    m, lo, hi, n = boot_mean([r["image_mass"] for r in rr])
    print(f"    image mass per generated token   {m:.4f}  [{lo:.4f}, {hi:.4f}]  n={n}")
    print("\n    docs/sink-location-by-image-type.md §16.1 measured 0.087 from the")
    print("    QUESTION's tokens at the cold start, and §17.2 measured the generated-token")
    print("    readout at roughly two thirds of the prefill one. `alpha` is a per-PATCH")
    print("    mean of this, so it is this number divided by the patch count.")
    mm, _lo, _hi, _n = boot_mean([r["mean_alpha"] for r in rr])
    print(f"    mean alpha (per non-sink patch)  {mm:.3e}")


def report_decay(rr, args):
    """Finding 1. T1 and T2 live here."""
    print("\n" + "=" * 78)
    print("3. FINDING 1 -- does visual attention decay over the chain?")
    long = [r for r in rr if not r["short"]]
    rho, lo, hi, n = boot_median([r["decay_rho"] for r in long])
    print(f"    per-rollout Spearman(alpha_t, t):  median {rho:+.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  n={n}")
    v = np.asarray([r["decay_rho"] for r in long], dtype=float)
    v = v[np.isfinite(v)]
    if v.size:
        print(f"      falling (rho < 0) in {np.mean(v < 0):.3f} of rollouts; "
              f"rho <= -0.2 in {np.mean(v <= -0.2):.3f}; "
              f"rho >= +0.2 in {np.mean(v >= 0.2):.3f}")
        print(f"      deciles  " + "  ".join(f"{x:+.2f}" for x in
                                             np.percentile(v, [10, 30, 50, 70, 90])))
    aw = [(r["arg_window"], r["n_windows"]) for r in long
          if r["arg_window"] is not None and r["n_windows"] >= 3]
    if aw:
        pos = np.asarray([a / max(1, w - 1) for a, w in aw], dtype=float)
        early = float(np.mean(pos <= 1 / 3))
        print(f"\n    where the PEAK window sits (0 = first, 1 = last), n={len(aw)}:")
        print(f"      first third {early:.3f}   middle "
              f"{float(np.mean((pos > 1 / 3) & (pos < 2 / 3))):.3f}   "
              f"last third {float(np.mean(pos >= 2 / 3)):.3f}")
        print("      R_vis scores every window against this peak, so a LATE peak means the")
        print("      reward is penalising the early tokens Finding 1 says matter most.")
    else:
        early = float("nan")
    return {"rho": rho, "rho_lo": lo, "rho_hi": hi, "early": early}


def report_vis(rr, groups, correct, args):
    """R_vis's headroom. T3 and T4 live here."""
    print("\n" + "=" * 78)
    print("4. R_vis -- can GRPO see it?")
    for name, sub in (("all rollouts", rr), ("gated (acc x format > 0)", correct)):
        if not sub:
            continue
        m, lo, hi, n = boot_mean([r["r_vis"] for r in sub])
        sd, ng = LZ.within_group_sd([r["r_vis"] for r in sub], [r["group"] for r in sub])
        print(f"    {name:<26} R_vis {m:.4f} [{lo:.4f}, {hi:.4f}]  n={n}")
        print(f"    {'':<26} within-group sd {sd:.5f} over {ng} groups   "
              f"x omega {LZ.OMEGA_VIS} -> {sd * LZ.OMEGA_VIS:.6f} reward units")
    sd_acc, _ = LZ.within_group_sd([r["acc"] for r in rr], [r["group"] for r in rr])
    print(f"\n    for scale: within-group sd of the ACCURACY term is {sd_acc:.5f}")
    gs = _groups_of(rr)
    flat_n = sum(1 for g in gs if len({(r["acc"], r["format"]) for r in g}) == 1)
    print(f"    groups where accuracy AND format are constant across all rollouts: "
          f"{flat_n}/{len(gs)} = {flat_n / max(1, len(gs)):.3f}")
    print("      In those, the attention terms are the ENTIRE advantage. That is the")
    print("      mechanism docs/overlap-reward-hack-set-a.md measured at 240x amplification.")
    # The other side of the same coin, and the real denominator behind T3 and T6. The
    # gate is multiplicative, so the attention terms can only RANK rollouts against each
    # other inside a group that has at least two gated ones. A group with one correct
    # rollout still has its advantage nudged, but there is nothing there for the reward to
    # discriminate between, which is what "learnable" was supposed to mean.
    rankable = sum(1 for g in gs if sum(1 for r in g if r["acc"] > 0 and r["format"] > 0) >= 2)
    print(f"\n    groups with >= 2 GATED rollouts -- the only ones in which an attention")
    print(f"    term can re-rank anything: {rankable}/{len(gs)} = "
          f"{rankable / max(1, len(gs)):.3f}")

    r_len = pearson([r["r_vis"] for r in rr], [r["n_response"] for r in rr])
    r_alpha = pearson([r["r_vis"] for r in rr], [r["mean_alpha"] for r in rr])
    r_flat = pearson([r["r_vis"] for r in rr], [r["r_vis_flat"] for r in rr])
    print(f"\n    r(R_vis, response length) {r_len:+.3f}   "
          f"r(R_vis, mean alpha) {r_alpha:+.3f}   r(early-weighted, flat) {r_flat:+.3f}")
    print("      The second is the one §1 of the design predicted would be near zero:")
    print("      target_level is the rollout's own peak, so R_vis rewards FLATNESS and is")
    print("      blind to whether the model looked at the picture much or barely at all.")
    sd_c, _ = LZ.within_group_sd([r["r_vis"] for r in correct],
                                 [r["group"] for r in correct]) if correct else (float("nan"), 0)
    return {"sd_correct": sd_c, "r_len": r_len, "r_alpha": r_alpha}


def _groups_of(rr):
    by = {}
    for r in rr:
        by.setdefault(r["group"], []).append(r)
    return list(by.values())


def report_sinks(rr, correct, args):
    """Finding 2. T5 and T6 live here."""
    print("\n" + "=" * 78)
    print("5. FINDING 2 -- is there a sink set, and does it pull more than its share?")
    fr, lo, hi, n = boot_mean([r["sink_frac"] for r in rr])
    print(f"    |S| / m               {fr:.4f}  [{lo:.4f}, {hi:.4f}]  n={n}   "
          f"(median |S| = {int(np.median([r['n_sinks'] for r in rr]))} patches)")
    ra, rlo, rhi, _n = boot_mean([r["sink_ratio_mean"] for r in rr])
    print(f"    aS / aV               {ra:.3f}  [{rlo:.3f}, {rhi:.3f}]   "
          f"tau = {LZ.TAU}, so anything above that is penalised")
    empty = np.mean([r["n_sinks"] == 0 for r in rr])
    print(f"    rollouts with an EMPTY sink set (R_supp == 0 by short-circuit): {empty:.3f}")
    for name, sub in (("all rollouts", rr), ("gated (acc x format > 0)", correct)):
        if not sub:
            continue
        m, mlo, mhi, mn = boot_mean([r["r_supp"] for r in sub])
        sd, ng = LZ.within_group_sd([r["r_supp"] for r in sub], [r["group"] for r in sub])
        print(f"    {name:<26} R_supp {m:.4f} [{mlo:.4f}, {mhi:.4f}]  n={mn}")
        print(f"    {'':<26} within-group sd {sd:.5f} over {ng} groups   "
              f"x omega {LZ.OMEGA_SUPP} -> {sd * LZ.OMEGA_SUPP:.6f} reward units")
    sd_c, _ = LZ.within_group_sd([r["r_supp"] for r in correct],
                                 [r["group"] for r in correct]) if correct else (float("nan"), 0)
    return {"ratio": ra, "ratio_lo": rlo, "sd_correct": sd_c, "sink_frac": fr}


def report_border(rr, args):
    """The cross-check the validation ladder asks for. T7 and T8 live here."""
    print("\n" + "=" * 78)
    print("6. IS `S` THE BORDER, AND DOES IT MOVE WITH THE QUERY?")
    have = [r for r in rr if r["n_sinks"] > 0 and r.get("ring")]
    if not have:
        print("    no rollout had a non-empty sink set")
        return {"jaccard": float("nan"), "cv": float("nan")}
    j, jlo, jhi, jn = boot_mean([r["ring"]["jaccard"] for r in have])
    on, olo, ohi, _n = boot_mean([r["ring"]["sink_on_ring"] for r in have])
    en, elo, ehi, _n = boot_mean([r["ring"]["enrichment"] for r in have])
    area, _lo, _hi, _n = boot_mean([r["ring"]["ring_area"] for r in have])
    print(f"    Jaccard(S, ring)        {j:.3f}  [{jlo:.3f}, {jhi:.3f}]  n={jn}")
    print(f"    share of S on the ring  {on:.3f}  [{olo:.3f}, {ohi:.3f}]   "
          f"(the ring is {area:.3f} of the patches)")
    print(f"    enrichment of S there   {en:.3f}  [{elo:.3f}, {ehi:.3f}]   1.0 = chance")
    cv, clo, chi, cn = boot_mean([r["sink_cv"] for r in have])
    print(f"\n    across-query CV of the S columns  {cv:.3f}  [{clo:.3f}, {chi:.3f}]  n={cn}")
    print(f"      Pre-registered at {THRESHOLDS['T7_cv']}, taken unchanged from")
    print("      docs/sink-location-by-image-type.md §16.4, which measured 1.2-2.6 for the")
    print("      border and concluded the ring is a PEAK and not a sink. A sink needs")
    print("      magnitude AND query-invariance; R_supp's premise is the second one.")
    return {"jaccard": j, "cv": cv, "on_ring": on, "enrichment": en}


def report_verdict(facts):
    print("\n" + "=" * 78)
    print("7. THE VERDICT -- docs/laser-go-no-go.md §4, scored mechanically")
    T = THRESHOLDS
    vis_sd = facts["vis"]["sd_correct"] * LZ.OMEGA_VIS
    supp_sd = facts["sink"]["sd_correct"] * LZ.OMEGA_SUPP
    # Each row carries the number its verdict is read off, so a statistic that could not
    # be estimated prints `n/a` instead of a FAIL. A threshold nothing was measured
    # against has not been failed, and reading one as a failure is how a thin run turns
    # into a confident no-go.
    rows = [
        ("T1", "visual attention decays over the chain", facts["decay"]["rho"],
         f"median rho {facts['decay']['rho']:+.3f} <= {T['T1_decay_rho']}"
         f" and CI [{facts['decay']['rho_lo']:+.3f}, {facts['decay']['rho_hi']:+.3f}] "
         f"excludes 0",
         facts["decay"]["rho"] <= T["T1_decay_rho"] and facts["decay"]["rho_hi"] < 0),
        ("T2", "the decay is early", facts["decay"]["early"],
         f"peak in the first third for {facts['decay']['early']:.3f} "
         f">= {T['T2_early_share']}",
         facts["decay"]["early"] >= T["T2_early_share"]),
        ("T3", "R_vis is learnable", vis_sd,
         f"within-group sd x omega = {vis_sd:.6f} >= {T['T3_vis_sd']}",
         vis_sd >= T["T3_vis_sd"]),
        ("T4", "R_vis is not just a length reward", facts["vis"]["r_len"],
         f"|r(R_vis, length)| = {abs(facts['vis']['r_len']):.3f} <= {T['T4_length_r']}",
         abs(facts["vis"]["r_len"]) <= T["T4_length_r"]),
        ("T5", "sinks pull more than their share", facts["sink"]["ratio"],
         f"aS/aV = {facts['sink']['ratio']:.3f} >= {T['T5_sink_ratio']}",
         facts["sink"]["ratio"] >= T["T5_sink_ratio"]),
        ("T6", "R_supp is learnable", supp_sd,
         f"within-group sd x omega = {supp_sd:.6f} >= {T['T6_supp_sd']}",
         supp_sd >= T["T6_supp_sd"]),
        ("T7", "the sinks are sinks, not peaks", facts["border"]["cv"],
         f"across-query CV {facts['border']['cv']:.3f} <= {T['T7_cv']}",
         facts["border"]["cv"] <= T["T7_cv"]),
        ("T8", "S is not simply the border", facts["border"]["jaccard"],
         f"Jaccard(S, ring) {facts['border']['jaccard']:.3f} <= {T['T8_jaccard']}",
         facts["border"]["jaccard"] <= T["T8_jaccard"]),
    ]
    verdict = {}
    for tag, claim, value, detail, good in rows:
        estimated = np.isfinite(value)
        verdict[tag] = bool(good) if estimated else None
        print(f"    {'PASS' if good else ('n/a ' if not estimated else 'FAIL')}  "
              f"{tag}  {claim}")
        print(f"           {detail}")
    def _arm(tags):
        bad = [t for t in tags if verdict[t] is False]
        return "GO" if not bad else f"NO-GO (failed {', '.join(bad)})"

    vis_ok = all(verdict[t] for t in ("T1", "T3", "T4"))
    supp_ok = all(verdict[t] for t in ("T5", "T6", "T7"))
    unmeasured = [t for t, v in verdict.items() if v is None]
    print("\n    R_vis  ->", _arm(("T1", "T3", "T4")))
    print("    R_supp ->", _arm(("T5", "T6", "T7")))
    # T8 as pre-registered asks its question with the wrong statistic, and saying so is
    # not the same as moving it. Jaccard is held down by a SIZE mismatch: if S is a
    # handful of patches and the ring is a third of the grid, S can sit entirely on the
    # border and still score near zero. The containment number is what answers "is S the
    # border", it was collected, and it is reported here beside the verdict rather than
    # instead of it. The threshold stands as written.
    on_ring = facts["border"].get("on_ring", float("nan"))
    if verdict["T8"] and np.isfinite(on_ring) and on_ring >= 0.9:
        print("\n    T8 PASSES ON A STATISTIC THAT DOES NOT ANSWER ITS OWN QUESTION.")
        print(f"    {on_ring:.3f} of S sits on the border ring, at "
              f"{facts['border']['enrichment']:.2f}x its area share, so S IS the")
        print(f"    border -- a strict subset of it. Jaccard is only "
              f"{facts['border']['jaccard']:.3f} because S is far")
        print("    smaller than the ring, and Jaccard charges for that size mismatch.")
        print("    The pre-registered verdict is left as recorded; the substantive")
        print("    reading is the opposite of it, and T8 should be containment rather")
        print("    than Jaccard if this is ever run again.")
    if unmeasured:
        print(f"    NOT ESTIMATED: {', '.join(unmeasured)} -- a GO/NO-GO that leans on "
              f"one of these\n    is not supported by this run.")
    if not supp_ok and verdict["T7"] is False:
        print("\n    §5 of the design named this outcome in advance: if S is large but")
        print("    query-dependent, R_supp rewards looking away from the tokens the")
        print("    model's own queries are aimed at, and the correctness gate does not")
        print("    catch it because a rollout that was already right still gets paid.")


def stage_report(args):
    meta, _arrays = read_stage(args.out_dir, "collect", want_arrays=False)
    rr = flatten(meta)
    if not rr:
        print(f"no rollouts under {args.out_dir}")
        return 1
    print(f"{len(rr)} rollouts from {args.out_dir}")
    by_set = {}
    for r in rr:
        by_set.setdefault(r["set"], 0)
        by_set[r["set"]] += 1
    print("   " + "   ".join(f"{k}: {v}" for k, v in sorted(by_set.items())))

    groups, correct = report_counts(rr, args)
    report_budget(rr)
    facts = {"decay": report_decay(rr, args)}
    facts["vis"] = report_vis(rr, groups, correct, args)
    facts["sink"] = report_sinks(rr, correct, args)
    facts["border"] = report_border(rr, args)
    report_verdict(facts)

    print("\n" + "=" * 78)
    short = np.mean([r["short"] for r in rr])
    gated = np.mean([r["acc"] > 0 and r["format"] > 0 for r in rr])
    print(f"Read section 1 first. {short:.1%} of rollouts score R_vis = 0 by the "
          f"short-circuit,\nand only {gated:.1%} clear the accuracy-and-format gate that "
          f"lets either attention term\nreach the reward at all. Both bound everything "
          f"section 7 says.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "collect", "report", "monitor"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dataset", default=",".join(DEFAULT_DATASETS),
                    help="comma-separated; prompts are drawn evenly from each")
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--n-rollouts", type=int, default=TRAINER_ROLLOUTS,
                    help="rollouts per prompt. run_grpo.sh uses 8, and the within-group "
                         "spread that decides T3/T6 is only meaningful at the trainer's own "
                         "group size")
    ap.add_argument("--temperature", type=float, default=TRAINER_TEMPERATURE)
    ap.add_argument("--max-new-tokens", type=int, default=TRAINER_MAX_COMPLETION)
    ap.add_argument("--flat-stability", action="store_true",
                    help="score R_vis with the unweighted variant. train.sh sets "
                         "APPLY_EARLY_WEIGHTED_STABILITY=True, so the weighted one is the "
                         "default here; both are collected either way")
    ap.add_argument("--flush-every", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--selftest-rows", type=int, default=2)
    ap.add_argument("--selftest-tokens", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.dataset = [d for d in args.dataset.split(",") if d]
    args.expect_rollouts = args.n_rollouts

    if args.stage == "report":
        return stage_report(args)
    if args.stage == "monitor":
        IV.monitor(Path(args.out_dir), args.interval, args.once, "")
        return 0
    if not args.model:
        raise SystemExit("--model is required for this stage")
    if args.stage == "selftest":
        return stage_selftest(args)
    return stage_collect(args)


if __name__ == "__main__":
    sys.exit(main())
