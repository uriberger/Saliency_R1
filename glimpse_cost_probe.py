#!/usr/bin/env python3
"""What would a GLIMPSE-based grounding reward cost per GRPO step?

Deliverable 1 of docs/glimpse-reward-brief.md: a measurement, before any reward code.
GLIMPSE (docs/saliency-maps.md map 6) costs one backward per TARGET TOKEN plus one eager
replay of every propagated layer for that token, where the incumbent gradient reward
(`trl/grad_maps.py`) amortises one vmapped backward over a chunk of whole STEPS. That is
a different complexity class, and the colocated GRPO step is already ~40 s. So: time and
peak-memory both map producers on the *same* teacher-forced cases and report the ratio.

The case is the trainer's, not a synthetic one. `per_device_train_batch_size=1 x
grad_accum=8 x num_generations=8` (launch_grpo_qwen3_overlap_colocated_job.sh) means one
rank calls `_compute_grad_step_maps` with **8 completions per optimizer step**, so this
generates 8 completions per prompt at `--max-new-tokens 1024`, segments them with the same
FLAN-T5 observe-step classifier the reward uses, and treats one completion as one case.
The headline number is therefore the per-sample total: seconds added to one rank's step.

Both producers run on byte-identical input -- one `teacher_forced_case` per completion,
the same spans -- and both inside `frozen_params`, which is what the trainer must do
anyway (see the ZeRO-3 note in `_compute_grad_step_maps`). Peak memory is reported twice:
allocated including the resident ~16.3 GiB of weights, which is the number that decides
whether it fits next to vLLM and DINO, and the delta above the resting allocation, which
is the number that scales with N.

`--glimpse-layer-fracs` prices the one obvious lever: the paper's ablation loses nothing
at 0.6, and cutting the propagated stack cuts both the backward and the replays.

    srun --jobid=$JOBID --overlap -n1 bash -lc '... python glimpse_cost_probe.py \
        --n-samples 2 --out outputs/glimpse_cost/run1.json'

Nothing here writes to the training path; it only reads the two map producers.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent


def repo_path(rel: str) -> Path:
    """Repo-relative path, falling back to the central tree (see overlap_probe)."""
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# saliency_viz already loads overlap_probe / intervene_probe / grad_maps exactly once,
# so going through it keeps a single copy of each -- and the glimpse producer measured
# here is literally the one `--stage scan` runs.
SV = _load_module("_gc_saliency_viz", "saliency_viz.py")
PROBE, IV, GM, OSTEPS, FC = SV.PROBE, SV.IV, SV.GM, SV.OSTEPS, SV.FC


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------
class Phases:
    """Per-phase seconds inside one glimpse_map call.

    Every phase is synchronised before it is read, so these are wall-clock GPU time and
    not launch time. The sync also serialises the pipeline, which inflates the total a
    little; `--breakdown-check` reruns one case uninstrumented to price that.
    """

    KEYS = ("backward", "edge", "propagate")

    def __init__(self, device):
        self.device = device
        self.t = dict.fromkeys(self.KEYS, 0.0)
        self.n_backward = self.n_edge = 0

    @contextlib.contextmanager
    def phase(self, key: str):
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            self.t[key] += time.perf_counter() - t0


@contextlib.contextmanager
def instrument(prof: Phases | None):
    """Wrap the three GLIMPSE inner loops in timers, then put them back.

    Patching here rather than in saliency_viz.py keeps the measured code identical to the
    code that ships -- a probe that has to edit its subject is measuring something else.
    """
    if prof is None:
        yield
        return
    cls, orig_prop = SV.GlimpseGradCache, SV.glimpse_propagate
    orig_grads, orig_edge = cls.layer_grads, cls.edge

    def layer_grads(self, scalar):
        prof.n_backward += 1
        with prof.phase("backward"):
            return orig_grads(self, scalar)

    def edge(self, li, g_out):
        prof.n_edge += 1
        with prof.phase("edge"):
            return orig_edge(self, li, g_out)

    def propagate(row, mats, alphas):
        with prof.phase("propagate"):
            return orig_prop(row, mats, alphas)

    cls.layer_grads, cls.edge, SV.glimpse_propagate = layer_grads, edge, propagate
    try:
        yield
    finally:
        cls.layer_grads, cls.edge, SV.glimpse_propagate = orig_grads, orig_edge, orig_prop


class Meter:
    """Seconds and peak GiB around one map call, with the resting allocation subtracted.

    `empty_cache` before the reset is deliberate: without it the peak carries whatever
    the previous call left cached, and the two producers would not be comparable.
    """

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        self.base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.t0
        peak = torch.cuda.max_memory_allocated(self.device)
        self.peak_gib = peak / 2**30
        self.delta_gib = (peak - self.base) / 2**30
        return False


# ---------------------------------------------------------------------------
# the two producers, on one identical case
# ---------------------------------------------------------------------------
def run_grad(model, processor, inputs, prompt_len, comp_ids, steps, device, args):
    """`trl/grad_maps.py` exactly as `_compute_grad_step_maps` calls it."""
    ip = processor.image_processor
    ps = int(getattr(ip, "patch_size", 16))
    tps = int(getattr(ip, "temporal_patch_size", 2))
    case = PROBE.teacher_forced_case(inputs, comp_ids, device)
    spans = [(prompt_len + a, prompt_len + b) for _t, a, b in steps]
    grid = inputs["image_grid_thw"][0].tolist()
    return GM.step_grad_maps(model, case, spans, grid, ps, tps, target=args.grad_target,
                             span_chunk=args.grad_span_chunk)


def run_glimpse(model, processor, inputs, ids, prompt_len, steps, gh, gw, question,
                device, args, layer_frac):
    """`saliency_viz.glimpse_map` exactly as `--stage scan` calls it."""
    gargs = argparse.Namespace(
        glimpse_temp=args.glimpse_temp,
        glimpse_depth_temp=args.glimpse_depth_temp,
        glimpse_layer_frac=layer_frac,
        glimpse_target=args.glimpse_target,
        glimpse_token_weight=args.glimpse_token_weight,
    )
    return SV.glimpse_map(model, processor, inputs, ids, prompt_len, steps, gh, gw,
                          question, gargs, device)


# ---------------------------------------------------------------------------
# how the per-token cost scales with N
# ---------------------------------------------------------------------------
def scale_timing(args, device):
    """Seconds for ONE target token's map, against sequence length.

    The case measurement lands wherever the model's own chains land, which on set_a is
    N ~ 400. Training allows `--max_completion_length 1024`, so a long chain is 3x that,
    and GLIMPSE's per-layer work is `[N, N]` -- the two `[N, N]` fp32 passes per head in
    `glimpse_edge_matrix` plus the `[N] @ [N, N]` propagation. Whether the cost per token
    is linear or quadratic in N therefore decides whether the case numbers extrapolate or
    understate, and one number per N settles it.

    Text-only synthetic input, like `test_glimpse_gpu.test_scaling`: the map needs an
    image, but none of the cost above does -- what scales is `L x H x N^2`, and this
    isolates it from the tokeniser and the sampler.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    model.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    n_layers = sum(1 for m in model.modules()
                   if type(m).__name__ == "Qwen3VLTextAttention")
    out = []
    for n in [int(x) for x in args.scale.split(",") if x]:
        keep = min(n_layers, max(1, int(round(args.scale_layer_frac * n_layers))))
        first = n_layers - keep
        ids = torch.randint(1000, 20000, (1, n), device=device)
        cap = SV.GlimpseGradCache(model, first, args.glimpse_temp)
        rec = {"N": n, "n_layers_propagated": keep}
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with SV.checkpointing_off(model), torch.enable_grad():
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                res = model(input_ids=ids, use_cache=False,
                            logits_to_keep=torch.tensor([n - 2, n - 1], device=device))
                z = res.logits[0].float()
                del res
                cap.check()
                cap.mask = IV.causal_mask(n, dtype, device)
                torch.cuda.synchronize(device)
                rec["forward_seconds"] = time.perf_counter() - t0

                # Twice: the first pass pays allocator growth and cuBLAS algorithm
                # selection, which in the real path is amortised over ~100 tokens.
                for it in range(2):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    g_out = cap.layer_grads(z[0, n // 2])
                    torch.cuda.synchronize(device)
                    t_bwd = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    mats, g_l1 = [], []
                    for li in cap.layers:
                        e, g1 = cap.edge(li, g_out.pop(li))
                        mats.append(e)
                        g_l1.append(g1)
                    torch.cuda.synchronize(device)
                    t_edge = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    alphas = SV.glimpse_layer_alphas(g_l1, cap.layers,
                                                     args.glimpse_depth_temp)
                    SV.glimpse_propagate(n - 2, mats, alphas)
                    torch.cuda.synchronize(device)
                    t_prop = time.perf_counter() - t0
                    del mats, g_l1, alphas, g_out
                rec.update(backward_seconds=t_bwd, edge_seconds=t_edge,
                           propagate_seconds=t_prop,
                           token_seconds=t_bwd + t_edge + t_prop)
                del z
            rec["peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        except torch.cuda.OutOfMemoryError:
            rec["oom"] = True
        finally:
            cap.close()
            del ids
            gc.collect()
            torch.cuda.empty_cache()
        out.append(rec)
        if "oom" in rec:
            print(f"[scale] N={n:5d}  OOM", flush=True)
        else:
            print(f"[scale] N={n:5d} layers={keep}  fwd {rec['forward_seconds']:5.2f}s | "
                  f"per token {rec['token_seconds']:6.3f}s "
                  f"(bwd {rec['backward_seconds']:.3f} edge {rec['edge_seconds']:.3f} "
                  f"prop {rec['propagate_seconds']:.3f})  peak {rec['peak_gib']:5.1f}GiB",
                  flush=True)
    return {"config": vars(args), "scale": out}


def summarise_scale(res):
    v = [r for r in res["scale"] if "token_seconds" in r]
    if len(v) < 2:
        return "scaling: too few points"
    out = ["N      layers  fwd s   s/token   bwd     edge    prop   peak GiB   exponent"]
    prev = None
    for r in v:
        # local log-log slope: 1 = linear in N, 2 = quadratic
        exp = ""
        if prev:
            exp = (f"{np.log(r['token_seconds'] / prev['token_seconds']) / np.log(r['N'] / prev['N']):.2f}")
        out.append(f"{r['N']:<6d} {r['n_layers_propagated']:>6d} {r['forward_seconds']:>6.2f} "
                   f"{r['token_seconds']:>9.3f} {r['backward_seconds']:>7.3f} "
                   f"{r['edge_seconds']:>7.3f} {r['propagate_seconds']:>6.3f} "
                   f"{r['peak_gib']:>10.1f} {exp:>10}")
        prev = r
    lo, hi = v[0], v[-1]
    slope = np.log(hi["token_seconds"] / lo["token_seconds"]) / np.log(hi["N"] / lo["N"])
    out.append(f"\noverall exponent on N, {lo['N']} -> {hi['N']}: {slope:.2f} "
               "(1 = linear, 2 = quadratic)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def measure(args, device, on_case=None):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag="_gcost", split=args.split)
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    model.requires_grad_(False)
    clf = OSTEPS.OverlapStepsClassifier.load(args.steps_ckpt, device=device)
    tok = processor.tokenizer
    fracs = [float(f) for f in args.glimpse_layer_fracs.split(",") if f]

    torch.cuda.synchronize(device)
    weights_gib = torch.cuda.memory_allocated(device) / 2**30
    print(f"[cost] weights resident: {weights_gib:.2f} GiB   tf32={args.tf32}", flush=True)

    cases, warmed, checked = [], False, False
    for si, row in enumerate(rows):
        inputs, prompt_len, seqs = PROBE.generate(
            processor, model, row["image"], row["question"], args.num_generations,
            args.max_new_tokens, args.temperature, device)
        gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
        gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
        torch.cuda.empty_cache()

        taken = 0
        for c, (comp_ids, truncated) in enumerate(seqs):
            if args.max_cases and len(cases) >= args.max_cases:
                break
            if args.per_sample_cases and taken >= args.per_sample_cases:
                break
            _text, seg, reason = SV.segment_case(tok, clf, row["question"], comp_ids)
            if seg is None:
                print(f"[cost] s{si} c{c}: skipped ({reason})", flush=True)
                continue
            steps, format_ok = seg
            ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)],
                               device=device)
            n_tok = sum(b - a for _t, a, b in steps)
            rec = {"sample": si, "row_index": row["row_index"], "completion": c,
                   "seq_len": int(ids.shape[1]), "prompt_len": int(prompt_len),
                   "n_completion_tokens": len(comp_ids), "truncated": bool(truncated),
                   "format_ok": bool(format_ok), "grid": [gh, gw],
                   "n_steps": len(steps), "n_target_tokens": int(n_tok)}

            # One throwaway call each before the first timed one: the allocator grows and
            # cuBLAS picks its algorithms on the first pass, which would otherwise be
            # charged to case 0 and to whichever producer ran first.
            if not warmed:
                warmed = True
                one = [steps[0][:2] + (steps[0][1] + 1,)]     # a single-token span
                try:
                    with GM.frozen_params(model):
                        run_grad(model, processor, inputs, prompt_len, comp_ids, one,
                                 device, args)
                        run_glimpse(model, processor, inputs, ids, prompt_len, one, gh, gw,
                                    row["question"], device, args, fracs[0])
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[cost] warmup OOM (timings will include first-call cost): {e}",
                          flush=True)
                torch.cuda.empty_cache()

            try:
                with GM.frozen_params(model), Meter(device) as m:
                    gmaps = run_grad(model, processor, inputs, prompt_len, comp_ids,
                                     steps, device, args)
                rec["grad"] = {"seconds": m.seconds, "peak_gib": m.peak_gib,
                               "delta_gib": m.delta_gib, "shape": list(gmaps.shape)}
                del gmaps
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec["grad"] = {"oom": True}

            rec["glimpse"] = {}
            for frac in fracs:
                prof = Phases(device) if args.breakdown else None
                try:
                    with GM.frozen_params(model), instrument(prof), Meter(device) as m:
                        maps, info = run_glimpse(model, processor, inputs, ids, prompt_len,
                                                 steps, gh, gw, row["question"], device,
                                                 args, frac)
                    e = {"seconds": m.seconds, "peak_gib": m.peak_gib,
                         "delta_gib": m.delta_gib, "shape": list(maps.shape),
                         "n_layers_propagated": info["n_layers"] - info["first_layer"]}
                    if prof is not None:
                        e["phases"] = dict(prof.t)
                        e["n_backward"], e["n_edge"] = prof.n_backward, prof.n_edge
                    del maps
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    e = {"oom": True}
                rec["glimpse"][f"{frac:g}"] = e

            if args.breakdown and args.breakdown_check and not checked:
                # The sync-per-phase instrumentation serialises the pipeline. Rerun the
                # first frac uninstrumented ONCE, so the inflation is a number and not an
                # assumption -- doing it per case would double the whole run.
                checked = True
                with contextlib.suppress(torch.cuda.OutOfMemoryError):
                    with GM.frozen_params(model), Meter(device) as m:
                        run_glimpse(model, processor, inputs, ids, prompt_len, steps, gh,
                                    gw, row["question"], device, args, fracs[0])
                    rec["glimpse_uninstrumented_seconds"] = m.seconds

            cases.append(rec)
            taken += 1
            if on_case is not None:
                on_case(cases)          # checkpoint the JSON: a run that runs out of
                                        # wall-clock still reports every finished case
            g = rec.get("grad", {})
            gl = rec["glimpse"].get(f"{fracs[0]:g}", {})
            print(f"[cost] s{si} c{c}: N={rec['seq_len']} steps={rec['n_steps']} "
                  f"tok={n_tok}  grad {g.get('seconds', float('nan')):.2f}s "
                  f"{g.get('peak_gib', float('nan')):.1f}GiB | "
                  f"glimpse(f={fracs[0]:g}) {gl.get('seconds', float('nan')):.1f}s "
                  f"{gl.get('peak_gib', float('nan')):.1f}GiB", flush=True)
            del ids
            torch.cuda.empty_cache()

        if args.max_cases and len(cases) >= args.max_cases:
            break

    return {"config": vars(args), "weights_gib": weights_gib, "cases": cases}


def summarise(res, fracs):
    """Per-case medians and the per-sample totals -- the seconds one rank's step gains."""
    cases = res["cases"]
    if not cases:
        return "no cases measured"
    out = []

    def pairs(getter):
        """-> [(case, entry)] for the cases where this producer actually produced."""
        got = []
        for c in cases:
            d = getter(c)
            if d and "seconds" in d:
                got.append((c, d))
        return got

    n = len(cases)
    tok = [c["n_target_tokens"] for c in cases]
    seq = [c["seq_len"] for c in cases]
    steps = [c["n_steps"] for c in cases]
    out.append(f"{n} cases   N median {int(np.median(seq))} (max {max(seq)})   "
               f"steps/case median {int(np.median(steps))}   "
               f"target tokens/case median {int(np.median(tok))} (max {max(tok)})")

    rows = [("grad", pairs(lambda c: c.get("grad")))]
    for f in fracs:
        key = f"{f:g}"
        rows.append((f"glimpse f={key}",
                     pairs(lambda c, k=key: c.get("glimpse", {}).get(k))))

    out.append("")
    out.append(f"{'producer':<16} {'n':>3} {'s/case':>8} {'s/token':>9} {'peak GiB':>9} "
               f"{'delta GiB':>10} {'s / 8-case step':>16}")
    base = None
    for name, v in rows:
        if not v:
            out.append(f"{name:<16} {'--':>3}   OOM or not measured")
            continue
        s = np.array([d["seconds"] for _c, d in v])
        pk = np.array([d["peak_gib"] for _c, d in v])
        dl = np.array([d["delta_gib"] for _c, d in v])
        per_tok = s / np.array([max(c["n_target_tokens"], 1) for c, _d in v])
        out.append(f"{name:<16} {len(v):>3} {np.median(s):>8.2f} {np.median(per_tok):>9.3f} "
                   f"{np.median(pk):>9.1f} {np.median(dl):>10.2f} "
                   f"{8 * float(np.mean(s)):>16.1f}")
        if name == "grad":
            base = float(np.mean(s))
    if base:
        out.append("")
        for name, v in rows[1:]:
            if v:
                # Compare on the cases BOTH producers survived, or the ratio is a
                # comparison of two different samples of completions.
                both = [(c, d) for c, d in v if "seconds" in c.get("grad", {})]
                if not both:
                    continue
                num = float(np.mean([d["seconds"] for _c, d in both]))
                den = float(np.mean([c["grad"]["seconds"] for c, _d in both]))
                out.append(f"{name} is {num / den:.0f}x the gradient reward per case "
                           f"({len(both)} shared cases)")

    ph = [c["glimpse"][f"{fracs[0]:g}"].get("phases") for c in cases
          if c.get("glimpse", {}).get(f"{fracs[0]:g}", {}).get("phases")]
    if ph:
        tot = {k: float(np.sum([p[k] for p in ph])) for k in Phases.KEYS}
        wall = float(np.sum([c["glimpse"][f"{fracs[0]:g}"]["seconds"] for c in cases
                             if c["glimpse"][f"{fracs[0]:g}"].get("phases")]))
        out.append("")
        out.append(f"where the glimpse time goes (f={fracs[0]:g}, summed over cases):")
        for k, v in tot.items():
            out.append(f"  {k:<10} {v:8.1f}s  {100 * v / wall:5.1f}%")
        out.append(f"  {'other':<10} {wall - sum(tot.values()):8.1f}s  "
                   f"{100 * (wall - sum(tot.values())) / wall:5.1f}%   (sdpa forward, "
                   "logits, token weights)")
    chk = [c for c in cases if "glimpse_uninstrumented_seconds" in c]
    if chk:
        c = chk[0]
        inst = c["glimpse"][f"{fracs[0]:g}"]["seconds"]
        out.append(f"instrumentation overhead: {inst:.1f}s instrumented vs "
                   f"{c['glimpse_uninstrumented_seconds']:.1f}s not "
                   f"({100 * (inst / c['glimpse_uninstrumented_seconds'] - 1):+.0f}%)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/set_a")))
    p.add_argument("--split", default="train", choices=["train", "holdout", "all"])
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--max-cases", type=int, default=8,
                   help="stop after this many completions in total (0 = no cap). The "
                        "default 8 is one rank's optimizer step.")
    p.add_argument("--per-sample-cases", type=int, default=4,
                   help="completions to measure per prompt (0 = all --num-generations of "
                        "them). Below --num-generations so a fixed --max-cases spreads "
                        "over several images, since N and the step count vary by image.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-model",
                   default=str(repo_path("checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--steps-ckpt", default=os.environ.get(
        "OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    # grad side -- the trainer's own defaults
    p.add_argument("--grad-target", default="clogit", choices=list(GM.GRAD_TARGETS))
    p.add_argument("--grad-span-chunk", type=int, default=GM.SPAN_CHUNK_DEFAULT)
    # glimpse side -- saliency_viz's defaults, except the target, which is matched to the
    # gradient reward's clogit so the two producers score the same scalar.
    p.add_argument("--glimpse-temp", type=float, default=0.5)
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2)
    p.add_argument("--glimpse-layer-fracs", default="1.0,0.6")
    p.add_argument("--glimpse-target", default="clogit", choices=list(GM.GRAD_TARGETS))
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"])
    p.add_argument("--stage", default="cases", choices=["cases", "scale"],
                   help="cases = the real per-completion cost; scale = per-target-token "
                        "cost vs sequence length, on synthetic text-only input")
    p.add_argument("--scale", default="400,800,1200,1600,2400")
    p.add_argument("--scale-layer-frac", type=float, default=1.0)
    p.add_argument("--breakdown", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--breakdown-check", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=str(REPO / "outputs/glimpse_cost/cost.json"))
    args = p.parse_args()

    fracs = [float(f) for f in args.glimpse_layer_fracs.split(",") if f]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "scale":
        res = scale_timing(args, args.device)
        text = summarise_scale(res)
        res["summary"] = text
        out.write_text(json.dumps(res, indent=2, default=str))
        print("\n" + text + f"\n\nwrote {out}", flush=True)
        return

    def checkpoint(cases):
        out.write_text(json.dumps({"config": vars(args), "cases": cases}, indent=2,
                                  default=str))

    res = measure(args, args.device, on_case=checkpoint)
    text = summarise(res, fracs)
    res["summary"] = text
    out.write_text(json.dumps(res, indent=2, default=str))
    print("\n" + text + f"\n\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
