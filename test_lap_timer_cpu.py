#!/usr/bin/env python
"""`_lap` runs on every GRPO step of every run, so a typo in it is a run that dies at step 1.

The three methods it adds are called from the middle and the end of
`_generate_and_score_completions`, which needs a model, six ranks and two sidecars to
reach -- about twenty-five minutes of warm-up before the first line of them executes. They
are also pure Python with no tensor in them, so none of that is needed to find out whether
they work. This binds them to a stub and exercises every branch on CPU in well under a
second.

What it pins, in the order it would hurt:

  * OFF BY DEFAULT. `SR1_LAP` unset must mean not one of the three does anything -- no
    accumulator, no wandb call, and above all no `wait_for_everyone()`, because a barrier
    that appears on some ranks and not others is a hang rather than an error. Every
    Qwen3-VL run shares this trainer.
  * The accumulator RESETS at flush. It is keyed per span name and summed within a step;
    if flush did not clear it, every step would report the sum of the run so far and the
    series would look like a leak in whatever it measured.
  * A span is attributed to the mark that CLOSES it, and `_lap()` with no name only starts
    the clock -- the first span of a step must not be charged to the last span of the one
    before.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# Import the trainer's methods without importing the trainer: the module pulls in
# transformers, peft and a vLLM client at import time, none of which this needs.
import ast
import types

src = open(os.path.join(ROOT, "trl", "grpo_trainer_qwen3.py")).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("GRPOTrainer"))
wanted = {"_lap", "_lap_barrier", "_lap_flush"}
funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
check("found all three methods in the trainer", {f.name for f in funcs} == wanted,
      str(sorted(f.name for f in funcs)))

ns = {"os": os, "time": time}
exec(compile(ast.Module(body=funcs, type_ignores=[]), "<lap>", "exec"), ns)


class Accel:
    is_main_process = True

    def __init__(self):
        self.barriers = 0

    def wait_for_everyone(self):
        self.barriers += 1


class Args:
    report_to = []          # no wandb -> _lap_flush takes the stdout path only


class Stub:
    _lap = ns["_lap"]
    _lap_barrier = ns["_lap_barrier"]
    _lap_flush = ns["_lap_flush"]

    def __init__(self):
        self.accelerator = Accel()
        self.args = Args()
        self.state = types.SimpleNamespace(global_step=7)


def run(env):
    keep = {k: os.environ.get(k) for k in ("SR1_LAP", "SR1_LAP_STDOUT")}
    os.environ.pop("SR1_LAP", None)
    os.environ.pop("SR1_LAP_STDOUT", None)
    os.environ.update(env)
    try:
        s = Stub()
        s._lap()
        time.sleep(0.02)
        s._lap("alpha")
        time.sleep(0.01)
        s._lap_barrier("beta")
        s._lap_flush()
        return s
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# ---- off by default ---------------------------------------------------------------
off = run({})
check("SR1_LAP unset: no accumulator is created", not hasattr(off, "_lap_acc"))
check("SR1_LAP unset: NO BARRIER IS TAKEN", off.accelerator.barriers == 0,
      f"{off.accelerator.barriers} barriers")
check("SR1_LAP unset: no clock is started", not hasattr(off, "_lap_t"))

# ---- on ---------------------------------------------------------------------------
class Probe(Stub):
    """Capture the accumulator at flush time, which is the only moment it is complete."""
    captured = None

    def _lap_flush(self):
        Probe.captured = dict(getattr(self, "_lap_acc", {}) or {})
        return Stub._lap_flush(self)


os.environ["SR1_LAP"] = "1"
try:
    s = Probe()
    s._lap()
    time.sleep(0.03)
    s._lap("alpha")
    time.sleep(0.01)
    s._lap_barrier("beta")
    acc_before_flush = dict(s._lap_acc)
    s._lap_flush()

    check("both spans are recorded", set(acc_before_flush) == {"alpha", "beta"},
          str(sorted(acc_before_flush)))
    check("a span is charged to the mark that closes it",
          acc_before_flush.get("alpha", 0) > acc_before_flush.get("beta", 1),
          str(acc_before_flush))
    check("the span length is the wall time, not zero",
          0.025 < acc_before_flush.get("alpha", 0) < 1.0, str(acc_before_flush.get("alpha")))
    check("_lap_barrier rendezvoused exactly once", s.accelerator.barriers == 1,
          f"{s.accelerator.barriers}")
    check("flush CLEARS the accumulator", not s._lap_acc, str(s._lap_acc))
    check("flush clears the clock too, so the next step starts clean", s._lap_t is None)

    # A second step must not inherit the first: same object, fresh spans.
    s._lap()
    time.sleep(0.01)
    s._lap("alpha")
    check("a second step starts from zero", s._lap_acc["alpha"] < acc_before_flush["alpha"],
          f"{s._lap_acc['alpha']:.4f} vs {acc_before_flush['alpha']:.4f}")

    # Repeated marks inside one step SUM -- compute_loss-shaped spans are entered per
    # micro-step, and a span that overwrote would report one micro-step as the whole step.
    s._lap_flush()
    s._lap()
    for _ in range(3):
        time.sleep(0.01)
        s._lap("gamma")
    check("a repeated mark accumulates rather than overwrites", s._lap_acc["gamma"] > 0.025,
          str(s._lap_acc.get("gamma")))

    # A flush with nothing accumulated must be a no-op, not a crash: that is what happens
    # on a rank whose step raised before the first mark.
    s._lap_flush()
    s._lap_flush()
    check("an empty flush is a no-op", True)
finally:
    os.environ.pop("SR1_LAP", None)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
