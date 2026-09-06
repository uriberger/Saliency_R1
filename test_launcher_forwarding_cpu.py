#!/usr/bin/env python
"""Every knob that changes the RUN NAME must survive the SLURM -> --direct hop.

`launch_grpo_qwen3_overlap_colocated_job.sh` does not run training when it is invoked
without `--direct`. It submits a job whose payload re-invokes the same script with
`--direct` and a hand-written list of flags, and the inner invocation is the one that
trains. Anything absent from that list is dropped, and the drop is silent in the worst
possible way: `RUN_NAME`, the `.out` filename and the SLURM log directory are all built
in the OUTER invocation from the flags that WERE passed, so the artefacts on disk go on
advertising the experiment that did not run.

That is not hypothetical. `--chain-boxes` and `--rect-placement` were both missing, and
two arms of the five-arm mask-source experiment were lost:

    outputs/logs/..._chainlast.6577777.out   ->  "Grounding: once per observe step"
    outputs/logs/..._inhash0.6580781.out     ->  "Mask: CENTRED RECTANGLE"

both with a resolved run name that carried neither suffix, and `mask/n_placements`
logging 1.0 for all 798 points of what was meant to be a 12-placement arm.

The rule this file pins is the one that would have caught it, and it needs no list of
its own to maintain: **the SUFFIX block is the definition of what makes two runs
different experiments** (that is why it exists -- so two arms cannot share a checkpoint
directory or a wandb run). So every variable the SUFFIX block reads must appear in the
forwarding block. A new arm gets its check for free on the day it gets its name.

CPU only, no imports beyond the standard library, no GPU, ~0.1 s.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(ROOT, "launch_grpo_qwen3_overlap_colocated_job.sh")
SRC = open(LAUNCHER).read()

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def section(start_pat, end_pat, what):
    """The source between two anchors, so the checks read the script rather than a copy."""
    a = re.search(start_pat, SRC, re.M)
    assert a, f"anchor for {what} not found: {start_pat}"
    b = re.search(end_pat, SRC[a.end():], re.M)
    assert b, f"end anchor for {what} not found: {end_pat}"
    return SRC[a.end(): a.end() + b.start()]


VAR = re.compile(r"\$\{?([A-Z][A-Z0-9_]*)")

# ---------------------------------------------------------------- the two blocks
naming = section(r"^# -+ naming: .*$", r"^RUN_NAME=", "the naming block")
forward = section(r"bash \$SCRIPT_PATH --direct \\", r"^\s*\$EXTRA_ARGS\s*$", "the forwarding block")

check("found the naming block", "SUFFIX=" in naming and len(naming) > 500)
check("found the forwarding block", "--num-gpus" in forward and len(forward) > 500)

# Names ASSIGNED inside the naming block are derived there, so they are not inputs and
# cannot be forwarded (SUFFIX, N_HEADS, LORA_SLUG, MODEL_SLUG, REWARD_SLUG, ...).
derived = set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)=", naming, re.M))
read_by_naming = set(VAR.findall(naming)) - derived

# One name in the block is not an input, and it is excused only by something the test
# still has to prove. Keep this map tiny: it is the only place the generic rule bends.
#
#   REWARD_VARIANT  internal, and the launcher says so itself at the `case
#                   "$REWARD_VARIANT"` that defines SALIENCY_METHOD_R: "REWARD_VARIANT is
#                   an internal variable and does not survive a new process".
#                   SALIENCY_METHOD_R is its 1:1 spelling and is what gets forwarded.
CARRIED_BY = {"REWARD_VARIANT": "SALIENCY_METHOD_R"}
required = sorted(read_by_naming - set(CARRIED_BY))
forwarded = set(VAR.findall(forward))

check("the naming block reads a plausible number of knobs", len(required) >= 20, str(len(required)))

missing = [v for v in required if v not in forwarded]
check("every variable in the run name is forwarded to the --direct re-invocation",
      not missing, "dropped: " + ", ".join(missing))

# An excused variable is only excused while its carrier really is forwarded.
for excused, carrier in CARRIED_BY.items():
    if excused in read_by_naming:
        check(f"{excused} is excused only because {carrier} is forwarded", carrier in forwarded)

# The three that were actually lost, named explicitly so the regression is legible even
# if the generic rule above is ever relaxed.
for v, flag in (("CHAIN_BOXES", "--chain-boxes"),
                ("RECT_PLACEMENT", "--rect-placement"),
                ("RECT_SEED", "--rect-seed")):
    check(f"{flag} is forwarded", v in forwarded)

# ---------------------------------------------------------------- it must EXPAND right
# The static check above cannot tell `${CHAIN_BOXES:+--chain-boxes $CHAIN_BOXES}` from a
# mention in a comment, and it cannot tell whether an unset knob leaks an empty flag --
# which would break every existing command line rather than only the new arms. So expand
# the block for real, in bash, with nothing but the variables set.
def expand(**env):
    """Run the forwarding block through bash and return the argv it would produce."""
    defaults = dict(
        NUM_GPUS="7", MODEL="/m", OUTPUT_DIR="/o", MAX_COMPLETION_LENGTH="1024",
        NUM_GENERATIONS="8", GRAD_ACCUM="8", PER_DEVICE_BATCH="1", LEARNING_RATE="1e-5",
        W_OVERLAP="0.4", TOKEN_REDUCTION="mean", LORA_TARGETS="q_proj,v_proj",
        OVERLAP_HEADS="28,31", OVERLAP_LAYER="22", BOX_THRESHOLD="0.10",
        MAX_BOX_AREA="0.5", MAX_UNION_AREA="", OVERLAP_METRIC="mean_in",
        MASS_FLOOR_TAU="", PLACEBO="", MASKFREE="", MASKFREE_PARITY="false",
        RECT_FRAC="", RECT_PLACEMENT="centre", RECT_SEED="0",
        CHAIN_BOXES="", MISMATCH_BANK="", MISMATCH_SEED="0",
        SALIENCY_METHOD_R="attention", GRAD_TARGET="clogit", GRAD_NULL_OFFSETS="16",
        GRAD_LOGRATIO_CLIP="1.0", GLIMPSE_TARGET="clogit", GLIMPSE_LAYER_FRAC="1.0",
        GLIMPSE_TOKEN_CAP="0", GLIMPSE_DEPTH_TEMP="0.2", GLIMPSE_TEMP="1.0",
        GLIMPSE_TOKEN_WEIGHT="full", ROLLNULL_OFFSETS="16", ROLLNULL_CLIP="1.0",
        ROLLNULL_SEED="0", NATURAL_ONLY="false", QUESTION_BOXES="",
        ALLOW_MAP_CHANGE="false", BETA="0", LENGTH_GUARD_REF="",
        LENGTH_GUARD_WEIGHT="0.20", LENGTH_GUARD_BAND_LO="0.30",
        LENGTH_GUARD_BAND_HI="3.0", LENGTH_GUARD_KNEE="1.0",
        ALLOW_REGULATOR_CHANGE="false", DINO_PORT="8100", VLLM_PORT="8000",
        VLLM_GPU_MEM="0.90", VLLM_MAX_MODEL_LEN="4096", VLLM_ENFORCE_EAGER="False",
        EVAL_STEPS="100", BENCH_GPUS="1", VAL_SETS_DIR="/v",
        SHARE_SIDECAR_GPU="false", EXTRA_ARGS="", SCRIPT_PATH="LAUNCHER",
    )
    defaults.update(env)
    assigns = "\n".join(f"{k}={v!r}" for k, v in defaults.items())
    # `echo` in place of the real re-invocation: the block is a command line, so running
    # it under `echo` prints exactly the argv the job would have executed.
    script = f"set -u\n{assigns}\necho {forward.strip()}\n".replace("bash $SCRIPT_PATH", "")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


def after(argv, flag):
    """The token following `flag`, or None when the flag never made it. Returning None
    rather than raising keeps a dropped flag reporting as one failed check among many
    instead of aborting the run -- which is the state this file exists to describe."""
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


base = expand()
check("an ordinary per-step run emits no rect or chain flag at all",
      not any(f in base for f in ("--rect-placement", "--rect-seed", "--chain-boxes",
                                  "--overlap-rect-frac")),
      " ".join(f for f in base if "rect" in f or "chain" in f))

hashed = expand(RECT_FRAC="0.565", RECT_PLACEMENT="interior_hash", RECT_SEED="3")
for want in ("--overlap-rect-frac", "0.565", "--rect-placement", "interior_hash",
             "--rect-seed", "3"):
    check(f"interior_hash run forwards {want!r}", want in hashed)
check("interior_hash run forwards placement adjacent to its value",
      after(hashed, "--rect-placement") == "interior_hash", str(after(hashed, "--rect-placement")))
check("interior_hash run forwards the seed adjacent to its value",
      after(hashed, "--rect-seed") == "3", str(after(hashed, "--rect-seed")))

chained = expand(CHAIN_BOXES="last")
check("chain run forwards --chain-boxes last",
      after(chained, "--chain-boxes") == "last", str(after(chained, "--chain-boxes")))
check("chain run still emits no rect flag", "--overlap-rect-frac" not in chained)

# A centred rect is byte-identical to the incumbent on the trainer side, so forwarding
# `--rect-placement centre` must not change what the inner invocation resolves to. It is
# forwarded anyway (rather than only when != centre) because a knob that is sometimes
# absent is how this bug is written a second time.
centred = expand(RECT_FRAC="0.565")
check("a centred rect forwards its placement explicitly",
      after(centred, "--rect-placement") == "centre", str(after(centred, "--rect-placement")))

# ---------------------------------------------------------------- the inner side agrees
# Whatever is forwarded has to be a flag the arg loop accepts, or the inner invocation
# dies at parse time with the allocation already held.
argloop = section(r"while \[\[ \$# -gt 0 \]\]; do", r"^\s*esac", "the argument loop")
# Flags reached by `$([ x ] && echo --flag)` carry the closing paren of the substitution,
# so strip shell punctuation before matching. `--direct` is the hop itself, not a knob.
emitted = sorted({tok.rstrip(')"\'\\') for tok in forward.split() if tok.startswith("--")})
emitted = [f for f in emitted if f != "--direct" and len(f) > 2]
unknown = [f for f in emitted if not re.search(r"^\s*%s\)" % re.escape(f), argloop, re.M)]
check("every flag the forwarding block emits is one the arg loop parses",
      not unknown, "unparsed: " + ", ".join(unknown))
check("the forwarding block emits a plausible number of flags", len(emitted) >= 30, str(len(emitted)))

print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    sys.exit(1)
print(f"all launcher-forwarding CPU checks passed ({len(required)} name-bearing knobs pinned)")
