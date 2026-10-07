#!/usr/bin/env python
"""Every training flag the Omni launcher PARSES must survive the SLURM -> --direct hop.

`launch_grpo_omni_overlap_job.sh` does not train when it is invoked without `--direct`.
It submits a job whose payload re-invokes the same script with `--direct` and a
hand-written `ARGS` array, and the inner invocation is the one that trains. Anything the
outer parser accepts and the array does not repeat is replaced on the node by the DEFAULT
at the top of the file -- silently, because the banner prints the default and the run
looks healthy.

That is not hypothetical. `--max-completion-length` was missing, and it is the ONE
training hyper-parameter that differs from the Qwen3-VL runs. The 2026-09-28 run was
launched with 768, wrote to a directory named `_c768`, and trained at 1024: its wandb
config says `max_completion_length: 1024` and `completions/max_length` is 1024 on 29 of
its 30 logged steps. Thirty steps of memory evidence, and the setting they were supposed
to be evidence about was never applied. See docs/omni-training-harness.md §12.1.

`test_launcher_forwarding_cpu.py` pins the Qwen3-VL launcher with a rule keyed on the
RUN NAME: every variable the suffix block reads must be forwarded. That rule would have
caught this one too -- the cap is in the Omni's run name -- but it is weaker than it needs
to be here, because six further flags were dropped the same way and none of them is in the
name. They did not bite only because nobody had passed them. So the rule this file pins is
the strict one:

    every long option the node-side parser accepts is either FORWARDED or on the
    submit-only allowlist below, with a reason.

CPU only, no imports beyond the standard library, no GPU, ~1 s.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(ROOT, "launch_grpo_omni_overlap_job.sh")
SRC = open(LAUNCHER).read()

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# Consumed by submit_job in the OUTER invocation and meaningless on the node, or appended
# conditionally a few lines below the array. Each needs a reason, so that "add it to the
# allowlist" is never the quiet way to make this test pass.
SUBMIT_ONLY = {
    "--direct": "the hop itself",
    "--partition": "chooses the queue; the node is already on one",
    "--duration": "the wall clock submit_job asks for",
    "--exclude-hosts": "a submit_job constraint",
    "--exclude_hosts": "alias of the above",
    "--preflight": "default true; only its negation needs forwarding",
    "--no-preflight": "appended conditionally under [ PREFLIGHT = true ] || ...",
    "--sync-all-weights": "appended conditionally under [ SYNC_LORA_ONLY = true ] || ...",
    "--nvidia-api-key": "crosses in the environment, not the command line",
    "--openai-api-key": "crosses in the environment, not the command line",
    "--wandb-api-key": "crosses in the environment, not the command line",
    "--hf-token": "crosses in the environment, not the command line",
    "--no-saliency": "shorthand that resolves into --reward-variant none; the canonical "
                     "flag carries the value across the hop, so forwarding both would be a "
                     "duplicate. Same shape as --grad/--glimpse in the Qwen3-VL launcher.",
    "--": "the passthrough separator, appended last",
}

# ------------------------------------------------------------------ the static rule
# Long options the `case "$1" in ... esac` parser accepts. Kept as ARMS rather than a flat
# set, because `--dataset_name|--dataset)` is one knob under two spellings: forwarding
# either one carries the value, and demanding both would be demanding a duplicate.
case_block = SRC[SRC.index("while [[ $# -gt 0 ]]; do"):SRC.index("# ---------- the GPU layout")]
arms = []
for m in re.finditer(r"^\s{8}([-|a-z_0-9]+)\)", case_block, re.M):
    alias = [a for a in m.group(1).split("|") if a != "*"]
    if alias:
        arms.append(alias)
# The `--exclude-hosts|--exclude_hosts` and `--dataset_name|--dataset` arms wrap onto a
# continuation line, so pick those up too.
for m in re.finditer(r"^\s{12}([-|a-z_0-9]+)\)", case_block, re.M):
    alias = [a for a in m.group(1).split("|") if a != "*"]
    if alias:
        arms.append(alias)
parsed = {a for arm in arms for a in arm}

args_block = SRC[SRC.index("    ARGS=(\"--direct\""):SRC.index("    echo \"Submitting")]

check("found the option parser", len(parsed) > 20, f"{len(parsed)} options")
check("found the ARGS array", "--overlap-layer" in args_block, args_block[:60])

for arm in sorted(arms):
    if all(a in SUBMIT_ONLY for a in arm):
        continue
    label = "|".join(arm)
    check(f"{label} survives the hop", any(f'"{a}"' in args_block for a in arm))

# And the allowlist must not rot: an entry naming an option the parser no longer has is a
# reason nobody will re-read, sitting in the way of the next real failure.
for opt in sorted(SUBMIT_ONLY):
    if opt == "--":
        continue
    check(f"allowlist entry {opt} is still a real option", opt in parsed)

# ------------------------------------------------------------------ and it must EXPAND
# The static check cannot tell an array entry from a mention in a comment, nor whether the
# value lands adjacent to its flag. So run the submit path for real against a stub
# submit_job and read the command line it would have executed.
def submit(*extra):
    tmp = tempfile.mkdtemp(prefix="omni_fwd_")
    try:
        stub = os.path.join(tmp, "submit_job")
        with open(stub, "w") as fh:
            fh.write('#!/bin/bash\nwhile [[ $# -gt 0 ]]; do\n'
                     '  if [[ "$1" == "--command" ]]; then echo "CMD $2"; shift 2;\n'
                     '  elif [[ "$1" == "--name" ]]; then echo "NAME $2"; shift 2;\n'
                     '  else shift; fi\ndone\n')
        os.chmod(stub, 0o755)
        env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"])
        out = subprocess.run(["bash", LAUNCHER, *extra], capture_output=True, text=True,
                             env=env, cwd=ROOT, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        cmd = name = ""
        for line in out.stdout.splitlines():
            if line.startswith("CMD "):
                cmd = line[4:]
            elif line.startswith("NAME "):
                name = line[5:]
        return cmd.split(), name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def after(argv, flag):
    """The token following `flag`, or None when the flag never made it. Returning None
    rather than raising keeps a dropped flag reporting as one failed check among many."""
    i = argv.index(flag) if flag in argv else -1
    return argv[i + 1] if 0 <= i < len(argv) - 1 else None


argv, name = submit("--max-completion-length", "768", "--max-steps", "50")
check("the node command carries --direct", "--direct" in argv)
check("the cap reaches the node with its value",
      after(argv, "--max-completion-length") == "768", str(after(argv, "--max-completion-length")))
check("max-steps reaches the node", after(argv, "--max-steps") == "50", str(after(argv, "--max-steps")))
check("the selected layer reaches the node", after(argv, "--overlap-layer") == "19",
      str(after(argv, "--overlap-layer")))
check("the selected heads reach the node", after(argv, "--overlap-heads") == "4,9",
      str(after(argv, "--overlap-heads")))
# CONTAINS rather than ends-with: the cap is one of several name-bearing knobs now
# (--mamba-kernels appends after it), and a suffix test would fail every time a new one
# is added -- which is noise, not a finding. What must hold is that the cap is IN there.
check("a 768 run is named _c768", "_c768" in name, name)

# A non-default cap that is NOT the historical one, to prove the value is carried rather
# than a string that happens to read 768 somewhere in the script.
argv512, name512 = submit("--max-completion-length", "512")
check("an arbitrary cap is carried verbatim",
      after(argv512, "--max-completion-length") == "512", str(after(argv512, "--max-completion-length")))
check("and lands in the run name", "_c512" in name512, name512)

# The Mamba kernels are name-bearing too: they change the arithmetic of 23 of the 52
# layers, so a fused run must not land in a directory holding a naive run's checkpoints.
check("the kernel choice reaches the node", after(argv, "--mamba-kernels") == "fused",
      str(after(argv, "--mamba-kernels")))
check("a fused run says so in its name", name.endswith("_fused"), name)
argv_n, name_n = submit("--mamba-kernels", "naive")
check("naive is forwarded too", after(argv_n, "--mamba-kernels") == "naive",
      str(after(argv_n, "--mamba-kernels")))
check("naive carries no kernel suffix (it is the historical path)",
      not name_n.endswith("_fused") and not name_n.endswith("_naive"), name_n)

# The classifier device and the free-text tag. --tag exists so a performance experiment
# gets its own directory without the naming rule growing a suffix per knob, so the one
# thing it must do is reach the node: RUN_NAME is recomputed there for WANDB_RUN_ID, and a
# tag that only the submitting side knows about means two runs sharing one wandb run.
argv_t, name_t = submit("--steps-device", "cuda", "--tag", "t5gpu")
check("the classifier device reaches the node", after(argv_t, "--steps-device") == "cuda",
      str(after(argv_t, "--steps-device")))
check("the tag reaches the node", after(argv_t, "--tag") == "t5gpu", str(after(argv_t, "--tag")))
check("the tag lands LAST in the run name", name_t.endswith("_t5gpu"), name_t)
argv_u, name_u = submit()
check("no tag emits no --tag at all", "--tag" not in argv_u, " ".join(argv_u[-4:]))
check("and leaves the name unsuffixed", not name_u.endswith("_"), name_u)

# The no-saliency arm. Its weights list is THREE long, not four, because
# --reward_variant none builds reward_funcs as [format, accuracy, judge]; a four-value list
# would silently shift the judge's weight onto accuracy and drop the judge. And its name
# must not carry an overlap weight, a layer, heads or a metric -- none of them apply.
argv_n, name_n = submit("--no-saliency", "--max-steps", "3990")
check("--no-saliency resolves to --reward-variant none",
      after(argv_n, "--reward-variant") == "none", str(after(argv_n, "--reward-variant")))
check("the no-sal run is named for what it is", name_n.startswith("grpo-omni30b-nosal"), name_n)
check("and carries no overlap weight in its name", "wov" not in name_n, name_n)
check("the shorthand itself is not forwarded twice", "--no-saliency" not in argv_n)
argv_o, name_o = submit()
check("the default arm is still 'ours'", after(argv_o, "--reward-variant") == "ours",
      str(after(argv_o, "--reward-variant")))
check("and still names its overlap weight", "wov" in name_o, name_o)

# 1024 is the Qwen3-VL value, and the naming rule keys off it: at 1024 there is no suffix,
# because a run that matches the reference runs must not be advertised as deviating.
argv1024, name1024 = submit("--max-completion-length", "1024")
check("1024 is still forwarded", after(argv1024, "--max-completion-length") == "1024",
      str(after(argv1024, "--max-completion-length")))
check("1024 carries no _c suffix", "_c" not in name1024, name1024)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
