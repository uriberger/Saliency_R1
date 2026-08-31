#!/usr/bin/env python
"""Every relative import in trl/ must still resolve after patch_trl_qwen3.sh moves the file.

`trl/` is the tracked source, `trl_repo/` is what executes, and the two have DIFFERENT
layouts: the patch script copies `trl/grpo_trainer_qwen3.py` and the map modules into
`trl_repo/trl/trainer/`, the reward modules into `trl_repo/trl/rewards/`, and
`trl/grpo_vlm_qwen3.py` into `trl_repo/examples/scripts/`. A relative import is resolved
against the directory the file ENDS UP in, so `from .rewards.grad_rewards import ...` --
correct where it is written, because `trl/rewards/` is a sibling there -- becomes
`trl.trainer.rewards` in the copy and raises ModuleNotFoundError.

Nothing else catches this. Every CPU test loads the reward modules directly (or with a
stub package), and `trl/` imports fine in place; the layout only differs in the tree no
CPU test touches. So the failure surfaces on a GPU, inside a live GRPO step, at whichever
line first runs the bad import -- which for the gradient reward is the diagnostics drain,
AFTER generation, the re-forward, the backward and the DINO calls of step 0.

    python test_import_layout_cpu.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PATCH = REPO / "patch_trl_qwen3.sh"
TRL_REPO = REPO / "trl_repo"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('   ' + detail) if detail else ''}")


def copied_files() -> list[tuple[Path, Path]]:
    """-> [(source file, destination DIRECTORY)] parsed out of patch_trl_qwen3.sh.

    Parsed rather than hardcoded so a newly copied file is covered the day it is added
    to the patch script, which is the only place the mapping is written down.

    Shell variables are expanded, because the one file that has to be covered --
    grpo_trainer_qwen3.py -- is the one copied as `cp "$SRC" "$DST"` rather than with
    its paths inline. A version of this test that only matched the inline form parsed
    ten files and silently skipped the file the bug was in.
    """
    assign = re.compile(r'^\s*(\w+)="([^"]+)"\s*$')
    copy = re.compile(r'^\s*cp\s+"([^"]+)"\s+"([^"]+)"')
    var = re.compile(r"\$(\w+)")
    env = {"REPO": str(REPO), "TRL_REPO": str(TRL_REPO)}

    def expand(s: str) -> str:
        for _ in range(5):                       # nested vars: "$SRC" -> "$REPO/trl/..."
            new = var.sub(lambda m: env.get(m.group(1), m.group(0)), s)
            if new == s:
                break
            s = new
        return s

    out = []
    for line in PATCH.read_text().splitlines():
        if m := assign.match(line):
            # Record every assignment, so $SRC/$DST hold whatever the last one set.
            env[m.group(1)] = expand(m.group(2))
        elif m := copy.match(line):
            src, dst = Path(expand(m.group(1))), Path(expand(m.group(2)))
            if "$" not in str(src) and "$" not in str(dst) and src.is_file():
                out.append((src, dst.parent))
    return out


def relative_imports(src: Path) -> list[tuple[int, str, str]]:
    """-> [(lineno, first component after the single dot, the whole statement)].

    Only single-dot imports: `..x` reaches above the destination package and is a
    different question, and both files that use it (`..data_utils` and friends) are
    original TRL modules whose location does not change.
    """
    pat = re.compile(r"^\s*from\s+\.(\w+)")
    out = []
    for i, line in enumerate(src.read_text().splitlines(), 1):
        m = pat.match(line)
        if m:
            out.append((i, m.group(1), line.strip()))
    return out


def test_relative_imports_survive_the_copy():
    print("\n[layout] every `from .x` resolves in the directory the file is copied to")
    pairs = copied_files()
    check("the patch script's cp lines parse", len(pairs) >= 8, f"{len(pairs)} files")
    # Self-check: the file the bug was in must be among them, whatever form the patch
    # script writes its cp in. Without this the guard can go quietly blind again.
    covered = {p.name for p, _ in pairs}
    check("grpo_trainer_qwen3.py is covered", "grpo_trainer_qwen3.py" in covered,
          f"covered: {len(covered)} files")
    if not TRL_REPO.is_dir():
        print(f"  SKIP  trl_repo not present at {TRL_REPO}")
        return

    # What a copied file's relative imports may name: a sibling already in the
    # destination, or a sibling that the patch script puts there in the same run.
    landing: dict[Path, set[str]] = {}
    for src, dst in pairs:
        landing.setdefault(dst, set()).add(src.stem)

    for src, dst in pairs:
        siblings = {p.stem for p in dst.glob("*.py")} | {
            p.name for p in dst.iterdir() if p.is_dir()
        } | landing.get(dst, set())
        for lineno, name, stmt in relative_imports(src):
            check(
                f"{src.relative_to(REPO)}:{lineno}  .{name}",
                name in siblings,
                f"-> {dst.relative_to(TRL_REPO.parent)}/  [{stmt}]",
            )


def absolute_trl_imports(src: Path) -> list[tuple[int, str, str]]:
    """-> [(lineno, dotted module after `trl.`, the whole statement)].

    `from trl.rewards.overlap_rewards import ...` resolves the same in both trees, which
    is why it is the recommended form -- but only if the module it names is actually in
    the tree that executes.
    """
    pat = re.compile(r"^\s*from\s+(trl\.[\w.]+)\s+import\b")
    out = []
    for i, line in enumerate(src.read_text().splitlines(), 1):
        m = pat.match(line)
        if m:
            out.append((i, m.group(1), line.strip()))
    return out


def our_source_for(dotted: str) -> Path | None:
    """The file in THIS repo that `trl.a.b` names, or None if the module is upstream's.

    Upstream modules (trl.data_utils and friends) are already in trl_repo and are not
    ours to copy; only the ones we maintain need a cp line.
    """
    rel = Path(*dotted.split(".")[1:])
    for cand in (REPO / "trl" / rel.with_suffix(".py"), REPO / "trl" / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def test_absolute_imports_are_copied():
    """Every module we maintain and import absolutely must have a cp line.

    The relative-import check above cannot see this class of bug: an absolute import is
    spelled identically in both trees and resolves fine in `trl/`, so nothing fails until
    the module is missing from `trl_repo/` at runtime. That is how 069bd32 shipped
    `maskfree_rewards.py` without a cp line -- and because the trainer imports it at the
    top of its `reward_variant == "ours"` metrics block, BEFORE the is_active() guard, the
    consequence was an ImportError on the first metrics log of every attention-overlap
    run, not just the --maskfree ones it was written for. It surfaced only on a second
    cluster, because the first had the file copied in by hand.

    Deliberately independent of trl_repo/: the invariant is a property of the patch
    script, so this fails on a fresh clone too.
    """
    print("\n[layout] every `from trl.x` module we maintain has a cp line in the patch script")
    pairs = copied_files()
    sources = {p.resolve() for p, _ in pairs}
    seen = set()
    for src, _dst in pairs:
        for lineno, dotted, stmt in absolute_trl_imports(src):
            ours = our_source_for(dotted)
            if ours is None:
                continue                        # upstream module, already in trl_repo
            key = (src, dotted)
            if key in seen:
                continue
            seen.add(key)
            ok = ours.resolve() in sources
            check(
                f"{src.relative_to(REPO)}:{lineno}  {dotted}",
                ok,
                # Only on failure: the remedy is noise next to 20 passing lines, and the
                # relative-import check above already prints its context unconditionally
                # because there the destination directory IS the information.
                "" if ok else
                f"-> add `cp \"$REPO/{ours.relative_to(REPO)}\" \"$TRL_REPO/...\"` "
                f"to patch_trl_qwen3.sh  [{stmt}]",
            )


def main():
    test_relative_imports_survive_the_copy()
    test_absolute_imports_are_copied()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        print("\n  A `.x` failure: use an absolute `from trl.rewards.<mod> import ...` "
              "instead --\n  it resolves in both trees, and grpo_vlm_qwen3.py already "
              "imports that way.")
        print("  A `trl.x` failure: the import is fine, the module is just never copied "
              "into\n  trl_repo/. Add the cp line to patch_trl_qwen3.sh -- the line the "
              "message names.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
