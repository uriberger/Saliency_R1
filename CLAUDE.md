# Claude Code Instructions

## Shell

- The user works in the **fish** shell. Always write shell commands and snippets in fish syntax (e.g. `set -x VAR value`, `set -e VAR`, `set -q VAR`, `command; and command2`), not bash/POSIX syntax.

## Git Workflow

Several Claude sessions work on this repo at the same time. The directory a
session is launched in is the **central tree**: the shared, always-current copy
of `main`, which every session reads and none of them modifies. All change-work
happens in a disposable worktree under `.worktrees/`, managed by `./worktree.sh`.

### In the central tree: read only

- Read, grep, run analysis, inspect results, launch jobs — never write.
- **Never edit a file, and never run a git command that changes state there**:
  no `checkout`/`switch`, no `branch`, no `stash`, no `commit`, no `merge`, no
  `pull` that moves it. Another session is reading these files right now.
- It stays on `main`. Keep it current with `./worktree.sh sync` when needed.

### The moment a change is needed: open a worktree

```fish
./worktree.sh new fix/judge-retries
```

This branches off `main` into `.worktrees/fix-judge-retries/` and symlinks the
shared gitignored paths listed in `.worktree-links`. Then work **entirely inside
that directory** — all edits, commits and tests. Commit early and often, with
clear messages. If you were resumed and a worktree for this task already exists
(`./worktree.sh list`), continue in it instead of creating another.

### When the change is finished

Ask: *"Ready to merge `<branch>` into main and delete the worktree — shall I
proceed?"* Wait for explicit confirmation.

- If confirmed, run from the central tree (never from inside the worktree):
  ```fish
  ./worktree.sh done fix/judge-retries
  ```
  It rebases the branch onto the current `main`, fast-forwards `main`, then
  deletes the branch and the worktree. Nothing is pushed to `origin` unless you
  add `--push`, and only if the user asked for it.
- If declined, leave the worktree in place and report its path so the work can
  be resumed later. `./worktree.sh drop <branch>` discards it entirely.
- If `done` reports rebase conflicts, resolve them inside the worktree, commit,
  and rerun it. Do not resolve anything in the central tree.

### Cautions

- Ignored artifacts are **symlinked, not copied** — a worktree does not give you
  a private `outputs/` or `checkpoint/`. Write run outputs to a per-branch
  subdirectory so parallel sessions don't clobber each other.
- `trl_repo/` is shared the same way, and it is the copy that actually executes.
  Running `patch_trl_qwen3.sh` from a worktree re-patches it for every session
  and every running job — say so before doing it.
- The conda envs are shared too. Treat `patch_*.sh` and any `pip install` as
  global, not worktree-local.
- Never force-push, never rebase published commits, never commit directly to
  `main`, never delete `main`.
