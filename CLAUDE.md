# Claude Code Instructions

## Shell

- The user works in the **fish** shell. Always write shell commands and snippets in fish syntax (e.g. `set -x VAR value`, `set -e VAR`, `set -q VAR`, `command; and command2`), not bash/POSIX syntax.

## Git Workflow

- The canonical branch is `main`. All finished work lives there.
- **Before making any file changes**, create a new branch off `main` with a short descriptive name (e.g. `feat/add-trainer-flag`, `fix/cuda-path`). Never commit directly to `main`.
- Work on that branch for the duration of the session. If a branch for this session already exists (e.g. you were resumed), continue on it.
- **After finishing the changes**, ask: *"Ready to commit and merge into main — shall I proceed?"* Wait for explicit confirmation before doing anything.
  - If confirmed: commit all changes with a clear message, merge the branch into `main` (fast-forward if possible, otherwise a merge commit), then delete the branch locally.
  - If declined: leave the branch as-is and report its name so work can be resumed later.
- Never force-push, never rebase published commits, never delete `main`.
