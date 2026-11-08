# Running ERGO's two gates — the prompt for a session on the second cluster

Companion to [ergo-build-prompts.md](ergo-build-prompts.md). That document builds the
gates; this one runs them, from a Claude Code session started on
`cs-oci-ord-login-01` while the first cluster has no GPU hardware.

## 1. How to use this file

§2 **is** the prompt — paste it whole, heading included, into a fresh session on the
second cluster. It is written for a session driven from an **interactive allocation**
the user provides, not from batch submission, and it says so repeatedly because both
launchers submit by default.

Two things in it are inferences rather than facts checked from the first cluster, and
the prompt tells the session to verify both before asking for GPUs:

* **the repo path there** — `/home/uberger/scratch/research/saliency_r1` is what the
  Gate B agent reported, but nothing here can confirm it;
* **whether the cold start and the V\* snapshot exist on that filesystem at all.** The
  two clusters share no filesystem. The checkpoint was copied there for earlier work
  and `HF_HOME` was populated *here*; neither is guaranteed. Discovering it under
  `salloc` wastes the allocation, so it is pre-flight.

---

## 2. The prompt

You are continuing the ERGO baseline port on the **second cluster**
(`cs-oci-ord-login-01`). Everything you need was built and merged yesterday on the
other cluster; your job is to produce the two numbers that unblock it.

Start in the repo (confirm the path; it should be
`/home/uberger/scratch/research/saliency_r1`). **Read `CLAUDE.md` first and obey it
exactly** — the directory you start in is the read-only central tree, all edits go in
a worktree via `./worktree.sh new <branch>`, and the user's shell is **fish**.

### First: get current

The two clusters share no filesystem, so this clone is behind. Run
`./worktree.sh sync` before anything else. You need `main` at `87ad90f` or later;
`ergo_gate_lowres.py`, `ergo_zoom_gate.py` and both `launch_ergo_*_job.sh` must exist.

### What you are doing and why

`docs/ergo-build-prompts.md` is the plan for porting ERGO (arXiv:2509.21991) onto
Qwen3-VL. Day 1 ran: the environment and the reward-model decision are done, and both
blocking gates are **built, CPU-tested and completely unrun** — the first cluster lost
every GPU node. P5 through P11 are blocked until these two gates return numbers, and
one of the valid outcomes is cancelling the whole arm.

- **Gate A** (`ergo_gate_lowres.py`, `docs/ergo-gate-lowres.md`) — does Qwen3-VL-8B
  even have a low-resolution penalty for a crop to recover? V*, HR-Bench 4K and
  HR-Bench 8K, on vanilla Instruct and on our cold start, at visual-token caps of 640 /
  1280 / 16384, plus a V* arm at 640 with the ground-truth crop appended. 14 units,
  wants 8 GPUs, ~4 h.
- **Gate B** (`ergo_zoom_gate.py`, `docs/ergo-gate-zoom.md`) — will the cold start emit
  a parseable `<zoom>` box, and in which coordinate frame? 191 V* rows × 2 turns × 2
  models. One GPU, ~2 h.

### IMPORTANT: do not submit anything. Wait for an interactive allocation.

The user is going to hand you an **interactive shell allocation** and will tell you
when it is live. Until then:

- **Do not run `launch_ergo_gate_lowres_job.sh` or `launch_ergo_zoom_gate_job.sh`
  without `--dry-run`.** Those submit batch jobs. You are not submitting batch jobs.
- Use `--dry-run` on both to print the exact runner script each would have submitted —
  that is where the conda env, the environment variables and the real python command
  line come from. Read them; do not reconstruct the commands from the headers.
- Do all the login-node pre-flight below, then **stop and tell the user you are ready
  and waiting for the allocation**, saying how many GPUs and how much walltime each
  gate wants.

When the allocation exists, ask how many GPUs and how long it has, then run the python
entry points from the dry-run output directly inside it. Gate A **resumes** — a unit
with a harvested json is skipped, and it exits 17 while units are still owed and 0 when
the grid is complete — so it is the one that tolerates a short or interrupted
allocation. Gate B does not resume; give it a clean ~2 h.

### Pre-flight, all of it on the login node, before the allocation

A wasted GPU allocation is the thing to avoid. Verify, and fix what you can:

1. **The cold start exists here.** `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`
   was copied to the *other* cluster's filesystem. If it is missing or partial here,
   that is a blocker — report it immediately rather than discovering it under salloc.
2. **The V\* snapshot is in `HF_HOME`.** `craigwu/vstar_bench`, 191 rows. Gate B's
   runner sets `HF_HUB_OFFLINE=1`, so a compute node cannot fetch it. Fetch it from the
   login node now if it is absent. HR-Bench likewise.
3. **The conda envs the dry-run output names actually exist here**, and
   `assets/vstar_bench_boxes.json` is present (191 boxes; only V* ships boxes at all —
   HR-Bench ships none and has no crop arm).
4. **Run both CPU suites.** `python test_ergo_gate_lowres_cpu.py` needs the
   **`lmms_eval`** env — in `saliency_r1_qwen3` seven tests error on a missing import,
   which is the env, not a defect. `python test_ergo_zoom_gate_cpu.py` should be 50/50.
5. **`sinfo -h -N -o "%G" | grep -ci gpu`** — confirm this cluster actually has GPU
   hardware. A partition can be `State=UP` with `TotalNodes=0`, which accepts a job and
   parks it at `PartitionConfig` forever. That is exactly what stopped the first cluster.

### Traps that produce a confident number meaning nothing

- **`test_ergo_gate_lowres_cpu.py` prints a verdict table ending
  `VERDICT: PROCEED -- +21.0 points`. Those are synthetic fixture numbers.** Gate A has
  never produced a real result. Do not report them, ever.
- **`--force_simple` is not optional.** lmms-eval's `qwen3_vl` chat wrapper ignores
  `max_pixels` entirely (it passes no `image_kwargs` and sets `do_resize=False`), so
  without it every resolution arm is the same arm and the gap reads 0.0 — which is also
  the "abandon ERGO" answer. Confirm it is in the command you actually run.
- **Never `--use_cache` here.** The cache key omits the pixel budget, so one arm's
  answers get served back for every arm and the gap again reads exactly 0.0.
- **The factor is 32, not 28.** Qwen3-VL is patch-16 / merge-2, so one visual token is
  32×32 px. A literal `1280×28×28` buys **973** tokens, not 1280. Every cap must be
  `N×32×32`.
- **The "16384-token" arm on V\* is really ~3.3k tokens** — the images are 3 MP and are
  never enlarged to meet a cap. Say so in the writeup; ERGO's own table has the same
  property.
- Gate B's first scored lines must show **`processor_resized_under_us 0`** and
  **`coord hint: 'none'`**. Anything else and the frame table is measuring something
  other than what the model saw — stop and fix the pixel plumbing before reading on.

### How to read the results

**Gate A**, on the 16384-vs-640 gap:

- **≥ 10 points** — proceed.
- **3–10** — proceed only if the claim is the efficiency axis, not accuracy.
- **< 3** — **stop and say so.** "ERGO's premise does not hold on this backbone" is a
  publishable sentence, not a failure.

**Gate B**: report the zoom-parse rate per model, the format rate per turn, the winning
coordinate frame with its margin, and the median crop area. The frame is scored three
ways (structural, IoU, centre error) because they can disagree — V*'s targets are a
fraction of a percent of the image, so a crop containing the whole answer still scores
IoU ≈ 0. If it prints `DISAGREEMENT`, do not let IoU read as settled.

If the **zoom-parse rate on the cold start is below ~60%**, a two-turn cold start is
needed: +3–5 days, and the single thing that blows this arm's budget. **Say so loudly;
do not start it.**

### When you have the numbers

Write the real results into `docs/ergo-gate-lowres.md` and `docs/ergo-gate-zoom.md` on
a worktree branch, with the verdict in one line each. Report the branch path and the
two verdicts, and stop — **do not merge to `main`**; the user confirms that. If you do
merge later, note that pushing matters here: the other cluster can only see this work
through `origin`.
