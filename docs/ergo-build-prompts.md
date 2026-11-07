# ERGO on Qwen3-VL — agent prompts for build and training

Companion to [baseline-port-estimates.md](baseline-port-estimates.md) § *ERGO on
Qwen3-VL*. That document is the decision; this one is the execution — a sequence of
prompts, each sized for one coding agent, each with the acceptance test that says it
is done.

**Route A (TRL) throughout.** Per the estimate's §4, ERGO's two turns become two
independent training rows carrying the same advantage, so there is no observation
splice, no observation-token loss mask and no hand-rolled position ids. That is the
whole reason this baseline stays in our stack instead of going to verl, and it is why
it is the one arm an agent can build without the silent-correctness risk that governs
DeepEyes and Monet. **Do not let an agent "simplify" this into a concatenated
single-row trajectory.**

**Budget.** ~10–31 h of training on 8×A100 for one 250-step run, plus a comparable
amount of eval. The gates in P1/P2 cost under 3 GPU-hours and can end the project on
day 2, which is a good outcome, not a failure.

---

## 1. How to use this file

Each `### Prompt N` section **is** the prompt — paste the whole section, heading
included, into a fresh agent. Prepend §2 to every one of them.

Prompts at the same depth in §3's graph are independent and should be launched in
the same message so they run concurrently. A prompt that names a branch expects the
agent to create that worktree itself and work only inside it.

---

## 2. Shared preamble — prepend this to every prompt below

> You are working in `/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/saliency_r1`.
> Read `CLAUDE.md` first and obey it exactly.
>
> - The directory you start in is the **central tree** and is **read-only**. Read,
>   grep and launch jobs there; never edit a file and never run a state-changing git
>   command in it.
> - Open your own worktree for all edits: `./worktree.sh new <branch>`, then work
>   entirely inside `.worktrees/<branch-with-dashes>/`. Commit early and often.
> - The user's shell is **fish**. Write every command and snippet in fish syntax.
> - `checkpoint/`, `outputs/`, `cold_data/`, `trl_repo/`, `laser_fork/` are
>   **symlinks shared with every other session and every running job**, not private
>   copies. Write your run outputs to a per-branch subdirectory. Never run
>   `patch_trl_qwen3.sh` or `patch_laser_qwen3.sh` — they re-patch the shared clones
>   for everyone.
> - Never `git add -A`. Stage explicit paths.
> - Do not score test benchmarks while a training run is in flight; dispatch
>   `watch_bench_evals.sh` afterwards.
> - When your task is finished, report the branch path and stop. Do **not** merge to
>   `main` — the user confirms that.
>
> Relevant existing machinery, verified present:
>
> | path | what it is |
> |---|---|
> | `checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` | the cold start every arm starts from |
> | `trl/grpo_trainer_qwen3.py`, `trl/grpo_vlm_qwen3.py` | our GRPO trainer sources (`trl_repo/` is the clone that executes) |
> | `trl/scripts/vllm_serve.py` | our vLLM server; single-image today |
> | `trl/rewards/*.py` | reward modules; `openai_rewards.py` holds the NVIDIA-gateway judge |
> | `build_grpo_sets.py` | builds the GRPO corpora; `MAX_IMAGE_SIDE = 512` at :109, `union_bbox` at :133 |
> | `coldstart_gate.py` | the precedent for a cheap "did this checkpoint learn the format" gate |
> | `eval_mini/benchmarks.py` | the 13-benchmark mini suite table |
> | `cluster_env.sh` | `sr1_pick_partition` / `sr1_find_submit_job`; source it after DURATION is final |
> | `launch_laser_job.sh` | the `_job.sh` (submit) + `_run.sh` (execute in allocation) launcher split to copy |
> | `install_lmms_sinkshift.sh` + `lmms_eval_plugin/` | how an lmms-eval wrapper is registered from this repo |
> | `/home/uberger/scratch/research/lmms-eval` | the lmms-eval clone; wrappers in `lmms_eval/models/chat/` |

---

## 3. Order of work

```
day 1   P0 env ────┐
        P1 Gate A ─┼── all three in parallel; P1 and P2 are BLOCKING
        P2 Gate B ─┤
        P3 RM check┘
                   │
day 2-4            ├─ P4 benchmarks ──┐
                   ├─ P5 hi-res data ─┤
                   ├─ P6 two images ──┼── P7 two-phase rollout ──┐
                   ├─ P8 rewards ─────┤                          │
                   └─ P9 eval wrapper ┘                          │
                                                                 │
day 5              P10 launcher ─────────────────────────────────┤
day 5-7            P11 smoke + real run ─────────────────────────┘
```

**P1 and P2 gate everything after them.** Do not start P5–P11 until Gate A has
returned a number. If Gate A says the low-resolution penalty on Qwen3-VL is under 3
points, the correct action is to stop and write the sentence.

---

### Prompt 0 — a private TRL clone and environment for ERGO

Branch: `feat/ergo-env`

ERGO needs changes to the GRPO trainer (two images per row) and to the vLLM server
(multi-image prompts). Both live in `trl_repo/`, which is a **symlink shared with
every other session and every running job** — editing it would change what in-flight
Qwen3-VL runs execute.

Do what `trl_repo_nemotron` already does for the Omni work: give ERGO its own clone
and its own environment, so nothing you do can reach the live trainer.

1. Create `trl_repo_ergo/` as a clone of the same TRL fork and branch that
   `trl_repo/` is on (read `patch_trl_qwen3.sh` to learn which, and copy its
   structure — do not run it).
2. Add `trl_repo_ergo` to `.worktree-links` with a comment in the house style saying
   what it is and that it is shared.
3. Write `patch_trl_ergo.sh`, modelled on `patch_trl_nemotron.sh`: it copies our
   tracked `trl/` sources into `trl_repo_ergo/` and nothing else. State in its header
   that running it re-patches a shared clone.
4. Create a conda env named `ergo` by cloning `saliency_r1_qwen3`
   (`conda create -n ergo --clone saliency_r1_qwen3`), then `pip install -e
   trl_repo_ergo` into it so `trl` resolves there and not to `trl_repo`.

Verify and report: in the `ergo` env, `python -c "import trl, os;
print(os.path.dirname(trl.__file__))"` must print a path under `trl_repo_ergo`, and
the same command in `saliency_r1_qwen3` must still print `trl_repo`. If the second
one changed, you have broken every running job — fix it before reporting.

Pinned versions that must not move: `torch 2.7.1`, `transformers 5.13.0.dev0`,
`peft 0.19.1`. The vLLM side stays on `saliency_r1_qwen3_vllm` at **vLLM 0.11.0** —
vllm-project/vllm#29595 reports corrupted Qwen3-VL grounding output above 0.11.0,
and grounding output is this entire method.

---

### Prompt 1 — Gate A: does Qwen3-VL even have a low-resolution penalty? (BLOCKING)

Branch: `probe/ergo-gate-lowres`

ERGO's whole headroom is the accuracy that downsampling destroys and a crop
recovers. Their Table 1 on Qwen2.5-VL-7B / V*: 77.0 at 16384 tokens, 64.9 at 1280,
56.5 at 640, and 77.0 again at 640 **with the ground-truth crop appended**. If
Qwen3-VL-8B does not lose those points, the method has nothing to recover.

There is specific reason to doubt it does: DeepStack fuses multi-level ViT features
precisely to hold detail at fewer tokens, and per the Qwen3-VL tech report the
backbone already had tool-integrated RL with an `image_zoom_in_tool` in its own
post-training.

Measure it. Score `vstar_bench`, `hrbench_4k` and `hrbench_8k` on both
Qwen3-VL-8B-Instruct and
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, at pixel caps of
**640, 1280 and 16384 visual tokens**, plus a fourth arm at 640 tokens with the
ground-truth crop appended as a second image (V* and HR-Bench both ship boxes, so
this is a ~40-line task variant).

**The factor is 32, not 28.** Qwen3-VL is patch-16 / merge-2, so one visual token
covers 32×32 px, not Qwen2.5-VL's 28×28. A budget written `N×28×28` copied literally
buys ~23% fewer tokens than ERGO trained on. Express every cap as `N×32×32` and say
so in the report. Note also that `lmms_eval/models/chat/_chat_base.py:40` defaults
`IMAGE_KWARGS = {"max_pixels": 512*512}` — 256 visual tokens, far below even ERGO's
smallest setting — so nothing we have measured so far is at ERGO's operating point.

The task files `vstar_bench/` and `hrbench/` already exist under
`/home/uberger/scratch/research/lmms-eval/lmms_eval/tasks/`. Budget ~1.5–2.5 h on
8 A100s.

Report one table, and read it as:

- **gap ≥ 10 points** between 16384 and 640 — proceed.
- **gap 3–10** — proceed only if the claim is the efficiency axis, not accuracy.
- **gap < 3** — **stop and say so.** "ERGO's premise does not hold on this backbone"
  is a publishable sentence that cost one day.

Write the result to `docs/ergo-gate-lowres.md` and report the verdict in one line.

---

### Prompt 2 — Gate B: will the cold start emit `<zoom>`, and in which coordinate frame? (BLOCKING)

Branch: `probe/ergo-gate-zoom`

Two independent things that both read as "it works" until they don't.

**Format.** `build_coldstart_data.py:52` records Qwen3-VL-8B-Instruct at **format
0.000 under every prompt variant tried** — that is why our cold start exists. ERGO
needs `<think>` in *both* turns plus `<zoom>[x1,y1,x2,y2]</zoom>` in turn 1. If turn
1 is never well-formed, every reward in the GRPO group is 0, the advantage is 0, and
there is no gradient at all — a training run that burns a day and learns nothing.

**Coordinates.** Qwen2.5-VL emits absolute pixels in the post-`smart_resize` frame;
**Qwen3-VL emits a 0–1000 relative grid against the original image.** ERGO's
`src/ergo/utils.py:map_box` rescales by `original/resized`, which on Qwen3-VL crops
the wrong region while continuing to look healthy — the region reward then collapses
into the box-area reward and training looks fine.

Build a gate in the shape of `coldstart_gate.py` (read it first; it is the precedent
for exactly this class of check) that, on both Qwen3-VL-8B-Instruct and our cold
start, over ~200 V*/HR-Bench rows:

1. reports the rate at which a parseable `<zoom>` box is produced in turn 1;
2. reports the rate at which turn 2 matches `<think>…</think><answer>…</answer>`;
3. **decides the coordinate frame empirically** — for each emitted box, compute IoU
   against the ground-truth box under the 0–1000-relative reading and under the
   absolute-pixel reading, and report which interpretation wins. Do not assume;
   report the measurement.
4. reports the median crop area as a fraction of the original, so γ can be sanity-
   checked against ERGO's 0.6.

Use ERGO's released prompts verbatim from `nota-github/ERGO` `src/ergo/const.py`
(clone it to scratch, it is 627 lines and Apache 2.0), adjusted only for the
coordinate convention you measured.

~1 h on one GPU. Report: zoom-parse rate per model, format rate per turn, the
winning coordinate frame with its IoU margin, and median crop area.

**If the zoom-parse rate on the cold start is below ~60%**, a two-turn cold start is
needed (~2–5k SFT rows in ERGO's format, turn-1 boxes taken from the GT annotations
already in `cold_data/grpo_sets`). That is +3–5 days and is the single thing that
blows this arm's budget — say so loudly rather than starting it.

---

### Prompt 3 — the reward model decision (10 minutes, do it first)

Branch: `probe/ergo-reward-model`

ERGO's contribution is `r_region`: a **frozen** reward model, shown **only the crop
and the question** — never the original image — answers correctly or does not. That
withholding is the whole idea; DeepEyes and PixelReasoner show the RM both, so the
crop is never forced to be self-contained.

The paper uses Qwen2.5-VL-72B-Instruct, ~145 GB in bf16, which would be a 4-GPU
sidecar for the whole run — i.e. half our node. Check the cheap option first.

`trl/rewards/openai_rewards.py` already calls the NVIDIA gateway and its default
model is vision-capable. **Unverified, and it is the first thing to check:** that the
gateway accepts image parts in the message, and that a ~1 MP crop fits its payload
limit. Write a ten-line probe that sends one real V* crop plus its question and
prints the reply.

Report one of three outcomes:

1. **gateway works** — zero GPUs, ~128 image calls per step inside the existing
   `ThreadPoolExecutor` judge path. Take it.
2. **gateway rejects images** — fall back to a local **Qwen3-VL-32B** sidecar on 1–2
   A100s, modelled on `serve_grounding_dino.sh`. Cheapest local option and arguably
   the better control, since the RM then shares the policy's visual front-end. Flag
   that this costs 1–2 of the 8 GPUs and so makes every training estimate 1.15–1.35×
   longer.
3. **neither** — local Qwen2.5-VL-72B on 4 GPUs, paper-faithful and expensive.

Whatever you pick, note in the report that the RM choice is a deviation that belongs
in the paper's caption.

---

### Prompt 4 — put the visual-search benchmarks in the table

Branch: `feat/ergo-search-benchmarks`

`eval_mini/benchmarks.py` carries 13 benchmarks and **none of V*, HR-Bench or
TreeBench is among them** — only `mmerealworld` from the natural suite overlaps
ERGO's headline table. This is worth doing regardless of whether ERGO is built: our
effect lives in the visual-search half of the benchmark split, and these are the
canonical visual-search benchmarks.

Add `vstar_bench`, `hrbench_4k`, `hrbench_8k` and `spatialtreebench` to
`eval_mini/benchmarks.py`. Read the module docstring first — every field
(`suite`, `yaml`, `metric`, `scale`, `in_mean`, `group_by`, `item_metric`,
`item_decomposes`) exists for a reason and several were set from what the tasks
actually emitted rather than from the yaml. Set `scale` from a real results.json, not
from the task config.

Then baseline our existing arms on them so the ERGO row has something to sit next
to. Use `watch_bench_evals.sh` / `run_bench_eval.sh` as the dispatch path.

Acceptance: `make_mini_tasks.py` generates the four new configs, `bench_eval.py`
reduces them without a KeyError, and a short run produces a number for each on the
cold start that is within a few points of the published Qwen3-VL-8B figures
(V* ~77.5, HR-4K ~72.4, HR-8K ~68.1 for the Thinking variant — ours is Instruct +
cold start, so expect lower, but not 0 and not 100).

---

### Prompt 5 — training data at native resolution

Branch: `feat/ergo-hires-data`

`build_grpo_sets.py:109` and `trl/grpo_vlm_qwen3.py:122` both cap images at
`MAX_IMAGE_SIDE = 512`. **Training ERGO on 512 px images trains a no-op** — there is
no high-resolution detail left for the crop to recover, so `r_region` is satisfied by
any box containing the object and the method's actual claim is untestable.

Two options; build the second, and keep the first reachable by a flag.

- **Paper-faithful**: `ChenShawn/DeepEyes-Datasets-47k`, which ERGO reuses
  deliberately "to isolate the contribution of our method from dataset curation
  effects". ERGO takes the V* and ArxivQA halves
  (`data_0.1.2_visual_toolbox_v2.parquet`, `data_v0.8_visual_toolbox_v2.parquet`) and
  **not** the ThinkLite math half. 4.84 GB, images embedded, reachable from the login
  node.
- **Comparability-faithful** (build this): our own GRPO sets rebuilt at native
  resolution, so the ERGO arm and our arm differ only in method. Raise
  `MAX_IMAGE_SIDE` behind a parameter rather than editing the constant — the 512 cap
  must keep holding for every existing arm — and rebuild the V*/visdrone rows.

Note what ERGO's training does **not** need: ground-truth boxes. `r_region` is
supervised by the reward model's answer, not by an annotation. Boxes are needed only
to re-derive γ (see P8).

Watch the disk: native-resolution V*/visdrone is tens of GB and `cold_data` already
holds 201 GB. Lustre has room; the rebuild itself is CPU work, so send it to
`cpu_short` (50 GPU-free nodes, instant) rather than queueing behind a GPU
allocation.

Acceptance: a row sampled from the new set, decoded, has its long side at native
resolution; the old 512-capped sets are byte-identical to before; and the rebuild
ran on CPU nodes.

---

### Prompt 6 — two images per row, in the trainer and in the vLLM server

Branch: `feat/ergo-two-images`

Work in `trl_repo_ergo/` (see P0), **not** `trl_repo/`.

ERGO's turn 2 conditions on two images: the downsampled original and the crop. Our
stack is single-image throughout.

1. **`trl/grpo_trainer_qwen3.py`** — `_signature_columns = ["prompt", "image"]`
   (~line 1179) and `split_pixel_values_by_grid` assumes exactly one `image_grid_thw`
   row per sample. Generalise to a per-row image count. This touches the ZeRO-3 path,
   so keep the single-image case byte-identical in behaviour.
2. **`trl/scripts/vllm_serve.py`** (~lines 572–578) takes one base64 image per prompt
   and builds `multi_modal_data={"image": img}`. It needs to accept
   `list[list[str]]`. We own this file.

**The acceptance test is the point of this prompt.** Add a CPU test in the shape of
the repo's existing `test_*_cpu.py` files asserting that:

- a single-image row produces exactly the same `input_ids`, `image_grid_thw` and
  position ids as it did before the change (regression guard);
- a two-image row's `image_grid_thw` has two entries, and the number of
  `<|image_pad|>` tokens equals `sum(grid.prod() // merge_length)` over both;
- the processor built the conversation itself — grep your own diff and confirm you
  have not hand-written a splice, a loss mask, or a position-id computation anywhere.
  If you have, you have reintroduced the bug class Route A exists to avoid, and the
  change is wrong even if it trains.

Do not touch `trl_repo/`. Verify with `git -C trl_repo status --short` returning
clean before you report.

---

### Prompt 7 — the two-phase rollout and the advantage broadcast

Branch: `feat/ergo-two-phase-rollout`

Depends on P6. Work in `trl_repo_ergo/`.

Implement ERGO's rollout in `_generate_and_score_completions` as **two independent
training rows sharing one advantage**:

```
row 1:  prompt = [sys, user(low-res image, Q1)]                  completion = o_region
row 2:  prompt = [sys, user(low-res image, Q1),
                  assistant(o_region), user(crop, Q2)]           completion = o_acc
```

Both carry `Â_g`, computed once per rollout from the four rewards. This is gradient-
identical to the concatenated-trajectory form, because turn 1's text is a *prompt*
for turn 2 and carries no gradient there.

Sequence: generate turn 1 for all rollouts → parse `<zoom>` → crop in PIL on the main
process from the **original, full-resolution** image → `smart_resize` the crop back
under the same token budget → build turn-2 prompts → generate again at `n=1` → emit
2× rows.

Specific things to get right:

- **Coordinates**: use the frame P2 measured. If it is 0–1000 relative, the mapping
  is `x * w / 1000`, and `QUESTION_TEMPLATE1` must say so.
- **The `28` is a `32`.** Every `N×28×28` budget in ERGO's code and tables must be
  restated as `N×32×32` to mean the same token count.
- **Length caps**: `--max_prompt_length 2048` cannot hold turn 2 at N=1280
  (1280 + 1280 visual tokens + text + turn-1 output ≈ 3,300). Raise both caps.
- **`loss_type`**: a sequence-mean loss weights the two turns equally rather than by
  length. Do not silently pick one — expose it and state the choice in the launcher
  header.
- **Everything scores 0 if turn 1 produces no parseable box.** That is ERGO's rule,
  not a bug; implement it, and log the rate.

Log to W&B, per step: zoom-parse rate, mean crop area as a fraction of the original,
the four reward terms separately, and the fraction of groups whose advantage is
exactly 0. That last one is the early-warning signal for the Gate B failure mode.

---

### Prompt 8 — `trl/rewards/ergo_rewards.py`

Branch: `feat/ergo-rewards`

Independent of P6/P7; build against a stub. Four binary terms:

```
R = r_TCE + r_acc + r_format
r_TCE = 1.0 * r_region + 0.5 * r_box

r_region = 1 iff a FROZEN reward model, shown ONLY the crop and the question,
           answers correctly                              <- the contribution
r_box    = 1 iff Area(crop) / Area(original) <= gamma     (paper: gamma = 0.6)
r_acc    = 1 iff the policy's own turn-2 answer is correct
r_format = 1 iff BOTH turns match their tag structure
```

Follow the house style in `trl/rewards/` — read `overlap_rewards.py` and
`openai_rewards.py` first. The RM client comes from P3's verdict.

**`r_region` must never see the original image.** Assert it in the code and test it:
a test that passes an obviously wrong crop must score 0 even when the question is
trivially answerable from the full picture. If that test passes with the full image
accidentally in context, the reward degenerates into `r_box` and the method is gone.

**Re-derive γ on our data.** 0.6 came from the GT-box area distribution in
TreeVGR/VisCoT/V*/VGR. `build_grpo_sets.py` already carries `_boxes_abs` per row
(:362) and `union_bbox` (:133). An hour's work over `cold_data/grpo_sets`; report
both our value and theirs, and use ours.

Ship `test_ergo_reward_cpu.py` alongside, in the shape of the repo's other CPU
reward tests, covering: all four terms independently, the no-parseable-box → all-zero
rule, and the withholding assertion above.

---

### Prompt 9 — the `ergo_qwen3_vl` lmms-eval wrapper

Branch: `feat/ergo-eval-wrapper`

Independent of the trainer. A two-stage chat wrapper that runs ERGO's loop at
inference, so the ERGO arm can be scored on the same benchmarks as everything else.

Cheaper than an agentic wrapper because the loop is fixed at two turns with no
tool-call parsing. The house style is
`/home/uberger/scratch/research/lmms-eval/lmms_eval/models/chat/qwen3_vl_vga.py` and
`qwen3_vl_sinkshift.py`; `thyme.py` in the same directory is the closest structural
precedent for a two-stage loop. ERGO's own `src/ergo/eval.py` is the reference for
the protocol but is a standalone async OpenAI client, not an lmms-eval model.

**Register it the way this repo registers wrappers**: source of truth in
`lmms_eval_plugin/ergo_qwen3_vl.py` in *this* repo, installed into the clone by a new
`install_lmms_ergo.sh` modelled line-for-line on `install_lmms_sinkshift.sh`. Read
that script's header — a wrapper that lives only in an untracked clone exists on
exactly one machine, and we have two clusters.

Expose `--max-pixels` so the pixel-constraint sweep is configuration, not code, and
make the output directory carry the cap when it is set explicitly so sweeps cannot
collide.

Acceptance: `--model ergo_qwen3_vl` resolves; `--model qwen3_vl` still resolves to
exactly the class it did before (`install_lmms_sinkshift.sh --check` is the pattern);
and a 20-document run on `vstar_bench` produces non-zero accuracy with a zoom-parse
rate above 0.

---

### Prompt 10 — the launcher

Branch: `feat/ergo-launcher`

Copy the `_job.sh` / `_run.sh` split used by `launch_laser_job.sh` +
`launch_laser.sh`: the `_job.sh` builds a runner and hands it to `submit_job`, the
other shards work across the allocation's GPUs once it exists.

Requirements:

- `source cluster_env.sh` and resolve `PARTITION` with `sr1_pick_partition`
  **after** argument parsing, not next to the defaults — which partitions can hold
  the job depends on how long it asks for.
- Partitions cap at 2/4/8 h. A 250-step ERGO run is 10–31 h on 8 A100s, so **chunked
  autoresume is mandatory** — 2–4 jobs with `save_freq` and resume-from-latest. Each
  restart pays model load + vLLM engine init; keep chunks at the longest the
  partition allows.
- Write outputs to a per-branch subdirectory under the shared `outputs/` and
  `checkpoint/` symlinks so parallel sessions do not clobber each other.
- If P3 chose a local reward model, the launcher reserves its GPUs and the training
  ranks drop to 6–7. Make that explicit in the header with its cost.
- Header in the house style of `launch_grpo_qwen3_overlap_job.sh`: state what is not
  optional, what the first scored step should print, and what a mismatch there means.

---

### Prompt 11 — smoke, then the real run

Branch: `run/ergo-smoke`

**Smoke first, and read it before scaling.** 20 steps, logging on. The four things
that decide whether the real run is worth starting, in order:

1. **zoom-parse rate** — if this is near 0, Gate B was wrong or the prompt drifted.
   Stop.
2. **fraction of groups with zero advantage** — if most groups are all-zero, there is
   no gradient and the run is a space heater regardless of what the loss curve does.
3. **crop area distribution** — if it collapses to the γ threshold immediately, the
   policy has found `r_box` and is ignoring `r_region`. That is the degenerate
   solution; check that `r_region` is actually firing and actually blind to the
   original image.
4. **OOM at 2×1280 visual tokens and prompt truncation** — both show up in the first
   few steps or not at all.

Then the real run: 250 steps, global batch 128, 8 rollouts per example, lr 1e-6,
AdamW. **Fidelity note to carry in the writeup:** ERGO's Table 7 says 8 rollouts and
Appendix B's prose says 16. Pick one and state which. Appendix B's "half of each
mini-batch was allocated to longer rollouts" is not reproducible from the paper —
ignore it and say so.

Expect **10–31 h on 8 A100s**, 2–4 chunked jobs. Do not score test benchmarks while
it runs; dispatch `watch_bench_evals.sh` afterwards.

**The comparability trap, for the caption.** ERGO is an *efficiency* method — its
claim is accuracy at a reduced token budget, headline "+4.7 on V* at 23% of the
vision tokens". Our model has never been evaluated under a pixel cap at all; the
default is 512×512, 256 tokens. An honest table reports accuracy **at a matched pixel
budget for both arms**, plus vision-token counts and latency, or it is comparing a
method tuned for a budget against one that never saw it. Re-score our arms at the
same caps.

---

## 4. The traps, in one place

| trap | where it bites | what it looks like when it bites |
|---|---|---|
| 0–1000 relative coordinates | P2, P7 | every crop lands in the wrong place; `r_region` collapses into `r_box`; **training looks fine** |
| `IMAGE_FACTOR = 28` is 32 on Qwen3-VL | P1, P7 | every token budget is 23% off; every row of an efficiency table is at the wrong operating point |
| format 0.000 on the Instruct model | P2 | every reward in the group is 0, advantage is 0, no gradient, no error |
| `MAX_IMAGE_SIDE = 512` | P5 | the method trains a no-op — nothing for the crop to recover |
| `r_region` seeing the original image | P8 | the reward degenerates to box area; the paper's one idea is silently removed |
| editing `trl_repo/` instead of `trl_repo_ergo/` | P0, P6, P7 | in-flight runs on other branches change underneath their ranks |
| vLLM above 0.11.0 | P0 | corrupted Qwen3-VL grounding output (vllm#29595) — i.e. corrupted `<zoom>` boxes |
| γ = 0.6 taken on faith | P8 | the box reward is calibrated to someone else's box-size distribution |

## Sources

* [arXiv:2509.21991](https://arxiv.org/abs/2509.21991) — the paper (ICLR 2026)
* [nota-github/ERGO](https://github.com/nota-github/ERGO) — 627 lines, inference and eval only, Apache 2.0
* [nota-ai/ERGO-7B](https://huggingface.co/nota-ai/ERGO-7B) — the public checkpoint, for the cheap reference in P1
* [baseline-port-estimates.md](baseline-port-estimates.md) — the estimate this executes
