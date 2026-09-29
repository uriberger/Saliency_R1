# Handoff prompt: 50 clean steps of the Omni arm, and what a step costs

Run `launch_grpo_omni_overlap_job.sh` for **50 steps on the selected heads**, establish that
it does not OOM, measure where the step time goes, and — if it is too slow — propose the
fix with numbers behind it.

The reward is now `--overlap-layer 19 --overlap-heads 4,9`, selected on this model
(`docs/omni-head-selection.md`). The last Omni run trained the inherited `33 / 28,31`,
reached **step 30 of 50**, and then **every resume OOMed on its first backward**. So "50
steps without OOM" is an open problem, not a formality — it is most of this task.

## Read these first, in this order

| file | what it gives you |
|---|---|
| `launch_grpo_omni_overlap_job.sh` (the header) | the operating manual, and the four sections that matter here: PLAN A, THE MEMORY CEILING, THE SALIENCY LAYER AND HEAD PAIR, TWO ENVS |
| `docs/omni-training-harness.md` §10, §11 | the memory measured per rank, and exactly where the last run got to |
| `docs/omni-head-selection.md` | why 19 / 4,9, and the caveat that goes on any number read off this run |
| `trl/rewards/length_guard_rewards.py` | the lever §11 names and the Omni launcher does not expose |

Memory entries `omni-grpo-step-time`, `omni-completion-cap-is-the-ceiling`,
`omni-grpo-two-silent-traps` and `omni-head-selection-result` cover the same ground short.

## 1. The command, and the one flag that is not optional

```bash
bash launch_grpo_omni_overlap_job.sh --max-completion-length 768 --max-steps 50
```

**`--max-completion-length 768` is load-bearing and is NOT the default.** The launcher still
defaults to 1024, which is the Qwen3-VL value; at 1024 this run reaches step 12–15 and dies
on whichever micro-step is carrying a completion that reached the cap. 768 is what the
previous run used, it is in the run name (`_c768`), and any comparison against a Qwen3-VL
run has to state it.

*(Flipping that default was deliberately left undone — it is a memory-critical
hyper-parameter and nobody had asked. If this session confirms 768, flip it and say so.)*

The run name resolves to `grpo-omni30b-overlap__wov0.2_L19_h4-9_mean_in_c768`, which cannot
collide with the previous `_L33_h28-31_..._c768` checkpoint. **Do not resume that one** —
different heads means a different reward.

## 2. What "does not OOM" means here, and why 768 may not be enough

`docs/omni-training-harness.md` §10, per training rank, before a micro-step's forward:

```
                              allocated   reserved   peak      of 79.2 GB
through the DINO server          61.6       61.6     66.4
```

61.6 GB is weights. The forward adds **0.3 GB** — all 52 blocks recompute, so gradient
checkpointing is doing its job. What fills the card is the WORKING SET: a 128-expert
mixture materialises ~11.4 GB of intermediates for a ~1,400-position sequence and the
backward asks for one further ~5.5 GB block. **Both are linear in tokens.**

So the failure is not a fixed wall, it is a drift: **this reward lengthens chains**, the
working set grows with them, and a fixed cap is reached eventually rather than never. That
is exactly what happened — 30 steps at 768, then every resume died on its first backward.

**Instrument it rather than waiting for the crash.** `SR1_MEM_REPORT=N` prints the per-rank
allocated/reserved/peak for the first N micro-steps (`grpo_trainer_qwen3.py:3440`); the
launcher sets 6. Raise it, and **plot peak memory against step number and against
`completions/mean_length`**. If peak is climbing with mean length, 50 steps at 768 will not
finish and you know it by step 10 instead of step 31.

### The three levers, in the order to prefer them

1. **`--length-guard`** — the repo's own answer to "this reward lengthens chains", and the
   one that makes the cap HOLD instead of drift. `trl/rewards/length_guard_rewards.py` is
   already copied into `trl_repo_nemotron` by `patch_trl_nemotron.sh`, so the machinery is
   installed — but **the Omni launcher does not expose the flags**. Wiring it is a bounded
   job: copy the five `LENGTH_GUARD_*` defaults, the arg-parse cases and the forwarding
   lines from `launch_grpo_qwen3_overlap_colocated_job.sh` (search `--length-guard`). It is
   a reward term, so it is a second hyper-parameter change and the run name must say so.
2. **A lower cap (512).** Cheap and certain, but it truncates completions that FINISH — the
   previous policy's longest terminated completion was 594 tokens, so 512 cuts real answers
   where 768 only cuts runaways. Read `completions/clipped_ratio` before and after.
3. **A card bigger than 80 GB.** Changes nothing else, if one exists on either cluster.

## 3. The timing question, and where to look first

Measured on the previous run (§11):

```
step time     215.7 s   on the submitted (container) path
              97-124 s  on a bare `srun`
               34.0 s   training-side benchmark only (omni_train_step_bench.py)
```

At 215 s, 50 steps is **3 hours** and a 2 h `batch_short` chunk buys ~33 steps. At 100 s it
is 1h23m and fits one chunk. **So the container-vs-srun gap is worth more than any other
optimisation on the table, and nobody has explained it.** Measure it first: same config,
one chunk each way.

### Budget the step before optimising it

The reward side dominates, but the arithmetic does not currently close:

| | |
|---|---|
| training side, benchmarked | 34.0 s |
| generation: 8 rollouts of one real prompt on the vLLM server | 7.0–9.8 s |
| saliency capture: 8 teacher-forced forwards, ONE layer, no_grad | **~5 s, inferred** |
| **accounted** | **~50 s** |
| observed, bare srun | 97–124 s |
| observed, container | 215.7 s |

The ~5 s is inferred from this session's head-selection scan, which did the same
teacher-forced forward on the same model and measured **0.65 s per case** — and that
captured **all six** attention layers where the trainer captures one, under no_grad, batch
1, `logits_to_keep=1`, on a warm model. Treat it as an order of magnitude, not a
measurement: re-measure it in the trainer before acting on it.

If that holds, **50–170 s a step is unaccounted for** and it is not the model. The
candidates, in the order I would test them:

* **Grounding-DINO round trips.** One HTTP request per reward call, images base64'd.
  `--dino_api_base` is set (it must be — see §5), but the server is on GPU 0 of the same
  node and the payloads are large.
* **The FLAN-T5 observe classifier**, which §10 moved to CPU to save memory. That trade may
  now be costing more than it saves — it segments every completion of every rollout.
* **The container path itself** — filesystem, NCCL init per step, or the lustre mounts.
* **Weight sync.** Already scoped to 18 tensors by `SR1_VLLM_SYNC_LORA_ONLY=1`, so this
  should be small; confirm rather than assume.

**One thing that is already done — do not re-derive it.** The saliency capture does NOT
re-run the attention module in eager on this model. `grpo_trainer_qwen3.py:1878` reads
`fam.attention_weights_are_returned`, and `NemotronHAttention.forward` returns
`(output, weights)` because `nemotron_loader` pins eager. The re-run is a fallback.

## 4. Things that will cost you a day if you rediscover them

* **Two conda envs.** `nemotron` (torch 2.8, transformers 5.13.0.dev0) trains;
  `nemotron_vllm` (torch 2.11, vLLM 0.20.2) serves generation. The launcher handles both.
* **`--dino_api_base` is not optional.** Without it every training rank builds its own
  Grounding-DINO on its own already-full card: ~8 GB a rank, peak 74.1 GB against 66.4, and
  the visible symptom is 123 `[dino] CUDA OOM; retrying batch` lines in half an hour with
  the detector's own card idle at 1.4 GB. The launcher passes it; check the log says so.
* **The prompt opens `<think>`.** Completions carry only the closing tag. The trainer reads
  this off the actual prompt now, but anything new that parses the format has to know, or
  it reports `format 0.000` on a model that is fine.
* **`patch_trl_nemotron.sh` rewrites a SHARED clone.** Say so before running it; it patches
  `trl_repo_nemotron` for every session and every running job.
* **ptrace is off on these nodes**, so py-spy and gdb cannot attach. `faulthandler` +
  `dump_traceback_later` from inside the process is the way in — `omni_vllm_probe.py
  --watchdog` is the worked example.
* **Do not score test benchmarks during training.** Dispatch `watch_bench_evals.sh` after.
* **Read CLAUDE.md**: the launch directory is read-only, all changes happen in a worktree
  via `./worktree.sh new <branch>`, and the user works in fish.

## 5. A caveat that belongs on whatever this run produces

`docs/omni-head-selection.md` §4: the corpus the heads were selected on has a correctness
label that is **52% disputed**, because the Omni is a BASE checkpoint whose prose answers
`accuracy_reward`'s exact-string fallback cannot read. `19 / 4,9` is a defensible
improvement on an inherited pair — it is not a settled pair, and cold-starting the Omni is
the honest fix. A reward number off this run is a number about `19 / 4,9`, not about "the
Omni's saliency heads".

## Done means

* 50 steps on the board with no OOM, at a stated `--max-completion-length`, and peak memory
  plotted against step and against mean completion length so the next person can see
  whether it was about to fail.
* A step-time budget that CLOSES — every second of the observed step attributed, on both
  the container and the bare-`srun` paths, and the gap between them explained.
* If it is too slow: a named fix with a measured cost and a measured benefit, not a guess.
  If it is fast enough, say that plainly and say what "enough" was measured against.
