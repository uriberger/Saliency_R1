# ERGO on Qwen3-VL — port and training estimate

Assessment written 2026-11-07. The question: what would it cost to implement ERGO
(Lee, Shin et al., ICLR 2026, [arXiv:2509.21991](https://arxiv.org/abs/2509.21991)) on
Qwen3-VL-8B so the result is a comparable row next to our model?

**Verdict: ~14–18 person-days and ~400–1,000 A100-hours — roughly half of what
`docs/deepeyes-port-estimate.md` costs — but run the 1-day gate in §8 FIRST, because
ERGO's entire headroom is the low-resolution penalty, and Qwen3-VL may not have one.**

Unlike DeepEyes, this one is a good fit for our existing TRL stack: ERGO's rollout is
exactly two turns, not an open-ended agent loop, and two turns can be trained as two
ordinary `(prompt, completion)` rows sharing one advantage. No observation splicing, no
hand-rolled position ids, no loss-mask surgery. That removes the single argument that
sent DeepEyes to verl.

Repos cloned for this assessment (scratch, not under version control):

    /tmp/ERGO                                   # nota-github/ERGO @ main (eval only, 627 lines)
    /tmp/ergo_paper/ergo.txt                    # arXiv:2509.21991v2, HTML -> text


## 1. What the method is, mechanically

Qwen2.5-VL-7B-Instruct, GRPO, **250 steps**, global batch 128, 8 rollouts per example
(the appendix text says 16 — see §10), lr 1e-6, AdamW, 4×H100, **~150 GPU-hours**.
No cold start: the authors chose Qwen2.5-VL explicitly because its referring-detection
ability makes grounding work without one.

A rollout is **two fixed turns**, not a tool loop:

1. The image is downsampled to a pixel cap (`N×28×28`, with `N ∈ {640, 1280, 16384}`),
   and the policy emits a think trace plus a box:
   `<think>…</think><zoom>[x1, y1, x2, y2]</zoom>`.
2. That box is cropped from the **original, full-resolution** image, `smart_resize`d back
   under the same pixel cap, and appended as a fresh user turn. The policy answers from
   `[downsampled image, turn-1 output, crop, question]` as
   `<think>…</think><answer>…</answer>`.

The prompts are released verbatim in `src/ergo/const.py` (`SYSTEM_PROMPT`,
`QUESTION_TEMPLATE1`, `QUESTION_TEMPLATE2`) and the crop math in `src/ergo/utils.py`
(`crop_with_zoom`, `map_box`, `parse_zoom_bbox_from_text`).

The reward is four terms, all binary:

    R = r_TCE + r_acc + r_format
    r_TCE    = 1.0·r_region + 0.5·r_box

    r_region = 1 iff a FROZEN reward model, shown ONLY the crop and the question,
               answers correctly.                        <- the contribution
    r_box    = 1 iff Area(crop) / Area(original) <= 0.6
    r_acc    = 1 iff the policy's own turn-2 answer is correct
    r_format = 1 iff BOTH turns match their tag structure

Everything scores 0 if turn 1 produces no parseable box.

The one real idea is `r_region`'s **withholding of the original image from the reward
model**. Prior work (DeepEyes, PixelReasoner) shows the RM both the full image and the
crop, so the RM can answer from the full image and the crop is never forced to be
self-contained. ERGO shows it the crop alone, so the policy is pushed to crop regions
that *carry the answer*, which under downsampling means cropping context the model can
only infer (the straw near the coffee cup), not just the target it can see.

`r_box`'s γ = 0.6 comes from the GT-box area distribution in TreeVGR / VisCoT / V* / VGR
(their Fig. 3). We can re-derive it on our own training set in an hour — `build_grpo_sets.py`
already carries `_boxes_abs` per row and calls `union_bbox` on them.


## 2. What they released, and what they did not

`nota-github/ERGO` is **627 lines of Python, inference and evaluation only**:

    src/ergo/const.py     10    the three prompts
    src/ergo/infer.py    169    single-image HF demo of the two-stage loop
    src/ergo/eval.py     283    async OpenAI-API client against a vLLM server
    src/ergo/utils.py    165    smart_resize, box parse, crop, box->original mapping

There is **no training code, no reward implementation, no data pipeline, no config**.
The repo credits [VLM-R1](https://github.com/om-ai-lab/VLM-R1) as the RL codebase, but
VLM-R1 is 92 files of single-turn open-r1-multimodal GRPO (rec / gui / ovd tasks) with
no multi-turn path at all — so ERGO's two-turn rollout was custom and is unpublished.
There is nothing to lift. The repo has one issue, closed, about `uv`.

What *is* released and genuinely useful: the exact prompts, the exact crop/resize math,
the exact evaluation protocol, and the **`nota-ai/ERGO-7B` checkpoint** (4 shards, public).
Also useful: `r_box`'s threshold, the reward weights, and all the training hyperparameters
in Table 7.


## 3. What breaks on Qwen3-VL

**Coordinates changed scheme.** Qwen2.5-VL emits absolute pixel coordinates *in the
post-`smart_resize` frame* — which is why `utils.py` carries `width_scale = width / resize_w`
and `map_box()`. Qwen3-VL emits **relative coordinates on a 0–1000 grid against the
original image**. So `map_box` must be replaced with `x * w / 1000`, and
`QUESTION_TEMPLATE1` must say which space it wants. This is small but *silent*: get it
wrong and every crop lands in the wrong place, the region reward collapses to the
box-area reward, and training still looks like it is running.

**Format is not free.** ERGO needs `<think>` in both turns, and our own
`build_coldstart_data.py:53` records that Qwen3-VL-8B-Instruct **scores format 0.000
under every prompt variant tried** — which is why that cold start exists at all. If the
policy never emits well-formed turn-1 output, every reward in the group is 0, the
advantage is 0, and GRPO has no gradient. Mitigation is to start from
`checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, which already
emits `<think>…</think><answer>…</answer>` — and which is also the right starting point
for comparability with our model. Whether that checkpoint will also emit `<zoom>` is
Gate B in §8.

**Our images are capped at 512 px.** `build_grpo_sets.py:109` and
`trl/grpo_vlm_qwen3.py:122` both set `MAX_IMAGE_SIDE = 512`. Training ERGO on 512 px
images is training a no-op: there is no high-resolution detail left for the crop to
recover. The training set must be rebuilt at full resolution and the trainer cap raised,
which is what drives the step cost in §9.

**vLLM 0.11.0 is the version to stay on.** vllm-project/vllm#29595 reports corrupted
Qwen3-VL grounding output above 0.11.0, and `saliency_r1_qwen3_vllm` is on exactly
0.11.0. Do not upgrade it for this arm.


## 4. Two routes, and which to take

### Route A — our TRL trainer (recommended if ERGO is built alone)

The key observation: **ERGO's two turns can be two independent training rows with the
same advantage.**

    row 1:  prompt = [sys, user(low-res image, Q1)]                              completion = o_region
    row 2:  prompt = [sys, user(low-res image, Q1), assistant(o_region), user(crop, Q2)]
                                                                                 completion = o_acc

Both rows carry `Â_g` computed once per rollout from the four rewards. The gradient is
identical to the concatenated-trajectory form, because turn 1's text is a *prompt* for
turn 2 and carries no gradient there. This means **no observation-token masking, no
custom MRoPE, no `response_mask` surgery** — each row is a normal conversation the
Qwen3-VL processor builds itself. That is exactly the class of silent bug
(`parallel_env.py:282`) that made the DeepEyes port argue for verl, and here it simply
does not arise.

What does need building in `trl/grpo_trainer_qwen3.py`:

* **Two images per row.** The trainer is single-image throughout
  (`_signature_columns = ["prompt", "image"]`, line 1179), and
  `split_pixel_values_by_grid` assumes one `image_grid_thw` row per sample. Generalising
  it to a per-row image count is contained but touches the ZeRO-3 path.
* **Multi-image in the vLLM server.** `trl/scripts/vllm_serve.py:572-578` takes one
  base64 image per prompt and builds `multi_modal_data={"image": img}`. We own that
  file; it needs `list[list[str]]`.
* **A two-phase rollout** inside `_generate_and_score_completions`: generate turn 1 for
  all rollouts, crop in PIL on the main process, build turn-2 prompts, generate again at
  `n=1`, then emit 2× rows.
* **Advantage broadcast**: compute rewards per rollout, repeat each advantage twice.
  Note the normalisation wrinkle — a sequence-mean `loss_type` would weight the two
  turns equally rather than by length. State the choice in the launcher.
* **Length caps.** `--max_prompt_length 2048` cannot hold turn 2 at `N=1280`
  (1280 + 1280 vision tokens + text + turn-1 output ≈ 3,300). Both caps go up, with the
  memory consequence in §9.

### Route B — verl (recommended if the DeepEyes port happens anyway)

If `docs/deepeyes-port-estimate.md` is executed, ERGO is **incremental, not a second
project**: same framework, same data, same backbone, and ERGO's zoom is a degenerate
case of DeepEyes' `image_zoom_in_tool` with `max_turns=2`. The delta is the four reward
functions and the vision reward-model client — call it **4–7 days on top**.

Doing verl *only* for ERGO is not worth it: §5's 14–18 days would become ~20–25 with the
env and smoke work from the DeepEyes estimate's §5, to buy correctness insurance against
a bug class that Route A does not have.

**Decision rule: if DeepEyes is on the roadmap, do verl once and get both. If ERGO is
the only baseline, use our TRL trainer.**


## 5. Work breakdown (Route A)

| | Days |
|---|---|
| **Gate A** — the low-res penalty on Qwen3-VL-8B (§8). Stops here if it is small | 1 |
| **Gate B** — `<zoom>` emission + coordinate-frame probe on Qwen3-VL-8B-Instruct and on our cold start, mirroring `coldstart_gate.py` | 1 |
| Benchmarks: add `vstar_bench`, `hrbench_4k`, `hrbench_8k` (+ `spatialtreebench`) to `eval_mini/benchmarks.py`; baseline our existing arms on them | 1–2 |
| High-res training data: raise `MAX_IMAGE_SIDE`, rebuild V*/visdrone rows; or convert DeepEyes' V* + ArxivQA parquets (§6) | 1–2 |
| Trainer: two images per row (processor, `split_pixel_values_by_grid`) + multi-image `vllm_serve.py` | 2 |
| Trainer: two-phase rollout, PIL crop, 2× rows, advantage broadcast, W&B logging of zoom-parse rate and crop area | 2–3 |
| `trl/rewards/ergo_rewards.py`: region / box / accuracy / format, the vision-RM client, and a `test_ergo_reward_cpu.py` gate | 2 |
| lmms-eval `ergo_qwen3_vl` two-stage chat wrapper (precedent: `qwen3_vl_vga.py`, `qwen3_vl_sinkshift.py`, `thyme.py`) | 1–2 |
| Launcher + **debugging the first real rollouts**: format rate, zoom-parse rate, OOM at 2×1280 vision tokens, prompt truncation | 3–4 |
| **Total** | **14–18 days** |

**If Gate B fails** (the cold start will not emit `<zoom>`), add **3–5 days** for a
two-turn cold start: ~2–5k SFT rows in ERGO's format, with turn-1 boxes taken from the
GT annotations we already have in `cold_data/grpo_sets`. We have the whole cold-start
pipeline, so this is data work, not new machinery.


## 6. Data

Two options, and they answer different questions.

**Paper-faithful**: `ChenShawn/DeepEyes-Datasets-47k` — ERGO says it reuses DeepEyes'
curated data "to isolate the contribution of our method from dataset curation effects".
4.84 GB, 3 parquets, images embedded as bytes, reachable from the login node (checked).
ERGO uses the V* and ArxivQA halves — `data_0.1.2_visual_toolbox_v2.parquet` (0.99 GB)
and `data_v0.8_visual_toolbox_v2.parquet` (2.20 GB) — and not the ThinkLite math half.

**Comparability-faithful**: our own GRPO sets, rebuilt at full resolution. Same rows our
model trained on, so the ERGO arm and our arm differ only in method. This is the better
choice for a table in our paper; the cost is one rebuild of
`build_grpo_sets.py` with `MAX_IMAGE_SIDE` raised, and the disk for it (V*/visdrone at
native resolution, tens of GB — `cold_data` already holds 201 GB).

Note what ERGO's training does **not** need: GT boxes. The region reward is supervised by
the reward model's answer, not by an annotation. Boxes are only needed to re-derive γ.


## 7. Eval

Mostly already there. In `/home/uberger/scratch/research/lmms-eval/lmms_eval/tasks`:
`vstar_bench`, `hrbench`, `mme_realworld`, `spatialtreebench` all exist. VisualProbe
(MiniO3's) does not — optional.

What is missing is that **none of V*, HR-Bench or TreeBench is in our benchmark table** —
`eval_mini/benchmarks.py` has `mmerealworld` and nothing else from ERGO's headline six.
Adding them is a day, and it is worth doing regardless: per
[[benchmark-table-splits-search-vs-symbolic]] our effect lives in the visual-search half,
and V* / HR-Bench are the canonical visual-search benchmarks.

Two things do need writing:

* an **`ergo_qwen3_vl` lmms-eval wrapper** running the two-stage loop at inference. Their
  `eval.py` is the reference but is a standalone async OpenAI client, not an lmms-eval
  model — port the loop, keep our harness. `thyme.py` in our clone already does a
  crop-and-resubmit loop and is the closest precedent.
* a **pixel-constraint sweep**. `run_bench_eval.sh` already carries `--max-pixels`, so
  this is configuration, not code — but the table needs *our* model re-scored at
  640/1280×28×28 too, or the comparison is not matched. ERGO is an efficiency method;
  comparing it to a model evaluated at full resolution measures nothing.


## 8. Run this gate first — 1 day, and it is likelier than DeepEyes' to end the job

ERGO's headroom is *exactly* the gap its Table 1 reports for Qwen2.5-VL-7B on V*:

| Pixel const. | Qwen2.5-VL-7B on V* |
|---|---|
| 16384×28×28 | 77.0 |
| 1280×28×28 | 64.9 |
| 640×28×28 | 56.5 |
| 640×28×28 + GT crop appended | **77.0** |

The method recovers the 12–20 points that downsampling destroys. **If Qwen3-VL-8B does
not lose those points, there is nothing to recover and the arm is pointless.** There is
specific reason to doubt it does: Qwen3-VL's DeepStack fuses multi-level ViT features
precisely to preserve fine detail at fewer tokens, and per the DeepEyes estimate's §8 the
backbone already received tool-integrated RL with an `image_zoom_in_tool` as part of its
own post-training.

**The gate**: `vstar_bench` + `hrbench_4k` + `hrbench_8k` on Qwen3-VL-8B-Instruct and on
our cold start, at `--max-pixels` ∈ {640, 1280, 16384}×28×28, plus a fourth arm with the
GT crop appended (a ~40-line task variant — HR-Bench and V* both ship boxes). One day,
~10–20 A100-hours, and it reproduces their Table 1 on our backbone.

Read it as:

* **gap ≥ 10 points** — the method has room; proceed.
* **gap 3–10 points** — proceed only if the efficiency axis (tokens, latency) is the
  claim being made, not accuracy.
* **gap < 3 points** — stop. Report the measurement instead; "ERGO's premise does not
  hold on this backbone" is a publishable sentence and costs one day rather than a month.

**A second, cheaper reference while the gate runs**: serve the public `nota-ai/ERGO-7B`
and run their `src/ergo/eval.py` against our benchmark set. It is Qwen2.5-VL-7B so it
does not answer the Qwen3 question, but it gives the published method measured by our
harness, and it validates the eval wrapper before any training exists. Half a day.


## 9. GPU budget

Two independent anchors, which agree:

* **The paper**: ~150 H100-hours for one 250-step run. On A100 that is **~300 A100-hours**.
* **Token-volume scaling from our own runs**: a rollout is ~1,950 tokens for turn 1 and
  ~3,320 for turn 2 at `N=1280`, so ~5,300 per rollout against our ~1,360; and 128
  rollouts per step against our 48. That is **~10× our token volume per step**. From
  ~22.5 s/step on 8 H100s ([[grpo-step-time-is-mostly-fixed-cost]]) that gives roughly
  140–450 s/step on 8 A100s — **80–250 A100-hours** for 250 steps.

Take **~200–350 A100-hours per run**, and budget 2–3 runs (smoke, real, fix) →
**~400–1,000 A100-hours**. Add **~100–200** for the eval sweeps, which are not cheap here:
every document costs two forward passes, times 4 benchmarks, times 3 pixel caps, times
the arms being compared.

That is roughly **half the DeepEyes figure**, and the reason is the rollout shape:
DeepEyes pushes `max_response_length=20480` over up to 5 generate rounds; ERGO is two
rounds of ≤512 tokens.

**The reward model is the one open cost.** The paper uses frozen Qwen2.5-VL-72B-Instruct
— ~145 GB in bf16, so 2 A100-80GBs at best and realistically 4, as a sidecar for the
whole run (3 runs × 4 GPUs × ~8 h ≈ 100 A100-hours). Three options:

1. **Gateway** — `trl/rewards/openai_rewards.py` already calls the NVIDIA gateway, and
   its default `azure/openai/gpt-4o-mini` is vision-capable. Zero GPUs; ~128 image calls
   per step is well inside what the existing `ThreadPoolExecutor` judge path handles.
   **Unverified**: that the gateway accepts image parts and that a ~1 MP crop fits its
   payload limit. Ten minutes to check, and it is the first thing to check.
2. **Local Qwen2.5-VL-72B** — paper-faithful, 4 A100s, mirrors `serve_grounding_dino.sh`.
3. **Local Qwen3-VL-32B** — the same-family analogue, 1–2 A100s. Cheapest local option
   and arguably the better control, since the RM then shares the policy's visual
   front-end.

**Cluster blocker, unchanged**: `sinfo -h -N -o "%G" | grep -ci gpu` returns **0** here —
`oci-nrt-cs-001` still has no GPU hardware ([[gpu-partitions-can-be-empty-of-hardware]]).
All of this runs on `cs-oci-ord-login-01`, which is A100, hence the 2× on every
H100-derived figure above.


## 10. Fidelity notes to settle before writing it up

* **Rollouts per sample: 8 or 16?** Table 7 says 8; Appendix B's prose says "Sixteen
  rollouts were sampled per training example". Pick one and say which.
* **"Half of each mini-batch was allocated to longer rollouts to avoid VRAM
  bottlenecks"** (Appendix B) is not reproducible from the paper. Ignore it and say so.
* **γ = 0.6** was derived from *their* data's GT-box areas. Re-derive it on ours — one
  hour with `union_bbox` over `cold_data/grpo_sets` — and report both.
* **No cold start** is a claim about Qwen2.5-VL, not a property of the method. If Gate B
  forces a cold start on Qwen3-VL, that is a finding worth one sentence, not a failure.
* **The comparison axis.** ERGO is an efficiency method and our model is not. The honest
  table reports accuracy *at a matched pixel budget*, plus vision-token counts and
  latency. Our model has never been evaluated under a pixel cap.


## Sources

* [arXiv:2509.21991](https://arxiv.org/abs/2509.21991) — the paper (ICLR 2026)
* [nota-github/ERGO](https://github.com/nota-github/ERGO) — inference and eval only, Apache 2.0
* [nota-ai/ERGO-7B](https://huggingface.co/nota-ai/ERGO-7B) — the public checkpoint
* [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) — the credited RL codebase; single-turn only
* [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k) — the training data ERGO reuses
* [Qwen3-VL blog](https://qwen.ai/blog?id=qwen3-vl) and `cookbooks/2d_grounding.ipynb` — the 0–1000 coordinate change
* `docs/deepeyes-port-estimate.md` — the sibling assessment this one is scoped against
