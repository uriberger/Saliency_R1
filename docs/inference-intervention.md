# Doing it at inference instead — 2026-09-07

The conclusion this project is converging on is that moving attention toward the middle
of the picture improves accuracy, and `--overlap_rect_frac` — a reward computed against a
fixed centred rectangle — is the arm that shows it most cleanly. If that reading is right,
training may not be needed: the same thing could be done directly while the model answers.

Two ways to do it, both built here:

1. **Move the attention.** Take weight off the picture's border and give it to the middle,
   inside the attention itself, at generation time. `sink_shift.py`.
2. **Pick the best of several answers.** Sample N, keep the one whose own attention map
   scores highest on `mean_in`. `best_of_n_probe.py` offline, `--best-of N` on the GPU.

**Idea 2 is already answered, and the answer is no.** Section 4. Idea 1 is built, gated by
a selftest, and not yet run.

## 1. What the edit does

The picture becomes a grid of patches, typically 10 rows x 16 columns = 160 tokens. Named
sets of patches on that grid:

| set | what | on 10x16 |
|---|---|---|
| `frame` | the one-patch border | 48 patches (0.300) |
| `rect` | the centred rectangle `--overlap_rect_frac 0.565` scores | 96 (0.600) |
| `ring2` | the one-patch ring immediately inside the frame | 40 |
| `core` | everything at least two patches from the border | 72 |

An attention row is one query position, one layer, one head: weights over every earlier
token, summing to 1. Slice out the weights on image patches, call it `w`. Every arm is the
same operation with different sets:

```
w'_src = w_src − α · w_src
w'_dst = w_dst + α · (Σ over src of w) · t_dst        t sums to 1 over dst
```

The row still sums to 1, the image/text split is unchanged, and no weight on a text token,
a BOS token or any other sink outside the picture is touched. `α = 0` is exactly the
identity. `t` is proportional to what the destination already had, so the edit removes the
sink without inventing somewhere to look.

### Why `frame` and not "everything outside the rectangle"

Those are two different sets, and the difference is 16 patches — columns 1 and 14, rows 1
to 8, the two vertical strips just inside the left and right edges:

```
        col 0         5         10        15
row 0   # # # # # # # # # # # # # # # #        #  frame  (source)
row 1   # ? R R R R R R R R R R R R ? #        R  rect   (destination)
 ...    # ? R R R R R R R R R R R R ? #        ?  neither
row 9   # # # # # # # # # # # # # # # #
```

Those `?` patches are **not** sinks — the sink is the frame, which holds ~half the image
attention and 76–85% of the map peaks. Draining them too would make the edit do two things
at once: remove the sink *and* narrow the field of view. So the source is the frame alone,
and the `?` patches keep exactly what they had. Uri's decision, 2026-09-07. `rect` is the
destination either way, because that is the set the reward scored.

### The arms

| arm | source → destination | what it answers |
|---|---|---|
| `centre` | frame → rect | the treatment |
| `core` | frame → core | a stricter middle |
| `outward` | frame → ring2 | **the wrong-place control.** Same source, same mass, still off the frame, aimed away from the middle |
| `flat` | all → all, evenly | is it just evenness? |
| `reverse` | rect → frame | the sign check: this must hurt |
| `text` | the same MASS moved among the text tokens | is any nudge of this size enough? |

**`centre` minus `outward` at matched α is the result. `centre` alone is not.** Anything
that perturbs attention at this magnitude will move some answers; only the contrast says
whether the *destination* mattered.

`flat` is a treatment, not a control. `--overlap_rect_frac` correlates **0.932** with the
box-blind `flatness` statistic ([per-completion-masks.md](per-completion-masks.md)), and
`--maskfree flatness` already recovered 71% of `mean_in`'s natural gain
([next-reward-experiments.md](next-reward-experiments.md)). "Move it to the middle" and
"even it out" have never been separated. This separates them.

## 2. Where it plugs in, and why not a hook

`intervene_probe.py` and `flow_intervene_probe.py` recover the softmax weights by re-running
the attention module in eager mode inside a forward hook. That is correct for one
teacher-forced pass over an answer that already exists, and **wrong while the model is
writing**: a decode step re-run with `past_key_values=None` attends to itself alone.

So this registers an attention implementation instead. Transformers resolves
`config._attn_implementation` through `ALL_ATTENTION_FUNCTIONS` inside
`Qwen3VLTextAttention.forward`, **after** rotary embedding and **after** the KV cache
update, so the registered function sees the real keys and values including everything
cached, and one code path serves prefill and decode. Verified against transformers 5.13 in
both the `saliency_r1_qwen3_vllm` and `lmms_eval` environments.

Only the text decoder is switched — the vision tower keeps its own implementation, because
the text config is a different object. Layers the arm does not touch fall through to the
fused kernel, so a narrow arm costs almost nothing.

**Batch size 1 only.** The picture is located from the prompt's own token ids; left padding
in a wider batch moves every image column. A wider batch is refused, not edited wrongly.
The bench harness already runs at 1.

## 3. How much there is to move — read this before reading any null

The edit can only shift as much of an attention row as that row spends on the picture.
From the stored probes (`outputs/overlap_probe/20260809-021810-crossrun-val_natural-plus`),
at **layer 22 heads 28/31 — the pair the reward trained**:

| model | image mass of a row | border's share of it | movable at α=1 |
|---|---|---|---|
| cold start | 0.00396 | 0.520 | **0.00206** |
| `mean_in` 8k | 0.00896 | 0.481 | 0.00431 |
| `mean_in` set_a cp2000 | 0.01374 | 0.492 | 0.00676 |
| `auroc` set_a cp2500 | 0.01408 | 0.518 | 0.00729 |

So the narrow arm, at full strength, moves **0.2%–0.7% of one attention row, in 2 heads of
1,152.** A null there is the expected outcome and it is **not** evidence about the idea —
it is evidence that there was nothing to move. It is run first because it is the
configuration the training result came from, and because `--stage survey` prices the broad
arm from the same run.

The broad arm (every layer, every head) is where any effect must come from. That is also
the only configuration that has ever moved this model causally: `flow_intervene_probe.py`
edited every layer up to a cutoff and shifted log P(gold) by 0.726 nats, against a flat
noise floor for every single-layer and single-head edit (0 of 288 cells).

## 4. Idea 2 is a null, measured for free

The stored probes hold, per completion, `accuracy_reward` and the LLM judge's
`openai_reward` **alongside the maps** — so best-of-8 by saliency needed no GPU at all.
`best_of_n_probe.py`, 11 checkpoints x 30 questions x 8 sampled answers, `val_natural`.
Report: `outputs/best_of_n/crossrun_val_natural.txt`.

Median of `picked − random` across the 11 models, where `random` is the group mean (the
exact expectation of plain sampling) and argmax ties are averaged:

| selector | accuracy | judge | above 0 (accuracy) |
|---|---|---|---|
| `rect_mean_in` — the rectangle | −0.0125 | +0.0003 | 5/11 |
| `true_mean_in` — the DINO union | −0.0071 | −0.0083 | 5/11 |
| `flatness` — no mask | −0.0089 | +0.0025 | 4/11 |
| `low_frame` — minus border mass | −0.0089 | −0.0004 | 3/11 |
| `short` — minus token count | −0.0125 | −0.0113 | 4/11 |
| **`hash` — deterministic noise** | **+0.0083** | **+0.0037** | **6/11** |

**No saliency statistic beats the noise selector.** `oracle − random` is +0.07 to +0.09,
so the headroom is real and it is the selectors that are empty. This reproduces, through a
different route, the already-established `r(overlap reward, accuracy reward) = −0.019 ±
0.051` within a group.

The GPU version exists anyway (`--best-of N`), because the offline test uses each model's
own sampled answers at temperature 1 on 30 questions, and someone may want it on the
benchmark. It is not worth GPU hours on this evidence.

Caveats: 30 questions; the 11 models are one cold start plus ten descendants, so "5/11" is
consistency along one trajectory, not eleven replications.

## 5. Running it

```fish
set OUT outputs/sink_shift/coldstart_trained
set M checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged

# 1. price the experiment -- about a minute
bash launch_sink_shift.sh --stage survey --gpus 1 --out-dir $OUT --model $M

# 2. the gate. The launcher refuses to run without a PASS in this log.
bash launch_sink_shift.sh --stage selftest --gpus 1 --out-dir $OUT --model $M

# 3. the narrow arm, two alphas first
bash launch_sink_shift.sh --stage run --gpus 8 --out-dir $OUT --model $M \
    --scope trained --arms centre,outward,flat,reverse --alphas 0.5,1.0 \
    --rows-per-split 128

python sink_shift_probe.py --stage report --out-dir $OUT

# 4. then the broad arm, into its OWN out-dir
bash launch_sink_shift.sh --stage run --gpus 8 --out-dir outputs/sink_shift/coldstart_all \
    --model $M --scope all --arms centre,outward,flat,reverse --alphas 0.25,0.5
```

Cost: one row is one greedy generation at batch size 1, ~10 s. 128 rows x 2 splits x
(1 baseline + 4 arms x 2 alphas) over 8 GPUs is roughly 40 minutes per scope.

The selftest must pass and it checks three things: `α=0` reproduces the un-hooked greedy
generation **token for token**, uninstalling puts the model back, and at `α>0` the border's
share of the picture's attention really falls to `(1-α)` of itself. The last is the lesson
`flow_intervene_probe.py` paid for: when the thing actuated is not the thing measured, a
null with a flat manipulation check says nothing.

### On the benchmark

Two files outside this repo, both additive, chosen so that **no concurrent evaluation can
notice they exist**:

- `lmms-eval/lmms_eval/models/chat/qwen3_vl_sinkshift.py` — new file, subclasses the chat
  `Qwen3_VL` and installs the edit. lmms-eval imports only the model actually requested,
  so `--model qwen3_vl` never opens it.
- `lmms-eval/lmms_eval/models/__init__.py` — **one added key** in
  `AVAILABLE_CHAT_TEMPLATE_MODELS`. Adding a key cannot change what another key resolves
  to; verified afterwards that `qwen3_vl` still resolves to the identical class. Written by
  atomic rename, so a process reading it mid-write sees the old whole file or the new whole
  file. Both are left uncommitted in that fork, on `main`.

No conda environment was modified. `SINK_SHIFT_ALPHA` defaults to **0**, which is the
identity, so selecting this model type without configuring it evaluates the stock model and
says so loudly in the log.

```fish
set -x BENCH_MODEL_TYPE qwen3_vl_sinkshift
set -x SINK_SHIFT_ALPHA 0.5
set -x SINK_SHIFT_ARM centre
set -x SINK_SHIFT_LAYERS all
set -x SINK_SHIFT_HEADS all
bash run_bench_eval.sh --run-dir <dir> ...
```

`BENCH_MODEL_TYPE` is a new passthrough in `run_bench_eval.sh`; unset, that script is
unchanged. Read the result against the measured **~0.013** seed floor, and against Uri's
`--w-overlap 0` control rather than `baseline/grpo-no-saliency`, which starts from a
different model.

## 6. What would falsify what

- `centre ≈ outward` at every α → the magnitude, not the direction. Inference-time steering
  is a dead end and the training result is not "the middle".
- `flat ≈ centre` → it was never about the middle, only about evenness. The rect-frac
  reading needs rewording, and `--maskfree flatness` is the cheaper way to get it.
- `centre > outward` **and** `reverse` below baseline → a real, signed, weight-free effect,
  and the training arm is the expensive way to get it.
- Every arm null at `--scope all` with the survey showing real leverage → the attention
  distribution over the picture does not carry the effect, which would agree with results 4
  and 5 in [probe-results.md](probe-results.md) and disagree with the rect-frac reading.

Read no cell whose format-valid rate fell below `--min-format` (default 0.9). That is broken
generation, not an effect — `flow_intervene_probe.py` at α=1 had box and roll agreeing to
0.0003 nats for exactly that reason.

## 7. Files

| file | what |
|---|---|
| `sink_shift.py` | the edit, the patch sets, and the attention implementation it registers. `install(model, arm=..., alpha=...)` |
| `sink_shift_probe.py` | `selftest` / `survey` / `run` / `report` / `monitor`, plus `--best-of N` |
| `launch_sink_shift.sh` | shards `run` over a node's GPUs; refuses to start without a passing selftest |
| `test_sink_shift_cpu.py` | 87 CPU checks. Includes that `rect` agrees with `overlap_rewards._centre_rect_mask` patch for patch on every grid from 4x4 to 21x25, so the intervention and the training arm cannot drift into different experiments |
| `best_of_n_probe.py` | idea 2, offline, from the stored probes |

## 8. Caveats

- **Nothing here has been run on a GPU yet.** Every number in section 3 and section 4 comes
  from files already on disk; sections 1, 2 and 5 describe code that passes its CPU tests
  and has not yet met a model.
- **The validation accuracy is this harness's own.** The baseline is its `α=0` run, not the
  `val/*/accuracy` curve from training, which came from vLLM. Pair within one harness; do
  not read an absolute difference against the training curve.
- **`mean_in` going up is not a result here.** The map's peak sits on the border 82% of the
  time and `mean_in` divides by it, so draining the border raises `mean_in` by arithmetic.
  It is a contract check, never a readout.
- **One rectangle fraction.** Everything uses 0.565, the value the arms were launched at.
  `--rect-frac` moves it, and nothing has been measured at any other value.
