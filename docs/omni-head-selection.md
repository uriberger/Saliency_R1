# Selecting the Omni's saliency layer and head pair

Written 2026-09-29, branch `feat/omni-head-selection`. `docs/omni-training-harness.md` is
the harness this feeds; `OMNI_HEAD_SELECTION_PROMPT.md` is the handoff that asked for it.

The Omni GRPO run trains `--overlap-layer 33 --overlap-heads 28,31`, and **neither number
was selected**. 33 is the attention layer nearest Qwen3-VL's layer 22 in relative depth
(63% against 61%), which is the entire argument for it. 28 and 31 were chosen on
Qwen3-VL-8B by `head_correlation_probe.py`, and on any other model the same two indices
name two arbitrary heads. The layer and the head pair ARE the reward's definition, so
those 30 steps are a harness smoke test and not a baseline to continue from.

This is what replaces both.

## Why it is cheaper here than it was on Qwen3-VL

**192 cells, not 1,152.** `hybrid_override_pattern` gives the Omni an attention matrix at
only 6 of its 52 decoder layers — `[5, 12, 19, 26, 33, 42]` — so 6 x 32 heads is the whole
space and the parity split has far less multiplicity to survive.

It also **dissolves the layer question**: the scan covers every attention layer there is,
so the layer comes out of the same split as the heads rather than being argued for from
depth and defended afterwards.

## The port, and the one thing the handoff did not know

`head_correlation_probe.py` and `intervene_probe.py --stage prepare` were Qwen3-VL-only in
four places each, and each now asks `vlm_family`:

| was | now |
|---|---|
| `type(m).__name__ == "Qwen3VLTextAttention"` | `in fam.attn_classes` |
| `IMAGE_TOKEN_ID = 151655` | `fam.image_token_id` — **18** here |
| `image_grid_thw[0, 1:] // 2` | `fam.token_grid` — from `imgs_sizes` over a 32px token |
| `getattr(transformers, config.architectures[0])` | `nemotron_loader.load_any` |

The handoff said to check whether `prepare` was model-agnostic before assuming it was. It
was not, and **the failure is silent**. The Omni's chat template ends its generation prompt
at `<|im_start|>assistant\n<think>\n`, so a completion carries only the closing tag;
`judge_format` wants exactly one of each and scores every completion of a perfectly
well-behaved model as malformed. Unported, `prepare` writes a cases file with **zero cases
and no error**, and nothing in the log says "template" — it says `bad_format` on every row.
This is the same fault as §9 of the harness doc, one stage earlier, and the fix is the same
one: read `<think>\s*$` off the actual prompt rather than declaring it per family.

Two things that are not renames:

* **The capture reads the module's own weights.** Qwen3-VL runs under sdpa, which discards
  the softmax weights, so each hook re-runs its module in eager. The Omni is loaded eager
  throughout (`nemotron_loader` pins it — the wrapper declares no SDPA support) and
  `NemotronHAttention.forward` already returns `(attn_output, attn_weights)`. Re-running it
  would cost a second attention per layer, and a rebuilt causal mask to do it, to recover a
  tensor the module just handed over. `Family.attention_weights_are_returned` is that fact.
* **`prepare` stores the model's own answer.** It generated the whole completion in one
  pass, so the tokens after `</think>` ARE what that chain answered. The scan otherwise
  re-derives it with a greedy decode off the forward's KV cache — which is exactly the
  cache a hybrid decoder does not build on a plain forward (`use_cache=False` is in
  `NemotronVL.forward_defaults`, deliberately). Cases prepared before this field existed do
  not carry it and the old path still runs for them.

`run` and `selftest` are **not** ported and refuse a non-Qwen3-VL model by name: the
`Intervener` rebuilds a module's output from its attention weights through `v_proj`, the
GQA expansion and `o_proj`, which is a real port with its own gate.

### What holds Qwen3-VL still

`test_probe_family_cpu.py` restates each replaced expression and checks the family's answer
against it — 64 checks, CPU only, no weights. And `--stage report` on
`outputs/head_corr/coldstart_setA` is byte-identical to `main`'s output but for one word of
prose (the incumbent row count became configurable).

## THE LABEL IS HALF GRADING NOISE, AND THAT IS THE RESULT THAT MATTERS MOST

The scan ranks heads on *does this head's attention predict getting the answer right*.
"Right" is `grpo_vlm_qwen3.accuracy_reward`, which parses the gold with math_verify and,
when that yields nothing — which it does for `C`, for `horses`, for `Yes` — falls back to

```python
float(answer_text.lower() == solution.strip().lower())
```

an **exact string match**. On a cold-started Qwen3-VL that costs almost nothing, because it
was trained into the terse format. The Omni was not. From the smoke run's 18 completions:

| what the Omni wrote | gold | `accuracy_reward` | actually |
|---|---|---|---|
| `(C) water supply` | C | 0.0 | right |
| `(C) working` | C | 0.0 | right |
| `\boxed{A}` | A | 0.0 | right |
| `This is my answer. \nD` | D | 0.0 | right |
| `This is my answer. singing` | singing | 0.0 | right |
| `A` | A | 1.0 | right |
| `\boxed{D}` | A | 0.0 | wrong |

**0.167 strict against 0.611 soft on the same completions, 8 of 18 disagreeing.**
Qwen3-VL's cold start reads 0.452 on this corpus, so the Omni is not a worse model here —
it is a worse-*parsed* one.

And the second row of that table is its own finding: the base Omni **echoes the system
prompt's own illustration**, emitting the literal string `This is my answer.` before
answering. `SYSTEM_PROMPT` ends `"i.e., <think>\nThis is my reasoning.\n</think>\nThis is
my answer."` and the model copies it. A cold-started checkpoint would not.

So `--stage report` **audits the label before it ranks anything**, printing both
accuracies and how many completions disagree. Strict stays the primary number and the
default, because it is what the reward actually optimises; `--regrade soft` re-runs
everything on the soft label, and says loudly that it did. The gap between the two is how
much of the ranking is grading noise, and it is not something to pick a winner from
without looking at.

The grader is `answer_grading.py`, which was `fig1_multistep.py`'s — written for this exact
artefact, with `this is my answer` in its own `_BOILERPLATE` and the LogicVista parser in
its docstring. Moved rather than copied. It grew one rule, `\boxed{LETTER}`, which the Omni
writes and which every earlier pattern missed (`{}` is neither a bracket nor a
parenthesis); across all 24,476 answer strings stored under `outputs/fig1-multistep`
exactly zero contain one, so it cannot move a published number. Counted, not assumed.

## --max-union IS PRE-REGISTERED AT 0.5

Fixed in `launch_omni_head_selection.sh` before any Omni number existed. Every map measured
so far reads lower the larger the DINO union gets — r(union, auroc) = -0.55 averaged over
all 1,152 Qwen3-VL heads — and the median step's union covers 54% of the patch grid, so a
pooled level mixes two different questions. Above ~0.5 coverage the union has stopped
localising the thing the step names. A threshold chosen after seeing the result is a
researcher degree of freedom, and the confirmation half of the parity split is single use.
0.5 is what the Qwen3-VL report used and what `launch_head_correlation.sh`'s own `[next]`
line names.

## Running it

```
bash launch_omni_head_selection.sh                 # submit: 2 h, one 8-GPU node
STAGE=scan bash launch_omni_head_selection.sh      # just the scan, cases already built
```

```
GPU 0-6   7 prepare shards, one whole 62 GB copy of the Omni each
GPU 7     Grounding-DINO, served
```

The detector is served, not built per shard, for the reason the GRPO run learned the hard
way: a local Grounding-DINO costs ~8 GB on a card already holding 62, and the symptom is
`[dino] CUDA OOM; retrying batch` rather than an error. The scan needs no detector and uses
all 8.

Measured: **~23 s a sample** for prepare at 7-way concurrency (18.5 s on one card alone),
and **1.2 s a case** for the scan — the scan captures all six attention layers in the one
teacher-forced forward, so it is minutes, not hours. 1,000 samples is ~1 h of generation.

`smoke_omni_head_select.sh` is the same path on 24 samples and 2 GPUs, ~40 minutes, and it
checks the three failures by name rather than by exit code.

## Reading the result

**Read the parity split, not the ranking.** Heads are ranked on odd-indexed rows and
re-scored on the even ones. A head that survives is a candidate; one that does not is
selection noise — and on 192 cells that is a far stronger statement than it was on 1,152.

# THE RESULT

Job 7092099, 2026-09-29, `outputs/omni_head_select/setA/`. 1,000 samples of set_a →
**746 cases, 3,795 grounded observe steps**, of which `--max-union 0.5` keeps 2,083 steps
from 578 completions. (Qwen3-VL's arm: 1,157 cases, 3,471 steps, 807 completions kept.)
242 rows dropped `bad_format` — 24%, against zero if the `<think>` fix were missing.

## 1. The incumbent is near the bottom

`--overlap-layer 33 --overlap-heads 28,31` ranks **148–176 of 192** in every report, and
survives the parity split in 5 of 16 (setup × metric × label × controls) combinations —
always with a NEGATIVE sign. Whatever the right heads are, these are not them, and that
part of the question is settled.

## 2. The pick: `--overlap-layer 19 --overlap-heads 4,9`

Of all 192 cells, **layer 19 heads 4 and 9 are the only two that are positive on all four
halves** — select and held-out, under both labels — with union size, patch count, step
token count and answer length controlled:

| cell | strict select | strict HELD OUT | soft select | soft HELD OUT |
|---|---|---|---|---|
| **L19 H4** | +0.0958 | **+0.1083** | +0.0948 | +0.0320 |
| **L19 H9** | +0.0534 | **+0.1211** | +0.0756 | +0.0376 |
| L19 H6 | +0.0734 | +0.0872 | +0.0618 | −0.0066 |
| L33 H28 (incumbent) | — | — | — | −0.0062 |

The held-out r being *larger* than the select-half r is the signature of an effect rather
than of selection. Layer 19 is also the top layer by max|r| on `auroc`/step, raw and
partial. L19H4 is the least union-confounded of the strong cells (r(auroc, union) = −0.134
against −0.41 at L33H27), and adding the head's own image mass as a control moves it by
0.0003.

**This is the best-supported positive pair on this corpus. It is not a strong result**, and
the next two sections are why.

## 3. The largest reproducible structure in the scan is NEGATIVE and layer-wide

All 32 heads of layer 26, and most of layer 42, survive the parity split under both labels
with a negative sign: the more those heads attend to the objects a step names, the *less*
likely the completion is right. Mean partial r ≈ −0.047 at layer 26.

It is not the union confound (it survives residualising on union) and it is not image mass
(controlling each head's own mass moves layer 26 from −0.0474 to −0.0427). A per-head
statistic that is identical across all 32 heads of a layer is a LAYER property, so it is
not a head-selection result — but it is the strongest thing in the scan, and an overlap
reward pointed anywhere near layers 26 or 42 would be rewarding a statistic that predicts
being wrong.

## 4. The label is 52% disputed, and that is the real blocker

**746 completions: strict 0.121, soft 0.638, 386 disagree.** Section "THE LABEL IS HALF
GRADING NOISE" above is the mechanism; this is its size on the full corpus.

The two labels do not agree on which heads look predictive. In the four reports' own
survivor blocks, **no cell survives under both labels**: the positive family (L19, L12) is
strict-only and the negative family (L26, L42) is soft-only. Only by widening to all 192
cells and all 16 analysis variants does L19 H4/H9 come through both.

The root cause is not the grader. It is that **the Omni is a base checkpoint**. Qwen3-VL's
head selection ran on `coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` — a model
taught to emit `<think> … </think> <terse answer>`. The Omni was never cold-started, so it
answers in prose, wraps its letters (`(C) working`, `\boxed{A}`), and echoes the system
prompt's own illustration. `accuracy_reward`'s exact-string fallback cannot read any of
that, and r(strict correctness, answer length) = **−0.251**: a quarter of "was it right" is
"was it terse".

**So the honest next step is a cold-started Omni, not a training arm.** `build_coldstart_data.py`
and `launch_coldstart_job.sh` are the existing machinery. Re-running this scan on that
checkpoint would cost the same ~1.5 h and would be answering the question that was asked;
on a base checkpoint this scan is answering a noisier one.

## If the arm is run anyway

```
bash launch_grpo_omni_overlap_job.sh --overlap-layer 19 --overlap-heads 4,9
```

That is a defensible improvement on `33 / 28,31` — selected rather than inherited, positive
under both labels, and replacing a pair that ranks 148–176 of 192. Any number read off it
still has to say that the head pair was chosen on a corpus whose correctness label is 52%
disputed.

## Reproducing the tables

```
python head_correlation_probe.py --stage report \
    --out-dir  outputs/omni_head_select/setA/scan \
    --cases-dir outputs/omni_head_select/setA/cases \
    --max-union 0.5 --controls union,npatch,ntok,alen \
    --incumbent-layer 33 --incumbent-heads 28,31 --regrade off    # and --regrade soft
```

The four logs are in `outputs/omni_head_select/setA/logs/report_maxunion0.5_*.log`
(`{off,soft}` × `{raw,partial}`), and each writes its own `corr_*.npz`.
