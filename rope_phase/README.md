# RoPE phase drift in multimodal attention

Self-contained probe. Nothing here imports from the surrounding repository, and
the whole directory can be moved elsewhere as-is; the only external things it
needs are a checkpoint and a dataset with `image` and `problem` columns, both
given by flag or environment variable.

## The question

In Qwen-VL's M-RoPE a text token at index `p` carries position `(p, p, p)`, and a
patch at row `r`, column `c` of an image anchored at `s` carries `(s, s+r, s+c)`.
The three per-axis offsets reaching the attention logit are therefore

```
dt = d          dh = d - r          dw = d - c          with  d = p - s
```

so every patch dependence enters **only** through `d - r` and `d - c`. Adding 1 to
`d` is mathematically identical to relabelling the grid `(r-1, c-1)`: the model
cannot distinguish "the query moved one token further from the image" from "the
image's coordinate frame slid one patch diagonally".

Two consequences:

- The positional overlay RoPE lays on the image **translates one patch per
  generated token**, so a chain of thought sweeps it across and off the grid many
  times over.
- Spatial resolution and distance-stability are the *same* parameter `theta`.
  Discriminating adjacent rows needs `theta * |r - r'|` to be non-negligible;
  invariance to query displacement needs `theta * delta` to be negligible. With an
  image ~30 patches across and a CoT ~10^3 tokens long, you cannot have both.

The t-axis offset does not depend on the patch, so it moves image-vs-text *mass*
but cannot change the *shape* of the profile over patches. This probe is about the
shape only; the mass effect is the separate, already-documented "visual fading".

## The measurement

The overlay is faint next to content, but it depends on exactly one variable, `d`,
and `d` is arbitrary with respect to what is in the picture. So: bucket hundreds of
thousands of (token, head) attention rows by the phase of one RoPE channel, average
inside each bucket — content mushes out, the overlay adds coherently — subtract the
grand mean, and ask whether consecutive buckets are the same picture shifted by the
predicted number of patches.

Bucketing by a channel of angular frequency `theta` into `B` buckets makes
consecutive buckets differ in `d` by `2*pi/(theta*B)`, and the overlay shifts by
that many patches. **Nothing is fitted.** For Qwen3-VL-8B: `h8` → +1.00 row/bucket,
`h4` → +2.00 (same channel, half the buckets), `w10` → +1.02 columns/bucket.

Only the channel you bucketed on stays coherent within a bucket, so each binning
should march along *its own* axis and stay put on the other — a two-way
dissociation, sharper than the diagonal.

## Result (2026-08-16)

Qwen3-VL-8B-Instruct, 256 cases, 40,679 tokens, 1152 heads:

| binning | predicted | heads at predicted | matched null |
|---|---|---|---|
| `h8`  | +1.00 row/bucket | **68.6%** | 0.3% |
| `h4`  | +2.00 rows/bucket | **77.9%** | 2.9% |
| `w10` | +1.02 cols/bucket | **59.1%** | 0.3% |
| `decoy8` | +2.53 | 9.9% | 8.9% (at chance) |

Residual power 64.9x the matched floor. The 2-D test gives modal `(row +1, col 0)`
for `h8` and `(row 0, col +1)` for `w10`, both against a 0.0% null.

**Qwen2.5-VL-7B control: flat.** Same code, same data. Its `h8` score curve is
symmetric (+0.12 / +0.35 / +0.13 at -1/0/+1) where Qwen3-VL's is asymmetric
(+0.02 / +0.48 / +0.60), and 774/784 heads pick no shift. Its `w10` is reported
*degenerate*: that channel turns 0.057 rad over the entire observed `d` range,
0.9% of a cycle. Severity is a config knob — Qwen3's interleaved M-RoPE puts H/W on
frequency indices 1 and 2 (0.786 / 0.618 rad per position); Qwen2.5's chunked
layout puts them at 16 and 40 (0.032 / 0.000178).

**Effect size, aggregate: positional share 0.0087** — under 1% of the variance in a
token's image-attention profile, stable across 32- and 256-case runs.

**But that aggregate is a mean over a very heavy tail, and quoting it alone
understates the effect by orders of magnitude.** Per head, the share of a single
token's profile explained by position is:

| median | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|
| 0.013% | 0.20% | 2.0% | 18.4% | **51.2%** (L12 H2) | 1.02% |

So a tenth of all heads have over 2% of their profile driven by position and one in
a hundred has over 18%. "Under 1%" describes no head in particular: the typical head
is 75x below it and the top head is 50x above. Detectability (69% of heads show the
direction) and magnitude are different questions and neither implies the other.

A free consistency check, and a better argument than any single %@pred: going from
32 to 256 cases leaves `power(h8)` essentially unchanged (3.66e-2 -> 3.57e-2) while
the `perm8` floor falls 5.7x for 8x the tokens. That is a fixed signal sitting on
noise that averages away as 1/n.

Layer profile (fraction of heads marching, `h8`): 0.88–1.00 through L0–L14, collapse
at L15/16 (0.16/0.09), L20 (0.00), L27 (0.00), L29 (0.03), back to 1.00 at L35.

Full reports in `results/`.  The raw bucket sums those came from (~250 MB per
run) were left in the host repo at `outputs/rope_phase/{e0_pilot,e0_qwen3_256,
e0_qwen25}/scan/` rather than copied here; only the reports travel with this
directory.  Re-running regenerates them.

## E1 — the causal arm

E0 is observational. E1 intervenes on `position_ids` and nothing else: the tokens,
pixels, weights and causal mask are byte-identical between arms, so any difference
is attributable to position alone. "Tail" is everything after the image.

| arm | intervention | what it isolates |
|---|---|---|
| `null` | +delta to **every** token, image included | logits must come back identical — validates the harness and that nothing else reads absolute position |
| `full` | +delta to all three tail axes | the honest "query is further from the image" condition |
| `t` | +delta to the tail's t axis only | every patch shares the image's t index, so this offset is *constant across patches*: it can move mass but cannot reshape the profile. The visual-fading arm |
| `hw` | +delta to the tail's h/w axes only | tail↔tail and image↔image offsets are untouched, so this moves **exactly** the cross-modal spatial offsets. The hypothesis, surgically isolated |
| `fix` | tail h/w frozen at the first post-image value | removes the p-dependence outright, so its curve must be **flat**. The positive control |

**Phase bucketing.** The first build averaged the profile over every post-image
token before comparing, which cancelled the very thing worth measuring: tokens in a
case sit at hundreds of different distances, spanning ~30 turns of the fast row
clock, so their stripes are at ~30 phases and averaging wipes them out. Adding a gap
shifts every phase equally but cannot un-cancel what already cancelled. That build
saw only the slow channels.

This build buckets by phase first — as E0 does, and using each token's *unmodified*
distance, so the same tokens share a bucket in every arm at every gap and the
comparison is exactly paired. That turns E1 into a **calibration**: a bucket at gap
N should be that bucket at gap 0 translated by N, so the fitted shift should give
back N — and must **wrap**, since the row clock repeats every 8 patches. A gap of 8
is indistinguishable from no gap; a gap of 5 reads as −3.

**The prediction is a sawtooth of period 8, with nothing fitted.** No decay story
produces a sawtooth. Half a period (a gap of 4) is genuinely ambiguous between +4
and −4, and the report scores it as such rather than pretending otherwise.

Stated before running: `null` at arithmetic noise; `t` recovers shift 0 at every gap,
because its offset is identical for every patch and cannot reshape anything; `hw` and
`full` trace the sawtooth; `fix` flat at 0.

### E1 result (2026-08-18, Qwen3-VL-8B, 19 cases, gaps 0-20)

**Shape — confirmed, strongly.** Similarity of the phase-bucketed profile to gap 0:

| gap | 0 | 2 | **4** | 6 | **8** | 10 | **12** | 14 | **16** | 18 | **20** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `null` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `t` | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| `hw` | 1.00 | 0.61 | **0.20** | 0.56 | **0.94** | 0.56 | **0.18** | 0.54 | **0.92** | 0.56 | **0.20** |
| `fix` | 0.51 | 0.50 | 0.50 | 0.50 | 0.50 | 0.49 | 0.50 | 0.50 | 0.49 | 0.49 | 0.49 |

Three clean cycles on the predicted period. The column bucketing oscillates on *its*
own period instead (minima ~5 and ~15, maxima 0, 10, 20). Four predictions, four
hits — **on this readout.**

**The other readout, the one billed above as "the prediction", only half-fired, and
the number belongs here rather than in the report alone.** Recovering the imposed
gap as a fitted shift gives `hw`/`full` **52% of gaps** against a 14% floor on the
row clock, and **14% — exactly the floor — on the column clock**. Every miss is in
the same direction: the fit returns 0 wherever the predicted shift is small (gaps 1,
2, 6, 7, 9, 10 …).

There is a mechanical reason, and it is trap 2 wearing a new hat. E1 compares bucket
*b* at gap N with bucket *b* at gap 0 — **the same tokens** — so the bucket's
content residual is identical on both sides and contributes a term that peaks at
shift 0. E0 does not have this problem: it compares bucket *k+1* with bucket *k*,
which are different tokens. The same term is why the similarity floor is +0.20
rather than the −1.00 a pure translating sinusoid would give; read literally, about
40% of the bucket residual is the marching component and 60% is content that did not
cancel. Fixing it would need a shift test that does not put the same tokens on both
sides.

The similarity oscillation is untouched by this and remains the strong result. But
"the prediction is a sawtooth, with nothing fitted" should not be quoted without the
52%.

**Behaviour — negative.** Fraction of greedy tokens that change:

| `null` (a no-op) | `t` | `hw` | `full` | `fix` |
|---|---|---|---|---|
| **1.0-1.5%** | 1.1-1.6% | **1.1-1.8%** | 1.2-2.1% | **~30%** |

bf16 arithmetic alone flips 1.2% of tokens over a 250-token completion, and `hw`
barely clears it — with **no period-8 structure in any arm**. `fix` flips 30%, so the
readout spans 1.2 to 30 per cent and the null in between is a measurement, not
insensitivity. Only 5.5% of positions are near-indifferent to begin with.

**So the drift massively reshapes where the model looks and does not change what it
says.** Where a VLM looks and what it then says are weakly coupled — which is worth
knowing on its own, and is direct causal evidence for something attention-attribution
work usually has to infer.

Untested, and where a one-to-two patch displacement should actually bite: pointing
and bounding boxes, dense document and chart reading, counting, and within-answer
self-consistency (ask the same spatial question twice at different depths — the
disagreement should be worse at half a period than at a full one).

Two analysis traps this went through, both worth knowing before reusing the code.
Averaging the profile over tokens before comparing cancels the fast channel, because
tokens span ~30 turns of it; bucket by phase first. And subtracting the across-bucket
mean is not optional: leave the shared content in and it sits identically on both
sides of every comparison, pinning the fitted shift at zero however far the pattern
actually moved.

Mass and shape are reported separately throughout — conflating them is exactly how
this effect gets mistaken for visual fading.

`test_rope_phase_e1_cpu.py` does not test the implementation against itself: it
enumerates the pairwise offsets that actually reach the attention logit
(tail↔tail, image↔image, tail↔image) and asserts which ones each arm is allowed to
move. That is where an off-by-one would silently invalidate the GPU run.

```fish
SCRIPT=rope_phase_e1.py N_SAMPLES=20 OUT_DIR=$PWD/outputs/e1 bash submit_rope_phase_job.sh
```

Cost is ~105 forwards per case (5 arms x 21 gaps): about 13 min for 20 cases on one
H100, measured.

## E2 — does it move where the model thinks things ARE?

E1's flip count was measured over prose, which never asks the model to care about
one or two patches. E2 builds a task that does: a grey canvas with one coloured
square, a controlled number of **patches** above or below the midline, and the
question "top half or bottom half?" read straight off the two answer tokens'
logits. The offset is a dial; the readout is where the answer flips sign — the
model's perceived midline, in the same units as the drift.

Baseline curve (job 6163837): monotone, unsaturated, crossing at −0.71 patches,
moving 3.3 logits per patch with a colour-to-colour spread of ~0.5, so the midline
is pinned to about ±0.08 patches. Run the `pilot` stage first — a saturated model
has no crossing point to move, and the sweep would measure nothing while looking
like it had.

**Perceived-midline shift, in patches, versus the gap:**

| gap | 4 | 16 | 64 | 128 | 192 | **256** |
|---|---|---|---|---|---|---|
| `null` (arithmetic floor) | 0.00 | +0.02 | −0.01 | 0.00 | +0.01 | **0.00** |
| `t` | −0.10 | −0.12 | −0.18 | −0.26 | −0.17 | **−0.34** |
| `hw` | −0.03 | −0.21 | −0.31 | −0.36 | −0.47 | **−0.36** |
| `full` | −0.10 | −0.28 | −0.40 | −0.52 | −0.59 | **−0.63** |

**The effect is real**: 30x the arithmetic floor, on an instrument resolving ±0.08
patches. Unlike E1's flip count, this task can see it — which is partly a lesson
about the task, not only about the effect.

**It is small**: at a realistic reasoning length the whole effect is 0.63 patches,
about 20 px on a 768 px canvas, under 3% of image height, growing roughly
logarithmically in the gap. So attention moves 1-2 patches and judgement moves a
fifth of that. `fix` is noisy here (wandering, no trend) because it is far
off-distribution; do not read it.

**Retracted: "it is monotone with no wrap, so it is not the stripes."** That
argument was wrong and it is corrected here rather than deleted, because it is an
easy one to make twice. The overlay has no period to return at. Only the *fastest*
H channel repeats every 8; the next repeats every 16.48, and at a gap of 8 it sits
at 0.486 of a cycle — very nearly antiphase, close to the worst possible gap, not
restored. H and W do not share a period either (7.9956 against 10.1747, ratio
1.2725), so the joint phase never repeats at all.

E1's wrap is a property of the *analysis*, not of the physics: phase-bucketing on
the h8 clock isolates the one component that does repeat. A behavioural readout
sees the sum of the whole ladder, which does not. Asking E2 and E3 to dip at 8, 16
and 24 was asking for a signature the mechanism never predicted.

What survives, and it is the stronger statement anyway, is the **dissociation of
magnitudes**: a gap of 4 translates the attention overlay by exactly 4 patches — a
sixth of the image — and moves the perceived midline by 0.03. The `hw` arm's
contribution is real (0.36 patches at gap 256, 4.5x the arithmetic floor) and
should not have been discounted for having "the wrong shape".

## E3 — growing the distance with real tokens

E1 and E2 both reached in and rewrote position ids. E3 asks the same question the
way it actually happens: the same filler placed **before** the image (condition A,
positionally inert for the image-to-question distance) or **after** it (B, which
grows that distance by exactly the filler length, asserted from the model's own
`get_rope_index` — A held at 56 for all 25 lengths, B grew to 80).

**The control invalidated the design, which is the result.** Condition A moved the
perceived midline **0.65 patches** on its own. So B−A is not positional and its
−0.93 must not be read as such: it is 4x what E2 measures for the same distance
while changing nothing but ids.

Two reasons A is not inert, and the second is the bigger one. The filler's meaning
acts on the judgement differently depending on where it sits; and in B the filler
ends up ~576 tokens (the whole image block) *closer to the answer* than in A, so
its content has far more influence there. **Content moves this judgement several
times harder than the distance those words create**, which retroactively justifies
the artificial-looking position-id intervention: the natural version is confounded
by a larger effect. No curve dips at 8/16/24 — see the retraction above for why
that was never the right thing to look for.

## E4 — freeze the cross-modal position, and find the best place to freeze it

The intervention: give each post-image token **two position identities** — its real
one when attending to text, and a frozen one (`h = w = s + d0`, the same for every
token) when attending to image patches. The overlay then stops moving.

This is not E1's `fix`. `fix` pinned the tail's h/w in the position ids, which also
flattened tail-to-tail spatial offsets and cost +168% NLL — nearly all of it damage
to *text-to-text* attention, nothing to do with the image. Restricting the frozen
identity to image columns cannot be expressed through `position_ids` at all
(offsets are differences of per-token values, so "change tail↔image but not
tail↔tail" has no solution), so `rope_phase_e4.py` patches attention instead. Both
self-checks are bitwise exact: the hand-written attention reproduces the library's
kernel, and the frozen path reproduces the real one when handed identical phases.

`d0` is a free parameter — *every* value removes the drift, so correctness does not
pick one. It is chosen on **error against ground truth**, never on how little it
disturbs the model: that criterion is minimised by doing nothing at all, so
surprise-at-own-text is kept as a rejection filter (`guard`) and never as the
objective.

Two stimulus families with exact ground truth: E2's square, and 120 real images
from `set_c` (train split; gqa, openimages, textcap, v7w, docvqa, textvqa,
infographicsvqa) with their annotated boxes, scaled and **translated** on a grey
canvas so the box centre sits a controlled number of patches off the midline.
Offsets are whole patches — a sub-patch shift re-aligns content to the patch grid
and changes the features, which showed up in the pilot as a 7.5-logit sawtooth on a
docvqa page.

### Result (2026-08-19, Qwen3-VL-8B)

| | synthetic square | real images (63 usable of 120) |
|---|---|---|
| drift, gap 0 → 512 | **−0.778 ± 0.074** | **−0.356 ± 0.056** |
| best frozen arm's residual | −0.484 | −0.209 ± 0.035 |
| **share of the drift removed** | **38%** | **41%** |

- **The drift's behavioural cost on real images, measured for the first time:
  −0.356 ± 0.056 patches** over a 512-token answer. It had only ever been measured
  on a coloured square.
- **Best `d0` is 24 = `max(gh, gw)`** — the principled default, the distance of the
  first token after the image, the value that makes the intervention an exact no-op
  at the start of an answer. Best on synthetic; statistically tied with everything
  in 24..47 on real images.
- **The off-distribution controls behaved as predicted.** `d0` = 0 and 12 put the
  query's row counter *inside* the image's own rows, where no text token ever sits.
  They are the worst arms on both families and cost the most NLL.
- **No periodicity in `d0`** — the error varies smoothly, as the incommensurate
  channel ladder above requires.
- **It removes only ~40% of the length-induced error, consistently.** The remaining
  60% is the t axis — image-vs-text mass, "visual fading" — which this intervention
  deliberately leaves alone because it provably cannot reshape the profile. That
  matches E2, where `t` alone reproduced over half the effect.
- **It is nearly free**: under 1% NLL and 1.1–1.5% token flips against E1's measured
  bf16 floor of 1.2%.

**One result that does not replicate, and should not be quoted without this.** On
the synthetic square, freezing at 24 also nearly eliminates the *standing* bias at
gap 0 (−0.675 → −0.069, a 90% cut). On real images it does not (+0.179 → +0.391,
slightly worse). That gain is a property of the synthetic stimulus. On real data the
honest claim is that freezing buys the ~40% of the length effect and nothing more.

**Superseded: "this cannot be patched at inference, it would have to be trained
for."** That was inferred from `fix`'s +168% NLL, which measured a different lesion.
The surgical version costs under 1%. Circle-RoPE and DIPE still have the better
argument for doing it in training — they get the whole effect rather than 40% — but
the inference-time patch is cheap and works.

**Read the median, not the mean.** The crossing is intercept over slope, and a
shallow slope sends it to infinity: 11% of real stimuli land beyond ±5 patches, and
one of them moves a mean of 111 by half a patch. The mean put the real-image drift
at +0.05 ± 0.68 — no effect, useless error bar — where the median put it at −0.37.
The mean was measuring its own tail.

## Controls

- `perm8` — the `h8` buckets with their **labels permuted per case**. Counts per
  bucket per case are preserved exactly, as is all content; only the alignment of
  the overlay across cases is destroyed. An iid random bucket draw is *not* a
  matched floor and must not be used: a case's `d` values are consecutive integers,
  so they fill the phase buckets almost perfectly evenly and that case's content
  cancels far better under the real binning than under a random one. Measured, the
  iid floor sits ~5x high, which would rig the comparison.
- `h4` — same channel, half the buckets, so the predicted rate doubles.
- `decoy8` — a frequency that is not a RoPE channel. Weak by construction, since a
  decoy bucket spans ~2.5 periods of the fast channel; read it as a rate check, not
  a null.

## Running it

```fish
python test_rope_phase_cpu.py                 # 7 checks, seconds, no GPU

set -x ROPE_PHASE_DATASET /path/to/dataset
bash submit_rope_phase_job.sh                 # 32 cases, ~5 min on one H100
N_SAMPLES=256 bash submit_rope_phase_job.sh   # ~21 min

python rope_phase_probe.py --stage report --out-dir outputs/e0
```

Use a **fresh `--out-dir` per run**: the report refuses to merge shards scanned with
different `n_samples`/`num_shards`, because a 1-shard run writes `shard00` covering
every case and a later 8-shard run would double-count most of the corpus.

`test_rope_phase_cpu.py` plants a marching overlay of known amplitude and rate under
synthetic content, pushes it through the same accumulation and shift-test code the
GPU scan uses, and checks both that the planted rate comes back and that nothing
comes back when nothing is planted.

## Files

| | |
|---|---|
| `rope_phase_probe.py` | E0 observational probe: `--stage scan` (GPU) / `--stage report` (CPU) |
| `test_rope_phase_cpu.py` | CPU tests of the E0 analysis, incl. plant-and-recover |
| `rope_phase_e1.py` | E1 causal probe: the position-id gap sweep |
| `test_rope_phase_e1_cpu.py` | CPU tests of the E1 arms and the shift recovery (13) |
| `rope_phase_e2.py` | E2 pointing probe: perceived-midline shift |
| `test_rope_phase_e2_cpu.py` | CPU tests of the stimulus and crossing estimator (6) |
| `rope_phase_e3.py` | E3 filler-position probe; `--stage check` validates the tokenizer end |
| `rope_phase_e4.py` | E4 the frozen cross-modal query, and the `d0` sweep |
| `test_rope_phase_e4_cpu.py` | CPU tests of the intervention, the splice and resume (11) |
| `submit_rope_phase_job.sh` | single-GPU batch submission (what produced `results/`) |
| `launch_rope_phase.sh` | 8-way fan-out on a held interactive node (untested) |
| `results/` | reports from the runs above |

## Prior art, and what is not done

The *magnitude* half of this story is covered — Circle-RoPE (2505.16416), DIPE
(2603.10863), CCA-LLaVA (2410.15926) are all about attention mass and visual
fading. Circle-RoPE and DIPE both incidentally remove this drift, since they stop
text tokens' spatial coordinates from advancing, but neither states the invariance
as their motivation. The *shape drift* measured here appears unclaimed.

**What is measured, end to end.** E0 found the march and E1 confirmed it causally.
E2 put its behavioural cost at 0.63 patches on a purpose-built task; E3 showed the
natural-tokens version of that question is confounded by content; E4 measured the
cost on real annotated images (0.36 patches over a 512-token answer) and showed it
can be removed at inference for under 1% NLL -- but only ~40% of it, the rest being
the fading channel.

**Not done.** The 60% that fading contributes is untouched and it is the larger
half. Whether any of this shows up in generated text has not been tested: every
behavioural number here comes from reading two answer-token logits, never from a
completion the model actually wrote. Dense OCR and counting are still untested at
length. And the per-head causal test is open: a handful of heads carry 33-51% of the
drift while the median carries 0.01%, so an intervention restricted to those heads
would be far gentler than one applied to all 1152.
