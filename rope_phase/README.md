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

**Effect size is small: positional share 0.0087**, i.e. under 1% of the variance in
a token's image-attention profile, stable across 32- and 256-case runs. The march
is statistically overwhelming and behaviourally tiny. Quote both together.

Layer profile (fraction of heads marching, `h8`): 0.88–1.00 through L0–L14, collapse
at L15/16 (0.16/0.09), L20 (0.00), L27 (0.00), L29 (0.03), back to 1.00 at L35.

Full reports in `results/`.  The raw bucket sums those came from (~250 MB per
run) were left in the host repo at `outputs/rope_phase/{e0_pilot,e0_qwen3_256,
e0_qwen25}/scan/` rather than copied here; only the reports travel with this
directory.  Re-running regenerates them.

## E1 — the causal arm (built, not yet run)

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

Predictions, stated before running: `null` at fp noise; `t` leaves shape alone but
moves mass; `hw` drifts the profile centroid with delta and **ripples at period 8.0
(rows) / 10.2 (cols)** — a monotone curve would be a decay story, and no decay story
produces a ripple at a pre-specified period; `fix` flat. Deltas are swept densely
(0–16 by 1, then out to 256) because log spacing would miss the signature entirely.

Mass and shape are reported separately throughout — conflating them is exactly how
this effect gets mistaken for visual fading.

`test_rope_phase_e1_cpu.py` does not test the implementation against itself: it
enumerates the pairwise offsets that actually reach the attention logit
(tail↔tail, image↔image, tail↔image) and asserts which ones each arm is allowed to
move. That is where an off-by-one would silently invalidate the GPU run.

```fish
SCRIPT=rope_phase_e1.py N_SAMPLES=20 OUT_DIR=$PWD/outputs/e1 bash submit_rope_phase_job.sh
```

Cost is ~120 forwards per case (5 arms x 24 deltas), so roughly 15–25 min for 20
cases on one H100 — estimated from E0's rate, not measured.

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
| `test_rope_phase_e1_cpu.py` | CPU tests that each E1 arm perturbs only what it claims |
| `submit_rope_phase_job.sh` | single-GPU batch submission (what produced `results/`) |
| `launch_rope_phase.sh` | 8-way fan-out on a held interactive node (untested) |
| `results/` | reports from the runs above |

## Prior art, and what is not done

The *magnitude* half of this story is covered — Circle-RoPE (2505.16416), DIPE
(2603.10863), CCA-LLaVA (2410.15926) are all about attention mass and visual
fading. Circle-RoPE and DIPE both incidentally remove this drift, since they stop
text tokens' spatial coordinates from advancing, but neither states the invariance
as their motivation. The *shape drift* measured here appears unclaimed.

The E1 harness (below) is built and CPU-tested but **has not been run on a GPU**, so
there is as yet no measurement of what the drift costs.
