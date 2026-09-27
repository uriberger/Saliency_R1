# Shrinking the Omni's frozen base: 8 bits keeps the attention, 4 bits flattens it

Written 2026-09-27, branch `probe/omni-quant`. `omni_quant_probe.py` produced it,
`outputs/sink_location/xmodel/omni_quant/*.npz` holds the arrays.

## Why

Training the overlap reward on `Nemotron-3-Nano-Omni-30B-A3B` is expensive for one
reason: 66 GB of bfloat16 weights do not fit beside a generation server on an 80 GB
card, so the trainer cuts them into pieces across cards and re-collects them 16 times a
step. Under LoRA the base is **frozen** — nothing is written back into it — so it can be
stored in fewer bits. If it fits whole on one card, the cutting-up and its traffic go
away, and the step-time problem goes with them.

That is only worth building if the shrunken model still looks in the same places, because
where it looks is what the reward scores.

## The measurement

8 pictures from the boxed corpus, one per source type, the same prompts and the same scan
at each setting. Compared per (layer, head): 48 layer-maps and 1,536 cells. The RADIO
tower and the `mlp1` projector are left full-size in both shrunken runs — they are ~600M
of 33B, they are where the border stamp comes from, and the grid every patch statistic is
defined on is their output. Shrinking them would confound the measurement with the thing
being measured.

## Result

```
                           8-bit          4-bit
attention map correlation  0.9969 mean    0.9841 mean
                           0.9734 min     0.9568 min
                           2.1% < 0.99    72.9% < 0.99
most-attended patch same   91.2%          76.6%
peak on the top-left patch 11.9% (bf16 11.8%)   10.7%

corner_share                -0.6%          -5.2%
corner_tl_share             +0.5%          -4.1%
ring_share                  -0.3%          -3.2%
peak_in_ring                -0.2%          -3.7%
first_patch_share           +0.5%          -4.1%
peak_share                  -0.7%          -3.1%
image_mass                  +0.5%          -0.0%
entropy_norm                +0.0%          +0.8%
```

```
mode     peak GPU   fits 80GB   load s   s/picture   vs bf16
bf16        63.9G         yes      113         1.0     1.00x
int8        34.8G         yes       87         2.5     2.42x
nf4         20.6G         yes       46         1.0     0.95x
```

**8 bits preserves where the model looks.** Every quoted statistic moves by less than one
percent, in no consistent direction. The map correlation is 0.997 and only 2% of
layer-maps fall below 0.99. The most-attended patch moves in 9% of cells, but the
*distribution* of where peaks land does not — top-left holds 11.8% of cells at full size
and 11.9% at 8 bits — so those are near-ties flipping, not the lever moving.

**4 bits flattens the map, systematically.** Every concentration statistic falls by
3–5% and the flatness measure rises. That is one consistent direction, not noise: the
attention spreads out. The most-attended patch moves in a quarter of all cells. For a
reward that divides by the maximum, a systematically smaller maximum is the lever itself
moving.

**8 bits is 2.4x slower per forward; 4 bits is free.** This is the known shape of
bitsandbytes: its 8-bit path splits out large activations and pays for it, while its
4-bit path expands back to bfloat16 and runs at full speed.

## What this does and does not settle

Settled: both fit on one card with room to spare, so the weight-splitting can be dropped.
Both shrink the parts that matter — the experts, the Mamba projections and the attention
projections are all `nn.Linear`, which is what the shrinking acts on. And 8 bits does not
move the measurement.

**Not settled, and it is the number that decides this:** the step time. Everything above
is a forward pass at batch 1 on ~270 picture-tokens. The reason to shrink was to delete
the cross-card traffic, and this measurement cannot see that cost at all. A forward that
is 2.4x slower can still be a large net win if it removes something bigger. Twenty real
training steps would say.

**Also not settled:** 4-bit LoRA training is the well-trodden path — it is what QLoRA
means and it is what most tooling assumes. 8-bit LoRA training works but is used far
less. So the accurate option here is also the less-tested one, and the fast, common
option is the one that bends the numbers.

## Recommendation

Train at 8 bits if the step time allows it. Nothing in the measurement has to be
re-baselined and any attention number quoted off the run stands beside the existing
full-size tables.

If the step time forces 4 bits, it is still usable — the reward compares eight rollouts
of the same model, and a flattening that hits all eight equally largely cancels. But then
**every** attention number quoted off that run has to be compared against a 4-bit
baseline, never against the full-size cross-model table. `corner_share` falling 5% is not
negligible next to the effects this project measures.

## The step time, measured — and it says don't quantize

20 steps, `omni_train_step_bench.py`, the real settings: LoRA r=16 on
q/k/v_proj in the six attention layers, lr 1e-5, 8 micro-steps of one sequence of
303 + 1024 = 1,327 positions, a no-grad saliency re-forward each micro-step, beta 0 so no
reference forward, no gradient checkpointing (the launcher does not use it).

```
8-bit base, one card, NO weight-splitting
    median   68.1s      mean 66.4s   sd 3.3s   min/max 59.5 / 70.7
    peak GPU 63.8 GB of 79.2        headroom 15.4 GB
    3,990 steps = 75 h, training side only

16-bit base, one card              OUT OF MEMORY
    76.8 GB in use, needed 2.75 GB more
```

**The gradient check passed first.** All 36 LoRA tensors finite and non-zero after one
optimizer step, gradient norms 3.0e-04 to 6.3e-01. The learning signal does travel back
through the 23 Mamba layers on the torch fallback. That was the riskiest unknown on the
list and it is answered: the plan is not dead on correctness.

**But quantising does not buy the step time.** 68 s on the training side alone is 75
hours for a 3,990-step run, before generation and reward. The 8-bit matmul is 2.4x slower
than bfloat16, and that penalty is larger than the weight-splitting it was meant to
replace. Trading a communication cost for a bigger compute cost is not a trade.

**And the memory is not where I said it was.** Peak training memory at 8 bits is 63.8 GB
against 34.8 GB at inference — so roughly **30 GB is activations, not weights**. That is
what actually put 16-bit over the edge: 62 GB of weights plus the same 30 GB of
activations is 92 GB, and the run died 2.75 GB short of finishing an allocation.

Which reframes the whole thing. The launcher does not use gradient checkpointing —
recomputing intermediate results during the backward pass instead of storing them. Turning
it on typically cuts activation memory by an order of magnitude, for about 30% more
compute. At 16 bits that would be roughly 62 GB of weights plus a few GB of activations:
**it would fit on one card, at full compute speed, with no quantization and no fidelity
question at all.** Scaling the measured 68 s by 1/2.42 for bfloat16 compute and 1.3 for
the recomputation lands near 37 s — an estimate, not a measurement, and the obvious thing
to measure next.

So the recommendation above is superseded on the training question, though not on the
fidelity one: if you ever do quantise, 8 bits is the setting that does not move the
attention. But the first thing to try is **16-bit with gradient checkpointing on one
card**, which avoids the whole question.

## The answer: 16 bits with recompute, and no quantization at all

Recompute = gradient checkpointing, throwing the forward pass's intermediates away and
re-running them during the backward. The launcher does not use it. Turning it on is what
the activation finding above pointed at, and it settles the question.

```
setting                          fits 80GB   step (training side)   peak GPU   3,990 steps
16-bit, no recompute                    NO                      -      > 79              -
16-bit + recompute                     yes    34.0s  (sd 0.5)     73.3 GB           38 h
 8-bit, no recompute                   yes    68.1s  (sd 3.3)     63.8 GB           75 h
 8-bit + recompute                     yes    85.3s  (sd 3.2)     48.6 GB           95 h
```

**16 bits with recompute is twice as fast as anything involving 8 bits**, and it needs no
quantization, so the fidelity question disappears with it. Recompute costs ~36% more
compute, and 16-bit arithmetic is 2.4x faster than 8-bit; the second wins easily. Adding
8-bit and recompute together is the worst of both.

**The headroom is 5.9 GB, and that turns out to be enough.** Measured on a 1,327-position
sequence, and `max_prompt_length 2048` made that look risky. It is not: on `set_a`'s
50,000 rows the questions are **8-24 tokens** (p50 8, p100 24) and the Omni's pictures
come to **266-286 tokens** (200 real images, none over 512x512). So the longest real
sequence is

    286 picture + ~48 text + 1024 completion = 1,358 positions

against the 1,327 benchmarked -- a 2% difference. `max_prompt_length 2048` is a cap
nothing approaches. **No second card is needed for the trainer, and asking for one would
cost roughly 2x in step time** (see `docs/omni-gpu-layout.md`).

The variable to watch is not the question length but the PICTURE: the Omni is
native-resolution and climbs to 3,328 tokens on a large image. This corpus is capped at
512x512. A training set with bigger pictures would change this conclusion and nothing
else in this file.

## Recomputation is safe here, but the first test said otherwise and was wrong

The first version compared the gradients with recompute off and on, got a worst relative
difference of 5.16e-01, and concluded recomputation was broken. It had no control.

With the control -- two passes with recompute OFF, to establish what the same computation
twice costs on this model -- at both precisions:

```
                       off vs off (control)        off vs on (signal)
16-bit                 6.50e-01, cos 0.768         3.09e-01, cos 0.955
 8-bit                 4.89e-01, cos 0.874         5.88e-01, cos 0.861
```

**The model does not reproduce its own gradients run to run**, at 16 bits as well as 8, so
this is not a bitsandbytes artefact. The recompute difference is the same size as the
noise, and at 16 bits it is smaller. So there is no evidence against recomputation -- and
none for it either; this test cannot resolve it while the baseline is that loud.

Two things worth saying about that number before anyone quotes it. It is a **worst
per-tensor relative** difference over 36 tensors, so whichever tensor has the smallest
gradient norm dominates it, and the minimum norms here are ~2e-04 against a median of
~3e-02. A global metric over the concatenated gradient would very likely be far smaller.
And GRPO is comparatively robust to gradient noise. Neither of those has been measured;
they are the reason not to treat the nondeterminism as alarming, not a reason to ignore it.

## Caveats

- 8 pictures. Enough to see a 3–5% systematic shift; not enough for a confidence interval
  on any single statistic.
- One query set (`text`, the scan's primary). The generated-token set was not measured.
- The Mamba layers ran on the slow torch fallback in all three runs, so they are
  compared like with like, but none of these are the numbers a run with the fast kernels
  installed would produce.
