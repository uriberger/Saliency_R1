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

## Caveats

- 8 pictures. Enough to see a 3–5% systematic shift; not enough for a confidence interval
  on any single statistic.
- One query set (`text`, the scan's primary). The generated-token set was not measured.
- The Mamba layers ran on the slow torch fallback in all three runs, so they are
  compared like with like, but none of these are the numbers a run with the fast kernels
  installed would produce.
