# How the 8 cards divide for an Omni run

Written 2026-09-27. Retracts a hedge in an earlier draft: "or give the trainer two cards"
was offered as a fix for a memory problem that measurement says does not exist, and it
would have cost roughly 2x in step time.

Two words used throughout. A **copy** is one complete set of the model's weights. **Batch
8** means the trainer does 8 small batches before each weight update (`grad_accum`), and
what the run actually holds fixed is the product `per_device x processes x grad_accum`,
which is 48 rollouts per update on the Qwen3-VL runs.

## What the Qwen3-VL run does now

```
GPU 0      detector (Grounding-DINO)
GPU 1      generation server
GPU 2-7    6 training processes, ONE copy shredded across all six
```

Six processes x batch 8 = 48. The shredding is the thing that makes a Nemotron step
slow: 66 GB of weights are re-collected 16 times per step.

## Plan A — one whole copy per card. This is the one to use.

```
GPU 0      detector
GPU 1      generation server
GPU 2-7    6 training processes, each holding a WHOLE copy (73.3 GB)
```

Nothing is shredded, so nothing is re-collected; the only traffic per step is averaging
the 1.22M LoRA gradients, which is nothing. Batch stays 8, six processes, 48 rollouts per
update, unchanged. **Measured: 34.0 s** on the training side.

Headroom per card is 5.9 GB, and the measured sequence lengths say that is enough — see
`docs/omni-quantization.md`. The longest real sequence on `set_a` is 1,358 positions
against the 1,327 benchmarked.

## Plan B — one copy spread across a pair of cards. What "two cards" meant, and why not.

```
GPU 0      detector
GPU 1      generation server
GPU 2+3    training copy 1   (layers 0-25 here, 26-51 there)
GPU 4+5    training copy 2
GPU 6+7    training copy 3
```

Each card holds about half the weights, ~31 GB, leaving ~49 GB free. The memory pressure
disappears completely.

**But six processes become three**, and the batch has to double from 8 to 16 to keep the
same 48 rollouts per update. Each copy does twice the work, so the step roughly doubles —
about 68 s, which is back where 8 bits was.

And splitting by layer does not make a pair faster than a single card: while the first
card computes layers 0-25 the second is idle, and then they swap. A pair buys memory, not
speed. Paying 2x in time to solve a memory problem that measurement says does not exist
is the wrong trade, which is why Plan A is the recommendation and this section exists
only to say so explicitly.

(There is a second way to split a copy across two cards — the shredding the current run
uses, `ZeRO-3` — which spreads every layer rather than assigning whole layers. It costs
the same parallelism as Plan B *and* adds the re-collection traffic, so it is strictly
worse here.)

## The one thing genuinely unsettled: the generation server

Plan A gives the generation server a single card, and the Omni is 62 GB of weights at
full size. That leaves under 18 GB for the key-value scratch space generation needs. It
may be enough at `max_model_len 4096` with 8 rollouts; it has not been measured, and
vLLM cannot serve this model at the installed version anyway
(`docs/omni-training-blockers.md`).

If it turns out to need two cards, the trainer loses one:

```
GPU 0      detector
GPU 1-2    generation server across two cards
GPU 3-7    5 training processes, one whole copy each
```

Five processes, so the batch goes from 8 to 10 to hold 48 rollouts, and the step lands
near 42 s rather than 34 s. That is the realistic downside case, and it is still
comfortably better than any 8-bit arrangement.

## Summary

| plan | training copies | batch | step (training side) | note |
|---|---|---|---|---|
| A: whole copy per card | 6 | 8 | **34.0 s** measured | recommended |
| A': generation server on 2 cards | 5 | 10 | ~42 s estimated | if the server needs it |
| B: copy across a pair | 3 | 16 | ~68 s estimated | solves a problem you do not have |
| current shredding, 16-bit | 6 | 8 | not measured | what Plan A replaces |
