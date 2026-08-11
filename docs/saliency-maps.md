# Saliency maps: every definition, in one place

Five maps have been used to ask "was this reasoning step looking at the objects it
names", plus one intervention that edits the flow rather than measuring it. The maths
lives in the probes' module docstrings, which is the right place to maintain it but the
wrong place to compare them. This page states the notation once and defines each map
against it.

Nothing here is a result. Results are in [probe-results.md](probe-results.md); the state
of the investigation is in [HANDOFF.md](HANDOFF.md).

---

## Notation

| symbol | meaning |
|---|---|
| `p, q` | positions in the full sequence (prompt + image + chain), `1..P` |
| `I` | the image-token positions, `\|I\| = M` (the patch grid, `gh × gw`) |
| `l` | decoder layer, `0..35` |
| `h` | attention head, `0..31` |
| `A^{l,h}_{p,q}` | post-softmax attention, query `p`, key `q`. Causal, rows sum to 1 |
| `S` | a step's **token span** — the positions its text occupies, `{a … b-1}` |
| `U ⊆ I` | image tokens inside that step's Grounding-DINO union |
| `e_j` | image embedding `j` as it enters the language model |
| `W_O^h` | head `h`'s output-projection block; `v^h_q` its value vector at `q` |

A step's span comes from `prompt_len + tok_a` to `prompt_len + tok_b`
(`flow_correlation_probe.py:394-398`). `a-1`, the token immediately before the span, is
used by the increment.

Every map below produces a length-`M` vector over image patches, which is then scored
against `U` by the two metrics in [Scoring](#scoring).

---

## 1. Direct map

`head_correlation_probe.py`. The simplest thing: head `h` of layer `l`, attention from
the step's own tokens straight to the image patches.

```
direct^{l,h}_j = mean_{p in S} A^{l,h}_{p, I_j}
```

One map per (layer, head) — 1,152 of them. This is what the GRPO overlap reward paid
for, at L22 heads 28/31.

**What it assumes:** that the model reads the patches *while writing the step*. The
information-flow literature says otherwise — the image is absorbed into text positions
in early layers, and later text reads those positions rather than the patches. A direct
map at layer 22 cannot see that path at all, which is what motivated maps 2–4.

---

## 2. Rollout, heads merged by the mean (`rollout_mean`)

`flow_correlation_probe.py`. Follows image content through intermediate text positions.
`sal_p` is a length-`M` vector: *how much of what sits at position `p` is traceable to
each patch*.

```
sal^{(0)}_p = one-hot at j   if p is image token j,  else 0

sal^{(l)}_p = a · Σ_{q<=p} w^{(l)}_{p,q} sal^{(l-1)}_q  +  (1-a) · sal^{(l-1)}_p
```

with `a = 0.5` (Abnar & Zuidema's retention constant) and `w^{(l)} = mean_h A^{l,h}`.
The first term is what layer `l`'s attention pulls in; the second is what the position
keeps, because a block *adds* its output to the residual stream.

The step's map is `mean_{p in S} sal^{(l)}_p`, read at the **last layer** — it already
contains every layer below it, so no layer has to be selected. Per-layer readouts are
recorded as secondaries.

Three things this gets right that a single-layer form does not:

- the inherited map is indexed at `l-1`, not `l`. Position `q`'s value vector at layer
  `l` is built from `q`'s residual *after* layer `l-1`. A single-layer recursion instead
  sums paths of arbitrary hop count through one layer — paths the architecture cannot
  execute, since `h` hops need `h` distinct layers.
- the recursion starts at the **image tokens**, so image-to-image mixing is carried. By
  layer 22 "image token j" is not purely patch `j`.
- `mass(sal^{(l)}_p) ≤ 1` at every layer by induction, since attention rows sum to 1 and
  the initial masses are 0 or 1. So the total is interpretable as "fraction of this
  position traceable to the image".

---

## 3. Rollout, heads merged by value norm (`rollout_wnorm`)

Identical recursion, different edge weight. Heads are **summed, not averaged**: each
writes into the residual through its own column block of `o_proj`, and the block output
is `Σ_h W_O^h (Σ_q A^h_{p,q} v^h_q)`. So the honest edge weight is the magnitude of that
source's total contribution:

```
w^{(l)}_{p,q} ∝ ‖ Σ_h A^{l,h}_{p,q} · W_O^h v^h_q ‖₂        (row-normalised to sum to 1)
```

This is also the value-norm correction: raw attention overweights sinks, which take a
large share of every row while carrying near-zero-norm values. It is the only form that
can express cancellation between heads.

Computed via a per-source Gram matrix `G[q,h,g] = ⟨W_O^h v^h_q, W_O^g v^g_q⟩` rather than
materialising `P·P·d` floats (`edge_weights_wnorm`).

The plain mean (map 2) is the max-entropy default when no value information is used —
a convention, not a fact about the architecture, which is why both are run.

---

## 4. Increment (`incL`)

`sal` is cumulative in the sequence: by the time the model writes step 5, those
positions already carry whatever the question tokens and steps 1–4 pulled in. A high
score can therefore reflect a completion-level property rather than per-step grounding.
The increment removes the inherited baseline:

```
inc^{(l)} = mean_{p in S} sal^{(l)}_p  −  sal^{(l)}_{a-1}
```

Both terms come from **the same single forward pass** — this is a difference of two
observations, not an ablation. Nothing is re-run without the prior content.

Entries can be negative, so the question it answers shifts from *"is there mass on the
named objects"* to *"did the named objects **gain** more than the background did"*. That
is why it is scored by AUROC only: `mean_in_v2` divides by the map's mean, which an
increment can drive through zero.

One `incL` per layer since 2026-08-07; before that, the last layer only.

---

## 5. Gradient map (`grad`)

No flow model at all — differentiate the thing directly. For a step spanning `t_a..t_b`,
teacher-forced on the model's own chain:

```
F   = Σ_{n=a..b-1} log P(t_n | t_<n, image, prompt)
g_j = ‖ ∂F / ∂e_j ‖₂
```

This counts every path through every layer and head, with no `a`, no head-merge
convention and no rollout approximation, so it is the control on whether the rollout's
approximations cost anything. It cannot be decomposed per head, but it can be a
differentiable loss unchanged.

Four variants: `gnorm` / `gxi` exclude the deepstack injection paths, `gnorm_ds` /
`gxi_ds` include them. `gxi` is gradient-times-input, `|⟨e_j, ∂F/∂e_j⟩|`, usually the
less noisy.

### 5b. The training-time gradient map (`trl/grad_maps.py`)

The GRPO gradient reward (`--reward_variant grad`) uses the same idea with two changes,
each of which closes a hole the probe version does not have to care about:

```
F_S = mean_{n in S} [ z_{t_n,n} − (1/V) Σ_v z_{v,n} ]        centered logit
G_j = ‖ ∂F_S / ∂x_j ‖₂                                       x_j = token j's 32×32 px
```

- **the centered logit instead of the log-prob.** `∂ log P(t)/∂z = onehot(t) − p` vanishes
  as `p_t → 1`, so a log-prob map shrinks everywhere for a step the model is sure of and
  the reward would pay for uncertainty. The raw logit does not saturate but carries a
  common-mode component shared by the whole vocabulary — the *same map for every step* —
  so shaping it once would lift every step at no per-step cost. Subtracting the vocabulary
  mean removes that channel and keeps the non-saturating property.
- **w.r.t. pixels, not embeddings.** The vision tower stays in the graph, so the deepstack
  taps are counted automatically and there is no `_ds` variant to choose. ~10% more
  compute (the tower is 0.55 B params over ~1024 patches against 8 B over ~900 tokens).

Mean-over-tokens versus sum differs by exactly `1/|S|`, which the roll-null ratio below
cancels; the mean is used so the logged raw norms are comparable across step lengths.

---

## 6. The intervention edit

`flow_intervene_probe.py`. The only entry here that *changes* the model rather than
measuring it. Maps 1–5 are correlational; so is every result derived from them alone.

Carrying full `sal` per position would cost `O(P·M)`. It is not needed: the recursion is
**linear in `sal`**, so any linear functional of it obeys the same recursion. Two scalars
per position suffice:

```
u_q  union-traceable mass    u^{(0)}_q = 1 for q in U,  else 0
m_q  image-traceable mass    m^{(0)}_q = 1 for q in I,  else 0
```

At layer `l`, for query `p in S`:

```
E_p   = { q <= p : m^{(l-1)}_q > 0 }                    eligible keys
T_q   = u^{(l-1)}_q / Σ_{r in E_p} u^{(l-1)}_r          target, sums to 1 over E_p
M^h_p = Σ_{q in E_p} A^{l,h}_{p,q}                      this head's mass on E_p

A'^{l,h}_{p,q} = (1-α)·A + α·M^h_p·T_q      q in E_p
A'^{l,h}_{p,q} = A                          otherwise
```

In words: **hold fixed how much this head reads from image-carrying positions, and
re-allocate that mass across them in proportion to how much boxed-object content each
one holds.** Eligible keys include text positions that absorbed image content earlier —
that is the whole point, and it is what distinguishes this from the direct intervention
in `intervene_probe.py`.

Row sums and every `M^h_p` are preserved exactly, and no mass lands on `q > p`.
`T` is normalised **per query row** over `q <= p`; a single global normalisation would
place mass after `p`.

Conditions, each a separate forward: `box` (`T` from `U`), `roll` (`T` from `U` rolled to
a random offset, matched area — the wrong-place control), and `α = 0` (no edit, same rows
rebuilt through the same path, so the eager-rebuild difference cancels). Read
`box − roll`, never `box` alone.

---

## Scoring

Both metrics score a map's length-`M` vector against the step's union mask.

| metric | definition | chance |
|---|---|---|
| `mean_in_v2` | in-mask mean ÷ overall mean | **1.0** |
| `auroc` | Mann-Whitney rank of in-mask patches vs out, average ranks for ties | **0.5** |
| `logratio` | `log ‖g_U‖ − log N_0`, where `N_0² = (1/K) Σ_k ‖g_{U'_k}‖²` over `K` translates of `U` | **0.0** |

`logratio` (`trl/rewards/grad_rewards.py`) is the roll-null: the map scored against
*itself*, on the same mask moved elsewhere. Three properties the other two lack:

- **exact size invariance.** On a flat map the score is 0 for a union of any area — the
  translate has the same area by construction. `‖g_U‖` alone grows like `√|U|`, and even
  `‖g_U‖ − ‖g_out‖` is monotone in `|U|` for a flat map, so both pay for bigger boxes.
- **self-limiting.** Under a uniform translation `E|U ∩ U'| / |U| = |U|/n`, so a union
  covering most of the image is compared against itself and the ratio collapses to 1.
  That is a *pull toward smaller unions*, not merely a cap on large ones.
- **scale invariance.** Any `G → cG` cancels, which removes the confidence and
  image-sensitivity confounds that a subtractive control would only shift.

`N_0` pools the **squared** norms over the `K` placements before the log, so one control
landing on a dead region cannot dominate. Offsets are drawn in-frame (no border wrap) and
exclude the identity. What it does not close: a radial prior in `G` still lets a centred
box beat its translates for free — `grad/ecc` is the monitor, not a fix.

`head_correlation_probe.metrics`. Ties get average ranks because attention maps have
many near-identical near-zero patches and `argsort` would break those arbitrarily.
Degenerate unions (empty or full) are skipped — there is no in/out contrast to score.

**Read the level next to any correlation.** A map can predict correctness while sitting
on the wrong side of chance, in which case "more overlap goes with being right" is really
"less anti-grounding goes with being right" — a different claim.

**Union area confounds the level.** Every map reads lower the larger the union gets
(`r(union, auroc) = −0.55` over all 1152 heads), and the median step's union covers 54%
of the grid. `--max-union` on both correlation probes' `report` stage caps it; both
reports open with a union-decile table.

---

## Parameters, and which were ever varied

| parameter | value | swept? |
|---|---|---|
| `a`, rollout retention (`ROLLOUT_A`) | 0.5 | **never** |
| `α`, intervention strength | 0.25 / 0.5 / 1.0 | yes — but α=1.0 is off-manifold, see result 5 |
| layer cutoff `L` (intervention) | 8 / 16 / 24 / 35 | yes |
| readout layer (rollout) | last (35) | per-layer secondaries recorded |
| `box_threshold` (DINO) | 0.1 | no |
| `max_box_area` | 0.5 | no — caps each **box**, not the union |
| `tight_union` | 0.35 | no |

Two distinct things are called `a`/`alpha` and it is worth keeping them apart: the
rollout retention constant, fixed at 0.5 throughout and never tested, and the
intervention strength, which was swept.

---

## Deepstack, which affects maps 2–6

Qwen3-VL does not feed the image in once. The vision encoder taps features at **vision**
layers 8/16/24, and those three tensors are **added into the residual stream at the
image-token positions** at **LM layers 0, 1, 2** — `layer_idx in
range(len(deepstack_visual_embeds))`, `modeling_qwen3_vl.py:835`.

`deepstack_visual_indexes = [8, 16, 24]` in the config names the layers the features are
tapped **from**, not the ones they land **in**. Reading it as a set of LM layers puts any
re-seed in the wrong place.

- **Maps 2–4** do not model the addition. It only pushes image positions further toward
  their own patch, so `sal` there stays a valid attribution, but the mixing ratio is not
  exact.
- **Map 5** captures both the merged embeds and every deepstack tensor as leaves, which
  is what the `_ds` variants are.
- **Map 6** re-seeds `u` and `m` at layers 0/1/2, because fresh image content enters
  there that the recursion never saw arrive. Without it every mass downstream is
  understated and the targets built from `u` are wrong.

---

## Where the code is

| file | maps |
|---|---|
| `head_correlation_probe.py` | 1, and the scoring both correlation probes share |
| `flow_correlation_probe.py` | 2, 3, 4, 5 |
| `trl/grad_maps.py` | 5b — the training-time pixel gradient, shared with `saliency_viz.py` |
| `trl/rewards/grad_rewards.py` | the `logratio` roll-null scoring, and the reward built on it |
| `flow_intervene_probe.py` | 6 |
| `intervene_probe.py` | the *direct* intervention (not a map — edits step→image attention at one layer) |
| `test_flow_correlation_cpu.py`, `test_flow_intervene_cpu.py` | the algebra, against naive references |
