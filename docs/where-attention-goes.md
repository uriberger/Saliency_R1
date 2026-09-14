# Where does the attention actually go?

**A survey of the "VLM attention is not on the object" claims, and what each one says it lands on instead.**

Compiled 2026-09-14. Scope: work from 2024 onward, weighted to 2025–2026, on large
vision–language models and on the vision encoders they inherit from.

This exists because our own result — that attention mass concentrates on the **outer ring of
the image**, at 2.6x the border's area share, in all 12 image types — needs a related-work
paragraph that says honestly what has and has not been claimed before. The short version: the
literature's answer is overwhelmingly **background**, given as a semantic property of the
patches, and almost never as a geometric one.

## How to read the verification column

Claims marked **V** were read in the primary source. Claims marked **S** come from search
summaries of the primary source and were not opened directly — they are reliable enough to
orient by and **should be checked before being quoted in the paper**.

---

## 1. The five answers in the literature

### 1.1 Background, and low-information patches (the dominant answer)

| paper | venue | claim | ver. |
|---|---|---|---|
| DAMRO, [arXiv:2410.04514](https://arxiv.org/abs/2410.04514) | EMNLP 2024 | The LLM decoder's attention over image tokens is highly consistent with the ViT's, and both "tend to focus on particular background tokens rather than the referred objects". Attributed to a flaw in the visual encoder. | S |
| Darcet et al., *Vision Transformers Need Registers*, [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) | ICLR 2024 | High-norm artifact tokens (~2% of the sequence, ~10x norm) appear "in low-informative background areas", driven by local redundancy — artifact patches are ~2.5x more likely to sit in regions similar to their neighbours. They store global rather than local information. | S |
| *Vision Transformers Don't Need Trained Registers*, [arXiv:2506.08010](https://arxiv.org/abs/2506.08010) | 2025 | Fewer than 10 neurons dictate where the outliers appear; ablating them **moves** the outliers. A mechanistic account with no positional component. | S |

This is the family our result is closest to, and the one it has to be distinguished from.

### 1.2 Fixed positions — but fixed with respect to the *text*, not the image

| paper | venue | claim | ver. |
|---|---|---|---|
| Kang et al., *See What You Are Told: Visual Attention Sink in LMMs*, [arXiv:2503.03321](https://arxiv.org/abs/2503.03321) | ICLR 2025 | Sinks identified by massive activation in fixed hidden-state dimensions; the same visual tokens win regardless of the query. Spatially characterised as background. | **V** |

Treated in detail in §2, because it is the paper a reviewer is most likely to raise.

### 1.3 The lower part of the image (positional / recency, from RoPE decay)

| paper | venue | claim | ver. |
|---|---|---|---|
| MCA-LLaVA, [arXiv:2507.09184](https://arxiv.org/abs/2507.09184) | 2025 | "Image alignment bias": image tokens are flattened in raster order, and RoPE's long-term decay makes instruction tokens attend preferentially to tokens later in that order — i.e. the **bottom** of the image. | S |
| VisPruner, [arXiv:2412.01818](https://arxiv.org/abs/2412.01818) | ICCV 2025 | Attention-based token pruning preserves the lower part of the image, sometimes pure padding; bias is present from the first layer and stronger in shallow layers. | S |
| Attention Debiasing for Token Pruning, [arXiv:2508.17807](https://arxiv.org/abs/2508.17807) | 2025 | A content-agnostic recency bias: raw attention rises with token index, so it can be estimated by averaging over many images. | S |

**This is the closest existing work to a *geometric* claim** — it is about position rather than
content. But the geometry it describes is a 1-D gradient along raster order (top→bottom), not a
ring, and its stated mechanism is positional encoding in the LLM, not the vision encoder.

### 1.4 Not the image at all — text and instruction tokens

| paper | venue | claim | ver. |
|---|---|---|---|
| Attention Hijackers, [arXiv:2503.08216](https://arxiv.org/abs/2503.08216) | 2025 | Specific instruction tokens disrupt visual attention. | S |
| Modality Bias in LVLMs, [arXiv:2508.02419](https://arxiv.org/abs/2508.02419) | 2025 | Separates generative from discriminative hallucination by whether visual or textual reliance dominates. | S |

### 1.5 The dissent: the attention is fine, the premise is wrong

| paper | venue | claim | ver. |
|---|---|---|---|
| *Same Attention, Different Truths*, [arXiv:2608.07302](https://arxiv.org/abs/2608.07302) | 2026 | Real and hallucinated objects receive **equally strong** visual attention in mid-to-late layers. The problem is not how much the model attends but what it does with it. | S |
| *MLLMs Know Where to Look*, [arXiv:2502.17422](https://arxiv.org/abs/2502.17422) | ICLR 2025 | Models "consistently know where to look, even when they provide the wrong answer" — the failure is perceiving small detail, not locating it. | S |
| Adaptive attention calibration, [arXiv:2505.21472](https://arxiv.org/abs/2505.21472) | 2025 | Methodological: raw single-layer attention weights do not reliably reflect token importance, since embeddings are progressively contextualised. | S |

The last row cuts against our own measurements as much as anyone's and should be cited rather
than avoided.

---

## 2. What the canonical sink paper actually claims

Kang et al., ICLR 2025 — read directly, since it is both the most-cited VLM sink result and the
one whose "fixed locations" phrasing is most often over-read.

**The criterion is activation-based; position never enters.** A visual token is a sink if its
value in fixed hidden-state dimensions (e.g. {1415, 2533} for LLaMA2-7B) exceeds tau = 20.

**"Fixed" means invariant to the text token, not across images:**

> "irrelevant visual tokens exist in fixed locations, regardless of the specific text token"

> "whether the text token is *knife* or *cup*, the model consistently attends to the same
> irrelevant visual tokens"

It cannot mean fixed patch indices across images: their own background statistics are computed
against per-image segmentation masks, which only makes sense if the positions move with the
picture.

**The spatial characterisation is semantic:**

> "Visual sink tokens are mostly located in the background, which is less informative."

**And the effect is weak.** Table 6, LLaVA-1.5-7B, background defined as "all regions except the
main object":

| | Pascal-VOC | MS-COCO |
|---|---|---|
| visual **sink** tokens in background | 90.5% | 93.7% |
| **all** visual tokens in background (base rate) | 82.9% | 90.5% |
| **enrichment over base rate** | **1.09x** | **1.04x** |

The headline is close to the base rate, because most of an image is background. Most citations
of this paper omit the all-tokens row.

**Borders, edges, corners and periphery are never mentioned.**

They also show that **removing** the sink tokens costs no performance — the opposite polarity to
treating the concentration as a peak worth optimising against.

---

## 3. The border question

### What exists

One sentence, in *Attention Sink in Transformers: A Survey*,
[arXiv:2604.10098](https://arxiv.org/abs/2604.10098) (verified verbatim):

> "they are spatially concentrated at image boundaries, correlating strongly with background
> regions rather than foreground objects"

It is attributed to a single reference **[29]**, which is **unresolved**: the arXiv HTML
truncates before the bibliography, and both the survey PDF and the ICLR proceedings PDF exceed
the fetcher's size limit. **Resolve this before claiming priority.**

The only border sentence found in Darcet et al. concerns the *registers* — "register 3 tends to
focus on border areas, while other registers focus on more centered areas" — i.e. the fix, not
the artifact tokens.

### What does not exist

No primary source found states that attention mass concentrates on the image periphery as a
geometric fact, conditioned on image content. The mechanisms on offer are all semantic
(background, redundancy) or 1-D positional (raster-order recency).

### Why this matters, and the confound to address head-on

**Border and background are confounded in every paper above.** Image borders are
disproportionately background, so "background" explains their observations without anyone
needing to look at position. The contribution is the *dissociation*, not the observation:

1. the bias holds at 2.6x the border's share **across all 12 image types**, including those
   where the border is not background;
2. it is a **peak**, not a sink — the opposite polarity to a literature that treats these tokens
   as low-value and safely deletable;
3. the reward consequence — a centred rectangle covers 0% of the ring holding half the mass — is
   a claim this literature is not positioned to make.

### One honesty check on the headline contrast

Our 2.6x and Kang et al.'s 1.04–1.09x measure **different quantities**: theirs is a fraction of
sink *tokens* against the fraction of all tokens; ours is attention *mass* against area share.
Both are enrichments over a base rate, so the comparison is fair in spirit, but it must be
stated rather than tabled as like for like.

---

## 4. Suggested related-work framing

> Prior work that finds VLM attention off-target attributes it to the *content* of the attended
> patches — background or locally redundant regions (Darcet et al., 2024; Liu et al., 2024; Kang
> et al., 2025) — or to a 1-D positional bias along raster order induced by the language model's
> positional encoding (Tian et al., 2025). We show instead that the effect is geometric and
> inherited from the vision encoder: attention concentrates on the outer ring of the image at
> 2.6x its area share, and survives conditioning on image type, which dissociates it from the
> background explanation.

Cite the survey's boundary sentence openly rather than letting a reviewer find it.

---

## Bibliography

| ref | arXiv | venue |
|---|---|---|
| DAMRO: Dive into the Attention Mechanism of LVLM to Reduce Object Hallucination | [2410.04514](https://arxiv.org/abs/2410.04514) | EMNLP 2024 |
| Darcet et al., Vision Transformers Need Registers | [2309.16588](https://arxiv.org/abs/2309.16588) | ICLR 2024 |
| Vision Transformers Don't Need Trained Registers | [2506.08010](https://arxiv.org/abs/2506.08010) | 2025 |
| Kang et al., See What You Are Told: Visual Attention Sink in LMMs | [2503.03321](https://arxiv.org/abs/2503.03321) | ICLR 2025 |
| MCA-LLaVA: Manhattan Causal Attention | [2507.09184](https://arxiv.org/abs/2507.09184) | 2025 |
| VisPruner / Beyond Text-Visual Attention | [2412.01818](https://arxiv.org/abs/2412.01818) | ICCV 2025 |
| Attention Debiasing for Token Pruning | [2508.17807](https://arxiv.org/abs/2508.17807) | 2025 |
| Attention Hijackers | [2503.08216](https://arxiv.org/abs/2503.08216) | 2025 |
| Modality Bias in LVLMs | [2508.02419](https://arxiv.org/abs/2508.02419) | 2025 |
| Same Attention, Different Truths | [2608.07302](https://arxiv.org/abs/2608.07302) | 2026 |
| MLLMs Know Where to Look | [2502.17422](https://arxiv.org/abs/2502.17422) | ICLR 2025 |
| Adaptive attention calibration | [2505.21472](https://arxiv.org/abs/2505.21472) | 2025 |
| Attention Sink in Transformers: A Survey | [2604.10098](https://arxiv.org/abs/2604.10098) | 2026 |
| To Sink or Not to Sink: Visual Information Pathways in LVLMs | [2510.08510](https://arxiv.org/abs/2510.08510) | ICLR 2026 |
| When Sinks Help or Hurt: Unified Framework for Attention Sink in LVLMs | [2604.03316](https://arxiv.org/abs/2604.03316) | 2026 |
