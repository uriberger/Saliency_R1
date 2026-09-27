# Porting the sink-location scan to `NVIDIA-Nemotron-Nano-12B-v2-VL`

**Status: blocked at the last step, deliberately.** Written 2026-09-27, branch
`probe/nemotron-12b-vl`.

## Why it was attempted

Picking a second backbone to train the overlap reward on. The Omni
(`Nemotron-3-Nano-Omni-30B-A3B`) is the strategic target — it is what NVIDIA ships — but
it is 33B and its 66 GB of weights do not leave room for a colocated vLLM server on one
80 GB card. `NVIDIA-Nemotron-Nano-12B-v2-VL` looked like the cheap stepping stone: same
Nemotron-H hybrid, same C-RADIOv2-H tower, 13.2B and dense.

Two things had to be true first. **Does it tile?** and **does it have the corner peak the
reward's `/max` normalisation leans on?**

## Question 1: it tiles, hard. Answered.

The Omni is native-resolution — it buckets a picture to an aspect-matched size, emits one
grid and reports it in `imgs_sizes`. This checkpoint is InternVL:
`image_processing.dynamic_preprocess` picks the closest aspect ratio up to
`max_num_tiles: 12`, cuts the resize into 512x512 tiles and appends a whole-picture
thumbnail whenever there is more than one tile. Measured on 12 pictures of the boxed
corpus, both processors on the same images:

```
picture                   orig WxH | 12B tiles +thumb 12B tokens |  Omni grid Omni tokens
gqa-00001.png              512x384 |        12    yes       3328 |      14x19         266
openimages-00000.png       512x306 |         6    yes       1792 |      13x21         273
vsr-00000.png              512x512 |         1     no        256 |      16x16         256
infographicsvqa-00003.png  220x512 |        10    yes       2816 |      25x11         275
docvqa-00003.png           386x512 |        12    yes       3328 |      19x15         285

mean visual tokens   12B  2517     Omni  269
```

`imgs_sizes` does not exist on this processor at all (it emits `pixel_values` and
`num_patches`), so `NemotronVL.grids_for` raises on the first picture.

Two consequences. **The ring is ambiguous under tiling** — a tile's outer ring is an
interior edge of the picture, and the thumbnail covers the picture a second time at a
different scale, so a peak can be counted twice in two places. `InternVL` already hit
this and the project's answer was to run the primary comparison with tiling OFF;
`NemotronVLV2` does the same, forcing one tile and one 16x16 grid over the whole picture.
**And the Omni is nine times cheaper per image than the 12B**, which inverts the cost
argument that motivated the stepping stone. (Caveat: the corpus is capped at 512px. The
Omni is native-resolution, `max_num_patches: 13312`, so on larger pictures it climbs to
the same 3,328-token ceiling. The comparison above is this corpus, not the models' limits.)

## Question 2: blocked on the measurement seam, not on the science

The scan never fired. It locates the picture correctly and then accumulates nothing:

```
locate_image_runs : runs [256] grids [(1, 16, 16)]
AFTER THE FORWARD
  scan.n_forwards : 0
  _acc['text'] layers: []
  result()        : None
```

`sink_location.SinkScan.install` works by registering an implementation in
`transformers`' `ALL_ATTENTION_FUNCTIONS` and flipping `_attn_implementation`. **This
checkpoint has no such dispatch point.** Its decoder is the pre-5.x transformers pattern:

```python
# NVIDIA-Nemotron-Nano-12B-v2-Base/modeling_nemotron_h.py:754
self.mixer = NEMOTRONH_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
```

— one class per implementation (`NemotronHAttention`, `NemotronHFlashAttention2`,
`NemotronHSdpaAttention`), **chosen at construction**, and `NemotronHAttention.forward`
calls `torch.nn.functional.scaled_dot_product_attention` directly. `ALL_ATTENTION_FUNCTIONS`
appears 0 times in the file. Flipping `_attn_implementation` after loading cannot reach it.

The Omni's copy of the same file is the modern one:

```python
# Nemotron-3-Nano-Omni.../modeling_nemotron_h.py:963
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward)
```

which is why the Omni row of `sink-location-cross-model.md` exists at all.

### The seam that would work, if this is resumed

`NEMOTRONH_ATTENTION_CLASSES` is the model's own extension point. Register a subclass of
`NemotronHAttention` under a new key, set `llm_config._attn_implementation` to that key
**before** `from_pretrained`, and the scan is chosen at construction. Cleaner than
patching `forward`, and it does not re-derive the attention maths.

The cost is that it inverts the probe's lifecycle: `SinkScan.install()` is called on a
model that already exists, and this would have to be decided at load time. Reckon on the
adapter plus a selftest that holds it against stock sdpa, not a one-liner.

## What the 12B cost to get that far, and what that says

Six defects, in order. **Every one of them is already fixed in the Omni**, which is the
finding that matters more than any of them individually — the 12B is the older and
rougher release, not the simpler one.

| # | what | the Omni |
|---|---|---|
| 1 | `modeling_nemotron_h.py` raises at import without `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`, and every Mamba layer's `MambaRMSNormGated.forward` is a call to it | lazy-loads the kernels, falls back to torch |
| 2 | never sets `all_tied_weights_keys`, so `from_pretrained` dies in `mark_tied_weights_as_initialized` after all 25 GB have loaded | sets `self.all_tied_weights_keys = {}` |
| 3 | `extract_feature` does not cast `pixel_values`, so a float32 processor output meets a bfloat16 RADIO | casts to `vision_model.config.torch_dtype` |
| 4 | `forward` returns `past_key_values=outputs.past_key_values` on an output whose field is `cache_params` — **`forward()` is broken as shipped**; only `generate()` works | correct |
| 5 | `prepare_inputs_for_generation` indexes a `cache_position` transformers 5.13 no longer passes | — |
| 6 | attention does not route through `ALL_ATTENTION_FUNCTIONS` | it does |

1–5 are fixed on this branch and the checkpoint now loads, runs a forward and generates.
6 is the one that needs a design decision.

## What is on the branch

- `vlm_family.NemotronVLV2` — the family: one tile, a 16x16 grid, the dtype cast.
- `sink_location_probe` — `_shim_tied_weights_keys`, `_shim_cache_params_alias`,
  `_shim_cache_position`, `_vendor_mamba_rmsnorm`, `_shim_single_process_group`.
- `vendor/mamba_ssm_min/` — one upstream Triton file, Apache-2.0, verbatim from
  state-spaces/mamba v2.2.5, with `sink_selftest_mamba_rmsnorm.py` holding it down.
  Vendored rather than installed because the conda envs are shared.
- `debug_12b_scan.py` — the diagnostic that produced the `n_forwards: 0` result above.
- `outputs/sink_location/xmodel/box_nemotron12b/` — out-dir, corpus symlinked to the
  shared 1,800-picture boxed corpus.

The selftest gets as far as the attention checks and passes them
(`|scan - eager| = 0.00e+00`, same next token, agreement to token 304 of 304) — those
compare stock paths to each other and do not depend on the scan firing. It then fails at
the geometry block, which is where `measure()` returns None.

One thing to look at if this is resumed: under `--system-prompt none` the greedy decode
came out degenerate — `' (the answer to the question is a function of the function of the
function of'`. That may be the neutral prompt rather than the port, but a model writing
that would make the `gen` query set worthless, and it should be resolved before any
number off this checkpoint is quoted.
