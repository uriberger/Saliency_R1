#!/usr/bin/env python
"""Everything the sink-location experiment knows about the model it is measuring.

`docs/sink-location-by-image-type.md` is the design and `sink_location.py` is the
measurement. Both were written against Qwen3-VL, and the question
`docs/sink-location-cross-model.md` asks is whether the result is a property of that
model or of VLMs -- which cannot be answered without running the SAME measurement on a
different family. This module is the seam: one class per family, selected off
`config.model_type`, holding the handful of facts that differ.

WHAT DIFFERS, AND WHY EACH ONE IS HERE RATHER THAN BRANCHED INLINE

    the image token, and the delimiters around it   locating the picture in the prompt
    the patch grid                                  Qwen3-VL varies it per picture;
                                                    InternVL and LLaVA-1.5 are fixed
    the decoder's attention class                   which modules the scan installs on
    the module that emits the LLM-facing rows       where the permutation hook goes
    the VIEW BOX                                    see below
    the prompt                                      each model's own chat template

THE VIEW BOX IS THE ONE THAT WILL BITE YOU. Qwen3-VL and InternVL hand the whole picture
to the encoder; LLaVA-1.5's processor resizes the short side to 336 and then CENTRE-CROPS
to 336x336, so its 24x24 grid covers a centred square of the picture and nothing else.
`view_box` is that square in normalised picture coordinates, and every piece of geometry
that relates a pixel to a patch -- the per-patch content statistics, the frame check, the
arms' `patch_correspondence` -- composes through it. Without it, `follow content` on a
rotated picture would be decoded in the wrong frame and would answer the
content-versus-position question confidently and backwards, which is exactly the failure
mode the selftest exists to catch.

    import vlm_family as VF
    fam = VF.family_for(model, processor)     # -> Qwen3VL() / InternVL() / Llava15()
    scan = SL.install(model, family=fam)
"""

from __future__ import annotations

import numpy as np

# The families, by `config.model_type`. `family_for` is the only way to get one, so a
# model this file has never heard of fails loudly at the seam rather than silently
# measuring a Qwen3-VL-shaped hole in a different model.
_REGISTRY = {}


def register(cls):
    for t in cls.model_types:
        _REGISTRY[t] = cls
    return cls


def family_for(model=None, processor=None, config=None, model_type=None):
    """The adapter for this model. -> a bound Family instance."""
    if model_type is None:
        cfg = config if config is not None else getattr(model, "config", None)
        model_type = getattr(cfg, "model_type", None)
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise SystemExit(
            f"sink_location has no adapter for model_type {model_type!r}; "
            f"known families are {sorted(_REGISTRY)}. Add one to vlm_family.py -- do not "
            "branch on the model inside the measurement.")
    return cls().bind(model=model, processor=processor, config=config)


#: The decoder attention class for each text tower a connector-style VLM can be built on.
#: Several of these architectures are SOCKETS, not models -- the same
#: `LlavaForConditionalGeneration` holds Vicuna + CLIP in LLaVA-1.5 and Qwen2 + SigLIP in
#: llava-interleave-qwen -- so the class has to be read off the text config rather than
#: assumed. Guessing installs the scan on nothing, and a model that reports empty cells
#: looks exactly like a model with no effect.
TEXT_ATTENTION = {
    "llama": "LlamaAttention", "qwen2": "Qwen2Attention", "qwen3": "Qwen3Attention",
    "mistral": "MistralAttention", "gemma": "GemmaAttention",
    "gemma2": "Gemma2Attention", "gemma3_text": "Gemma3Attention",
    "phi3": "Phi3Attention", "olmo2": "Olmo2Attention",
}


def text_attention(config, default):
    mt = getattr(getattr(config, "text_config", None), "model_type", None)
    if not mt:
        return default
    cls = TEXT_ATTENTION.get(mt)
    if cls is None:
        raise SystemExit(
            f"a {mt!r} text tower: add its attention class to vlm_family.TEXT_ATTENTION. "
            "Guessing would install the scan on nothing and report empty cells as a "
            "result.")
    return (cls,)


# ---------------------------------------------------------------------------
class Family:
    """The surface `sink_location.py` is allowed to know about a model.

    Subclasses fill in the class attributes and override only the methods whose answer
    is not the common one. Every default here is the answer that is true for a model
    which shows the encoder the whole picture on a fixed grid, which is two of the three.
    """

    name = ""
    model_types = ()
    #: the decoder's attention module class(es). The scan registers its own attention
    #: implementation and switches these over to it; the vision tower's attention has a
    #: different class name in all three families, which is what keeps it untouched.
    attn_classes = ()
    #: the module whose forward OUTPUT carries the rows the language model will consume,
    #: one row per image token, in token order. That is where A9's permutation goes.
    row_classes = ()
    #: True when this family should be run with the project's own trainer system prompt
    #: (so the published Qwen3-VL numbers reproduce), False when it gets its own.
    uses_project_prompt = False

    image_token_id = None
    vision_start_ids = ()
    vision_end_ids = ()
    #: token strings to resolve against the tokenizer at bind time, when the ids are not
    #: a fixed part of the family.
    start_tokens = ()
    end_tokens = ()

    def __init__(self):
        self.system_prompt = None
        self.config = None
        self.processor = None

    # -- binding ---------------------------------------------------------
    def bind(self, model=None, processor=None, config=None):
        self.config = config if config is not None else getattr(model, "config", None)
        self.processor = processor
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            if self.start_tokens:
                self.vision_start_ids = self._ids(tok, self.start_tokens)
            if self.end_tokens:
                self.vision_end_ids = self._ids(tok, self.end_tokens)
        if self.image_token_id is None and self.config is not None:
            got = getattr(self.config, "image_token_id", None)
            if got is None:
                got = getattr(self.config, "image_token_index", None)
            self.image_token_id = None if got is None else int(got)
        return self

    @staticmethod
    def _ids(tok, names):
        out = []
        for n in names:
            i = tok.convert_tokens_to_ids(n)
            if i is not None and i >= 0:
                out.append(int(i))
        return tuple(out)

    # -- the prompt ------------------------------------------------------
    #: Processor keyword arguments this family must pin. Empty for most; see InternVL.
    proc_defaults = {}

    def image_arg(self, images):
        """How this processor wants the pictures: flat, or nested one list per sample."""
        return list(images)

    def build_inputs(self, processor, images, question, device, **proc_kwargs):
        """The prompt, at batch size 1, with one or more pictures.

        Each model gets its OWN chat template -- putting Qwen3-VL's `<think>` system
        prompt in front of LLaVA-1.5 would measure an off-distribution model, and the
        thing being compared is where attention lands, which is a property of the model
        and not of our prompt conventions. `--system-prompt none` is what puts all three
        on the same footing when the cross-model tables are produced.
        """
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        kw = dict(self.proc_defaults)
        kw.update(proc_kwargs)
        return processor(text=[text], images=self.image_arg(images),
                         return_tensors="pt", padding=True, padding_side="left",
                         add_special_tokens=False, **kw).to(device)

    def teacher_forced_case(self, prompt_inputs, comp_ids, device):
        """prompt ++ one completion, as the single measured forward.

        The multimodal inputs are carried over verbatim and the completion is appended as
        text. Families with extra per-token multimodal bookkeeping (Qwen3-VL's
        `mm_token_type_ids`) extend it; forgetting to would make this forward differ from
        the one the training run computes its reward on.
        """
        import torch

        ids = torch.cat([prompt_inputs["input_ids"],
                         torch.tensor([comp_ids], device=device)], dim=1)
        case = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        for k in self.passthrough_inputs:
            if prompt_inputs.get(k) is not None:
                case[k] = prompt_inputs[k]
        return case

    passthrough_inputs = ("pixel_values",)

    # -- the grid --------------------------------------------------------
    #: tokens per tile as (gh, gw), for families whose grid does not depend on the
    #: picture. `grids_for` uses it and never has to ask the processor.
    fixed_grid = None
    #: one grid cell, in the ENCODER's own input pixels. A10 cuts its blocks at exactly
    #: this size so the processor's resize is the identity and the shuffle is lossless.
    encoder_px = 32

    def grids_for(self, runs, inputs):
        """-> [(t, gh, gw)] one entry per TILE, in column order.

        One tile per picture in every family except InternVL with tiling switched on,
        where a picture becomes up to 12 tiles plus a thumbnail and each tile is its own
        16x16 grid. Returning tiles rather than pictures is what lets the tiled arm be
        scored on the geometry the encoder actually saw.
        """
        gh, gw = self.fixed_grid
        out = []
        for run in runs:
            n = int(run.numel())
            if n % (gh * gw):
                raise RuntimeError(
                    f"{self.name}: an image run of {n} tokens is not a multiple of the "
                    f"{gh}x{gw} tile this family emits")
            out += [(1, gh, gw)] * (n // (gh * gw))
        return out

    def grid_of(self, processor, image):
        """The grid THIS picture will get. Used by the frame check, per transform."""
        return tuple(self.fixed_grid)

    def view_box(self, image):
        """The part of the picture the patch grid covers, normalised. (u0, v0, u1, v1).

        The whole picture, unless the processor crops.
        """
        return (0.0, 0.0, 1.0, 1.0)

    def patch_px(self, image):
        """One grid cell, in the PICTURE's own pixels. Used by the padding arms.

        Derived from the view rather than assumed, because a padded picture is only 'one
        patch of border' if the pad is the width of a patch in the frame the grid uses.
        Needs no processor: every family whose grid depends on the picture overrides it.
        """
        u0, v0, u1, v1 = self.view_box(image)
        W, H = image.size
        gh, gw = self.fixed_grid
        return max(1, int(round(min((u1 - u0) * W / max(1, gw),
                                    (v1 - v0) * H / max(1, gh)))))

    # -- the vision side -------------------------------------------------
    def row_module(self, model):
        """The module whose output holds the LLM-facing image rows."""
        for m in model.modules():
            if type(m).__name__ in self.row_classes:
                return m
        raise RuntimeError(f"{self.name}: no module of {self.row_classes} on this model")

    def permute_rows(self, out, make_perm):
        """Apply a row permutation to that module's output. -> (new out, perm).

        The default is for a module that returns a plain [batch, tokens, dim] tensor,
        which is what both projectors do. `batch` is the TILE axis, and the rows reach
        the language model in row-major order, so flattening before permuting is what
        makes a tiled picture's permutation a permutation of its tokens.
        """
        import torch  # noqa: F401

        flat = out.reshape(-1, out.shape[-1])
        perm = make_perm(int(flat.shape[0]), out.device)
        return flat[perm].reshape(out.shape), perm

    def row_norms(self, out):
        """||row|| per image token, as a numpy array in token order."""
        return out.reshape(-1, out.shape[-1]).detach().float().norm(dim=-1).cpu().numpy()

    def deepstack_norms(self, out):
        """Norms at any extra injection point this family has. -> list or None."""
        return None


# ---------------------------------------------------------------------------
@register
class Qwen3VL(Family):
    """Native resolution, a per-picture grid, and three extra injection points.

    The published result (`docs/sink-location-by-image-type.md` §16-17) is this family,
    so nothing here may change: an adapter that silently moves the baseline invalidates
    the comparison the other two families exist to make.
    """

    name = "qwen3_vl"
    model_types = ("qwen3_vl",)
    attn_classes = ("Qwen3VLTextAttention",)
    row_classes = ("Qwen3VLVisionModel", "Qwen3VLVisionTransformerPretrainedModel")
    uses_project_prompt = True
    image_token_id = 151655              # <|image_pad|>
    vision_start_ids = (151652,)
    vision_end_ids = (151653,)
    passthrough_inputs = ("pixel_values", "image_grid_thw")

    def image_arg(self, images):
        # one list per sample, which is what this processor's batching expects
        return [list(images)]

    def teacher_forced_case(self, prompt_inputs, comp_ids, device):
        import torch

        case = super().teacher_forced_case(prompt_inputs, comp_ids, device)
        if prompt_inputs.get("mm_token_type_ids") is not None:
            zeros = torch.zeros(1, len(comp_ids), dtype=torch.long, device=device)
            case["mm_token_type_ids"] = torch.cat(
                [prompt_inputs["mm_token_type_ids"], zeros], dim=1)
        return case

    def grids_for(self, runs, inputs):
        thw = inputs if not hasattr(inputs, "get") else inputs.get("image_grid_thw")
        if thw is None or len(thw) != len(runs):
            raise RuntimeError(
                f"{len(runs)} image token runs but "
                f"{0 if thw is None else len(thw)} grids: refusing to guess which "
                "picture is which")
        out = []
        for run, g in zip(runs, thw):
            t, h, w = (int(x) for x in g)
            gh, gw = h // 2, w // 2          # this processor merges 2x2 patches per token
            if int(run.numel()) != t * gh * gw:
                raise RuntimeError(
                    f"image run of {int(run.numel())} tokens against a {t}x{gh}x{gw} "
                    "grid: the patch merge assumption is wrong for this model")
            out.append((t, gh, gw))
        return out

    def grid_of(self, processor, image):
        got = processor(text=["x"], images=[[image]], return_tensors="pt",
                        add_special_tokens=False)
        t, h, w = (int(x) for x in got["image_grid_thw"][0])
        return (h // 2, w // 2)

    def patch_px(self, image):
        # 32 exactly -- this processor's own token is 2x2 patches of 16px, and the
        # picture's size is rounded to a multiple of it, so deriving it from the
        # picture's width would round to 31 on some sizes and change the padding arms
        # away from the published run.
        return 32

    def permute_rows(self, out, make_perm):
        pool = out.pooler_output
        perm = make_perm(int(pool.shape[0]), pool.device)
        out.pooler_output = pool[perm]
        feats = getattr(out, "deepstack_features", None)
        if feats:
            # The three DeepStack features are injected into the LLM's early layers with
            # the same row indexing, so the permutation is only honest if all three move
            # with it. This is the piece the other two families do not have.
            out.deepstack_features = [f[perm] for f in feats]
        return out, perm

    def row_norms(self, out):
        pool = getattr(out, "pooler_output", None)
        if pool is None:
            return None
        return pool.detach().float().norm(dim=-1).cpu().numpy()

    def deepstack_norms(self, out):
        feats = getattr(out, "deepstack_features", None)
        if not feats:
            return None
        return [f.detach().float().norm(dim=-1).cpu().numpy() for f in feats]


# ---------------------------------------------------------------------------
@register
class InternVL(Family):
    """A Qwen3 text tower with someone else's eyes -- the best-controlled comparison.

    InternVL3.5-8B's language model is Qwen3, 36 layers and 32 heads, the same shape as
    Qwen3-VL-8B's. So this family holds the LLM nearly fixed and swaps the vision encoder
    and the connector: if the ring survives, it is not about the language model; if it
    dies, the encoder is implicated directly.

    TILING, AND THE TRAP INSIDE IT. `max_dynamic_patch: 12` means a picture can become
    twelve 448px tiles plus a thumbnail, each its own 16x16 grid, and then "the outer
    ring" is ambiguous -- the ring of a tile is an interior edge of the picture. The
    primary comparison therefore runs with tiling OFF, one 16x16 grid whose geometry
    matches Qwen3-VL's, and the tiled configuration is a separate, clearly labelled arm.

    `crop_to_patches` HAS TO BE PASSED. `image_processor.crop_to_patches` is False on
    this repo, and the processor tiles anyway -- `InternVLProcessor.__call__` carries its
    own default and overrides the attribute. Reading the attribute and believing it gives
    seven tiles where the analysis assumes one, and every column statistic then describes
    the top-left corner of the picture while claiming to describe the picture. That is
    what `proc_defaults` pins, and the selftest's frame check is what would have caught
    it if it had not been caught here.
    """

    name = "internvl"
    model_types = ("internvl",)
    attn_classes = ("Qwen3Attention",)             # the text tower IS Qwen3
    row_classes = ("InternVLMultiModalProjector",)
    start_tokens = ("<img>",)
    end_tokens = ("</img>",)
    fixed_grid = (16, 16)                          # 448/14 = 32, pixel-shuffled by 0.5
    proc_defaults = {"crop_to_patches": False}

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        if cfg is not None:
            # 256 tokens per tile is `image_seq_length`; derive rather than assume, so a
            # different downsample_ratio fails here instead of mislabelling the geometry.
            n = int(getattr(cfg, "image_seq_length", 256))
            side = int(round(n ** 0.5))
            if side * side != n:
                raise SystemExit(f"internvl: {n} tokens per tile is not a square grid")
            self.fixed_grid = (side, side)
            vc = getattr(cfg, "vision_config", None)
            px = getattr(vc, "image_size", 448) if vc is not None else 448
            self.encoder_px = int((px[0] if isinstance(px, (list, tuple)) else px) / side)
        return self


# ---------------------------------------------------------------------------
@register
class Llava15(Family):
    """A frozen CLIP encoder at its native resolution -- the best falsification target.

    LLaVA-1.5 is the clean case geometrically (fixed 336px, one 24x24 grid, no tiling)
    and the hard case for the claim: the literature reports a BOTTOM-of-image bias for
    this family (MCA-LLaVA, VisPruner), its CLIP tower is frozen at the resolution it was
    trained at so it barely interpolates position embeddings, and its connector is a
    two-layer MLP. If the ring is going to fail anywhere, it is here.

    THE CENTRE CROP. The stock `llava-hf` processor resizes the short side to 336 and
    centre-crops to 336x336 -- which is how LLaVA-1.5 is normally run, and is therefore
    what this uses. The consequence is that the 24x24 grid covers a centred square of the
    picture, not the picture, and `view_box` is that square. Every pixel-to-patch
    statement in the experiment composes through it.
    """

    name = "llava"
    model_types = ("llava",)
    attn_classes = ("LlamaAttention",)
    row_classes = ("LlavaMultiModalProjector",)
    # No delimiter tokens at all: `<image>` expands in place, with nothing around it.
    # `span_index` reports an empty vision_start/vision_end span, which is a fact about
    # this model rather than a gap in the measurement.
    fixed_grid = (24, 24)                          # 336/14

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        vc = getattr(self.config, "vision_config", None)
        if vc is not None:
            px = int(getattr(vc, "patch_size", 14))
            side = int(getattr(vc, "image_size", 336)) // px
            self.fixed_grid, self.encoder_px = (side, side), px
        self.attn_classes = text_attention(self.config, self.attn_classes)
        return self

    # -- the centre crop, in closed form ---------------------------------
    def view_box(self, image):
        """The part of the picture this processor's grid covers.

        TWO CASES, and reading the wrong one puts every patch statistic in the wrong
        frame. `size` with a `shortest_edge` plus `do_center_crop` -- LLaVA-1.5's
        CLIPImageProcessor -- resizes the short side and CENTRE-CROPS, so the grid covers
        a centred square. `size` with an explicit height and width -- SigLIP's processor,
        which llava-interleave-qwen uses -- resizes the whole picture to that square, so
        the grid covers all of it.

        The crop case reproduces `CLIPImageProcessor`'s arithmetic exactly rather than
        approximating it: the short side goes to `shortest_edge`, the long side is
        TRUNCATED, then `center_crop` takes `(size - crop) // 2` off the top and the
        left. A half-pixel of slop here is harmless; getting the direction wrong is not.
        """
        ip = getattr(self.processor, "image_processor", None)
        size, crop = getattr(ip, "size", None), getattr(ip, "crop_size", None)
        short = _size_get(size, "shortest_edge")
        if short is None or not getattr(ip, "do_center_crop", False):
            return (0.0, 0.0, 1.0, 1.0)          # resized to a square: the whole picture
        short = int(short)
        ch = int(_size_get(crop, "height") or short)
        cw = int(_size_get(crop, "width") or short)
        W, H = image.size
        if W <= H:
            nw, nh = short, int(short * H / W)
        else:
            nh, nw = short, int(short * W / H)
        left, top = (nw - cw) // 2, (nh - ch) // 2
        return (left / nw, top / nh, (left + cw) / nw, (top + ch) / nh)


# ---------------------------------------------------------------------------
@register
class Idefics3(Family):
    """A Llama-3 text tower on a SigLIP encoder -- the cell the other four do not fill.

    Qwen3-VL, InternVL3.5 and llava-interleave-qwen all run a Qwen language model, and
    LLaVA-1.5 is the only non-Qwen one, which leaves "the language model's queries favour
    border keys" confounded with everything that differs between those checkpoints. This
    is a second non-Qwen decoder -- Llama-3-8B, 32 layers by 32 heads -- on a SigLIP tower
    that WAS trained inside the VLM, which is the combination none of the others has.

    SPLITTING, like InternVL's tiling: `do_image_splitting` is True by default and cuts a
    picture into sub-images plus a global view. Pinned off, so one 13x13 grid covers the
    whole picture. The processor then resizes to 364x364 with no padding -- checked, the
    pixel attention mask comes back fully valid -- so the view box is the whole picture
    and the border of the grid really is the border of the image.
    """

    name = "idefics3"
    model_types = ("idefics3", "smolvlm")
    attn_classes = ("LlamaAttention",)
    row_classes = ("Idefics3Connector", "SmolVLMConnector")
    # One token is used on BOTH sides of the picture, so it is reported entirely in
    # `vision_start` rather than counted twice by naming it as the closer as well.
    start_tokens = ("<fake_token_around_image>",)
    fixed_grid = (13, 13)
    encoder_px = 28
    proc_defaults = {"do_image_splitting": False}

    def image_arg(self, images):
        return [list(images)]

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        vc = getattr(cfg, "vision_config", None)
        if vc is not None:
            px = int(getattr(vc, "patch_size", 14))
            # pixel shuffle by `scale_factor` in the connector, exactly as InternVL's
            # 2x2 shuffle does: 26x26 patches of 14px become 13x13 tokens of 28px.
            sf = int(getattr(cfg, "scale_factor", 2))
            side = (int(getattr(vc, "image_size", 364)) // px) // sf
            self.fixed_grid, self.encoder_px = (side, side), px * sf
        self.attn_classes = text_attention(cfg, self.attn_classes)
        return self


# ---------------------------------------------------------------------------
# geometry helpers that need the view box
# ---------------------------------------------------------------------------
def _size_get(size, key):
    """One field of a processor's size spec, whether it is a dict or a `SizeDict`.

    transformers hands these back in both shapes depending on the processor and the
    version, and `"shortest_edge" in size` raises on one of them. A view box read off the
    wrong branch silently frames every patch statistic on the wrong region.
    """
    if size is None:
        return None
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def view_crop(image, view):
    """The part of the picture the grid covers, as a picture of its own.

    The per-patch content statistics and the frame check's marker both have to be read on
    the region the model saw. Handing them the whole picture where the model centre-cropped
    it would line every covariate up against the wrong patch.
    """
    u0, v0, u1, v1 = view
    if (u0, v0, u1, v1) == (0.0, 0.0, 1.0, 1.0):
        return image
    W, H = image.size
    box = (int(round(u0 * W)), int(round(v0 * H)),
           max(int(round(u0 * W)) + 1, int(round(u1 * W))),
           max(int(round(v0 * H)) + 1, int(round(v1 * H))))
    return image.crop(box)


def block_permute(image, grid, block_px, perm=None, seed=0, mode="shuffle"):
    """A10 -- shuffle the PIXEL BLOCKS that will become grid cells, before the encoder.

    A9 shuffles the encoder's OUTPUT rows and shows the attractor travels with the token,
    which rules out the language model's positional slot. It does not rule out the
    ENCODER's own position embeddings, which could have stamped the token on the way
    through. This does: the pixels of grid cell j are moved to cell `perm[j]` before the
    vision tower runs, so if the attractor appears at whatever content now occupies the
    top-left OF THE VIT GRID, the encoder's position embedding is writing it, and if it
    follows the original content instead, it is content-driven.

    The picture is first resized to the ENCODER'S OWN input size -- `block_px` pixels per
    grid cell -- so the blocks are integer, the shuffle is lossless, and the processor's
    own resize is the identity rather than a second resampling that would blur every
    block boundary. That resize is still not free on the source side, so
    `mode="identity"` runs it with the identity permutation and is the baseline this arm
    is paired against.

    -> (image, perm) where slot j of the new picture holds the block that was at perm[j].
    """
    from PIL import Image

    gh, gw = int(grid[0]), int(grid[1])
    n = gh * gw
    bw = bh = max(1, int(block_px))
    base = image.resize((bw * gw, bh * gh), Image.BICUBIC)
    if perm is None:
        if mode == "identity":
            perm = np.arange(n)
        elif mode == "shuffle":
            perm = np.random.default_rng(int(seed)).permutation(n)
        else:
            raise ValueError(f"unknown block permutation mode {mode!r}")
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f"block_permute needs a permutation of {n} blocks")
    if mode == "identity":
        return base, perm
    out = Image.new("RGB", base.size)
    for j in range(n):
        src = int(perm[j])
        sr, sc = divmod(src, gw)
        dr, dc = divmod(j, gw)
        out.paste(base.crop((sc * bw, sr * bh, (sc + 1) * bw, (sr + 1) * bh)),
                  (dc * bw, dr * bh))
    return out, perm
