#!/usr/bin/env python
"""Move attention off the image border and into the middle, WHILE the model answers.

The training result this exists to test: `--overlap_rect_frac` rewards attention mass
inside a fixed centred rectangle and moves the benchmark. If that is really "look at the
middle, not at the edge", then doing it directly at inference should work too, with no
weights changed and no training at all. This module is the doing-it-directly half;
`sink_shift_probe.py` is the harness that runs and measures it.

---------------------------------------------------------------------------
The grid, and the two sets of patches
---------------------------------------------------------------------------
Qwen3-VL turns a picture into a grid of patches, one token each -- typically 10 rows by
16 columns = 160 tokens. Every name below is a set of patches on that grid:

    frame   the one-patch border.       48 of 160 patches (0.300 of the grid)
    rect    the centred rectangle the reward scores, at frac 0.565.
                                        96 of 160 (0.600), and it touches no frame patch
    ring2   the one-patch ring immediately INSIDE the frame.
                                        40 of 160 -- off the frame, but as far from the
                                        middle as the grid allows
    core    everything at least two patches from the border.
                                        72 of 160 -- a stricter middle than `rect`
    all     every patch

`frame` is not an arbitrary choice. It is 30% of the patches and it carries about HALF
of all the attention the row spends on the image, at 2.6-3.1x the interior's per-patch
density, with 76-85% of map peaks on it. That is the sink this module drains.

Note that `rect` and `ring2` overlap (24 patches): the rectangle's own outermost row and
column are inside the second ring. They are still the two most different destinations
this grid can offer, because the rectangle is 60% of the grid and nothing the same size
can be moved anywhere else -- see docs/per-completion-masks.md, which measured that.

---------------------------------------------------------------------------
The edit
---------------------------------------------------------------------------
One attention row is one query position, one layer, one head: a list of weights over
every earlier token, summing to 1. Slice out just the weights on image patches and call
that `w`. Every arm is the same two-line operation on `w`, differing only in which
patches are the source and which are the destination:

    take     alpha * (the weight sitting on `src`)
    give     that exact amount to `dst`, split in proportion to `t`

    w'_src = w_src - alpha * w_src
    w'_dst = w_dst + alpha * (sum of w over src) * t_dst          t sums to 1 over dst

The row still sums to 1, the image/text split does not move, and no weight on a text
token, a BOS token or any other sink outside the picture is touched. alpha=0 is exactly
the identity. `src` and `dst` may overlap; the take happens before the give.

`t` has two settings. `shape` splits the gift in proportion to what each destination
patch already had, so the edit removes the sink without inventing somewhere to look --
this is the default. `uniform` spreads it evenly, which is the literal rectangle mask.

ARMS
    centre    frame -> rect            the treatment: drain the sink into the middle
    core      frame -> core            the same, aimed at a stricter middle
    outward   frame -> ring2           same source, same mass, still off the frame, but
                                       aimed AWAY from the middle. The wrong-place
                                       control: `centre` minus `outward` is the result
    flat      all   -> all, uniform    no middle at all, just evenness. The box-blind
                                       arm, because --overlap_rect_frac correlates 0.932
                                       with `flatness` and the two have never been
                                       separated
    reverse   rect  -> frame           the sign check. If `centre` helps, this must hurt
    text      the same MASS moved among the text tokens instead, leaving the picture
                                       untouched. Answers "is any nudge of this size
                                       enough", not "does the middle matter"

HOW MUCH IS ACTUALLY BEING MOVED. At layer 22 heads 28/31 -- the heads the reward
trained -- the whole picture receives 0.4% to 1.4% of an attention row, and the frame
holds about half of that. So at alpha=1 those two heads move ~0.2-0.7% of one row, in 2
of the model's 1,152 heads. A null there is expected and is NOT evidence about the idea;
it is evidence that there was nothing to move. `--stage survey` measures the same
quantity for every layer and head, which is how the broad arm's leverage is known before
it is run.

---------------------------------------------------------------------------
Where it plugs in
---------------------------------------------------------------------------
Not a forward hook on the attention module. The existing probes (`intervene_probe.py`,
`flow_intervene_probe.py`) re-run the module in eager mode to recover the softmax
weights, which is correct for one teacher-forced pass over an answer that already exists
and WRONG while the model is writing: a decode step re-run with `past_key_values=None`
attends to itself alone.

Instead this registers an attention implementation. Transformers resolves
`config._attn_implementation` through `ALL_ATTENTION_FUNCTIONS` inside
`Qwen3VLTextAttention.forward`, AFTER rotary embedding and AFTER the KV cache update, so
the registered function sees the real keys and values including everything cached. The
same code path serves prefill and decode.

Only the text decoder is switched. The vision tower keeps its own implementation,
because the text config is a different object from the vision config.

Layers that are not being edited fall through to the stock fast kernel, so the cost of a
narrow arm is close to zero. On an edited layer the fused kernel is replaced by an
explicit softmax: no extra cost while writing (the query is one token), and one
[heads, q, kv] tensor per layer while reading the prompt.

    import sink_shift
    ss = sink_shift.install(model, arm="centre", alpha=0.5, layers=[22], heads=[28, 31])
    ...                                  # generate as usual
    ss.diagnostics()                     # did the edit land?
    ss.uninstall()
"""

from __future__ import annotations

import math

import torch

IMAGE_TOKEN_ID = 151655          # Qwen3-VL's <|image_pad|>, as overlap_probe.py uses
IMPL_NAME = "sink_shift"
RECT_FRAC = 0.565                # the fraction --overlap_rect_frac is launched at

ROWS_CHOICES = ("after_image", "generated", "all")
TARGETS = ("shape", "uniform")

# arm -> (source set, destination set, how the gift is split, matched-mass control?)
ARMS = {
    "centre":  ("frame", "rect",  "shape",   False),
    "core":    ("frame", "core",  "shape",   False),
    "outward": ("frame", "ring2", "shape",   False),
    "flat":    ("all",   "all",   "uniform", False),
    "reverse": ("rect",  "frame", "shape",   False),
    "text":    ("frame", "rect",  "shape",   True),
}


# ---------------------------------------------------------------------------
# geometry -- flat patch indices on a gh x gw grid, pure and CPU-testable
# ---------------------------------------------------------------------------
def _grid_bool(gh, gw):
    return torch.zeros(gh, gw, dtype=torch.bool)


def frame_set(gh, gw):
    """The one-patch border."""
    m = _grid_bool(gh, gw)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


def rect_set(gh, gw, frac=RECT_FRAC):
    """The centred rectangle, identical to `overlap_rewards._centre_rect_mask`.

    Reimplemented here in torch rather than imported, because this module must run
    inside an attention kernel with no numpy round-trip -- and `test_sink_shift_cpu.py`
    asserts the two agree patch for patch on every grid it can reach, so they cannot
    drift apart silently.
    """
    s = math.sqrt(min(1.0, max(0.0, float(frac))))
    rows = min(gh, max(1, int(round(gh * s))))
    cols = min(gw, max(1, int(round(gw * s))))
    m = _grid_bool(gh, gw)
    r0, c0 = (gh - rows) // 2, (gw - cols) // 2
    m[r0:r0 + rows, c0:c0 + cols] = True
    return m


def ring2_set(gh, gw):
    """The one-patch ring immediately inside the frame."""
    m = _grid_bool(gh, gw)
    if gh < 3 or gw < 3:
        return m
    m[1, 1:gw - 1] = m[gh - 2, 1:gw - 1] = True
    m[1:gh - 1, 1] = m[1:gh - 1, gw - 2] = True
    return m


def core_set(gh, gw):
    """Every patch at least two away from the border."""
    m = _grid_bool(gh, gw)
    if gh < 5 or gw < 5:
        return m
    m[2:gh - 2, 2:gw - 2] = True
    return m


def all_set(gh, gw):
    return torch.ones(gh, gw, dtype=torch.bool)


SETS = {"frame": frame_set, "rect": rect_set, "ring2": ring2_set,
        "core": core_set, "all": all_set}


def patch_set(name, gh, gw, frac=RECT_FRAC):
    """A boolean [gh, gw] grid by name. `rect` is the only one that takes `frac`."""
    if name not in SETS:
        raise ValueError(f"unknown patch set {name!r}; have {sorted(SETS)}")
    return rect_set(gh, gw, frac) if name == "rect" else SETS[name](gh, gw)


# ---------------------------------------------------------------------------
# the edit -- pure tensor algebra, no model
# ---------------------------------------------------------------------------
def build_target(w, dst, allowed=None):
    """The split of the gift over `dst`, one distribution per row. -> same shape as w.

    In proportion to what each destination already had. A row whose destinations are all
    at zero has no shape to preserve, so it falls back to an even split -- otherwise the
    gift would have nowhere to go and the mass would silently vanish, breaking the one
    invariant this module has.

    `allowed` marks the columns a row is permitted to attend to. It only ever matters for
    the `text` arm, where a destination can sit beyond the causal horizon; putting mass
    there would place weight on a token the model has not written yet.
    """
    sel = dst.expand_as(w) if dst.dim() == 1 else dst
    if allowed is not None:
        sel = sel & allowed
    t = torch.where(sel, w, torch.zeros_like(w))
    s = t.sum(-1, keepdim=True)
    n = sel.sum(-1, keepdim=True).clamp_min(1)
    even = sel.to(w.dtype) / n.to(w.dtype)
    return torch.where(s > 0, t / s.clamp_min(1e-30), even)


def move_fraction(w, src, dst, alpha, target="shape", allowed=None):
    """Take `alpha` of the weight on `src`; give it to `dst`.

    w      [..., N] float32, one row per (head, query). Rows need not sum to 1 -- this
                    operates on a SLICE of the row, and conserves that slice's total.
    src    [N] bool the patches that lose weight
    dst    [N] bool the patches that gain it
    alpha  float    0 = identity, 1 = the source is emptied

    Returns (w', moved) with `moved` [..., 1] the mass each row actually shifted.
    The take is applied before the give, so overlapping src and dst behave sensibly:
    src == dst == all with target="uniform" is exactly a blend toward flat.
    """
    if alpha == 0.0:
        return w, torch.zeros_like(w[..., :1])
    moved = alpha * torch.where(src.expand_as(w), w, torch.zeros_like(w)).sum(-1, keepdim=True)
    t = (build_target(w, dst, allowed) if target == "shape"
         else _even_target(w, dst, allowed))
    out = w - alpha * torch.where(src.expand_as(w), w, torch.zeros_like(w)) + moved * t
    return out, moved


def move_mass(w, src, dst, mass, target="shape", allowed=None):
    """Move an ABSOLUTE amount rather than a fraction, drained in proportion to `src`.

    This is what the `text` arm needs: it must shift exactly as much weight as `centre`
    would have shifted, so that the two differ in where the weight went and in nothing
    else. `mass` is [..., 1]. A row holding less than `mass` on its sources gives all it
    has and no more, so no weight can ever go negative.
    """
    s_w = torch.where(src.expand_as(w), w, torch.zeros_like(w))
    have = s_w.sum(-1, keepdim=True)
    take = torch.minimum(mass, have)
    frac = take / have.clamp_min(1e-30)
    t = (build_target(w, dst, allowed) if target == "shape"
         else _even_target(w, dst, allowed))
    return w - frac * s_w + take * t, take


def _even_target(w, dst, allowed=None):
    sel = dst.expand_as(w) if dst.dim() == 1 else dst
    if allowed is not None:
        sel = sel & allowed
    n = sel.sum(-1, keepdim=True).clamp_min(1)
    return sel.to(w.dtype) / n.to(w.dtype)


# ---------------------------------------------------------------------------
# the state the attention implementation reads
# ---------------------------------------------------------------------------
class SinkShift:
    """Configuration, the current prompt's picture layout, and the landing check.

    One instance is installed on one model. It owns the pre-hook that finds the image
    tokens, the swapped attention implementation, and the counters that say whether the
    edit did anything -- which is not optional here. An arm whose frame mass did not fall
    did not run, and a null from it means nothing.
    """

    def __init__(self, model, arm="centre", alpha=0.5, layers=None, heads=None,
                 rows="after_image", rect_frac=RECT_FRAC, target=None):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; have {sorted(ARMS)}")
        if rows not in ROWS_CHOICES:
            raise ValueError(f"unknown rows {rows!r}; have {ROWS_CHOICES}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.model = model
        self.arm, self.alpha, self.rows, self.rect_frac = arm, float(alpha), rows, rect_frac
        src, dst, tgt, self.mass_matched = ARMS[arm]
        self.src_name, self.dst_name = src, dst
        self.target = target or tgt
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; have {TARGETS}")
        self.layers = None if layers is None else set(int(x) for x in layers)
        self.heads = None if heads is None else sorted(int(x) for x in heads)
        # per-prompt, set by the pre-hook
        self.img_cols = None          # [n_img] long, absolute key positions
        self.src_cols = None          # [n_img] bool over img_cols
        self.dst_cols = None
        self.prompt_len = 0
        self.first_row = 0            # no query position below this is ever edited
        self.grids = []
        self._sets = {}               # name -> bool over img_cols, for the diagnostics
        self._handles = []
        self._prev_impl = None
        self._text_cfg = None
        self.reset_diagnostics()

    # -- diagnostics ------------------------------------------------------
    def reset_diagnostics(self):
        self._d = {"rows_edited": 0, "forwards": 0, "layers_touched": set(),
                   "frame_before": 0.0, "frame_after": 0.0, "rect_before": 0.0,
                   "rect_after": 0.0, "image_mass": 0.0, "moved": 0.0, "n": 0}

    def diagnostics(self):
        """What the edit actually did, averaged over every row it touched.

        `frame_share_after` is the contract: for `centre` at alpha it must be about
        (1 - alpha) times `frame_share_before`. If it is not, the arm did not run.
        """
        n = max(1, self._d["n"])
        return {
            "arm": self.arm, "alpha": self.alpha,
            "rows_edited": self._d["rows_edited"],
            "forwards": self._d["forwards"],
            "layers_touched": sorted(self._d["layers_touched"]),
            "image_mass": self._d["image_mass"] / n,
            "frame_share_before": self._d["frame_before"] / n,
            "frame_share_after": self._d["frame_after"] / n,
            "rect_share_before": self._d["rect_before"] / n,
            "rect_share_after": self._d["rect_after"] / n,
            "row_mass_moved": self._d["moved"] / n,
        }

    # -- per-prompt layout ------------------------------------------------
    def _locate_images(self, input_ids, image_grid_thw):
        """Absolute key positions of every image patch, and the source/destination sets.

        Runs of `IMAGE_TOKEN_ID` are matched to rows of `image_grid_thw` in order. Each
        run gets its OWN grid, so a prompt with two pictures of different shapes gets two
        correctly-shaped borders rather than one averaged wrong one.
        """
        ids = input_ids[0]
        is_img = ids == IMAGE_TOKEN_ID
        if not bool(is_img.any()):
            return False
        pos = torch.nonzero(is_img, as_tuple=True)[0]
        brk = torch.nonzero(pos[1:] - pos[:-1] != 1, as_tuple=True)[0]
        starts = [0] + (brk + 1).tolist()
        ends = (brk + 1).tolist() + [pos.numel()]
        runs = [pos[a:b] for a, b in zip(starts, ends)]

        if image_grid_thw is None or len(image_grid_thw) != len(runs):
            raise RuntimeError(
                f"{len(runs)} image token runs but "
                f"{0 if image_grid_thw is None else len(image_grid_thw)} grids: refusing "
                "to guess which picture is which")

        cols, src, dst, grids = [], [], [], []
        for run, thw in zip(runs, image_grid_thw):
            t, h, w = (int(x) for x in thw)
            gh, gw = h // 2, w // 2            # Qwen3-VL merges 2x2 patches into a token
            if run.numel() != t * gh * gw:
                raise RuntimeError(
                    f"image run of {run.numel()} tokens against a {t}x{gh}x{gw} grid: "
                    "the patch merge assumption is wrong for this model")
            s = patch_set(self.src_name, gh, gw, self.rect_frac).reshape(-1)
            d = patch_set(self.dst_name, gh, gw, self.rect_frac).reshape(-1)
            cols.append(run)
            src.append(s.repeat(t))
            dst.append(d.repeat(t))
            grids.append((t, gh, gw))

        dev = input_ids.device
        self.img_cols = torch.cat(cols).to(dev)
        self.src_cols = torch.cat(src).to(dev)
        self.dst_cols = torch.cat(dst).to(dev)
        self.grids = grids
        # The diagnostic sets are cached per prompt. Two prompts can have the same patch
        # COUNT and different grid shapes, so the cache is keyed on the shapes and not on
        # the count, and is dropped outright here rather than being checked for staleness.
        self._sets = {}
        self.prompt_len = int(input_ids.shape[1])
        self.first_row = (int(self.img_cols.max()) + 1 if self.rows == "after_image"
                          else self.prompt_len if self.rows == "generated" else 0)
        return True

    def _pre_hook(self, module, args, kwargs):
        """Find the picture at the start of each prompt; leave it alone while writing.

        A forward carrying image tokens is a fresh prompt. A forward without them is
        either a decode step of the prompt already located, or -- if nothing has been
        located yet -- a text-only request, which this module simply does not edit.
        """
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None or ids.dim() != 2:
            return None
        if ids.shape[0] != 1:
            raise RuntimeError(
                f"batch of {ids.shape[0]}: sink_shift locates the picture per prompt and "
                "is only correct at batch size 1")
        if bool((ids == IMAGE_TOKEN_ID).any()):
            self._locate_images(ids, kwargs.get("image_grid_thw"))
        return None

    # -- row selection ----------------------------------------------------
    def rows_to_edit(self, q_start, q_len, device):
        """Which query positions in this forward get edited. -> [R] long, may be empty."""
        pos = torch.arange(q_start, q_start + q_len, device=device)
        return torch.nonzero(pos >= self.first_row, as_tuple=True)[0]

    def edits_layer(self, layer_idx):
        return self.layers is None or int(layer_idx) in self.layers

    def head_index(self, n_heads, device):
        if self.heads is None:
            return None
        bad = [h for h in self.heads if h >= n_heads]
        if bad:
            raise ValueError(f"head {bad} out of range for {n_heads} heads")
        return torch.tensor(self.heads, device=device, dtype=torch.long)

    # -- install / uninstall ----------------------------------------------
    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, _make_attention(self))
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self._text_cfg = cfg
        self._prev_impl = cfg._attn_implementation
        cfg._attn_implementation = IMPL_NAME
        # Every attention module holds its own reference to a config object. Qwen3-VL's
        # text attentions share the text config, but setting it on each module's own
        # config is what actually decides the lookup, so do both rather than rely on
        # them being the same object.
        for m in self.model.modules():
            if type(m).__name__ == "Qwen3VLTextAttention" and hasattr(m, "layer_idx"):
                m.config._attn_implementation = IMPL_NAME
        h = self.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._handles.append(h)
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._text_cfg is not None and self._prev_impl is not None:
            self._text_cfg._attn_implementation = self._prev_impl
            for m in self.model.modules():
                if type(m).__name__ == "Qwen3VLTextAttention":
                    m.config._attn_implementation = self._prev_impl
        self._prev_impl = None


def install(model, **kwargs):
    """Build a SinkShift for `model` and switch the text decoder over to it."""
    return SinkShift(model, **kwargs).install()


# ---------------------------------------------------------------------------
# the attention implementation
# ---------------------------------------------------------------------------
def _repeat_kv(x, n_rep):
    """Grouped-query expansion: [b, kv_heads, s, d] -> [b, kv_heads * n_rep, s, d]."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def _make_attention(state: SinkShift):
    """The function registered under `sink_shift`, closed over one SinkShift."""

    def sink_shift_attention_forward(module, query, key, value, attention_mask,
                                     dropout=0.0, scaling=None, is_causal=None, **kwargs):
        # Layers this arm does not touch never leave the fused kernel.
        if not (state.edits_layer(getattr(module, "layer_idx", -1))
                and state.img_cols is not None and state.alpha > 0.0):
            return _sdpa(module, query, key, value, attention_mask, dropout, scaling,
                         is_causal, **kwargs)

        n_rep = getattr(module, "num_key_value_groups", 1)
        k, v = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
        q_len, kv_len = query.shape[2], k.shape[2]
        q_start = kv_len - q_len                     # absolute position of the first query
        rows = state.rows_to_edit(q_start, q_len, query.device)
        if rows.numel() == 0:
            return _sdpa(module, query, key, value, attention_mask, dropout, scaling,
                         is_causal, **kwargs)

        if scaling is None:
            scaling = module.head_dim ** -0.5
        logits = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
        allowed_row = None
        if attention_mask is not None:
            m = attention_mask[..., :kv_len].float()
            logits = logits + m
            allowed_row = torch.isfinite(m) & (m > torch.finfo(torch.float32).min / 2)
        elif q_len > 1:
            qpos = torch.arange(q_start, q_start + q_len, device=query.device)[:, None]
            kpos = torch.arange(kv_len, device=query.device)[None, :]
            causal = kpos <= qpos
            logits = logits.masked_fill(~causal, float("-inf"))
            allowed_row = causal[None, None]
        a = torch.softmax(logits, dim=-1)

        a = _apply_edit(state, a, rows, allowed_row, module)

        out = torch.matmul(a.to(v.dtype), v).transpose(1, 2).contiguous()
        return out, None

    return sink_shift_attention_forward


def _sdpa(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kw):
    """The stock fast path, for layers and rows this arm leaves alone."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    k, v = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    if attention_mask is not None:
        attention_mask = attention_mask[..., :k.shape[2]]
    causal = (query.shape[2] > 1 and attention_mask is None
              and (is_causal if is_causal is not None else getattr(module, "is_causal", True)))
    out = torch.nn.functional.scaled_dot_product_attention(
        query, k, v, attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
        is_causal=bool(causal))
    return out.transpose(1, 2).contiguous(), None


def _apply_edit(state: SinkShift, a, rows, allowed_row, module):
    """Rewrite the selected rows of `a` [b, H, q, kv], and record what moved.

    Everything below happens on the image columns only, except the `text` arm, which is
    handed the complementary set. Heads not selected keep their rows untouched. `a` came
    out of our own softmax and is not shared, so the rows are written back into it rather
    than into a second full copy -- at a 1,500-token prompt that copy would be ~300 MB.
    """
    heads = state.head_index(a.shape[1], a.device)
    hsel = slice(None) if heads is None else heads
    sub = a[:, hsel][:, :, rows, :]              # [b, H', R, kv]; advanced indexing copies

    img = state.img_cols
    w_img = sub[..., img]                                            # [b, H', R, n_img]
    tot = sub.sum(-1, keepdim=True).clamp_min(1e-30)
    img_mass = w_img.sum(-1, keepdim=True)
    denom = img_mass.clamp_min(1e-30)
    frame, rect = _cached_set(state, "frame"), _cached_set(state, "rect")
    before_frame = w_img[..., frame].sum(-1, keepdim=True) / denom
    before_rect = w_img[..., rect].sum(-1, keepdim=True) / denom

    if state.mass_matched:
        # `text`: shift the amount `centre` would have shifted, but among the tokens that
        # are NOT the picture, so the picture's own distribution is left exactly alone.
        want = state.alpha * w_img[..., state.src_cols].sum(-1, keepdim=True)
        notimg = torch.ones(a.shape[-1], dtype=torch.bool, device=a.device)
        notimg[img] = False
        w_txt = sub[..., notimg]
        every = torch.ones(int(w_txt.shape[-1]), dtype=torch.bool, device=a.device)
        allow_txt = (None if allowed_row is None
                     else allowed_row[:, :1, rows, :][..., notimg])
        new_txt, moved = move_mass(w_txt, every, every, want, target="uniform",
                                   allowed=allow_txt)
        sub[..., notimg] = new_txt
        new_img = w_img
    else:
        new_img, moved = move_fraction(w_img, state.src_cols, state.dst_cols,
                                       state.alpha, state.target)
        sub[..., img] = new_img

    d = state._d
    d["rows_edited"] += int(rows.numel())
    d["forwards"] += 1
    d["layers_touched"].add(int(getattr(module, "layer_idx", -1)))
    d["image_mass"] += float((img_mass / tot).mean())
    d["frame_before"] += float(before_frame.mean())
    d["frame_after"] += float((new_img[..., frame].sum(-1, keepdim=True) / denom).mean())
    d["rect_before"] += float(before_rect.mean())
    d["rect_after"] += float((new_img[..., rect].sum(-1, keepdim=True) / denom).mean())
    d["moved"] += float((moved / tot).mean())
    d["n"] += 1

    if heads is None:
        a[:, :, rows, :] = sub
    else:
        a[:, heads[:, None], rows[None, :], :] = sub
    return a


def _cached_set(state, name):
    """The named patch set over the CURRENT prompt's image columns, built once.

    Cleared by `_locate_images` whenever a new prompt arrives, so it can never describe
    the previous picture's grid.
    """
    got = state._sets.get(name)
    if got is not None:
        return got
    parts = [patch_set(name, gh, gw, state.rect_frac).reshape(-1).repeat(t)
             for (t, gh, gw) in state.grids]
    val = torch.cat(parts).to(state.img_cols.device)
    state._sets[name] = val
    return val
