#!/usr/bin/env python
"""CPU checks for the sink-location experiment. No GPU, no model, no dataset.

    python test_sink_location_cpu.py

What each group is for:

  geometry      the ring, the radial depths and the four edges are what the write-up says
                they are, and `ring_set(gh, gw, 0)` agrees patch for patch with
                `sink_shift.frame_set` -- the set the INTERVENTION drains. The two are
                written separately, so nothing but a test keeps the measurement and the
                edit describing the same patches.
  reducer       the endpoints are arithmetic, and the arithmetic is checked against maps
                whose answer is known by hand: a flat map gives enrichment 1.0 for every
                set including the controls, a delta map puts the peak where it was put,
                and the causal column correction removes exactly the gradient it is there
                to remove. That last one is the trap: attention is causal and the picture
                is raster-ordered, so the uncorrected image->image column mean has a
                built-in top-row bias of the shape this whole experiment is testing for.
  locate        `locate_image_runs` agrees with `SinkShift._locate_images`, including two
                pictures of different shapes in one prompt, and refuses a mismatched grid
                rather than guessing.
  attention     the scan's attention implementation reproduces stock SDPA exactly -- it
                must, it edits nothing -- and the columns it records are the columns a
                hand computation gives.
  transforms    THE COORDINATE FRAME. `patch_correspondence` agrees with a direct numpy
                rotation for every rotation and flip, survives a round trip, and the
                inverse maps land inside the source exactly where the padding says they
                should. An off-by-one here answers the content-versus-position question
                confidently and backwards, and nothing downstream would notice.
  probe         the storage round-trips, resume survives a torn line, and the statistics
                helpers do what their names say.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import sink_location as SL  # noqa: E402
import sink_shift as SS  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


GRIDS = [(gh, gw) for gh in range(4, 18) for gw in (5, 8, 10, 16, 21)]


# ---------------------------------------------------------------------------
def test_geometry():
    print("\ngeometry")
    bad = [(gh, gw) for gh, gw in GRIDS
           if not np.array_equal(SL.ring_set(gh, gw, 0),
                                 SS.frame_set(gh, gw).numpy())]
    check("ring depth 0 IS sink_shift's frame, on every grid", not bad, str(bad[:3]))

    bad = [(gh, gw) for gh, gw in GRIDS
           if abs(SL.ring_set(gh, gw, 0).mean() - SL.ring_area_frac(gh, gw)) > 1e-12]
    check("ring_area_frac is the ring's actual area", not bad, str(bad[:3]))

    bad = [(gh, gw) for gh, gw in GRIDS
           if not np.array_equal(SL.ring_set(gh, gw, 1), SS.ring2_set(gh, gw).numpy())]
    check("depth 1 IS sink_shift's second ring", not bad, str(bad[:3]))

    d = SL.depth_map(7, 11)
    check("depth_map counts from the nearest edge",
          d[0, 0] == 0 and d[3, 5] == 3 and d.max() == 3, f"max {d.max()}")
    check("every patch has exactly one depth",
          sum(int((SL.depth_map(7, 11) == k).sum()) for k in range(4)) == 77)

    e = SL.edge_sets(6, 9)
    check("the four sides cover the ring",
          np.array_equal(e["top"] | e["bottom"] | e["left"] | e["right"],
                         SL.ring_set(6, 9)))
    check("there are four corners and they are on the ring",
          e["corner"].sum() == 4 and (e["corner"] & ~SL.ring_set(6, 9)).sum() == 0)

    for gh, gw in GRIDS:
        want = int(SL.ring_set(gh, gw).sum())
        got = int(SL.random_contiguous_set(gh, gw, want, seed=3).sum())
        if got != want:
            check(f"the control block is ring-sized on {gh}x{gw}", False, f"{got}!={want}")
            break
    else:
        check("the control block is exactly ring-sized on every grid", True,
              f"{len(GRIDS)} grids")

    s = SL.named_sets(10, 16)
    check("named_sets are flat and grid-sized",
          all(v.shape == (160,) for v in s.values()), str({k: v.shape for k, v in
                                                           list(s.items())[:2]}))
    check("interior is the complement of the ring",
          np.array_equal(s["interior"], ~s["ring"]))


# ---------------------------------------------------------------------------
def _cells(map_flat, gh, gw, **kw):
    """One (layer, head) cell from a hand-written map."""
    m = np.asarray(map_flat, dtype=float).reshape(1, 1, -1)
    return SL.reduce_cells(m, None, 1, m.sum(-1), gh, gw, kw.pop("kv_len", 1000), **kw)


def test_reducer():
    print("\nreducer")
    gh, gw = 10, 16
    flat = np.ones(gh * gw)
    st, peak = _cells(flat, gh, gw)
    sets = SL.named_sets(gh, gw)
    bad = []
    for name, stat in (("ring", "ring_share"), ("top", "top_share"),
                       ("corner", "corner_share"), ("ctrl_block", "ctrl_block_share"),
                       ("depth1", "depth1_share"), ("deep", "deep_share")):
        e = st[0, 0, SL.STAT_INDEX[stat]] / sets[name].mean()
        if abs(e - 1.0) > 1e-9:
            bad.append(f"{name}={e:.4f}")
    check("a FLAT map gives enrichment 1.000 for every set", not bad, ", ".join(bad))
    check("a flat map has normalised entropy 1.0",
          abs(st[0, 0, SL.STAT_INDEX["entropy_norm"]] - 1.0) < 1e-9)
    check("a flat map's image mass is 1.0 when there is nothing else in the row",
          abs(st[0, 0, SL.STAT_INDEX["image_mass"]] - 1.0) < 1e-9)

    m = np.ones(gh * gw)
    m[0] = 100.0                                     # a corner
    st, peak = _cells(m, gh, gw, kv_len=200)
    check("the peak is where it was put", int(peak[0, 0]) == 0)
    check("...and it is on the ring", st[0, 0, SL.STAT_INDEX["peak_in_ring"]] == 1.0)
    check("...and peak_uniform_x prices it against the WHOLE row, not the picture",
          abs(st[0, 0, SL.STAT_INDEX["peak_uniform_x"]] - 100.0 / m.sum() * 200) < 1e-6,
          f"{st[0, 0, SL.STAT_INDEX['peak_uniform_x']]:.3f}, "
          f"want {100.0 / m.sum() * 200:.3f}")
    m2 = np.ones(gh * gw)
    m2[5 * gw + 8] = 100.0                           # the middle
    st2, peak2 = _cells(m2, gh, gw)
    check("an interior peak is not on the ring",
          st2[0, 0, SL.STAT_INDEX["peak_in_ring"]] == 0.0)
    check("a spiked map is less flat than a flat one",
          st2[0, 0, SL.STAT_INDEX["entropy_norm"]] < 0.99)

    ring = SL.ring_set(gh, gw).reshape(-1)
    m3 = np.where(ring, 3.0, 1.0)
    st3, _ = _cells(m3, gh, gw)
    want = 3.0 / (3.0 * ring.mean() + 1.0 * (1 - ring.mean()))
    check("a border 3x brighter than the interior gives that exact enrichment",
          abs(st3[0, 0, SL.STAT_INDEX["ring_share"]] / ring.mean() - want) < 1e-9,
          f"{st3[0, 0, SL.STAT_INDEX['ring_share']] / ring.mean():.4f} vs {want:.4f}")

    # THE CAUSAL TRAP. Give every image query an identical, flat row over the patches it
    # is allowed to see. Uncorrected, the column sums fall off a cliff from patch 0 to
    # patch N-1 and the top row looks like a sink. Corrected, it is flat again.
    n = gh * gw
    scan = SL.SinkScan(torch.nn.Module())
    ids = torch.zeros(1, n, dtype=torch.long)
    ids[0, :] = SL.IMAGE_TOKEN_ID
    scan._locate(ids, torch.tensor([[1, gh * 2, gw * 2]]))
    null = scan.column_null("image")
    allowed = np.tril(np.ones((n, n)))
    w = allowed / allowed.sum(1, keepdims=True)           # position-blind: uniform in-row
    col = w.sum(0).reshape(1, 1, -1)
    e_of = lambda st: (st[0, 0, SL.STAT_INDEX["top_share"]]
                       / SL.named_sets(gh, gw)["top"].mean())
    e_raw = e_of(SL.reduce_cells(col, None, n, col.sum(-1), gh, gw, n)[0])
    e_cnt = e_of(SL.reduce_cells(col, None, n, col.sum(-1), gh, gw, n,
                                 col_null=np.arange(n, 0, -1, dtype=float) / n)[0])
    e_fix = e_of(SL.reduce_cells(col, None, n, col.sum(-1), gh, gw, n, col_null=null)[0])
    check("uncorrected, causal masking alone manufactures a top-row sink", e_raw > 1.5,
          f"enrichment {e_raw:.2f}")
    check("dividing by the QUERY COUNT does not fix it -- the obvious fix is wrong",
          e_cnt > 1.5, f"enrichment {e_cnt:.2f}")
    check("dividing by the position-blind null does", abs(e_fix - 1.0) < 1e-9,
          f"enrichment {e_fix:.6f} (raw {e_raw:.2f}, by count {e_cnt:.2f})")

    st4, _ = SL.reduce_cells(np.zeros((1, 1, n)), None, 1, np.zeros((1, 1)), gh, gw, 10)
    check("a head that never looks at the picture returns NaN, not a division artefact",
          bool(np.all(np.isnan(st4[0, 0, [SL.STAT_INDEX["ring_share"],
                                          SL.STAT_INDEX["peak_share"]]]))))

    # peak_cv: two queries, the peak column identical in both -> CV 0
    cs = np.zeros((1, 1, n)); cs[0, 0, 0] = 2.0
    sq = np.zeros((1, 1, n)); sq[0, 0, 0] = 2.0                 # (1^2 + 1^2)
    st5, _ = SL.reduce_cells(cs, sq, 2, cs.sum(-1), gh, gw, 10)
    check("a column identical across queries has CV 0",
          abs(st5[0, 0, SL.STAT_INDEX["peak_cv"]]) < 1e-9)
    sq2 = np.zeros((1, 1, n)); sq2[0, 0, 0] = 4.0               # (2^2 + 0^2)
    st6, _ = SL.reduce_cells(cs, sq2, 2, cs.sum(-1), gh, gw, 10)
    check("a column present for one query in two has CV 1",
          abs(st6[0, 0, SL.STAT_INDEX["peak_cv"]] - 1.0) < 1e-9,
          f"{st6[0, 0, SL.STAT_INDEX['peak_cv']]:.4f}")


# ---------------------------------------------------------------------------
def _ids(grids, prefix=3, tail=4):
    n_img = sum(t * gh * gw for t, gh, gw in grids)
    ids = torch.zeros(1, prefix + n_img + tail + len(grids), dtype=torch.long)
    at = prefix
    for t, gh, gw in grids:
        ids[0, at] = SL.VISION_START_ID
        at += 1
        ids[0, at:at + t * gh * gw] = SL.IMAGE_TOKEN_ID
        at += t * gh * gw
    ids[0, at] = SL.VISION_END_ID
    return ids, torch.tensor([[t, gh * 2, gw * 2] for t, gh, gw in grids])


def test_locate():
    print("\nlocate")
    ids, thw = _ids([(1, 10, 16)])
    runs, grids = SL.locate_image_runs(ids, thw)
    st = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.0)
    st._locate_images(ids, thw)
    check("locate_image_runs agrees with SinkShift on the columns",
          torch.equal(torch.cat(runs), st.img_cols))
    check("...and on the grids", grids == st.grids, f"{grids} vs {st.grids}")

    ids2, thw2 = _ids([(1, 10, 16), (1, 6, 8)])
    runs2, grids2 = SL.locate_image_runs(ids2, thw2)
    st2 = SS.SinkShift(torch.nn.Module(), arm="centre", alpha=0.0)
    st2._locate_images(ids2, thw2)
    check("two pictures of different shapes in one prompt",
          grids2 == [(1, 10, 16), (1, 6, 8)] and grids2 == st2.grids)
    check("...and their columns are concatenated in prompt order",
          torch.equal(torch.cat(runs2), st2.img_cols)
          and int(runs2[0].max()) < int(runs2[1].min()))

    try:
        SL.locate_image_runs(ids2, thw2[:1])
        ok = False
    except RuntimeError:
        ok = True
    check("a grid list that does not match the runs is refused", ok)
    try:
        SL.locate_image_runs(ids, torch.tensor([[1, 8, 8]]))
        ok = False
    except RuntimeError:
        ok = True
    check("a grid whose token count is wrong is refused", ok)

    spans = SL.span_index(ids, runs)
    n = int(ids.numel())
    check("the spans cover the prompt exactly once, bar `first`",
          sum(int(spans[s].numel()) for s in SL.SPANS if s != "first") == n)
    check("the image span is the image", torch.equal(spans["image"], torch.cat(runs)))
    check("vision_start sits immediately before the picture",
          int(spans["vision_start"][0]) == int(spans["image"][0]) - 1)
    check("post_image holds no image and no delimiter",
          int(spans["post_image"].numel()) > 0
          and not bool(torch.isin(spans["post_image"], spans["image"]).any()))


# ---------------------------------------------------------------------------
class FakeAttn(torch.nn.Module):
    """Just enough of Qwen3VLTextAttention for the registered function to run."""

    def __init__(self, layer_idx, heads=4, kv_heads=2, head_dim=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = heads // kv_heads
        self.head_dim = head_dim
        self.is_causal = True


def _scan_state(gh=6, gw=8, prefix=3, tail=4):
    scan = SL.SinkScan(torch.nn.Module())
    ids, thw = _ids([(1, gh, gw)], prefix=prefix, tail=tail)
    scan._locate(ids, thw)
    return scan, int(ids.numel())


def test_attention():
    print("\nattention")
    scan, n = _scan_state()
    fn = SL._make_scan_attention(scan)
    mod = FakeAttn(layer_idx=0)
    g = torch.Generator().manual_seed(21)
    q = torch.randn(1, 4, n, 8, generator=g)
    k = torch.randn(1, 2, n, 8, generator=g)
    v = torch.randn(1, 2, n, 8, generator=g)

    ref, _ = SS._sdpa(mod, q, k, v, None, 0.0, None, True)
    got, _ = fn(mod, q, k, v, None, scaling=None)
    check("the scan reproduces stock SDPA exactly", torch.allclose(ref, got, atol=1e-5),
          f"max |diff| {float((ref - got).abs().max()):.2e}")

    # the columns it recorded, against a hand computation on the same weights
    logits = (q.float() @ SS._repeat_kv(k, 2).float().transpose(2, 3)) * (8 ** -0.5)
    kpos = torch.arange(n)[None, :]
    logits = logits.masked_fill(kpos > torch.arange(n)[:, None], float("-inf"))
    a = torch.softmax(logits, dim=-1)
    rows = scan.rows_for("text", 0, n, q.device)
    want = a[0, :, rows, :][..., scan.img_cols].sum(1)
    res = scan.result()
    check("the recorded columns are the attention that was computed",
          torch.allclose(torch.from_numpy(res["text"]["col_sum"][0]), want, atol=1e-5),
          f"max |diff| "
          f"{float((torch.from_numpy(res['text']['col_sum'][0]) - want).abs().max()):.2e}")
    check("the query count is per layer, not per layer times layers",
          res["text"]["n_rows"] == int(rows.numel()),
          f"{res['text']['n_rows']} vs {int(rows.numel())}")
    check("the text query set sits entirely after the picture",
          int(rows.min()) > int(scan.img_cols.max()))
    check("the span budget sums to the whole row",
          abs(float(sum(res["spans"][s][0].sum() for s in SL.SPANS if s != "first"))
              - float(res["text"]["row_total"][0].sum())) < 1e-3)

    scan2, n2 = _scan_state()
    fn2 = SL._make_scan_attention(scan2)
    fn2(FakeAttn(layer_idx=0), q, k, v, None, scaling=None)
    fn2(FakeAttn(layer_idx=1), q, k, v, None, scaling=None)
    r2 = scan2.result()
    check("two layers stack in index order, and neither inflates the row count",
          r2["text"]["col_sum"].shape[0] == 2
          and r2["text"]["n_rows"] == res["text"]["n_rows"])

    cn = scan.column_null("image")
    check("the image->image null falls from the first column to the last",
          cn[0] > cn[-1] > 0, f"{cn[0]:.3e}..{cn[-1]:.3e}")
    check("the text query set needs no column correction",
          scan.column_null("text") is None)

    scan3 = SL.SinkScan(torch.nn.Module())
    ids3, thw3 = _ids([(1, 4, 5)])
    ids3 = torch.cat([ids3, ids3], dim=0)
    try:
        scan3._pre_hook(None, (ids3,), {})
        ok = False
    except RuntimeError:
        ok = True
    check("a batch wider than 1 is refused, not edited wrongly", ok)


# ---------------------------------------------------------------------------
def test_transforms():
    print("\ntransforms")
    from PIL import Image

    gh0, gw0 = 7, 11
    base = np.arange(gh0 * gw0).reshape(gh0, gw0)

    for arm, npfn, shape in (("rot90", lambda a: np.rot90(a), (gw0, gh0)),
                             ("rot180", lambda a: np.rot90(a, 2), (gh0, gw0)),
                             ("hflip", lambda a: a[:, ::-1], (gh0, gw0))):
        _im, inv, _m = SL.transform(arm, Image.new("RGB", (gw0 * 32, gh0 * 32)))
        corr = SL.patch_correspondence(inv, shape[0], shape[1], gh0, gw0)
        want = npfn(base).reshape(-1)
        check(f"{arm}: the correspondence IS the numpy rotation",
              np.array_equal(corr, want),
              f"{corr[:4]} vs {want[:4]}")

    _im, inv, _m = SL.transform("rot90", Image.new("RGB", (gw0 * 32, gh0 * 32)))
    c1 = SL.patch_correspondence(inv, gw0, gh0, gh0, gw0)
    _im2, inv2, _m2 = SL.transform("rot90", Image.new("RGB", (gh0 * 32, gw0 * 32)))
    c2 = SL.patch_correspondence(inv2, gh0, gw0, gw0, gh0)
    check("rot90 four times is the identity, through the grid decoder",
          np.array_equal(c1[c2[c1[c2]]], np.arange(gh0 * gw0)))

    _im, inv, _m = SL.transform("identity", Image.new("RGB", (320, 224)))
    check("identity maps every patch to itself",
          np.array_equal(SL.patch_correspondence(inv, gh0, gw0, gh0, gw0),
                         np.arange(gh0 * gw0)))

    # a real rotation of a real picture, decoded through content rather than through the
    # inverse map: the two must agree, or one of them is wrong
    a = np.zeros((gh0 * 32, gw0 * 32, 3), dtype=np.uint8)
    a[2 * 32:3 * 32, 5 * 32:6 * 32] = 255
    im = Image.fromarray(a, "RGB")
    got = int(np.argmax(SL.content_stats(im, gh0, gw0)["mean_grey"]))
    check("content_stats finds a bright patch where it was painted",
          got == 2 * gw0 + 5, f"patch {got}, want {2 * gw0 + 5}")
    for arm, shape in (("rot90", (gw0, gh0)), ("rot180", (gh0, gw0)),
                       ("hflip", (gh0, gw0))):
        tim, inv, _m = SL.transform(arm, im)
        corr = SL.patch_correspondence(inv, shape[0], shape[1], gh0, gw0)
        peak = int(np.argmax(SL.content_stats(tim, *shape)["mean_grey"]))
        check(f"{arm}: the moved content lands where the correspondence says",
              corr[peak] == 2 * gw0 + 5, f"corr says {corr[peak]}")

    tim, inv, meta = SL.transform("pad_grey_1", im)
    check("pad keeps the picture's pixel size, so the grid cannot move",
          tim.size == im.size, f"{im.size} -> {tim.size}")
    corr = SL.patch_correspondence(inv, gh0, gw0, gh0, gw0)
    check("pad marks its border as showing nothing from the source",
          all(corr[i] == -1 for i in np.flatnonzero(SL.ring_set(gh0, gw0).reshape(-1))),
          f"{int((corr < 0).sum())} of {corr.size} patches are padding")
    check("...and the interior still shows the source",
          bool((corr[SL.depth_map(gh0, gw0).reshape(-1) >= 1] >= 0).all()))

    tim, inv, meta = SL.transform("zoom60", im)
    corr = SL.patch_correspondence(inv, gh0, gw0, gh0, gw0)
    check("zoom keeps the size and shows only the middle of the source",
          tim.size == im.size and (corr >= 0).all()
          and set(np.unique(corr % gw0)) <= set(range(1, gw0 - 1)),
          f"columns shown: {sorted(set(np.unique(corr % gw0)))}")

    tim, _inv, meta = SL.transform("donut", im)
    inner = np.asarray(tim.convert("L"))[gh0 * 16, gw0 * 16]
    check("donut blanks the middle and keeps the border",
          tim.size == im.size and abs(int(inner) - int(meta["fill"][0])) <= 1)

    tim, inv, meta = SL.transform("canvas0", im)
    corr = SL.patch_correspondence(inv, gh0, gw0, gh0, gw0)
    check("canvas0 puts the picture in the top-left quarter",
          (corr[:2] >= 0).all() and corr[-1] == -1,
          f"{int((corr >= 0).sum())} of {corr.size} patches show the picture")

    check("every declared arm can be built",
          all(SL.transform(a, im)[0] is not None for a in SL.ARMS),
          f"{len(SL.ARMS)} arms")
    check("a picture too small to pad is refused, not padded wrongly",
          SL.transform("pad_grey_2", Image.new("RGB", (64, 64)))[0] is None)


# ---------------------------------------------------------------------------
def test_probe():
    print("\nprobe")
    spec = importlib.util.spec_from_file_location(
        "_sl_probe", os.path.join(ROOT, "sink_location_probe.py"))
    SP = importlib.util.module_from_spec(spec)
    sys.modules["_sl_probe"] = SP
    spec.loader.exec_module(SP)

    with tempfile.TemporaryDirectory() as td:
        sink = SP.Sink(td, "scan", 0, flush_every=2)
        for i in range(3):
            arr = {"stats": np.full((2, 3, len(SL.STAT_NAMES)), float(i)),
                   "maps": np.full((2, 20 + i), float(i))}
            # An OPTIONAL field, present on one unit of a flushed part and absent from the
            # other. `perm` is exactly this in the real run -- only the permutation arms
            # have one -- and an earlier encoding dropped it from every part it shared
            # with another arm, which was all of them.
            if i == 1:
                arr["perm"] = np.arange(20 + i)[::-1].copy()
            sink.write(f"k{i}", {"key": f"k{i}", "type": "t", "grid": [4, 5],
                                 "ring_area_frac": SL.ring_area_frac(4, 5)}, arr)
        sink.close()
        meta, arrays = SP.read_stage(td, "scan")
        check("the storage round-trips every unit", len(meta) == 3 and len(arrays) == 3)
        check("...with fixed-shape fields intact",
              float(arrays["k2"]["stats"][0, 0, 0]) == 2.0)
        check("...and ragged fields sliced back to their own shapes",
              arrays["k0"]["maps"].shape == (2, 20)
              and arrays["k2"]["maps"].shape == (2, 22))
        check("...and a field only ONE unit in a part has survives, on that unit alone",
              "perm" in arrays["k1"] and "perm" not in arrays["k0"]
              and "perm" not in arrays["k2"],
              f"perm on {[k for k in arrays if 'perm' in arrays[k]]}")
        check("...and an index field comes back as exact integers, not rounded floats",
              np.array_equal(arrays["k1"]["perm"], np.arange(21)[::-1]),
              str(arrays["k1"]["perm"][:4]))

        with open(os.path.join(td, "scan_shard0.jsonl"), "a") as fh:
            fh.write("{ half a line\n")
        check("resume survives a torn last line",
              len(SP.Sink(td, "scan", 0).done()) == 3)

    p = SP.holm(np.array([0.01, 0.04, 0.5]))
    check("holm steps down and stays monotone",
          abs(p[0] - 0.03) < 1e-12 and p[1] >= p[0] and p[2] >= p[1], str(np.round(p, 4)))
    m, lo, hi, n = SP.boot_mean([1.0] * 20, n_boot=200)
    check("boot_mean on a constant has a zero-width CI",
          abs(m - 1) < 1e-12 and abs(hi - lo) < 1e-12 and n == 20)
    m2, _lo, _hi, n2 = SP.boot_paired({i: 1.0 for i in range(10)},
                                      {i: 0.0 for i in range(6)}, n_boot=200)
    check("boot_paired uses only the pictures both ran", abs(m2 - 1) < 1e-12 and n2 == 6,
          f"n={n2}")
    check("_mode is the label the cells agreed on, not their average",
          SP._mode([3, 3, 100]) == 3)
    check("ci_excludes reads a CI the way the verdicts do",
          SP.ci_excludes(1.6, 1.9, 1.5) and not SP.ci_excludes(1.4, 1.6, 1.5))

    # THE COORDINATE FRAME, on a stand-in for the processor's grid. The GPU selftest runs
    # the identical function against the real one; this is what makes a mistake in it cost
    # a second rather than a queued job.
    grid_fn = lambda im: (max(1, im.size[1] // 32), max(1, im.size[0] // 32))  # noqa: E731
    for size in ((512, 320), (320, 512), (352, 224)):
        bad, n = SP.frame_check(grid_fn, size)
        check(f"every transform decodes where it claims, at {size}", not bad,
              ", ".join(bad) if bad else f"{n} arms")

    check("every corpus type declares a loader and a note",
          all({"kind", "note"} <= set(v) for v in SP.CORPUS.values()),
          f"{len(SP.CORPUS)} types")
    check("the trained cell is the pair the reward shaped",
          (SP.TRAINED_LAYER, SP.TRAINED_HEADS) == (22, (28, 31)))
    check("the content keys the report indexes are the ones the scan writes",
          set(SP.CONTENT_KEYS) <= set(SL.content_stats(
              __import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 64)), 2, 2)))


def main():
    print("sink_location CPU checks")
    test_geometry()
    test_reducer()
    test_locate()
    test_attention()
    test_transforms()
    test_probe()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
