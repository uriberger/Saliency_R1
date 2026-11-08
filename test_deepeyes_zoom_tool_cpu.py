#!/usr/bin/env python
"""The zoom tool's five cases, on CPU, with no GPU and no ray cluster.

    conda activate deepeyes
    bash patch_deepeyes_zoom_tool.sh          # install the tool into deepeyes_fork
    python test_deepeyes_zoom_tool_cpu.py

WHAT IT TESTS, AND WHY IT IS THE INSTALLED COPY

It imports `verl.tools.image_zoom_in_tool` -- the copy inside `deepeyes_fork/` that a
training run executes -- and refuses to run if that copy has drifted from the tracked
source in `deepeyes/`. A test that imports the tracked file directly would pass on a
fork nobody patched, which is the one outcome that would be worse than no test.

Nothing here needs a GPU and nothing needs `ray.init`: the tool's rate-limiter pool is
constructed lazily and `execute` never routes through it. That is what makes these unit
tests and not a job submission -- which matters, because this cluster currently has
zero GPU nodes.

THE FIVE CASES docs/deepeyes-build-prompts.md Prompt 3 asks for

  1  coordinate conversion in every frame   rel1000, abs_original, abs_smartresize,
                                            all three mapped onto ONE region of ONE
                                            image, so a frame mix-up is visible as a
                                            different crop rather than as an error
  2  the 32 px floor                        a 30x30 box is sub-token on Qwen3-VL and
                                            must be grown; MIN_DIMENSION is 32
  3  an out-of-bounds box                   clamped when it overlaps, rejected when it
                                            does not
  4  a degenerate box                       zero area, inverted, and aspect > 100
  5  the crop source is the ORIGINAL image  loading must not smart-resize, a declared
                                            size that disagrees must fail the run, and
                                            cropping the already-resized copy must be
                                            shown to lose the detail

plus three that guard the way the tool is wired rather than what it computes:

  6  bbox_frame has no default              unset and unrecognised both raise
  7  the YAML path verl actually uses       build_tool_config -> initialize_tools_from_config
  8  raw-vs-converted logging               the first N calls land in a JSONL
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    return bool(ok)


def raises(fn, *args, **kwargs) -> tuple[bool, str]:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        return True, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
    return False, "no exception"


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
#: Deliberately NOT multiples of 32 or 28. An image whose dimensions survive `create`
#: unchanged is an image that was not smart-resized on the way in, and that is case 5's
#: first half.
ORIG_W, ORIG_H = 1601, 1201

#: Arm the `abs_smartresize` frame with a view that is a real downscale, so the three
#: readings are actually distinguishable. 1024 tokens * 32px * 32px = 1,048,576 px.
VIEW_MAX_TOKENS = 1024


def make_image(w: int = ORIG_W, h: int = ORIG_H):
    """A gradient with a 4 px red cross near the target region.

    4 px is the point: it is one eighth of a visual token, so it survives in the
    original and is annihilated by the ~1.35x downscale the "view" applies. Counting
    red pixels in the crop is therefore a direct measurement of whether the tool
    cropped the original or the copy the model was already shown.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(0, h, 8):  # cheap, non-uniform background
        for x in range(0, w, 8):
            for dy in range(min(8, h - y)):
                for dx in range(min(8, w - x)):
                    px[x + dx, y + dy] = ((x * 7) % 256, (y * 11) % 256, 90)
    d = ImageDraw.Draw(img)
    cx, cy = int(w * 0.70), int(h * 0.70)
    d.rectangle([cx - 2, cy - 2, cx + 1, cy + 1], fill=(255, 0, 0))
    return img


def count_red(img) -> int:
    return sum(1 for p in img.convert("RGB").getdata() if p[0] > 200 and p[1] < 70 and p[2] < 70)


def view_of(img, max_tokens: int = VIEW_MAX_TOKENS, factor: int = 32):
    from qwen_vl_utils.vision_process import smart_resize

    h_bar, w_bar = smart_resize(
        img.height, img.width, factor=factor, min_pixels=4 * factor**2, max_pixels=max_tokens * factor**2
    )
    return int(w_bar), int(h_bar)


def main() -> int:
    # ---------------------------------------------------------------- import
    try:
        from verl.tools.image_zoom_in_tool import FRAMES, ImageZoomInTool
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except Exception as e:
        print(f"FATAL: cannot import verl.tools.image_zoom_in_tool: {e}")
        print("       conda activate deepeyes; bash patch_deepeyes_zoom_tool.sh")
        return 2

    import verl.tools.image_zoom_in_tool as installed

    from deepeyes import zoom_tool_config as ztc

    installed_path = Path(installed.__file__).resolve()
    tracked = (REPO / "deepeyes/image_zoom_in_tool.py").read_text()
    print(f"installed tool: {installed_path}")
    if installed_path.read_text() != tracked:
        print("FATAL: the installed tool differs from the tracked source.")
        print("       bash patch_deepeyes_zoom_tool.sh")
        return 2
    if "laser_fork" in str(installed_path):
        print("FATAL: verl resolves inside laser_fork. Wrong conda env -- use `deepeyes`.")
        return 2

    schema = OpenAIFunctionToolSchema.model_validate(ztc.TOOL_SCHEMA)

    def build(**cfg) -> ImageZoomInTool:
        base = {
            "type": "native",
            "patch_size": 16,
            "merge_size": 2,
            "view_patch_size": 16,
            "view_merge_size": 2,
            "view_max_tokens": VIEW_MAX_TOKENS,
            "crop_source_check": "strict",
            "log_first_n_calls": 0,
        }
        base.update(cfg)
        return ImageZoomInTool(base, schema)

    img = make_image()
    view_w, view_h = view_of(img)
    resized = img.resize((view_w, view_h))

    def ck(image=None, **over):
        """create_kwargs for the ORIGINAL, as the data builder would write them."""
        out = {
            "image": image if image is not None else img,
            "original_size": [ORIG_W, ORIG_H],
            "view_size": [view_w, view_h],
        }
        out.update(over)
        return out

    async def zoom(tool, bbox, create_kwargs=None, label=""):
        iid, _ = await tool.create(create_kwargs=create_kwargs if create_kwargs is not None else ck())
        try:
            return await tool.execute(iid, {"bbox_2d": bbox, "label": label})
        finally:
            await tool.release(iid)

    print()
    print("=" * 78)
    print("CASE 6  bbox_frame is explicit or nothing")
    print("=" * 78)
    ok, det = raises(build)
    check(ok, "unset bbox_frame raises at construction", det)
    ok, det = raises(build, bbox_frame="pixels")
    check(ok, "unrecognised bbox_frame raises", det)
    ok, det = raises(build, bbox_frame="")
    check(ok, "empty bbox_frame raises", det)
    check(
        build(bbox_frame="abs_orig").bbox_frame == "abs_original"
        and build(bbox_frame="abs_resized").bbox_frame == "abs_smartresize"
        and build(bbox_frame="REL1000").bbox_frame == "rel1000",
        "P4's spellings alias onto the three frames",
        f"FRAMES={list(FRAMES)}",
    )

    print()
    print("=" * 78)
    print("CASE 1  coordinate conversion, three frames onto one region")
    print("=" * 78)
    # The same region of the same image, addressed three ways.
    target = [0.5 * ORIG_W, 0.5 * ORIG_H, ORIG_W, ORIG_H]
    raw_by_frame = {
        "abs_original": target,
        "rel1000": [500, 500, 1000, 1000],
        "abs_smartresize": [0.5 * view_w, 0.5 * view_h, view_w, view_h],
    }
    print(f"  original={ORIG_W}x{ORIG_H}  view={view_w}x{view_h}  target(original px)="
          f"{[round(v, 1) for v in target]}")
    converted = {}
    for frame, raw in raw_by_frame.items():
        t = build(bbox_frame=frame)
        conv = t.convert_bbox(raw, (ORIG_W, ORIG_H), (view_w, view_h))
        converted[frame] = conv
        print(f"  {frame:<16} raw={raw}  ->  converted={[round(v, 1) for v in conv]}")
        check(
            all(abs(a - b) <= 1.0 for a, b in zip(conv, target, strict=True)),
            f"{frame}: raw box maps onto the target region (+-1 px)",
            f"{[round(v, 1) for v in conv]}",
        )

    # The whole reason the frame is a config key: read the same numbers in the wrong
    # frame and you get a different, perfectly valid-looking crop.
    r = run(zoom(build(bbox_frame="rel1000"), raw_by_frame["rel1000"]))
    w = run(zoom(build(bbox_frame="abs_original"), raw_by_frame["rel1000"]))
    right, wrong = r[0].image[0], w[0].image[0]
    check(
        right.size != wrong.size,
        "the same numbers under the wrong frame crop a different region",
        f"rel1000 -> {right.size}, abs_original -> {wrong.size} (both succeed; nothing errors)",
    )

    # The free diagnostic: which frames could these numbers have come from at all.
    # A box beyond the view's extent cannot be post-smart_resize pixels; a box beyond
    # 1000 cannot be a 0-1000 grid. Over a run this aggregate is the cheap P4.
    p = ImageZoomInTool.frame_plausibility([1200, 900, 1500, 1100], (ORIG_W, ORIG_H), (view_w, view_h))
    check(
        p == {"rel1000": False, "abs_original": True, "abs_smartresize": False},
        "frame_plausibility rules out the frames a box cannot have come from",
        f"[1200,900,1500,1100] on {ORIG_W}x{ORIG_H}/{view_w}x{view_h} -> {p}",
    )
    p = ImageZoomInTool.frame_plausibility([200, 150, 800, 700], (ORIG_W, ORIG_H), (view_w, view_h))
    check(
        all(p.values()),
        "a small box is plausible under all three, and says nothing",
        f"[200,150,800,700] -> {p}",
    )

    print()
    print("=" * 78)
    print("CASE 2  the floor is 32 px, one visual token, not Qwen2.5-VL's 28")
    print("=" * 78)
    t = build(bbox_frame="abs_original")
    check(ImageZoomInTool.MIN_DIMENSION == 32, "class MIN_DIMENSION is 32", f"{ImageZoomInTool.MIN_DIMENSION}")
    check(t.min_dimension == 32 and t.patch_factor == 32,
          "instance floor is patch_size*merge_size = 32", f"min_dimension={t.min_dimension}")
    ok, det = raises(build, bbox_frame="abs_original", min_dimension=28)
    check(ok, "min_dimension=28 is refused as sub-token", det)

    resp, rew, info = run(zoom(t, [700, 700, 730, 730]))  # 30x30: legal at 28, sub-token at 32
    crop = resp.image[0] if resp.image else None
    check(
        info["success"] and crop is not None and min(crop.size) >= 32,
        "a 30x30 box is grown to at least one token",
        f"crop={crop.size if crop else None}",
    )
    resp, rew, info = run(zoom(t, [700, 700, 701, 701]))  # 1x1
    crop = resp.image[0] if resp.image else None
    check(
        info["success"] and crop is not None and min(crop.size) >= 32,
        "a 1x1 box is grown to at least one token",
        f"crop={crop.size if crop else None}",
    )
    # And the floor is still a floor: a tiny box in a tiny image cannot be grown.
    tiny = make_image(40, 40)
    resp, rew, info = run(
        zoom(t, [18, 18, 19, 19], create_kwargs={"image": tiny, "original_size": [40, 40],
                                                 "view_size": view_of(tiny)})
    )
    check(info["success"] and min(resp.image[0].size) >= 32,
          "a tiny image still yields at least one token", f"crop={resp.image[0].size}")

    print()
    print("=" * 78)
    print("CASE 3  out-of-bounds boxes")
    print("=" * 78)
    t = build(bbox_frame="rel1000")
    resp, rew, info = run(zoom(t, [900, 900, 1300, 1400]))  # overlaps, overflows
    check(
        info["success"] and resp.image[0].size[0] <= ORIG_W and resp.image[0].size[1] <= ORIG_H,
        "a box overflowing the edge is clamped, not rejected",
        f"crop={resp.image[0].size} <= {ORIG_W}x{ORIG_H}",
    )
    resp, rew, info = run(zoom(t, [1100, 1100, 1200, 1200]))  # entirely outside
    check(
        (not info["success"]) and rew == -0.05 and resp.image is None,
        "a box entirely outside the image is rejected with -0.05",
        f"reward={rew} text={(resp.text or '')[:60]!r}",
    )
    resp, rew, info = run(zoom(t, [-400, -400, -100, -100]))  # negative, entirely outside
    check((not info["success"]) and rew == -0.05, "a wholly negative box is rejected", f"reward={rew}")
    resp, rew, info = run(zoom(t, [-100, -100, 400, 400]))  # negative but overlapping
    check(info["success"], "a partly negative box is clamped to the origin", f"crop={resp.image[0].size}")

    print()
    print("=" * 78)
    print("CASE 4  degenerate boxes")
    print("=" * 78)
    t = build(bbox_frame="abs_original")
    for name, bbox in [
        ("zero area", [700, 700, 700, 700]),
        ("inverted x", [900, 700, 500, 900]),
        ("inverted y", [500, 900, 900, 700]),
        ("zero width", [700, 700, 700, 900]),
        ("aspect ratio > 100", [10, 700, 1590, 702]),
    ]:
        resp, rew, info = run(zoom(t, bbox))
        check((not info["success"]) and rew == -0.05 and resp.image is None,
              f"{name} is rejected with -0.05", f"{bbox} -> {(resp.text or '')[:50]!r}")
    for name, bbox in [("missing", None), ("three numbers", [1, 2, 3]), ("empty", [])]:
        resp, rew, info = run(zoom(t, bbox))
        check((not info["success"]) and rew == -0.05, f"bbox_2d {name} is rejected", f"reward={rew}")

    print()
    print("=" * 78)
    print("CASE 5  the crop source is the ORIGINAL image")
    print("=" * 78)
    t = build(bbox_frame="abs_original")
    iid, _ = run(t.create(create_kwargs=ck()))
    held = t._instance_dict[iid]["image"]
    run(t.release(iid))
    check(
        held.size == (ORIG_W, ORIG_H),
        "create() does not smart-resize on load",
        f"{ORIG_W}x{ORIG_H} in, {held.size[0]}x{held.size[1]} out "
        f"(verl's fetch_image would have rounded to a multiple of 28)",
    )

    # The detail test. Crop the same original-pixel region from the original and from
    # the already-downsampled copy, and count the 4 px red cross.
    region = [int(0.68 * ORIG_W), int(0.68 * ORIG_H), int(0.72 * ORIG_W), int(0.72 * ORIG_H)]
    resp, _, info = run(zoom(t, region))
    red_original = count_red(resp.image[0])
    sx, sy = view_w / ORIG_W, view_h / ORIG_H
    red_resized = count_red(
        resized.crop((int(region[0] * sx), int(region[1] * sy), int(region[2] * sx), int(region[3] * sy)))
    )
    check(
        red_original >= 12 and red_resized < red_original,
        "a crop of the original keeps detail the resized copy has lost",
        f"red px: original={red_original}, resized={red_resized} "
        f"(view is {ORIG_W}x{ORIG_H} -> {view_w}x{view_h})",
    )

    # The assertion that fails the run. A builder that passes `process_image`'s output
    # while honestly reporting the raw size is the realistic mistake, and it dies here.
    ok, det = raises(run, t.create(create_kwargs=ck(image=resized)))
    check(ok, "handing the already-resized image fails the run (strict)", det)
    ok, det = raises(run, t.create(create_kwargs={"image": img}))
    check(ok, "create_kwargs without original_size fails the run (strict)", det)
    ok, det = raises(run, t.create(create_kwargs=ck(image={"image": img, "max_pixels": 1000})))
    check(ok, "an image spec carrying resize instructions is refused", det)
    ok, det = raises(run, t.create(create_kwargs=ck(view_size=[999, 999])))
    check(ok, "a stated view_size the geometry cannot reproduce fails the run", det)

    # ... and the no-op it is guarding against, made visible. `off` is what a smoke
    # dataset uses; it must not raise, and the crop must then be demonstrably worse.
    loose = build(bbox_frame="abs_original", crop_source_check="off")
    resp, _, _ = run(zoom(loose, region, create_kwargs={"image": resized}))
    check(
        count_red(resp.image[0]) < red_original,
        "with the check off, the same call silently returns a worse, wronger crop",
        f"red px: {count_red(resp.image[0])} vs {red_original} -- original-frame coordinates land "
        f"somewhere else entirely on the resized copy, and nothing errors",
    )

    # min_expected_downscale: the corpus-level version of the same question.
    ok, det = raises(
        run,
        build(bbox_frame="abs_original", min_expected_downscale=2.0).create(create_kwargs=ck()),
    )
    check(ok, "min_expected_downscale rejects a corpus that cannot recover detail", det)

    print()
    print("=" * 78)
    print("CASE 7  the YAML path verl actually uses")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        cfg_path = ztc.build_tool_config(
            Path(td) / "image_zoom_in_tool_config.yaml", bbox_frame="rel1000", log_first_n_calls=0
        )
        tools = initialize_tools_from_config(str(cfg_path))
        check(
            len(tools) == 1 and tools[0].name == "image_zoom_in_tool" and tools[0].bbox_frame == "rel1000",
            "build_tool_config -> initialize_tools_from_config constructs the tool",
            f"{[t.name for t in tools]} frame={getattr(tools[0], 'bbox_frame', None)}",
        )
        # ... and the same path with no frame set must die rather than guess.
        import yaml

        bad = yaml.safe_load(cfg_path.read_text())
        bad["tools"][0]["config"].pop("bbox_frame")
        (Path(td) / "bad.yaml").write_text(yaml.safe_dump(bad))
        ok, det = raises(initialize_tools_from_config, str(Path(td) / "bad.yaml"))
        check(ok, "a tool YAML with no bbox_frame fails to initialise", det)

        # The data-builder helper, round-tripped through the tool's own check.
        ckw = ztc.build_create_kwargs(img, view_max_tokens=VIEW_MAX_TOKENS)
        tool = build(bbox_frame="rel1000")
        iid, _ = run(tool.create(create_kwargs=ckw))
        src = tool._instance_dict[iid]["source"]
        run(tool.release(iid))
        check(
            src["original_size"] == [ORIG_W, ORIG_H] and src["view_size"] == [view_w, view_h]
            and src["view_size_source"] == "stated",
            "build_create_kwargs satisfies the strict crop-source check",
            f"original={src['original_size']} view={src['view_size']} downscale={src['downscale']}",
        )
        check(
            ztc.MAX_ASSISTANT_TURNS == 5 and ztc.MAX_TOOL_CALLS == 4,
            "the turn budget is stated once: 5 generate rounds, 4 tool calls",
            f"max_assistant_turns={ztc.MAX_ASSISTANT_TURNS} max_user_turns={ztc.MAX_USER_TURNS}",
        )

    print()
    print("=" * 78)
    print("CASE 8  raw-vs-converted logging")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        ImageZoomInTool._log_count = 0
        ImageZoomInTool._crop_count = 0
        t = build(bbox_frame="rel1000", log_first_n_calls=3, log_dir=td, log_crops=1)
        for _ in range(5):
            run(zoom(t, [500, 500, 1000, 1000]))
        lines = []
        for p in Path(td).glob("zoom_calls.*.jsonl"):
            lines += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        check(len(lines) == 3, "exactly log_first_n_calls records are written", f"{len(lines)} records")
        if lines:
            rec = lines[0]
            check(
                rec.get("frame") == "rel1000"
                and rec.get("bbox_raw") == [500, 500, 1000, 1000]
                and rec.get("bbox_converted") != rec.get("bbox_raw")
                and rec.get("original_size") == [ORIG_W, ORIG_H],
                "a record carries the raw box, the converted box and the frame",
                json.dumps({k: rec.get(k) for k in
                            ("frame", "bbox_raw", "bbox_converted", "bbox_final", "crop_size", "downscale")}),
            )
        check(
            len(list(Path(td).glob("crop_*.png"))) == 1 and len(list(Path(td).glob("source_*.png"))) == 1,
            "log_crops dumps the crop and the source with the box drawn",
            f"{sorted(p.name for p in Path(td).glob('*.png'))}",
        )

    print()
    print("=" * 78)
    n_fail = sum(1 for ok, _, _ in RESULTS if not ok)
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"FAIL  {name}  --  {detail}")
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print("=" * 78)
    if n_fail == 0:
        print("NOTE: the coordinate FRAME is not decided by this test and cannot be --")
        print("      it proves all three conversions are correct, not which one Qwen3-VL")
        print("      emits. That is P4 (deepeyes_coord_format_probe.py), and it needs a GPU.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/home/uberger/scratch/cache/hf_cache")
    sys.exit(main())
