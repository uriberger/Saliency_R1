#!/usr/bin/env python
"""One chain as a short video: the picture and the question, then one frame per step.

`fig1_steps_figure.py` lays the same chain out as a static sheet -- input, then one
captioned panel per observe step, side by side. This plays it instead: frame 0 is the
picture and the question, and every frame after it is that picture with one step's
attention over it while the step's own sentence lights up in the chain on the right.
A talk or a project page wants the second one; a paper wants the first.

    python fig1_steps_video.py --run-dir outputs/saliency_viz/fig1b-realworld \
        --model ours --sample sample_069_row000069 --smooth 0.6 \
        --out outputs/fig1-multistep/video-clevr-vehicles

    <out>/chain.gif          the animation, per-frame durations, loops forever
    <out>/chain.mp4          the same at --fps, H.264, if PyAV can encode it
    <out>/frames/000.png     every distinct frame, in order, for a slide deck

The map is rendered by `fig1_steps_figure.overlay` rather than by a copy of it, so
`--smooth`, `--upsample`, `--overlay-mode`, `--norm` and `--alpha` mean exactly what they
mean there and a frame here is the same image as that figure's panel. In particular
**`--smooth` is cosmetic and sigma is in PATCHES**: read the grid off `maps.npz` before
carrying a sigma over from another sample, and never quote a number measured on a
smoothed map -- every AUROC in `docs/fig1-multistep.md` is `fig1_multistep.py` on the raw
grid.

Timing is per-state, not per-frame: each state (title card, each step, the answer card)
is one entry with a duration, the GIF gets that duration directly, and the MP4 repeats
the frame to fill it. `--fade` inserts a dissolve between consecutive states so the heat
is seen to move rather than to jump, which is the whole point of animating this at all.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fig1_steps_figure import BG, DIM, FG, draw_boxes, font as fallback_font, overlay, wrap

ACCENT = (255, 196, 61)
MUTED = (110, 110, 110)


def font(size, bold=False):
    """DejaVu Sans if matplotlib has it, which it always ships; Pillow's default if not.

    The default font is a hinted bitmap at small sizes and reads badly once a frame is
    scaled for a video; the figure script can live with it because its captions are read
    at 2x, and this one cannot.
    """
    try:
        from matplotlib import font_manager
        from PIL import ImageFont
        name = "DejaVu Sans:bold" if bold else "DejaVu Sans"
        return ImageFont.truetype(font_manager.findfont(name), size)
    except Exception:
        return fallback_font(size)


def fit_image(img, box_w, box_h):
    """Largest scale of `img` inside the box, preserving aspect."""
    s = min(box_w / img.size[0], box_h / img.size[1])
    return img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))),
                      Image.LANCZOS)


def answer_of(meta):
    """What the model actually said, i.e. everything after the chain closes."""
    gen = str(meta.get("generation", ""))
    tail = gen.split("</think>", 1)[1] if "</think>" in gen else gen
    tail = re.sub(r"<\|.*?\|>", " ", tail)
    return " ".join(tail.split())


class Canvas:
    """The fixed geometry every frame shares.

    Every frame has to be the same size -- a GIF with a changing canvas and an H.264
    stream with a changing resolution are both broken -- so the layout is solved once,
    against the longest step text, and each frame only repaints it.
    """

    def __init__(self, img, question, steps, args):
        self.args = args
        pad = self.pad = int(args.pad)
        self.img_w = args.image_width
        self.img_h = int(round(img.size[1] * args.image_width / img.size[0]))
        self.text_w = args.text_width
        self.w = pad * 3 + self.img_w + self.text_w

        self.f_q = font(args.font_size + 2, bold=True)
        self.f_meta = font(args.font_size - 1)
        self.q_lines = []
        for para in question.splitlines():
            self.q_lines += wrap(para, self.f_q, self.w - 2 * pad) or [""]
        self.q_lh = int(self.f_q.size * 1.42)
        self.head_h = pad + self.q_lh * len(self.q_lines) + pad

        # the chain column: shrink until the whole chain fits beside the picture, so no
        # step is ever cut off and the text never reflows between frames
        size = args.font_size
        while True:
            self.f_step = font(size)
            self.f_lbl = font(max(10, size - 2), bold=True)
            self.lh = int(self.f_step.size * 1.45)
            self.blocks = [wrap(f"{i + 1}. {s}", self.f_step, self.text_w - pad - 14)
                           for i, s in enumerate(steps)]
            need = (int(self.f_lbl.size * 2.2)
                    + sum(len(b) * self.lh + self.lh // 2 for b in self.blocks))
            if need <= self.img_h or size <= 11:
                break
            size -= 1
        self.h = self.head_h + self.img_h + pad
        if self.w % 2:
            self.w += 1
        if self.h % 2:
            self.h += 1

    def frame(self, panel, cur, steps, label, footer=None):
        """One frame: header, `panel` on the left, the chain on the right, `cur` lit."""
        pad = self.pad
        out = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(out)
        for i, ln in enumerate(self.q_lines):
            d.text((pad, pad + i * self.q_lh), ln, fill=FG, font=self.f_q)
        out.paste(panel, (pad, self.head_h))

        x = pad * 2 + self.img_w
        y = self.head_h
        d.text((x, y), label, fill=ACCENT if cur is not None else DIM, font=self.f_lbl)
        y += int(self.f_lbl.size * 2.2)
        for i, block in enumerate(self.blocks):
            live = cur is not None and i == cur
            done = cur is not None and i < cur
            colour = FG if live else (DIM if done else MUTED)
            if live:
                d.rectangle([x, y + 2, x + 3, y + len(block) * self.lh - 4], fill=ACCENT)
            for j, ln in enumerate(block):
                d.text((x + 14, y + j * self.lh), ln, fill=colour, font=self.f_step)
            y += len(block) * self.lh + self.lh // 2
        if footer:
            d.text((x + 14, y + self.lh // 2), footer, fill=ACCENT, font=self.f_lbl)
        return out


def encode_gif(path, states, scale):
    frames = [(f if scale == 1.0 else
               f.resize((int(f.size[0] * scale), int(f.size[1] * scale)), Image.LANCZOS))
              for f, _ms in states]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[int(ms) for _f, ms in states], loop=0, optimize=True)


def encode_mp4(path, states, fps):
    """H.264 at a constant frame rate; each state repeated to fill its duration."""
    try:
        import av
    except ImportError:
        return "PyAV is not installed in this environment"
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = states[0][0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "slow"}
        for img, ms in states:
            frame = av.VideoFrame.from_image(img)
            for _ in range(max(1, int(round(ms * fps / 1000.0)))):
                for pkt in stream.encode(frame):
                    container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
    except Exception as exc:                        # no libx264, or no encoder at all
        return f"{type(exc).__name__}: {exc}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a saliency_viz output root")
    ap.add_argument("--model", required=True, help="the subdirectory under it")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="glimpse")
    ap.add_argument("--steps", default=None,
                    help="comma-separated; default every observe step")
    ap.add_argument("--question", default=None,
                    help="header text, replacing the prompt the model was given. A "
                         "benchmark prompt carries scaffolding ('Answer with the option "
                         "letter only.', empty 'nan' choices) that a video does not want")
    ap.add_argument("--boxes", default=None,
                    help="a fig1_multistep.py json; draws each step's tight referent")
    ap.add_argument("--box-colour", default="#00ff66")
    # timing, in seconds
    ap.add_argument("--fps", type=int, default=25, help="mp4 frame rate")
    ap.add_argument("--hold-title", type=float, default=3.0)
    ap.add_argument("--hold-step", type=float, default=2.8)
    ap.add_argument("--hold-answer", type=float, default=3.5,
                    help="0 drops the closing answer card")
    ap.add_argument("--fade", type=float, default=0.4,
                    help="dissolve between consecutive states; 0 cuts")
    ap.add_argument("--fade-frames", type=int, default=6,
                    help="distinct images drawn per dissolve")
    # layout
    ap.add_argument("--image-width", type=int, default=760)
    ap.add_argument("--text-width", type=int, default=520)
    ap.add_argument("--font-size", type=int, default=19)
    ap.add_argument("--pad", type=int, default=22)
    ap.add_argument("--gif-scale", type=float, default=0.62,
                    help="the gif only; 256 colours at full size is a very large file")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    # render knobs, identical to fig1_steps_figure.py
    ap.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "rank"])
    ap.add_argument("--norm-lo", type=float, default=1.0)
    ap.add_argument("--norm-hi", type=float, default=99.0)
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian on the patch grid, sigma in PATCHES. Cosmetic only")
    ap.add_argument("--upsample", default="map", choices=["map", "rgb", "nearest"])
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--overlay-mode", default="blend", choices=["blend", "alpha"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps[args.cmap]

    sdir = Path(args.run_dir) / args.model / "samples" / args.sample
    if not (sdir / "maps.npz").exists():
        raise SystemExit(f"no maps.npz under {sdir}")
    meta = json.loads((sdir / "meta.json").read_text())
    z = np.load(sdir / "maps.npz")
    if args.map not in z.files:
        raise SystemExit(f"{sdir}/maps.npz has no `{args.map}` map (has {z.files})")
    maps = np.clip(z[args.map], 0, None).astype(np.float64)
    img = Image.open(sdir / "original.png").convert("RGB")

    want = ([int(x) for x in args.steps.split(",") if x != ""] if args.steps
            else list(range(len(meta["steps"]))))
    bad = [s for s in want if not 0 <= s < maps.shape[0]]
    if bad:
        raise SystemExit(f"step(s) {bad} outside 0..{maps.shape[0] - 1}")

    boxes_for = {}
    if args.boxes:
        for s in json.loads(Path(args.boxes).read_text())["steps"]:
            if s["sample"] == args.sample and s["model"] == args.model:
                boxes_for[s["step"]] = s.get("tight_boxes") or []

    texts = [meta["steps"][s]["text"] for s in want]
    shown_q = (args.question or str(meta.get("question", ""))).strip()
    canvas = Canvas(img, shown_q, texts, args)
    plain = fit_image(img, canvas.img_w, canvas.img_h)

    grid = tuple(meta.get("grid") or maps.shape[1:])
    print(f"[grid] {grid[0]}x{grid[1]} patches"
          + (f"  --smooth {args.smooth} = {args.smooth / grid[1]:.1%} of the width"
             if args.smooth else "  (unsmoothed)"))

    # (image, seconds) per state, before the dissolves are inserted
    states = [(canvas.frame(plain, None, texts, "input"), args.hold_title)]
    for k, si in enumerate(want):
        ov = overlay(img, maps[si], cmap, args)
        if boxes_for.get(si):
            ov = draw_boxes(ov, boxes_for[si], args.box_colour, width=2)
        states.append((canvas.frame(fit_image(ov, canvas.img_w, canvas.img_h), k, texts,
                                    f"step {k + 1} of {len(want)}  ·  {args.map}"),
                       args.hold_step))
    if args.hold_answer > 0:
        gold, said = meta.get("gt_answer"), answer_of(meta)
        states.append((canvas.frame(plain, None, texts, "answer",
                                    footer=f"{said}      (gold: {gold})"),
                       args.hold_answer))

    timed = []
    for i, (im, secs) in enumerate(states):
        if i and args.fade > 0 and args.fade_frames > 0:
            prev = states[i - 1][0]
            per = args.fade / args.fade_frames
            for j in range(1, args.fade_frames + 1):
                timed.append((Image.blend(prev, im, j / (args.fade_frames + 1)),
                              per * 1000.0))
        timed.append((im, secs * 1000.0))

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for i, (im, _ms) in enumerate(states):
        im.save(out / "frames" / f"{i:03d}.png")
    print(f"[out] {out/'frames'}/  ({len(states)} states, {canvas.w}x{canvas.h}, "
          f"{sum(s for _i, s in states):.1f}s)")

    if not args.no_gif:
        encode_gif(out / "chain.gif", timed, args.gif_scale)
        mb = (out / "chain.gif").stat().st_size / 1e6
        print(f"[out] {out/'chain.gif'}  ({len(timed)} frames, {mb:.1f} MB)")
    if not args.no_mp4:
        err = encode_mp4(out / "chain.mp4", timed, args.fps)
        if err:
            print(f"[warn] no mp4: {err}")
        else:
            mb = (out / "chain.mp4").stat().st_size / 1e6
            print(f"[out] {out/'chain.mp4'}  ({args.fps} fps, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
