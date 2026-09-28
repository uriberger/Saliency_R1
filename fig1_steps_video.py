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
    <out>/chain.json         every caption on the frame, next to the run's own text

`--model` repeats, and then each model gets a panel and the panels advance together on
one picture -- which is how the "ours looked there and was right, the baseline looked
here and was wrong" claim is shown rather than asserted. The chains are different
objects: step k of one is not step k of the other, they are only being played at the
same tempo, and a model that runs out of steps first shows its answer while the other
keeps going. `--layout rows` stacks them (portrait, `--image-width` ~560 for two);
`--layout columns` puts them side by side with the chain underneath, which is the only
way to get a landscape sheet out of a square photograph.

`--label`, `--step-text`, `--answer` and `--gold` let the frame say something other than
what the scan recorded -- a paper figure paraphrases a model's sentence to fit the
column, and prints "Two" where the completion said "C. Two". They change **captions
only**: the map under a step is always that step's. Everything overridden is written to
`chain.json` beside what it replaced, so no caption is unauditable. The tick and the
cross come from comparing `--answer` to `--gold`; `--verdict` forces them.

Each step's header reads `step k of n  ·  <map>`, which names the map the frame is drawn
from. That name is internal: `glimpse` means nothing to anyone outside this repo, so a
clip that leaves it -- a project page, a talk -- wants `--map-label ""` to drop the
suffix, or a name a reader will recognise.

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
OK = (76, 200, 110)
BAD = (235, 86, 86)


def kv(pairs, sep="="):
    """`--flag NAME=VALUE`, repeated, as a dict. Values may contain the separator."""
    out = {}
    for p in pairs or []:
        if sep not in p:
            raise SystemExit(f"expected NAME{sep}VALUE, got {p!r}")
        k, v = p.split(sep, 1)
        out[k.strip()] = v
    return out


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

    def __init__(self, img, question, chains, args):
        """`chains` is [(model name, [step text, ...])], one panel of the frame each.

        `--layout rows` stacks the models, each with its chain beside its picture: a tall
        frame, and the one to use when the picture is wide. `--layout columns` puts them
        side by side with the chain underneath: a wide frame, and the only way to get a
        landscape sheet out of a square photograph.
        """
        self.args = args
        self.layout = args.layout
        pad = self.pad = int(args.pad)
        n = len(chains)
        self.img_w = args.image_width
        self.img_h = int(round(img.size[1] * args.image_width / img.size[0]))
        if self.layout == "rows":
            self.text_w = args.text_width
            self.w = pad * 3 + self.img_w + self.text_w
        else:
            self.text_w = self.img_w
            self.w = pad * (n + 1) + self.img_w * n

        self.f_q = font(args.font_size + 4, bold=True)
        self.q_lines = []
        for para in question.splitlines():
            self.q_lines += wrap(para, self.f_q, self.w - 2 * pad) or [""]
        self.q_lh = int(self.f_q.size * 1.42)
        self.head_h = pad + self.q_lh * len(self.q_lines) + pad

        # one type size for every model -- two chains compared at two sizes are not
        # compared -- and in `rows` it shrinks until the longest one fits beside the
        # picture, so no step is ever cut off and the text never reflows between frames
        size = args.font_size
        while True:
            self.f_step = font(size)
            self.f_lbl = font(size, bold=True)
            self.lh = int(self.f_step.size * 1.45)
            self.lbl_h = int(self.f_lbl.size * 2.2)
            self.blocks = {name: [wrap(f"{i + 1}. {s}", self.f_step, self.text_w - 2 * pad)
                                  for i, s in enumerate(steps)]
                           for name, steps in chains}
            self.text_h = max(sum(len(b) * self.lh + self.lh // 2 for b in bl)
                              for bl in self.blocks.values()) + self.lh   # + the footer
            if self.layout != "rows" or self.lbl_h + self.text_h <= self.img_h or size <= 11:
                break
            size -= 1

        if self.layout == "rows":
            self.row_h = self.img_h + pad
            self.h = self.head_h + self.row_h * n + pad
        else:
            self.col_h = self.lbl_h + self.img_h + pad + self.text_h
            self.h = self.head_h + self.col_h + pad
        if self.w % 2:
            self.w += 1
        if self.h % 2:
            self.h += 1

    def _chain(self, d, x, y, key, cur, footer, label=None):
        """The chain with `cur` lit and the answer line; `label` above it if given."""
        if label is not None:
            d.text((x, y), label, fill=ACCENT if cur is not None else DIM, font=self.f_lbl)
            y += self.lbl_h
        top = y
        for i, block in enumerate(self.blocks[key]):
            live = cur is not None and i == cur
            done = cur is not None and i < cur
            colour = FG if live else (DIM if done else MUTED)
            if live:
                d.rectangle([x, y + 2, x + 3, y + len(block) * self.lh - 4], fill=ACCENT)
            for j, ln in enumerate(block):
                d.text((x + 14, y + j * self.lh), ln, fill=colour, font=self.f_step)
            y += len(block) * self.lh + self.lh // 2
        if footer:
            # on the last line of the block every panel reserves, not under whatever
            # this chain happened to need: two verdicts side by side have to line up
            y = top + self.text_h - self.lh
            said, gold, ok = footer
            head = f"{'✓' if ok else '✗'}  {said}"
            d.text((x + 14, y), head, fill=OK if ok else BAD, font=self.f_lbl)
            d.text((x + 14 + self.f_lbl.getlength(head + "   "), y),
                   f"(gold: {gold})", fill=DIM, font=self.f_lbl)

    def frame(self, panels):
        """One frame: the question on top, then `(name, picture, cur, footer)` a panel.

        `name` is `(key, display)` -- the run's subdirectory and what the frame calls it.
        `cur` is the index of the step to light, or None for a panel whose chain has not
        started or has finished; the answer card and a model that ran out of steps before
        the other did both land there.
        """
        pad = self.pad
        out = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(out)
        for i, ln in enumerate(self.q_lines):
            d.text((pad, pad + i * self.q_lh), ln, fill=FG, font=self.f_q)

        for k, ((key, label), picture, cur, footer) in enumerate(panels):
            if self.layout == "rows":
                top = self.head_h + k * self.row_h
                out.paste(picture, (pad, top))
                # centred against the picture: a two-step chain beside a tall photograph
                # otherwise hangs off the top of its row over a column of nothing. The
                # footer's line is always counted, so nothing shifts on the closing frame.
                need = self.lbl_h + self.text_h
                self._chain(d, pad * 2 + self.img_w,
                            top + max(0, (self.img_h - need) // 2),
                            key, cur, footer, label=label)
            else:
                left = pad + k * (self.img_w + pad)
                d.text((left, self.head_h), label,
                       fill=ACCENT if cur is not None else DIM, font=self.f_lbl)
                out.paste(picture, (left, self.head_h + self.lbl_h))
                self._chain(d, left, self.head_h + self.lbl_h + self.img_h + pad,
                            key, cur, footer)
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
    ap.add_argument("--model", required=True, action="append",
                    help="the subdirectory under it; repeatable, and then each model gets "
                         "its own row and the rows step together. A model whose chain is "
                         "shorter finishes early and shows its answer while the other "
                         "keeps going -- step k of one is NOT step k of the other")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="glimpse")
    ap.add_argument("--map-label", default=None,
                    help="what to call the map in each step's header; defaults to the "
                         "value of --map. Pass an empty string to print just 'step k of "
                         "n'. The map names are internal -- a reader outside this repo "
                         "does not know what `glimpse` is, so a video that leaves this "
                         "repo wants an empty string or a real name")
    ap.add_argument("--steps", default=None,
                    help="comma-separated; default every observe step")
    ap.add_argument("--question", default=None,
                    help="header text, replacing the prompt the model was given. A "
                         "benchmark prompt carries scaffolding ('Answer with the option "
                         "letter only.', empty 'nan' choices) that a video does not want")
    # what the frame CALLS things. Everything overridden here is recorded next to what it
    # replaced in <out>/chain.json, so a caption can always be checked against the run.
    ap.add_argument("--label", action="append", default=[], metavar="MODEL=NAME",
                    help="what to call a model on the frame, e.g. coldstart='Vanilla'")
    ap.add_argument("--step-text", action="append", default=[], metavar="MODEL:K=TEXT",
                    help="replace step K's caption. The paper's figure paraphrases the "
                         "model's sentences to fit; this is how to match it. It changes "
                         "the caption only -- the map under it is still that step's")
    ap.add_argument("--answer", action="append", default=[], metavar="MODEL=TEXT",
                    help="what the closing card says the model answered, e.g. 'Two' for "
                         "a run whose raw completion is 'C. Two'")
    ap.add_argument("--gold", default=None, help="likewise for the gold answer")
    ap.add_argument("--verdict", action="append", default=[], metavar="MODEL=ok|bad",
                    help="force the tick/cross; by default the answer is compared to "
                         "--gold, which is right whenever both are given in the same form")
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
    ap.add_argument("--layout", default="rows", choices=["rows", "columns"],
                    help="`rows` stacks the models with the chain beside the picture "
                         "(portrait); `columns` puts them side by side with the chain "
                         "underneath (landscape, and the only way to get a wide sheet "
                         "out of a square photograph). --text-width is ignored in columns")
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

    boxes_blob = json.loads(Path(args.boxes).read_text())["steps"] if args.boxes else []
    wanted = ([int(x) for x in args.steps.split(",") if x != ""] if args.steps else None)
    labels, answers = kv(args.label), kv(args.answer)
    verdicts, retext = kv(args.verdict), kv(args.step_text)
    for k in list(labels) + list(answers) + list(verdicts):
        if k not in args.model:
            raise SystemExit(f"--label/--answer/--verdict for unknown model {k!r}")

    loaded = []
    for model in args.model:
        sdir = Path(args.run_dir) / model / "samples" / args.sample
        if not (sdir / "maps.npz").exists():
            raise SystemExit(f"no maps.npz under {sdir}")
        meta = json.loads((sdir / "meta.json").read_text())
        z = np.load(sdir / "maps.npz")
        if args.map not in z.files:
            raise SystemExit(f"{sdir}/maps.npz has no `{args.map}` map (has {z.files})")
        maps = np.clip(z[args.map], 0, None).astype(np.float64)
        want = wanted if wanted is not None else list(range(len(meta["steps"])))
        bad = [s for s in want if not 0 <= s < maps.shape[0]]
        if bad:
            raise SystemExit(f"{model}: step(s) {bad} outside 0..{maps.shape[0] - 1}")
        boxes_for = {s["step"]: s.get("tight_boxes") or [] for s in boxes_blob
                     if s["sample"] == args.sample and s["model"] == model}
        raw = [meta["steps"][s]["text"] for s in want]
        shown = [retext.get(f"{model}:{s}", t) for s, t in zip(want, raw)]
        loaded.append(dict(model=model, sdir=sdir, meta=meta, maps=maps, want=want,
                           boxes=boxes_for, raw_texts=raw, texts=shown,
                           label=labels.get(model, model)))

    # the picture is the sample's, so every panel shows the same one and it is read once
    img = Image.open(loaded[0]["sdir"] / "original.png").convert("RGB")
    meta0 = loaded[0]["meta"]
    shown_q = (args.question or str(meta0.get("question", ""))).strip()
    canvas = Canvas(img, shown_q, [(m["model"], m["texts"]) for m in loaded], args)
    plain = fit_image(img, canvas.img_w, canvas.img_h)
    multi = len(loaded) > 1
    gold = args.gold if args.gold is not None else str(meta0.get("gt_answer"))

    def norm(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    for m in loaded:
        grid = tuple(m["meta"].get("grid") or m["maps"].shape[1:])
        print(f"[grid] {m['model']}: {grid[0]}x{grid[1]} patches"
              + (f"  --smooth {args.smooth} = {args.smooth / grid[1]:.1%} of the width"
                 if args.smooth else "  (unsmoothed)"))
        # every step's overlay up front: a state needs one panel per model at once
        m["panels"] = []
        for si in m["want"]:
            ov = overlay(img, m["maps"][si], cmap, args)
            if m["boxes"].get(si):
                ov = draw_boxes(ov, m["boxes"][si], args.box_colour, width=2)
            m["panels"].append(fit_image(ov, canvas.img_w, canvas.img_h))
        m["raw_answer"] = answer_of(m["meta"])
        m["said"] = answers.get(m["model"], m["raw_answer"])
        v = verdicts.get(m["model"])
        if v is not None and v not in ("ok", "bad"):
            raise SystemExit(f"--verdict {m['model']}={v!r}: expected ok or bad")
        m["ok"] = (v == "ok") if v is not None else norm(m["said"]) == norm(gold)
        m["footer"] = (m["said"], gold, m["ok"])

    map_label = args.map if args.map_label is None else args.map_label

    def panel(m, k):
        """Model `m` at position `k`; past the end of its chain it shows its answer."""
        tag = f"{m['label']}  ·  " if multi else ""
        if k is None:
            return ((m["model"], f"{tag}input"), plain, None, None)
        if k >= len(m["want"]):
            return ((m["model"], f"{tag}answer"), plain, None, m["footer"])
        label = f"{tag}step {k + 1} of {len(m['want'])}"
        if map_label:
            label += f"  ·  {map_label}"
        return ((m["model"], label), m["panels"][k], k, None)

    # (image, seconds) per state, before the dissolves are inserted
    states = [(canvas.frame([panel(m, None) for m in loaded]), args.hold_title)]
    for k in range(max(len(m["want"]) for m in loaded)):
        states.append((canvas.frame([panel(m, k) for m in loaded]), args.hold_step))
    if args.hold_answer > 0:
        states.append((canvas.frame(
            [((m["model"], f"{m['label']}  ·  answer" if multi else "answer"),
              plain, None, m["footer"]) for m in loaded]), args.hold_answer))

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

    # Every caption on the frame, next to the run's own text. --step-text, --answer,
    # --label and --gold each let a figure say something the scan did not; this is where
    # a reader checks what was replaced, and it is written whether or not anything was.
    shown = {"sample": args.sample, "run_dir": args.run_dir, "map": args.map,
             "layout": args.layout, "smooth": args.smooth,
             "question_shown": shown_q, "question_asked": str(meta0.get("question", "")),
             "gold_shown": gold, "gold_in_run": str(meta0.get("gt_answer")),
             "models": [{"model": m["model"], "label": m["label"], "steps": m["want"],
                         "answer_shown": m["said"], "answer_in_run": m["raw_answer"],
                         "correct": m["ok"],
                         "text_shown": m["texts"], "text_in_run": m["raw_texts"]}
                        for m in loaded]}
    (out / "chain.json").write_text(json.dumps(shown, indent=1))
    edits = sum(a != b for m in loaded for a, b in zip(m["texts"], m["raw_texts"]))
    print(f"[out] {out/'chain.json'}  ({edits} step caption(s) replaced)")

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
