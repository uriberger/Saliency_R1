"""Join the two generated-token panels into one figure with a single colourbar.

Each panel is drawn standalone, so each carries its own bar and its own footnote. Side
by side that is two identical bars and two identical notes. Both are cropped off the
pair and re-attached once:

  [ before grid ] [ after grid ] [ colourbar ]

The bar strip is taken from the BEFORE panel on purpose. The "> 16x" key is only drawn
on a panel that actually has a cell up there, and only the before panel does -- take the
after panel's bar and the violet corner would appear with nothing explaining it.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

SRC = '/tmp/fig5_gen/figures'
OUT = ('/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/uberger/research/'
       'saliency_r1/outputs/fig5-overrange')
os.makedirs(OUT, exist_ok=True)
BG = (252, 252, 251)

before = Image.open(f'{SRC}/heat_stats_gen_qwen3_vl.png').convert('RGB')
after = Image.open(f'{SRC}/heat_stats_gen_qwen3_vl_self_saliency.png').convert('RGB')
assert before.size == after.size
w, h = before.size

a = np.asarray(before).astype(int)
is_bg = np.abs(a - np.array(BG)).sum(axis=2) < 12

# the gutter between the grid and the colourbar: first wide background run right of middle
band = is_bg[int(h * 0.15):int(h * 0.70)].mean(axis=0)
runs, start = [], None
for x in range(w):
    if band[x] > 0.995:
        start = x if start is None else start
    elif start is not None:
        if x - start >= 6:
            runs.append((start, x))
        start = None
cut = next(s for s, _e in runs if s > w * 0.55)

# the footnote block: the lowest band of rows carrying ink, left of the cut
ink_rows = (~is_bg[:, :cut]).mean(axis=1)
note_rows = [y for y in range(int(h * 0.78), h) if ink_rows[y] > 0.004]
note_top = (min(note_rows) - 10) if note_rows else h
print(f'panel {w}x{h}: colourbar gutter x={cut}, footnote starts y={note_top}')

# grids with their own note painted out, so the pair carries one note, not two
def grid_only(img):
    g = img.crop((0, 0, cut, h)).copy()
    ImageDraw.Draw(g).rectangle([0, note_top, cut, h], fill=BG)
    return g

# above the footnote only: below it the strip still holds the tail of this panel's own
# note, which would reappear to the right of the shared one
bar = before.crop((cut, 0, w, note_top))   # carries the "> 16x" key
gap = 14
canvas = Image.new('RGB', (cut * 2 + gap + bar.width, h), BG)
canvas.paste(grid_only(before), (0, 0))
canvas.paste(grid_only(after), (cut + gap, 0))
canvas.paste(bar, (cut * 2 + gap, 0))

# the shared footnote, once, under the pair -- full width, not clipped at the gutter
note = before.crop((0, note_top, w, h))
canvas.paste(note, (0, note_top))

png = f'{OUT}/fig5_generated_tokens_overrange.png'
canvas.save(png)
canvas.save(f'{OUT}/fig5_generated_tokens_overrange.pdf', 'PDF', resolution=200)
print(f'wrote {png}  {canvas.size}  {os.path.getsize(png) / 1024:.0f} KB')

violet = (np.abs(np.asarray(canvas).astype(int) - np.array([0x4a, 0x3a, 0xa7])).sum(2) < 30).sum()
print(f'over-range (violet) pixels in the composite: {violet}')
