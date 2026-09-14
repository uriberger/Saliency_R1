#!/usr/bin/env python
"""Render one of this repo's docs/*.md to PDF, with no LaTeX.

There is no TeX on these nodes (no pdflatex, xelatex, wkhtmltopdf or weasyprint), so
`pandoc -o out.pdf` cannot work here -- pandoc has no engine to hand the document to. What is
available is pandoc itself for Markdown->HTML and the pure-Python xhtml2pdf for HTML->PDF, so
this script chains the two and supplies a stylesheet in the CSS subset xhtml2pdf implements.

xhtml2pdf is not installed in any shared conda env on purpose: those are shared with running
jobs, and a doc build is no reason to mutate one. Install it into a throwaway directory and
point PYTHONPATH at it:

    pip install --target /tmp/pdfvendor xhtml2pdf
    PYTHONPATH=/tmp/pdfvendor python md2pdf.py docs/where-attention-goes.md outputs/x.pdf

Only the CSS subset xhtml2pdf supports is used (@page, basic table and text properties), which
is why the stylesheet below looks dated: flexbox, grid and most of CSS3 are silently ignored.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CSS = """
@page { size: a4 portrait; margin: 1.9cm 1.8cm 2.1cm 1.8cm; }
body { font-family: Helvetica, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #16191d; }
h1 { font-size: 17pt; margin: 0 0 2mm 0; color: #0f1115; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm 0; color: #0f1115;
     border-bottom: 0.6pt solid #b9c0c8; padding-bottom: 1.1mm; }
h3 { font-size: 10.6pt; margin: 4.5mm 0 1.5mm 0; color: #26303a; }
p { margin: 0 0 2.2mm 0; text-align: left; }
strong { color: #0f1115; }
a { color: #1f4f8f; text-decoration: none; }
code { font-family: Courier, monospace; font-size: 8.8pt; background-color: #f0f2f4; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 3.5mm 0; font-size: 8.5pt; }
th { background-color: #eceff2; border: 0.5pt solid #b9c0c8; padding: 1.5mm 1.8mm;
     text-align: left; font-weight: bold; color: #0f1115; }
td { border: 0.5pt solid #ccd2d8; padding: 1.5mm 1.8mm; vertical-align: top; }
blockquote { margin: 2mm 0 2.6mm 5mm; padding-left: 3mm; border-left: 1.6pt solid #8d98a4;
             color: #2b333c; font-style: italic; }
hr { border: 0; border-top: 0.5pt solid #ccd2d8; margin: 5mm 0; }
li { margin-bottom: 1.1mm; }
"""

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/><style>{css}</style></head>
<body>{body}
<p style="margin-top:6mm; font-size:7.6pt; color:#6b7580;">
Generated from {src} by md2pdf.py. Claims marked S in the verification column were taken from
search summaries and not read in the primary source.</p>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown", type=Path)
    ap.add_argument("pdf", type=Path)
    args = ap.parse_args()

    if shutil.which("pandoc") is None:
        sys.exit("pandoc not on PATH; it is the Markdown->HTML half of this pipeline")
    try:
        from xhtml2pdf import pisa
    except ImportError:
        sys.exit("xhtml2pdf not importable -- see this script's docstring for the "
                 "pip install --target / PYTHONPATH recipe")

    body = subprocess.run(
        ["pandoc", "-f", "markdown+pipe_tables", "-t", "html", str(args.markdown)],
        capture_output=True, text=True, check=True).stdout

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with args.pdf.open("wb") as fh:
        # Returns a truthy err on failure rather than raising, so it has to be checked.
        status = pisa.CreatePDF(HTML.format(css=CSS, body=body, src=args.markdown.name), dest=fh)
    if status.err:
        sys.exit(f"xhtml2pdf reported {status.err} error(s) rendering {args.markdown}")
    print(f"[out] {args.pdf}  ({args.pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
