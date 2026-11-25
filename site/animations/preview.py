# ruff: noqa: E501  (embedded HTML and JavaScript)
"""Render frames of an animated SVG to PNG files for review.

    python3 site/animations/preview.py FILE.svg 0 2 4 6 [--dark] [--out DIR]

Each frame is the animation paused at that many seconds. The page's colour
variables are applied, so the frames match the site in light or dark mode.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

LIGHT = (
    "--bg:#fcfcfa;--surface:#f2f1ec;--text:#1d1d1b;--muted:#5d5d58;--rule:#dddcd4;"
    "--accent:#2a5db0;--cell-zero:#ebeae4;--cell-one:#1d1d1b;--s0:#2a5db0;"
    "--s1:#b25a00;--s2:#1f7a4d;--s3:#7a4fb0;--s4:#5d5d58;"
)
DARK = (
    "--bg:#161615;--surface:#222220;--text:#e9e8e2;--muted:#a3a29b;--rule:#3a3936;"
    "--accent:#8db4f2;--cell-zero:#2e2d2a;--cell-one:#e9e8e2;--s0:#8db4f2;"
    "--s1:#f0a85b;--s2:#7fd1a3;--s3:#c3a4f0;--s4:#a3a29b;"
)
PAGE = """<!doctype html><html><body style="margin:0;background:var(--bg);{vars}font-family:sans-serif">
<div style="width:720px;padding:16px">{svg}</div>
<script>
addEventListener('load', () => {{
  document.getAnimations().forEach(a => {{ a.pause(); a.currentTime = {ms}; }});
}});
</script></body></html>"""


def main(argv: list[str]) -> None:
    dark = "--dark" in argv
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path(".")
    skip = {argv[argv.index("--out") + 1]} if "--out" in argv else set()
    args = [a for a in argv if not a.startswith("--") and a not in skip]
    svg_path, times = Path(args[0]), [float(t) for t in args[1:]] or [0.0]
    svg = svg_path.read_text()
    out.mkdir(parents=True, exist_ok=True)
    for t in times:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.html"
            page.write_text(
                PAGE.format(vars=DARK if dark else LIGHT, svg=svg, ms=int(t * 1000))
            )
            target = out / f"{svg_path.stem}-{'dark' if dark else 'light'}-{t:g}s.png"
            subprocess.run(
                ["google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
                 "--window-size=760,420", f"--screenshot={target}", f"file://{page}"],
                check=True, capture_output=True,
            )  # fmt: skip
            print(target)


if __name__ == "__main__":
    main(sys.argv[1:])
