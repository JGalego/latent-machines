# ruff: noqa: E501  (embedded HTML and JavaScript)
"""Check an animated SVG for overlapping or clipped text.

    python3 site/animations/check.py FILE.svg [step_seconds] [loop_seconds]

The animation is paused every ``step_seconds`` (default 0.25) across the loop
(default 10 s). At each time, every visible <text> is measured. The check
reports text that leaves the viewBox, and pairs of text boxes that overlap.
Exit status is 1 if anything is reported. Arrows and shapes still need a look
at the frames from preview.py.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PAGE = """<!doctype html><body style="margin:0"><div style="width:720px">{svg}</div><pre id="out"></pre>
<script>
addEventListener('load', () => {{
  const svg = document.querySelector('svg');
  const vb = svg.viewBox.baseVal, box = svg.getBoundingClientRect();
  const sx = vb.width / box.width, sy = vb.height / box.height;
  const found = [];
  for (let t = 0; t <= {loop}; t += {step}) {{
    document.getAnimations().forEach(a => {{ a.pause(); a.currentTime = t * 1000; }});
    const items = [];
    svg.querySelectorAll('text').forEach(el => {{
      let node = el, opacity = 1, hidden = false;
      while (node && node !== svg.parentNode) {{
        const cs = getComputedStyle(node);
        opacity *= parseFloat(cs.opacity);
        if (cs.display === 'none' || cs.visibility === 'hidden') hidden = true;
        node = node.parentNode;
      }}
      if (hidden || opacity < 0.15 || !el.textContent.trim()) return;
      const r = el.getBoundingClientRect();
      items.push({{ text: el.textContent.trim(), x0: (r.left - box.left) * sx,
        y0: (r.top - box.top) * sy, x1: (r.right - box.left) * sx,
        y1: (r.bottom - box.top) * sy }});
    }});
    for (const i of items) {{
      if (i.x0 < -0.5 || i.y0 < -0.5 || i.x1 > vb.width + 0.5 || i.y1 > vb.height + 0.5)
        found.push({{ t, kind: 'outside viewBox', a: i.text }});
    }}
    for (let a = 0; a < items.length; a++) for (let b = a + 1; b < items.length; b++) {{
      const p = items[a], q = items[b];
      const w = Math.min(p.x1, q.x1) - Math.max(p.x0, q.x0);
      const h = Math.min(p.y1, q.y1) - Math.max(p.y0, q.y0);
      if (w > 1 && h > 2) found.push({{ t, kind: 'text overlap', a: p.text, b: q.text }});
    }}
  }}
  document.getElementById('out').textContent = JSON.stringify(found);
}});
</script></body>"""


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    step = float(argv[1]) if len(argv) > 1 else 0.25
    loop = float(argv[2]) if len(argv) > 2 else 10.0
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(PAGE.format(svg=path.read_text(), step=step, loop=loop))
        dom = subprocess.run(
            ["google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
             "--dump-dom", f"file://{page}"],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
    out = dom.split('<pre id="out">')[1].split("</pre>")[0]
    found = json.loads(out.replace("&quot;", '"').replace("&amp;", "&"))
    seen = set()
    for item in found:
        key = (item["kind"], item["a"], item.get("b"))
        if key not in seen:
            seen.add(key)
            print(
                f"{path.name} t={item['t']:g}s {item['kind']}: {item['a']!r} {item.get('b', '')!r}"
            )
    print(f"{path.name}: {'clean' if not found else str(len(seen)) + ' problems'}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
