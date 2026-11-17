"""Build the GitHub Pages site from a template, references, and run records.

``site/index.html`` is a template. The build resolves its markers and writes
static HTML, so every number on the page comes from a source file:

    {{cite:a,b}}                   citations, numbered by first use
    {{references}}                 the cited entries of site/references.json
    {{eqtag:name}}, {{eq:name}}    equation numbers: \\tag{} and "(n)"
    {{figure:name}}, {{fig:name}}  "Figure n" in captions and in prose
    {{table:name}}, {{tab:name}}   "Table n" in captions and in prose
    {{results}}                    tables generated from results/<EXP>/<run>/
    {{abstract_result}}, {{conformance_summary}}, {{experiment_status}}

Figures, tables, and equations are numbered in page order after results are
expanded. Run records are rendered by kind: lockstep conformance runs (E00,
E01), learned-controller runs (E02), and anything else as a list of links.
The results are also embedded as JSON for the interactive views in app.js.

Unknown markers or names, duplicate labels, and cited or listed references
that do not match fail the build.

    python3 site/build.py _site
"""

import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/JGalego/latent-machines/blob/main"
EXPERIMENTS = [f"E{n:02d}" for n in range(9)]
TITLES = {
    "E00": "memory conformance",
    "E01": "exact streaming machines",
    "E02": "learned interface comparison",
}
SPLITS = {
    "test_in_distribution": "2-8 statements",
    "test_length_32": "32 statements",
    "test_length_64": "64 statements",
    "test_held_out_pairs": "held-out record-value pairs",
}
MARKER = re.compile(r"\{\{(\w+)(?::([^{}]*))?\}\}")


class BuildError(Exception):
    pass


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def superscript_power(n: int) -> str:
    exponent = len(str(n)) - 1
    return f"10<sup>{exponent}</sup>" if n == 10**exponent else f"{n:,}"


# ---------- loading run records ----------


def kind(metrics: dict) -> str:
    variants = metrics.get("variants")
    if isinstance(variants, list) and variants and "actions_checked" in variants[0]:
        return "conformance"
    if "runs" in metrics and "summary" in metrics:
        return "learned"
    return "unknown"


def load_results(root: Path) -> dict[str, list[dict]]:
    experiments: dict[str, list[dict]] = {}
    for metrics_path in sorted(root.glob("results/*/*/metrics.json")):
        run_dir = metrics_path.parent
        run_path = run_dir / "run.json"
        try:
            metrics = json.loads(metrics_path.read_text())
            run = json.loads(run_path.read_text()) if run_path.exists() else {}
        except json.JSONDecodeError as error:
            raise BuildError(f"{metrics_path}: {error}") from error
        experiments.setdefault(run_dir.parent.name, []).append(
            {
                "kind": kind(metrics),
                "run_id": run.get("run_id", run_dir.name),
                "revision": (run.get("source") or {}).get("revision") or "",
                "config": run.get("config", {}),
                "cpu": (run.get("environment") or {}).get("cpu"),
                "metrics": metrics,
                "manifest": f"{REPO}/{metrics_path.relative_to(root)}",
                "report": (
                    f"{REPO}/{(run_dir.parent / 'README.md').relative_to(root)}"
                    if (run_dir.parent / "README.md").exists()
                    else None
                ),
            }
        )
    return experiments


# ---------- rendering ----------


def badge(label: str) -> str:
    cls = "residue" if label == "conforms with residue" else label
    return f'<span class="badge {esc(cls)}">{esc(label)}</span>'


def row_key(variant: dict) -> str:
    machine = variant.get("machine")
    return f"{machine} · {variant['workload']}" if machine else variant["workload"]


def intro(exp_id: str, runs: list[dict], detail: str) -> str:
    title = TITLES.get(exp_id, exp_id).capitalize()
    revisions = sorted({r["revision"][:7] for r in runs if r["revision"]})
    codes = ", ".join(f"<code>{esc(r)}</code>" for r in revisions)
    report = runs[0]["report"]
    link = (
        f' The <a href="{esc(report)}">report</a> gives the setup and limits.'
        if report
        else ""
    )
    plural = "s" if len(revisions) > 1 else ""
    return (
        f"<p><strong>{esc(title)} ({esc(exp_id)}).</strong> {detail} "
        f"Records come from revision{plural} {codes}.{link}</p>"
    )


def render_conformance(exp_id: str, runs: list[dict]) -> str:
    runs = sorted(
        runs, key=lambda r: (r["config"].get("horizon", 0), r["config"].get("seed", 0))
    )
    total = sum(r["metrics"]["totals"]["actions_checked"] for r in runs)
    rows = []
    for r in runs:
        t, variants = r["metrics"]["totals"], r["metrics"]["variants"]
        agree = sum(v["snapshot_resume"] == "agrees" for v in variants)
        rows.append(
            "<tr>"
            f"<td>{superscript_power(r['config'].get('horizon', 0))}</td>"
            f'<td class="num">{esc(r["config"].get("seed", ""))}</td>'
            f'<td class="num">{t["actions_checked"]:,}</td>'
            f'<td class="num">{t["mismatches"]}</td>'
            f'<td class="num">{t["max_residue"]}</td>'
            f'<td class="num">{agree}/{t["variants"]}</td>'
            f"<td>{badge(r['metrics']['status'])}</td>"
            f"<td>{esc(r['cpu'] or '')}</td>"
            f'<td><a href="{esc(r["manifest"])}">{esc(r["run_id"].split("-")[-1])}</a></td>'
            "</tr>"
        )
    largest = runs[-1]
    variants = largest["metrics"]["variants"]
    dtypes = list(dict.fromkeys(v["dtype"] for v in variants))
    keys = list(dict.fromkeys(row_key(v) for v in variants))
    cells = []
    for key in keys:
        tds = []
        for dtype in dtypes:
            match = [v for v in variants if row_key(v) == key and v["dtype"] == dtype]
            tds.append(
                f"<td>{badge(match[0]['classification'])}</td>"
                if match
                else "<td></td>"
            )
        cells.append(f'<tr><th scope="row">{esc(key)}</th>{"".join(tds)}</tr>')
    detail = f"{total:,} actions were checked in {len(runs)} runs."
    head = "".join(f"<th>{esc(d)}</th>" for d in dtypes)
    return f"""{intro(exp_id, runs, detail)}
    <p class="caption"><strong>{{{{table:{exp_id.lower()}-runs}}}}:</strong> {esc(exp_id)} runs. Mismatches count nonconforming steps; residue is Equation&nbsp;{{{{eq:residue}}}}; a snapshot agrees if the restored state equals the live state.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Actions per configuration</th><th>Seed</th><th>Actions checked</th><th>Mismatches</th><th>Max residue</th><th>Snapshots agree</th><th>Status</th><th>CPU</th><th>Record</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
    <p class="caption"><strong>{{{{table:{exp_id.lower()}-configs}}}}:</strong> Classification of each {esc(exp_id)} configuration. The table shows the run with {superscript_power(largest["config"].get("horizon", 0))} actions per configuration; with scripts enabled, choose another run and select a cell for its details.</p>
    <div class="explorer" data-experiment="{esc(exp_id)}">
      <div class="table-wrap"><table class="grid-matrix">
        <thead><tr><th></th>{head}</tr></thead>
        <tbody>{"".join(cells)}</tbody>
      </table></div>
    </div>"""


def accuracy_cell(stat: dict | None) -> str:
    if not stat:
        return "<td>–</td>"
    return (
        f'<td class="num">{stat["mean"]:.3f} '
        f'<span class="range">[{stat["min"]:.3f}, {stat["max"]:.3f}]</span></td>'
    )


def render_learned(exp_id: str, runs: list[dict]) -> str:
    parts = []
    for r in runs:
        m = r["metrics"]
        summary = m["summary"]
        seen = list(dict.fromkeys(s["split"] for s in summary))
        splits = [s for s in SPLITS if s in seen] + [s for s in seen if s not in SPLITS]
        arms = sorted({s["arm"] for s in summary})
        seeds = max((s["seeds"] for s in summary), default=0)
        mismatches = m.get("executor_mismatches")
        detail = (
            f'Run <a href="{esc(r["manifest"])}">{esc(r["run_id"].split("-")[-1])}</a> '
            f"has status {badge(m['status'])} with {seeds} training seed{'s' if seeds != 1 else ''} "
            f"and {mismatches} executor mismatches."
        )
        rows = []
        for metric, label in (("task_accuracy", "Task"), ("action_accuracy", "Action")):
            for arm in arms:
                cells = []
                for split in splits:
                    stat = next(
                        (
                            s.get(metric)
                            for s in summary
                            if s["arm"] == arm and s["split"] == split
                        ),
                        None,
                    )
                    cells.append(accuracy_cell(stat))
                rows.append(
                    f'<tr><th scope="row">{label} · {esc(arm)}</th>{"".join(cells)}</tr>'
                )
        head = "".join(f"<th>{esc(SPLITS.get(s, s))}</th>" for s in splits)
        name = f"{exp_id.lower()}-{r['run_id'].split('-')[-1]}"
        parts.append(f"""{intro(exp_id, [r], detail)}
    <p class="caption"><strong>{{{{table:{name}}}}}:</strong> {esc(exp_id)} accuracy by arm and test split: mean over training seeds, with the range in brackets. Action accuracy is undefined for arm A, which takes no actions.</p>
    <div class="explorer" data-experiment="{esc(exp_id)}" data-run="{esc(r["run_id"])}">
      <div class="table-wrap"><table>
        <thead><tr><th></th>{head}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
    </div>""")
    return "\n".join(parts)


def render_unknown(exp_id: str, runs: list[dict]) -> str:
    links = ", ".join(
        f'<a href="{esc(r["manifest"])}">{esc(r["run_id"])}</a>' for r in runs
    )
    return f"<p><strong>{esc(exp_id)}.</strong> Records without a renderer yet: {links}.</p>"


RENDERERS = {
    "conformance": render_conformance,
    "learned": render_learned,
    "unknown": render_unknown,
}


def render_results(experiments: dict[str, list[dict]]) -> str:
    parts = []
    for exp_id in sorted(experiments):
        runs = experiments[exp_id]
        for record_kind in ("conformance", "learned", "unknown"):
            group = [r for r in runs if r["kind"] == record_kind]
            if group:
                parts.append(RENDERERS[record_kind](exp_id, group))
    return (
        "\n".join(parts) if parts else '<p class="muted">No results recorded yet.</p>'
    )


# ---------- generated status text ----------


def conformance_stats(experiments: dict[str, list[dict]]) -> dict:
    runs = [r for rs in experiments.values() for r in rs if r["kind"] == "conformance"]
    variants = [v for r in runs for v in r["metrics"]["variants"]]
    return {
        "runs": len(runs),
        "configurations": len(variants),
        "exact": sum(v["classification"] == "exact" for v in variants),
        "actions": sum(r["metrics"]["totals"]["actions_checked"] for r in runs),
        "max_horizon": max((r["config"].get("horizon", 0) for r in runs), default=0),
    }


def abstract_result(stats: dict) -> str:
    if not stats["configurations"]:
        return "conformance runs have not been recorded yet"
    if stats["exact"] == stats["configurations"]:
        return (
            "these implementations matched their oracles with zero residue on every one of "
            f"the {stats['actions']:,} actions checked, with up to "
            f"{superscript_power(stats['max_horizon'])} actions per configuration"
        )
    failing = stats["configurations"] - stats["exact"]
    return (
        f"{failing} of {stats['configurations']} tested configurations did not match their "
        f"oracles exactly over {stats['actions']:,} checked actions (Section&nbsp;4.2)"
    )


def conformance_summary(stats: dict) -> str:
    if stats["exact"] == stats["configurations"]:
        return (
            f"All {stats['configurations']} configurations in the {stats['runs']} recorded "
            "conformance runs are classified as exact: no step failed to conform, the residue "
            "was zero throughout, and every snapshot restored the live state exactly."
        )
    return (
        f"{stats['exact']} of {stats['configurations']} configurations in the {stats['runs']} "
        "recorded conformance runs are classified as exact; the tables identify the others."
    )


def join_ids(ids: list[str]) -> str:
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]


def experiment_status(experiments: dict[str, list[dict]]) -> str:
    done = [e for e in EXPERIMENTS if e in experiments]
    rest = [e for e in EXPERIMENTS if e not in experiments]
    sentences = []
    if done:
        verb = "has" if len(done) == 1 else "have"
        sentences.append(f"{join_ids(done)} {verb} recorded results")
    if rest:
        sentences.append(f"{rest[0]} is next")
    if len(rest) > 1:
        sentences.append(f"{rest[1]}–{rest[-1]} are planned")
    return "; ".join(sentences) + "."


# ---------- markers ----------


def number(page: str, define: str, refer: str, label: str) -> str:
    names: dict[str, int] = {}
    for match in MARKER.finditer(page):
        if match.group(1) == define:
            name = match.group(2)
            if name in names:
                raise BuildError(f"duplicate {define} label {name!r}")
            names[name] = len(names) + 1

    def replace(match: re.Match) -> str:
        marker, name = match.group(1), match.group(2)
        if marker not in (define, refer):
            return match.group(0)
        if name not in names:
            raise BuildError(f"unknown {refer} reference {name!r}")
        n = names[name]
        if marker == define:
            return str(n) if label == "tag" else f"{label} {n}"
        return f"({n})" if label == "tag" else f"{label}&nbsp;{n}"

    return MARKER.sub(replace, page)


def cite(page: str, references: dict[str, str]) -> str:
    order: dict[str, int] = {}
    for match in MARKER.finditer(page):
        if match.group(1) == "cite":
            for key in match.group(2).split(","):
                if key not in references:
                    raise BuildError(f"unknown reference {key!r}")
                order.setdefault(key, len(order) + 1)
    unused = sorted(set(references) - set(order))
    if unused:
        raise BuildError(f"references never cited: {unused}")

    def replace(match: re.Match) -> str:
        if match.group(1) == "cite":
            links = [
                f'<a href="#ref-{key}">{order[key]}</a>'
                for key in match.group(2).split(",")
            ]
            return "[" + ", ".join(links) + "]"
        if match.group(1) == "references":
            items = "\n".join(
                f'      <li id="ref-{key}">{references[key]}</li>' for key in order
            )
            return f"<ol>\n{items}\n    </ol>"
        return match.group(0)

    return MARKER.sub(replace, page)


def results_data(experiments: dict[str, list[dict]]) -> str:
    """Compact JSON for the interactive views; '</' is escaped for inline script."""
    data = {
        exp_id: [
            {
                "kind": r["kind"],
                "run_id": r["run_id"],
                "revision": r["revision"],
                "horizon": r["config"].get("horizon"),
                "seed": r["config"].get("seed"),
                "manifest": r["manifest"],
                "variants": [
                    {
                        key: v.get(key)
                        for key in (
                            "variant", "dtype", "workload", "machine", "classification",
                            "snapshot_resume", "actions_checked", "max_residue",
                            "end_to_end_seconds", "action_sha256",
                        )
                    }
                    for v in r["metrics"].get("variants", [])
                ]
                if r["kind"] == "conformance"
                else [],
                "summary": r["metrics"].get("summary", []) if r["kind"] == "learned" else [],
            }
            for r in runs
        ]
        for exp_id, runs in experiments.items()
    }  # fmt: skip
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


def render_page(root: Path) -> str:
    page = (root / "site" / "index.html").read_text()
    references = json.loads((root / "site" / "references.json").read_text())
    experiments = load_results(root)
    stats = conformance_stats(experiments)
    values = {
        "results": render_results(experiments),
        "abstract_result": abstract_result(stats),
        "conformance_summary": conformance_summary(stats),
        "experiment_status": experiment_status(experiments),
        "results_data": results_data(experiments),
    }
    page = MARKER.sub(
        lambda m: (
            values[m.group(1)]
            if m.group(1) in values and m.group(2) is None
            else m.group(0)
        ),
        page,
    )
    page = number(page, "eqtag", "eq", "tag")
    page = number(page, "figure", "fig", "Figure")
    page = number(page, "table", "tab", "Table")
    page = cite(page, references)
    leftover = MARKER.search(page)
    if leftover:
        raise BuildError(f"unresolved marker {leftover.group(0)}")
    return page


def build(root: Path, out: Path) -> None:
    page = render_page(root)
    shutil.rmtree(out, ignore_errors=True)
    (out / "assets").mkdir(parents=True)
    (out / "index.html").write_text(page)
    for name in ("style.css", "app.js"):
        shutil.copy(root / "site" / name, out / name)
    for name in ("logo.svg", "logo.gif"):
        shutil.copy(root / "assets" / name, out / "assets" / name)
    (out / ".nojekyll").touch()


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "_site")
    try:
        build(ROOT, out)
    except BuildError as error:
        print(f"site build failed: {error}", file=sys.stderr)
        return 1
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
