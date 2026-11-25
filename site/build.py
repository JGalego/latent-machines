"""Build the GitHub Pages site from a template, references, and run records.

``site/index.html`` is a template. The build resolves its markers and writes
static HTML, so every number on the page comes from a source file:

    {{cite:a,b}}                   citations, numbered by first use
    {{references}}                 the cited entries of site/references.json
    {{eqtag:name}}, {{eq:name}}    equation numbers: \\tag{} and "(n)"
    {{figure:name}}, {{fig:name}}  "Figure n" in captions and in prose
    {{table:name}}, {{tab:name}}   "Table n" in captions and in prose
    {{results}}                    tables generated from results/<EXP>/<run>/
    {{toc}}                        contents list of the h2 and h3 headings after it
    {{abstract_result}}, {{conformance_summary}}, {{experiment_status}}
    {{length_charts}}, {{feedback_grid}}  figures drawn from E02 and E03 run records
    {{ledger}}                     hypothesis table from site/hypotheses.json

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
EXPERIMENTS = [f"E{n:02d}" for n in range(10)]
TITLES = {
    "E00": "memory conformance",
    "E01": "exact streaming machines",
    "E02": "learned interface comparison",
    "E03": "latent ticks and explicit events",
    "E04": "dataflow, specialization, and conditional units",
    "E05": "graph, semiring, and logic execution",
    "E06": "spatial and structured dynamical machines",
    "E07": "optimization-defined state updates",
    "E08": "heuristic search with an exact verifier",
    "E09": "state in the machine, not the context",
}
SPLITS = {
    "test_in_distribution": "2-8 statements",
    "test_length_32": "32 statements",
    "test_length_64": "64 statements",
    "test_length_256": "256 statements",
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


BADGE_CLASSES = {
    "conforms with residue": "residue",
    "within tolerance": "within",
    "exceeds tolerance": "exceeds",
}


def badge(label: str, in_scope: bool = True) -> str:
    cls = BADGE_CLASSES.get(label, label) if in_scope else "probe"
    mark = "" if in_scope else "†"
    return f'<span class="badge {esc(cls)}">{esc(label)}{mark}</span>'


def in_scope(variant: dict) -> bool:
    """False for variants a run declared outside its acceptance in advance:
    precision probes (E04) and inputs past a declared numerical limit (E08)."""
    return not variant.get("precision_probe") and variant.get(
        "within_declared_limits", True
    )


def approximate(variant: dict) -> bool:
    """True for variants checked against a declared tolerance, not exactly."""
    return variant.get("contract") == "approximate" or variant.get(
        "classification"
    ) in ("within tolerance", "exceeds tolerance")


def branches(runs: list[dict]) -> list[tuple[str | None, list[dict]]]:
    """Runs grouped by ``config.branch`` (E06 keeps exact and numerical apart)."""
    groups: dict[str | None, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["config"].get("branch"), []).append(r)
    return sorted(groups.items(), key=lambda item: item[0] or "")


def row_key(variant: dict) -> str:
    """The variant's name without its number format, which is the column."""
    parts = variant["variant"].split("/")
    if variant.get("dtype") in parts:
        parts.remove(variant["dtype"])
    return "/".join(parts)


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
    return "\n".join(
        render_branch(exp_id, branch, group) for branch, group in branches(runs)
    )


def snapshot_cell(variants: list[dict]) -> str:
    checked = [v for v in variants if v["snapshot_resume"] != "not applicable"]
    agree = sum(v["snapshot_resume"] == "agrees" for v in checked)
    return f"{agree}/{len(checked)}" if checked else "–"


def render_branch(exp_id: str, branch: str | None, runs: list[dict]) -> str:
    runs = sorted(
        runs, key=lambda r: (r["config"].get("horizon", 0), r["config"].get("seed", 0))
    )
    name = f"{exp_id.lower()}-{branch}" if branch else exp_id.lower()
    label = f"{exp_id} {branch} branch" if branch else exp_id
    tolerance = all(approximate(v) for r in runs for v in r["metrics"]["variants"])
    total = sum(r["metrics"]["totals"]["actions_checked"] for r in runs)
    probes = False
    rows = []
    for r in runs:
        variants = r["metrics"]["variants"]
        scoped = [v for v in variants if in_scope(v)]
        outside = sum(v.get("mismatches", 0) for v in variants if not in_scope(v))
        probes = probes or len(scoped) < len(variants)
        if tolerance:
            within = sum(v["classification"] == "within tolerance" for v in variants)
            checks = f'<td class="num">{within}/{len(variants)}</td>'
        else:
            mismatches = f"{sum(v.get('mismatches', 0) for v in scoped):,}"
            if outside:
                mismatches += f' <span class="range">(+{outside:,}†)</span>'
            residue = max((v.get("max_residue", 0) for v in scoped), default=0)
            checks = f'<td class="num">{mismatches}</td><td class="num">{residue}</td>'
        rows.append(
            "<tr>"
            f"<td>{superscript_power(r['config'].get('horizon', 0))}</td>"
            f'<td class="num">{esc(r["config"].get("seed", ""))}</td>'
            f'<td class="num">{r["metrics"]["totals"]["actions_checked"]:,}</td>'
            f"{checks}"
            f'<td class="num">{snapshot_cell(variants)}</td>'
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
                f"<td>{badge(match[0]['classification'], in_scope(match[0]))}</td>"
                if match
                else "<td></td>"
            )
        cells.append(f'<tr><th scope="row">{esc(key)}</th>{"".join(tds)}</tr>')
    unit = variants[0].get("unit")
    detail = f"{total:,} {esc(unit) + 's' if unit else 'actions'} were checked in {len(runs)} run{'s' if len(runs) != 1 else ''}."
    if branch:
        detail = f"{branch.capitalize()} branch: {detail}"
    head = "".join(f"<th>{esc(d)}</th>" for d in dtypes)
    if tolerance:
        columns = "<th>Within tolerance</th>"
        legend = "Each variant is checked against a reference under a tolerance declared before the run, not for exact equality."
    else:
        columns = "<th>Mismatches</th><th>Max residue</th>"
        legend = "Mismatches count nonconforming steps; residue is Equation&nbsp;{{eq:residue}}."
    dagger = (
        " † marks variants declared outside the acceptance before the run (precision probes or inputs past a declared numerical limit); their mismatches are reported but not counted against the run."
        if probes
        else ""
    )
    explorer_branch = f' data-branch="{esc(branch)}"' if branch else ""
    return f"""{intro(exp_id, runs, detail)}
    <p class="caption"><strong>{{{{table:{name}-runs}}}}:</strong> {esc(label)} runs. {legend} A snapshot agrees if the restored state equals the live state; variants without a snapshot step are not counted.{dagger}</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Checks per configuration</th><th>Seed</th><th>Checked</th>{columns}<th>Snapshots agree</th><th>Status</th><th>CPU</th><th>Record</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
    <p class="caption"><strong>{{{{table:{name}-configs}}}}:</strong> Classification of each {esc(label)} configuration. The table shows the run with {superscript_power(largest["config"].get("horizon", 0))} checks per configuration; with scripts enabled, choose another run and select a cell for its details.{dagger}</p>
    <div class="explorer" data-experiment="{esc(exp_id)}"{explorer_branch}>
      <div class="table-wrap"><table class="grid-matrix">
        <thead><tr><th></th>{head}</tr></thead>
        <tbody>{"".join(cells)}</tbody>
      </table></div>
    </div>{render_ram(name, label, largest)}{render_search(exp_id, largest)}"""


RAM_BADGES = {"deviates": "mismatch", "not applicable": "probe"}


def render_ram(name: str, label: str, run: dict) -> str:
    """E07: whether each optimizer-defined memory still behaves as memory/v1."""
    variants = [v for v in run["metrics"]["variants"] if "ram_conformance" in v]
    if not variants:
        return ""
    dtypes = list(dict.fromkeys(v["dtype"] for v in variants))
    keys = list(dict.fromkeys(row_key(v) for v in variants))
    rows = []
    for key in keys:
        tds = []
        for dtype in dtypes:
            match = [v for v in variants if row_key(v) == key and v["dtype"] == dtype]
            if not match:
                tds.append("<td></td>")
                continue
            ram = match[0]["ram_conformance"]
            cls = RAM_BADGES.get(ram, BADGE_CLASSES.get(ram, ram))
            tds.append(f'<td><span class="badge {esc(cls)}">{esc(ram)}</span></td>')
        rows.append(f'<tr><th scope="row">{esc(key)}</th>{"".join(tds)}</tr>')
    head = "".join(f"<th>{esc(d)}</th>" for d in dtypes)
    return f"""
    <p class="caption"><strong>{{{{table:{name}-ram}}}}:</strong> The same {esc(label)} configurations compared with the integer memory oracle. A deviation from RAM is a finding about the update rule, not a failure of the implementation, which is judged against its own reference above.</p>
    <div class="table-wrap"><table class="grid-matrix">
      <thead><tr><th></th>{head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>"""


def render_search(exp_id: str, run: dict) -> str:
    """E08's search arms: outcomes per arm and instance class."""
    search = run["metrics"].get("search")
    if not search:
        return ""
    classes = list(dict.fromkeys(s["class"] for s in search))
    arms = list(dict.fromkeys(s["arm"] for s in search))
    rows = []
    for arm in arms:
        tds = []
        for cls in classes:
            s = next((x for x in search if x["arm"] == arm and x["class"] == cls), None)
            if s is None:
                tds.append("<td></td>")
                continue
            text = f"{s['solved']}/{s['instances']}"
            if s.get("proved_unsat"):
                text += f" · {s['proved_unsat']} unsat"
            tds.append(f'<td class="num">{text}</td>')
        rows.append(f'<tr><th scope="row">{esc(arm)}</th>{"".join(tds)}</tr>')
    head = "".join(f"<th>{esc(c)}</th>" for c in classes)
    return f"""
    <p class="caption"><strong>{{{{table:{exp_id.lower()}-search}}}}:</strong> {esc(exp_id)} search arms at a fixed step budget: instances with a verified solution, and instances proved unsatisfiable. Only the complete solver can prove unsatisfiability; every other failure is unknown, not unsatisfiable.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Arm</th>{head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>"""


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
        present = {s["arm"] for s in summary}
        order = [a for a in r["config"].get("arms", []) if a in present]
        arms = order + sorted(present - set(order))
        seeds = max((s["seeds"] for s in summary), default=0)
        mismatches = m.get("executor_mismatches")
        detail = (
            f'Run <a href="{esc(r["manifest"])}">{esc(r["run_id"].split("-")[-1])}</a> '
            f"has status {badge(m['status'])} with {seeds} training seed{'s' if seeds != 1 else ''} "
            f"and {mismatches} executor mismatches."
        )
        rows = []
        metrics = [("task_accuracy", "Task"), ("action_accuracy", "Action")]
        if any("copy_fidelity" in s for s in summary):
            metrics.append(("copy_fidelity", "Copy"))
        notes = []
        if "A" in arms:
            notes.append(
                "Action accuracy is undefined for arm A, which takes no actions."
            )
        if len(metrics) > 2:
            notes.append(
                "Copy fidelity is the fraction of copies whose read and write "
                "targeted the right records."
            )
        note = "".join(f" {n}" for n in notes)
        for metric, label in metrics:
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
    <p class="caption"><strong>{{{{table:{name}}}}}:</strong> {esc(exp_id)} accuracy by arm and test split: mean over training seeds, with the range in brackets.{note}</p>
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


# ---------- story figures ----------

ARM_NAMES = {
    "E02": {
        "A": "no interface",
        "B": "text tool",
        "C": "typed interface",
        "D": "typed, delta-rule",
    },
    "E03": {
        "T": "text feedback",
        "L": "latent feedback",
        "S": "steering control",
        "H": "heads only",
        "N": "native loop",
    },
}
CHART_TITLES = {"E02": "E02: record updates", "E03": "E03: record updates with copies"}
LENGTH_SPLITS = {
    "test_in_distribution": "2–8",
    "test_length_32": "32",
    "test_length_64": "64",
}
VERDICTS = ("supported", "partly", "refuted", "null", "untested")


def latest_learned(experiments: dict[str, list[dict]], exp_id: str) -> dict | None:
    runs = [r for r in experiments.get(exp_id, []) if r["kind"] == "learned"]
    return runs[-1] if runs else None


def task_stats(run: dict) -> dict[str, dict[str, dict]]:
    """Task accuracy statistics as {arm: {split: {mean, min, max}}} for length splits."""
    stats: dict[str, dict[str, dict]] = {}
    for s in run["metrics"]["summary"]:
        if s.get("split") in LENGTH_SPLITS and s.get("task_accuracy"):
            stats.setdefault(s["arm"], {})[s["split"]] = s["task_accuracy"]
    return stats


def length_chart(exp_id: str, run: dict) -> str:
    """Task accuracy against program length, one labelled line per distinct arm."""
    stats = task_stats(run)
    order = [a for a in run["config"].get("arms", []) if a in stats] or sorted(stats)
    splits = [
        sp for sp in LENGTH_SPLITS if order and all(sp in stats[a] for a in order)
    ]
    if len(splits) < 2:
        return ""
    groups: list[dict] = []
    for arm in order:
        means = tuple(round(stats[arm][sp]["mean"], 6) for sp in splits)
        match = next((g for g in groups if g["means"] == means), None)
        if match:
            match["arms"].append(arm)
        else:
            groups.append({"arms": [arm], "means": means})
    width, height, left, right, top, bottom = 640, 320, 44, 190, 14, 46
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(i: int) -> float:
        return left + i * plot_w / (len(splits) - 1)

    def py(value: float) -> float:
        return top + (1 - value) * plot_h

    parts = []
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{py(tick):.1f}" y2="{py(tick):.1f}"/>'
            f'<text class="tick" x="{left - 6}" y="{py(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>'
        )
    for i, sp in enumerate(splits):
        parts.append(
            f'<text class="tick" x="{px(i):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{LENGTH_SPLITS[sp]}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        "statements per program (categorical; training used 2–8)</text>"
        f'<text class="tick" transform="translate(12 {top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle">task accuracy</text>'
    )
    gap, last_y = 14.0, -1e9
    for index, g in sorted(
        enumerate(groups), key=lambda item: py(item[1]["means"][-1])
    ):
        ys = [py(m) for m in g["means"]]
        g["label_y"] = last_y = max(ys[-1], last_y + gap)
        g["index"] = index
    for g in groups:
        arm = g["arms"][0]
        name = ARM_NAMES.get(exp_id, {}).get(arm, "")
        label = f"{', '.join(g['arms'])} {name} {g['means'][-1]:.3f}".replace("  ", " ")
        points = " ".join(f"{px(i):.1f},{py(m):.1f}" for i, m in enumerate(g["means"]))
        marks = []
        for i, sp in enumerate(splits):
            lo, hi = stats[arm][sp]["min"], stats[arm][sp]["max"]
            marks.append(
                f'<line class="whisker" x1="{px(i):.1f}" x2="{px(i):.1f}" y1="{py(lo):.1f}" y2="{py(hi):.1f}"/>'
                f'<circle cx="{px(i):.1f}" cy="{py(g["means"][i]):.1f}" r="3.5"/>'
            )
        parts.append(
            f'<g class="series s{g["index"] % 5}"><polyline points="{points}"/>{"".join(marks)}'
            f'<text x="{px(len(splits) - 1) + 10:.1f}" y="{g["label_y"] + 4:.1f}">{esc(label)}</text></g>'
        )
    arms_text = "; ".join(
        f"{', '.join(g['arms'])}: {g['means'][-1]:.2f} at the longest length"
        for g in groups
    )
    return (
        f'<svg class="length-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Task accuracy by program length. {esc(arms_text)}">{"".join(parts)}</svg>'
    )


def story_length_figure(experiments: dict[str, list[dict]]) -> str:
    charts, mismatches, seeds = [], [], 0
    for exp_id in ("E02", "E03"):
        run = latest_learned(experiments, exp_id)
        svg = length_chart(exp_id, run) if run else ""
        if svg:
            charts.append(
                f'<div class="chart-title">{esc(CHART_TITLES[exp_id])}</div>{svg}'
            )
            mismatches.append(run["metrics"].get("executor_mismatches"))
            seeds = max(
                seeds, max(s.get("seeds", 0) for s in run["metrics"]["summary"])
            )
    if not charts:
        return ""
    cause = (
        " The delta-rule executor matched its oracle on every action of every arm, so the drop comes from the controller's chosen actions, not from the machine."
        if all(m == 0 for m in mismatches)
        else ""
    )
    return (
        '<figure class="figure story-figure">'
        + "".join(charts)
        + '<figcaption class="caption"><strong>{{figure:story-length}}:</strong> '
        f"Task accuracy by program length, mean over {seeds} training seeds with the seed range as a whisker. "
        "Arms with identical results share a line." + cause + "</figcaption></figure>"
    )


def feedback_grid(experiments: dict[str, list[dict]]) -> str:
    """E03's four arms as a 2x2: trained input positions against feedback in the model."""
    run = latest_learned(experiments, "E03")
    if not run:
        return ""
    stats = task_stats(run)
    if not {"H", "S", "T", "L"} <= stats.keys() or any(
        sp not in stats[a]
        for a in "HSTL"
        for sp in ("test_length_32", "test_length_64")
    ):
        return ""

    def value(arm: str, split: str = "test_length_32") -> float:
        return stats[arm][split]["mean"]

    def cell(arm: str) -> str:
        name = ARM_NAMES["E03"][arm]
        return (
            f'<td><span class="big">{value(arm):.3f}</span>'
            f'<span class="small">{arm} · {esc(name)}</span>'
            f'<span class="small">64 statements: {value(arm, "test_length_64"):.3f}</span></td>'
        )

    row_effect = (value("S") + value("L") - value("H") - value("T")) / 2
    column_effect = (value("T") + value("L") - value("H") - value("S")) / 2
    names = ARM_NAMES["E03"]
    return f"""<figure class="figure story-figure">
      <div class="table-wrap"><table class="matrix2">
        <thead><tr><th></th><th>Feedback stays outside the model</th><th>Feedback re-enters the model</th></tr></thead>
        <tbody>
          <tr><th scope="row">Trained input positions</th>{cell("S")}{cell("L")}</tr>
          <tr><th scope="row">No trained positions</th>{cell("H")}{cell("T")}</tr>
        </tbody>
      </table></div>
      <figcaption class="caption"><strong>{{{{figure:story-grid}}}}:</strong> E03 task accuracy at 32 statements, mean over seeds. Averaging the cells, trained positions change accuracy by {row_effect:+.2f} and feedback re-entering the model by {column_effect:+.2f}. In {esc(names["T"])} the feedback is ordinary text tokens; in {esc(names["L"])} it is one trained vector per step. Latent arms use more model positions than the others, so work is not matched. One frozen 135M model, one task.</figcaption>
    </figure>"""


def render_ledger(root: Path) -> str:
    path = root / "site" / "hypotheses.json"
    if not path.exists():
        return ""
    rows = []
    for entry in json.loads(path.read_text()):
        if entry["verdict"] not in VERDICTS:
            raise BuildError(
                f"hypothesis verdict {entry['verdict']!r} is not one of {VERDICTS}"
            )
        links = []
        for label, target in entry["links"]:
            if not (root / target.split("#")[0]).exists():
                raise BuildError(f"hypotheses.json links to missing file {target!r}")
            links.append(f'<a href="{REPO}/{esc(target)}">{esc(label)}</a>')
        rows.append(
            f"<tr><td>{esc(entry['claim'])}"
            f'<span class="small">{esc(entry["note"])}</span></td>'
            f'<td><span class="verdict {entry["verdict"]}">{esc(entry["label"])}</span></td>'
            f'<td class="links">{" ".join(links)}</td></tr>'
        )
    return f"""<figure class="figure story-figure">
      <div class="table-wrap"><table class="ledger">
        <thead><tr><th>Hypothesis</th><th>Verdict</th><th>Evidence</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <figcaption class="caption"><strong>{{{{figure:story-ledger}}}}:</strong> Each hypothesis with its current verdict. Supported and refuted mean on the tested workloads and models only, with one to three seeds. Untested rows have no result.</figcaption>
    </figure>"""


# ---------- generated status text ----------


def conformance_stats(experiments: dict[str, list[dict]]) -> dict:
    """Counts over exact-contract variants inside their declared acceptance."""
    runs = [r for rs in experiments.values() for r in rs if r["kind"] == "conformance"]
    scoped = {
        r["run_id"]: [
            v for v in r["metrics"]["variants"] if in_scope(v) and not approximate(v)
        ]
        for r in runs
    }
    runs = [r for r in runs if scoped[r["run_id"]]]
    variants = [v for r in runs for v in scoped[r["run_id"]]]
    return {
        "runs": len(runs),
        "configurations": len(variants),
        "exact": sum(v["classification"] == "exact" for v in variants),
        "actions": sum(v["actions_checked"] for v in variants),
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
            f"All {stats['configurations']} configurations checked for exact equality in the "
            f"{stats['runs']} recorded conformance runs are classified as exact: no step failed "
            "to conform, the residue was zero throughout, and every snapshot restored the live "
            "state exactly."
        )
    return (
        f"{stats['exact']} of the {stats['configurations']} configurations checked for exact "
        f"equality in the {stats['runs']} recorded conformance runs are classified as exact; "
        "the tables identify the others. These counts leave out variants declared outside "
        "the acceptance before a run and variants checked against a tolerance."
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
    if len(rest) == 2:
        sentences.append(f"{rest[1]} is planned")
    elif len(rest) > 2:
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
                "branch": r["config"].get("branch"),
                "manifest": r["manifest"],
                "variants": [
                    {
                        key: v.get(key)
                        for key in (
                            "variant", "dtype", "workload", "machine", "classification",
                            "snapshot_resume", "actions_checked", "max_residue",
                            "end_to_end_seconds", "action_sha256", "unit",
                        )
                    }
                    | {"in_scope": in_scope(v)}
                    for v in r["metrics"].get("variants", [])
                ]
                if r["kind"] == "conformance"
                else [],
                "summary": r["metrics"].get("summary", []) if r["kind"] == "learned" else [],
                "arms": r["config"].get("arms", []),
            }
            for r in runs
        ]
        for exp_id, runs in experiments.items()
    }  # fmt: skip
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


HEADING = re.compile(r"<h([23])((?:\s[^>]*)?)>(.*?)</h\1>", re.S)


def slug(text: str) -> str:
    words = re.sub(r"<[^>]+>|&\w+;", " ", text)
    words = re.sub(r"^\s*[\d.]+\s*", "", html.unescape(words)).lower()
    return re.sub(r"[^a-z0-9]+", "-", words).strip("-")


def table_of_contents(page: str) -> str:
    """Give each h2 and h3 after the marker an id, and list them where {{toc}} is."""
    marker = page.index("{{toc}}")
    before, after = page[:marker], page[marker:]
    existing = set(re.findall(r'\sid="([^"]+)"', page))
    entries, used = [], set()

    def anchor(match: re.Match) -> str:
        level, attrs, inner = match.groups()
        own = re.search(r'id="([^"]+)"', attrs)
        # A heading that opens a section with an id links to that section.
        section = re.search(r'<section id="([^"]+)"[^>]*>\s*$', after[: match.start()])
        if own or section:
            name, tag = (own or section).group(1), match.group(0)
        else:
            name = slug(inner)
            if name in existing:
                raise BuildError(f"heading anchor {name!r} clashes with an existing id")
            tag = f'<h{level}{attrs} id="{name}">{inner}</h{level}>'
        if name in used:
            raise BuildError(f"duplicate heading anchor {name!r}")
        used.add(name)
        entries.append((int(level), name, inner.strip()))
        return tag

    after = HEADING.sub(anchor, after)
    items, open_sub = [], False
    for level, name, inner in entries:
        if level == 2:
            if open_sub:
                items.append("</ol></li>")
                open_sub = False
            elif items:
                items.append("</li>")
            items.append(f'<li><a href="#{name}">{inner}</a>')
        else:
            if not open_sub:
                items.append("<ol>")
                open_sub = True
            items.append(f'<li><a href="#{name}">{inner}</a></li>')
    items.append("</ol></li>" if open_sub else "</li>")
    toc = (
        '<nav class="toc" aria-labelledby="toc-title"><h2 id="toc-title">Contents</h2>'
        f"<ol>{''.join(items)}</ol></nav>"
    )
    return before + after.replace("{{toc}}", toc, 1)


DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)


def wrap_equations(page: str) -> str:
    """Wrap display math with its LaTeX source, without the generated tag, for copying."""

    def wrap(match: re.Match) -> str:
        tex = re.sub(r"\\tag\{[^{}]*\}", "", match.group(1))
        tex = " ".join(tex.split())
        return f'<div class="equation" data-tex="{esc(tex)}">{match.group(0)}</div>'

    # Only the article: scripts outside <main> configure KaTeX with "$$" delimiters.
    start, end = page.find("<main>"), page.find("</main>")
    if start < 0 or end < start:
        return DISPLAY_MATH.sub(wrap, page)
    return page[:start] + DISPLAY_MATH.sub(wrap, page[start:end]) + page[end:]


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
        "length_charts": story_length_figure(experiments),
        "feedback_grid": feedback_grid(experiments),
        "ledger": render_ledger(root),
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
    if "{{toc}}" in page:
        page = table_of_contents(page)
    page = wrap_equations(page)
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
