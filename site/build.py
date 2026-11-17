"""Build the GitHub Pages site into an output directory.

Copies the static page and logo, and summarizes every committed run record
under ``results/<experiment>/<run>/`` into ``data/results.json``, so the page
shows exactly what the records say.

    python3 site/build.py _site
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/JGalego/latent-machines/blob/main"
TITLES = {
    "E00": "memory conformance",
    "E01": "exact streaming machines",
}
VARIANT_FIELDS = (
    "variant",
    "dtype",
    "workload",
    "machine",
    "classification",
    "snapshot_resume",
    "actions_checked",
    "max_residue",
)


def summarize(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    variants = [
        {key: v[key] for key in VARIANT_FIELDS if key in v} for v in metrics["variants"]
    ]
    return {
        "run_id": run["run_id"],
        "revision": run["source"]["revision"],
        "horizon": run["config"]["horizon"],
        "seed": run["config"]["seed"],
        "cpu": run["environment"].get("cpu"),
        "status": metrics["status"],
        "totals": metrics["totals"],
        "snapshots_agree": sum(v["snapshot_resume"] == "agrees" for v in variants),
        "variants": variants,
        "manifest": f"{REPO}/{(run_dir / 'metrics.json').relative_to(ROOT)}",
    }


def main(out: str) -> None:
    target = Path(out)
    shutil.rmtree(target, ignore_errors=True)
    (target / "assets").mkdir(parents=True)
    (target / "data").mkdir()
    for name in ("index.html", "style.css", "app.js"):
        shutil.copy(ROOT / "site" / name, target / name)
    for name in ("logo.svg", "logo.gif"):
        shutil.copy(ROOT / "assets" / name, target / "assets" / name)
    (target / ".nojekyll").touch()

    experiments: dict[str, dict] = {}
    for metrics in sorted(ROOT.glob("results/*/*/metrics.json")):
        experiment = metrics.parent.parent.name
        report = metrics.parent.parent / "README.md"
        entry = experiments.setdefault(
            experiment,
            {
                "title": TITLES.get(experiment, ""),
                "report": f"{REPO}/{report.relative_to(ROOT)}"
                if report.exists()
                else None,
                "runs": [],
            },
        )
        entry["runs"].append(summarize(metrics.parent))
    (target / "data" / "results.json").write_text(
        json.dumps({"experiments": experiments}, indent=1) + "\n"
    )
    runs = sum(len(e["runs"]) for e in experiments.values())
    print(f"built {target}: {len(experiments)} experiments, {runs} runs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
