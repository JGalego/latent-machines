"""The site builder: markers, numbering, references, and record rendering."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def make_root(tmp_path: Path, template: str, references: dict[str, str]) -> Path:
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "index.html").write_text(template)
    (tmp_path / "site" / "references.json").write_text(json.dumps(references))
    return tmp_path


def add_run(
    root: Path, experiment: str, run_id: str, metrics: dict, config: dict
) -> None:
    run_dir = root / "results" / experiment / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    run = {"run_id": run_id, "source": {"revision": "abc1234def"}, "config": config}
    (run_dir / "run.json").write_text(json.dumps(run))


def conformance(classification: str = "exact") -> dict:
    variant = {
        "variant": "delta-eager/float32/overwrite", "dtype": "float32",
        "workload": "overwrite", "classification": classification,
        "snapshot_resume": "agrees", "actions_checked": 1000, "max_residue": 0.0,
    }  # fmt: skip
    totals = {
        "variants": 1,
        "actions_checked": 1000,
        "mismatches": 0,
        "max_residue": 0.0,
    }
    return {"status": "passed", "totals": totals, "variants": [variant]}


def test_real_site_builds_without_leftover_markers(tmp_path: Path) -> None:
    build.build(ROOT, tmp_path / "out")
    page = (tmp_path / "out" / "index.html").read_text()
    assert "{{" not in page
    references = json.loads((ROOT / "site" / "references.json").read_text())
    assert page.count('<li id="ref-') == len(references)
    captions = re.findall(r"<strong>Table (\d+):</strong>", page)
    assert captions == [str(n) for n in range(1, len(captions) + 1)]


def test_numbering_and_citations_follow_page_order(tmp_path: Path) -> None:
    template = (
        "{{cite:b}} {{figure:x}} {{fig:y}} {{figure:y}} {{tab:t}} {{table:t}} "
        "$$a\\tag{{{eqtag:e}}}$$ {{eq:e}} {{cite:a,b}} {{references}}"
    )
    root = make_root(tmp_path, template, {"a": "Entry A", "b": "Entry B"})
    page = build.render_page(root)
    assert page.startswith('[<a href="#ref-b">1</a>] Figure 1 Figure&nbsp;2 Figure 2')
    assert "Table&nbsp;1 Table 1" in page and "\\tag{1}" in page and "(1)" in page
    assert '[<a href="#ref-a">2</a>, <a href="#ref-b">1</a>]' in page
    assert page.index('id="ref-b"') < page.index('id="ref-a"')


@pytest.mark.parametrize(
    ("template", "references", "message"),
    [
        ("{{cite:missing}} {{references}}", {}, "unknown reference"),
        ("{{references}}", {"a": "Entry A"}, "never cited"),
        ("{{fig:nowhere}}", {}, "unknown fig"),
        ("{{table:t}} {{table:t}}", {}, "duplicate"),
        ("{{mystery}}", {}, "unresolved"),
    ],
)
def test_bad_markers_fail_the_build(
    tmp_path: Path, template: str, references: dict, message: str
) -> None:
    root = make_root(tmp_path, template, references)
    with pytest.raises(build.BuildError, match=message):
        build.render_page(root)


def test_each_record_kind_renders(tmp_path: Path) -> None:
    # Conformance captions cite the residue equation, so the template defines it.
    template = "$$r\\tag{{{eqtag:residue}}}$$ {{results}} {{experiment_status}}"
    root = make_root(tmp_path, template, {})
    add_run(
        root, "E00", "e00-x-abc1234-0001", conformance(), {"horizon": 1000, "seed": 0}
    )
    summary = [
        {"arm": arm, "split": "test_in_distribution", "seeds": 3,
         "task_accuracy": {"mean": 0.5, "min": 0.4, "max": 0.6}}
        for arm in "ABCD"
    ]  # fmt: skip
    learned = {
        "status": "completed",
        "executor_mismatches": 0,
        "summary": summary,
        "runs": [],
    }
    add_run(root, "E02", "e02-x-abc1234-0002", learned, {"seeds": [0, 1, 2]})
    add_run(root, "E05", "e05-x-abc1234-0003", {"status": "completed"}, {})
    page = build.render_page(root)
    assert "Table 1" in page and "Table 2" in page and "Table 3" in page
    assert "0.500" in page and "[0.400, 0.600]" in page
    assert "Records without a renderer yet" in page
    assert "E00, E02 and E05 have recorded results; E01 is next" in page


def test_status_text_reports_failures(tmp_path: Path) -> None:
    root = make_root(tmp_path, "{{abstract_result}} | {{conformance_summary}}", {})
    add_run(root, "E00", "e00-a-abc1234-0001", conformance(), {"horizon": 1000})
    add_run(
        root, "E00", "e00-b-abc1234-0002", conformance("mismatch"), {"horizon": 1000}
    )
    page = build.render_page(root)
    assert "1 of 2 tested configurations did not match" in page
    summary = (
        "1 of the 2 configurations checked for exact equality in the 2 recorded "
        "conformance runs are classified"
    )
    assert summary in page


def test_embedded_data_cannot_close_its_script_tag(tmp_path: Path) -> None:
    root = make_root(tmp_path, "{{results_data}}", {})
    metrics = conformance()
    metrics["variants"][0]["workload"] = "</script><script>alert(1)"
    add_run(root, "E00", "e00-x-abc1234-0001", metrics, {"horizon": 1000})
    assert "</script>" not in build.render_page(root)


def test_contents_lists_headings_with_unique_anchors(tmp_path: Path) -> None:
    template = (
        "<h2>Abstract</h2> {{toc}} "
        '<section id="intro"><h2>1&ensp;Introduction</h2>'
        "<h3>1.1&ensp;Scope</h3></section> "
        "<h2>References</h2>"
    )
    page = build.render_page(make_root(tmp_path, template, {}))
    toc = page[page.index('<nav class="toc"') : page.index("</nav>")]
    assert re.findall(r'href="#([^"]+)"', toc) == ["intro", "scope", "references"]
    assert "Abstract" not in toc
    ids = re.findall(r'\sid="([^"]+)"', page)
    assert len(ids) == len(set(ids))


def test_equations_carry_their_latex_without_the_tag(tmp_path: Path) -> None:
    template = "$$ a < b \\tag{{{eqtag:x}}} $$ {{eq:x}}"
    page = build.render_page(make_root(tmp_path, template, {}))
    assert '<div class="equation" data-tex="a &lt; b">' in page
    assert "\\tag{1}" in page and "(1)" in page


def test_math_delimiters_outside_main_are_left_alone(tmp_path: Path) -> None:
    template = (
        "<main>$$ x $$</main><script onload=\"f({left: '$$', right: '$$'})\"></script>"
    )
    page = build.render_page(make_root(tmp_path, template, {}))
    assert page.count('class="equation"') == 1
    assert "{left: '$$', right: '$$'}" in page


def test_probes_and_tolerance_variants_stay_out_of_the_exact_counts(
    tmp_path: Path,
) -> None:
    root = make_root(
        tmp_path, "{{eqtag:residue}} {{abstract_result}} | {{results}}", {}
    )
    probe = conformance("mismatch")
    probe["variants"][0] |= {"precision_probe": True, "mismatches": 7}
    probe["variants"].append(conformance()["variants"][0] | {"dtype": "float64"})
    add_run(root, "E04", "e04-a-abc1234-0001", probe, {"horizon": 1000})
    approx = conformance("exceeds tolerance")
    approx["variants"][0]["contract"] = "approximate"
    add_run(
        root,
        "E06",
        "e06-a-abc1234-0002",
        approx,
        {"horizon": 10, "branch": "numerical"},
    )
    page = build.render_page(root)
    assert "matched their oracles with zero residue on every one of the 1,000" in page
    assert "(+7†)" in page
    assert 'class="badge probe">mismatch†' in page
    assert "Numerical branch" in page and "Within tolerance" in page
    assert 'data-branch="numerical"' in page


def test_search_arms_render_as_a_table(tmp_path: Path) -> None:
    root = make_root(tmp_path, "{{eqtag:residue}} {{results}}", {})
    metrics = conformance()
    base = {"class": "small", "instances": 4, "solved": 2}
    metrics["search"] = [
        base | {"arm": "dpll", "proved_unsat": 2},
        base | {"arm": "walksat", "proved_unsat": 0},
    ]
    add_run(root, "E08", "e08-a-abc1234-0001", metrics, {"horizon": 10})
    page = build.render_page(root)
    assert "2/4 · 2 unsat" in page
    assert "search arms at a fixed step budget" in page
