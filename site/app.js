"use strict";

// ---------- Interactive results, from data embedded at build time ----------

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtInt = (n) => Number(n).toLocaleString("en-US");
const power = (h) => {
  const exp = Math.log10(h);
  return Number.isInteger(exp) ? `10${[...String(exp)].map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]).join("")}` : fmtInt(h);
};
const SPLITS = {
  test_in_distribution: "2–8 statements",
  test_length_32: "32 statements",
  test_length_64: "64 statements",
  test_held_out_pairs: "held-out pairs",
};

function badge(label) {
  const cls = label === "conforms with residue" ? "residue" : label;
  return `<span class="badge ${esc(cls)}">${esc(label)}</span>`;
}

function segmented(options, selected, onChange, label) {
  const group = document.createElement("div");
  group.className = "segmented";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", label);
  for (const [value, text] of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quiet";
    button.textContent = text;
    button.setAttribute("aria-pressed", String(value === selected));
    button.onclick = () => {
      for (const b of group.children) b.setAttribute("aria-pressed", String(b === button));
      onChange(value);
    };
    group.append(button);
  }
  return group;
}

const rowKey = (v) => (v.machine ? `${v.machine} · ${v.workload}` : v.workload);

function conformanceExplorer(root, runs) {
  const sorted = [...runs].sort((a, b) => a.horizon - b.horizon || a.seed - b.seed);
  const tableHost = root.querySelector(".table-wrap");
  const detail = document.createElement("div");
  detail.className = "detail muted";
  detail.textContent = "Select a cell for details.";

  function show(run) {
    const dtypes = [...new Set(run.variants.map((v) => v.dtype))];
    const keys = [...new Set(run.variants.map(rowKey))];
    const body = keys
      .map((key) => {
        const cells = dtypes.map((dtype) => {
          const index = run.variants.findIndex((v) => rowKey(v) === key && v.dtype === dtype);
          if (index < 0) return "<td></td>";
          const v = run.variants[index];
          return `<td><button type="button" class="cell-button" data-index="${index}" title="${esc(v.variant)}">${badge(v.classification)}</button></td>`;
        });
        return `<tr><th scope="row">${esc(key)}</th>${cells.join("")}</tr>`;
      })
      .join("");
    tableHost.innerHTML = `<table class="grid-matrix"><thead><tr><th></th>${dtypes.map((d) => `<th>${esc(d)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
    for (const button of tableHost.querySelectorAll(".cell-button")) {
      button.onclick = () => {
        const v = run.variants[Number(button.dataset.index)];
        detail.className = "detail";
        detail.innerHTML = `<strong>${esc(v.variant)}</strong> ${badge(v.classification)}
          <dl>
            <dt>Actions checked</dt><dd>${fmtInt(v.actions_checked)}</dd>
            <dt>Max residue</dt><dd>${esc(v.max_residue)}</dd>
            <dt>Snapshot</dt><dd>${esc(v.snapshot_resume)}</dd>
            <dt>End-to-end time</dt><dd>${esc(v.end_to_end_seconds)} s</dd>
            <dt>Input checksum</dt><dd><code>${esc(String(v.action_sha256 || "").slice(0, 16))}…</code></dd>
            <dt>Run</dt><dd><a href="${esc(run.manifest)}">${esc(run.run_id)}</a></dd>
          </dl>`;
      };
    }
  }

  const options = sorted.map((run, i) => [i, `${power(run.horizon)} · seed ${run.seed}`]);
  const last = sorted.length - 1;
  root.prepend(segmented(options, last, (i) => show(sorted[i]), "Run"));
  root.append(detail);
  show(sorted[last]);
}

// Bars of mean accuracy per arm, with whiskers spanning the range over seeds.
function accuracyChart(summary, split, metric) {
  const rows = summary.filter((s) => s.split === split && s[metric]).sort((a, b) => a.arm.localeCompare(b.arm));
  if (!rows.length) return '<p class="muted">No values for this metric and split.</p>';
  const width = 520, height = 220, left = 40, bottom = 28, top = 12;
  const plotH = height - bottom - top;
  const band = (width - left) / rows.length;
  const barW = Math.min(44, band * 0.5);
  const y = (v) => top + plotH * (1 - v);
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((t) => `<line class="grid" x1="${left}" x2="${width}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${left - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`)
    .join("");
  const bars = rows
    .map((s, i) => {
      const stat = s[metric];
      const cx = left + band * (i + 0.5);
      const h = Math.max(0, y(0) - y(stat.mean));
      const tip = `Arm ${s.arm}: mean ${stat.mean.toFixed(3)}, range ${stat.min.toFixed(3)}–${stat.max.toFixed(3)} over ${s.seeds} seed${s.seeds === 1 ? "" : "s"}`;
      return `<g class="bar-group"><title>${esc(tip)}</title>
        <rect class="hit" x="${cx - band / 2}" y="${top}" width="${band}" height="${plotH}"/>
        ${h >= 4
          ? `<path class="bar" d="M${cx - barW / 2},${y(0)} V${y(stat.mean) + 4} q0,-4 4,-4 H${cx + barW / 2 - 4} q4,0 4,4 V${y(0)} Z"/>`
          : `<rect class="bar" x="${cx - barW / 2}" y="${y(stat.mean)}" width="${barW}" height="${h}"/>`}
        <line class="whisker" x1="${cx}" x2="${cx}" y1="${y(stat.max)}" y2="${y(stat.min)}"/>
        <line class="whisker" x1="${cx - 6}" x2="${cx + 6}" y1="${y(stat.max)}" y2="${y(stat.max)}"/>
        <line class="whisker" x1="${cx - 6}" x2="${cx + 6}" y1="${y(stat.min)}" y2="${y(stat.min)}"/>
        <text class="value" x="${cx}" y="${y(stat.max) - 6}" text-anchor="middle">${stat.mean.toFixed(2)}</text>
        <text class="tick" x="${cx}" y="${height - 8}" text-anchor="middle">Arm ${esc(s.arm)}</text></g>`;
    })
    .join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(metric.replace("_", " "))} by arm">${grid}<line class="axis" x1="${left}" x2="${width}" y1="${y(0)}" y2="${y(0)}"/>${bars}</svg>`;
}

function learnedExplorer(root, run) {
  const seen = [...new Set(run.summary.map((s) => s.split))];
  const splits = [...Object.keys(SPLITS).filter((s) => seen.includes(s)), ...seen.filter((s) => !(s in SPLITS))];
  let split = splits[0];
  let metric = "task_accuracy";
  const chart = document.createElement("div");
  chart.className = "chart-host";
  const draw = () => (chart.innerHTML = accuracyChart(run.summary, split, metric));
  const controls = document.createElement("div");
  controls.className = "controls";
  controls.append(
    segmented(splits.map((s) => [s, SPLITS[s] || s]), split, (s) => ((split = s), draw()), "Test split"),
    segmented([["task_accuracy", "Task accuracy"], ["action_accuracy", "Action accuracy"]], metric, (m) => ((metric = m), draw()), "Metric")
  );
  root.prepend(controls, chart);
  draw();
}

function setupResults() {
  const source = document.getElementById("results-data");
  if (!source) return;
  const data = JSON.parse(source.textContent);
  for (const root of document.querySelectorAll(".explorer")) {
    const runs = (data[root.dataset.experiment] || []);
    if (root.dataset.run) {
      const run = runs.find((r) => r.run_id === root.dataset.run);
      if (run && run.summary.length) learnedExplorer(root, run);
    } else {
      const conformance = runs.filter((r) => r.kind === "conformance");
      if (conformance.length) conformanceExplorer(root, conformance);
    }
  }
}

// ---------- Playground: the delta rule in float32 next to an exact oracle ----------

const f32 = Math.fround;
const SLOTS = 8;
const BITS = 8;
let S;
let oracle;

const $ = (id) => document.getElementById(id);
const zeros = (n) => new Array(n).fill(0);
const oneHot = (index) => zeros(SLOTS).map((_, j) => (j === index ? 1 : 0));
const bitsOf = (word) => zeros(BITS).map((_, i) => (word >> i) & 1);

function normalize(vector) {
  let sum = 0;
  for (const x of vector) sum = f32(sum + f32(x * x));
  if ($("pg-norm").checked) {
    const scale = f32(1 / f32(Math.sqrt(f32(sum + 1e-6)))); // eps inside the square root
    return vector.map((x) => f32(x * scale));
  }
  const norm = Math.max(f32(Math.sqrt(sum)), 1e-6); // x / max(|x|, eps)
  return vector.map((x) => f32(x / norm));
}

// S <- alpha * (S - beta (S k) k^T) + beta v k^T, every operation rounded to float32.
function deltaStep(key, value, beta) {
  const alpha = $("pg-alpha").checked ? f32(0.999) : 1;
  const k = normalize(key);
  for (let i = 0; i < BITS; i++) {
    let sk = 0;
    for (let j = 0; j < SLOTS; j++) sk = f32(sk + f32(S[i][j] * k[j]));
    for (let j = 0; j < SLOTS; j++) {
      const retained = f32(S[i][j] - f32(beta * f32(sk * k[j])));
      S[i][j] = f32(f32(alpha * retained) + f32(beta * f32(value[i] * k[j])));
    }
  }
}

function decodeColumn(column) {
  return column.reduce((word, x, i) => word | ((x > 0.5 ? 1 : 0) << i), 0);
}

function read(slot) {
  deltaStep(zeros(SLOTS), zeros(BITS), 0);
  const q = normalize(oneHot(slot));
  const output = S.map((row) => row.reduce((sum, x, j) => f32(sum + f32(x * q[j])), 0));
  return decodeColumn(output);
}

function residue() {
  let worst = 0;
  for (const row of S) for (const x of row) worst = Math.max(worst, Math.min(Math.abs(x), Math.abs(x - 1)));
  return worst;
}

function log(text, bad = false) {
  const item = document.createElement("li");
  item.textContent = text;
  if (bad) item.style.color = "var(--bad)";
  $("pg-log").prepend(item);
}

function render() {
  const grid = $("pg-grid");
  const parts = ['<div class="label"></div>'];
  for (let j = 0; j < SLOTS; j++) parts.push(`<div class="head">slot ${j}</div>`);
  for (let i = BITS - 1; i >= 0; i--) {
    parts.push(`<div class="label">bit ${i}</div>`);
    for (let j = 0; j < SLOTS; j++) {
      const x = S[i][j];
      const pct = Math.max(0, Math.min(1, x)) * 100;
      parts.push(`<div class="cell" title="S[${i}][${j}] = ${x}" style="background: color-mix(in srgb, var(--cell-one) ${pct}%, var(--cell-zero))"></div>`);
    }
  }
  const decoded = zeros(SLOTS).map((_, j) => decodeColumn(S.map((row) => row[j])));
  parts.push('<div class="label">neural</div>');
  decoded.forEach((w, j) => parts.push(`<div class="word${w === oracle[j] ? "" : " bad"}">${w}</div>`));
  parts.push('<div class="label">oracle</div>');
  oracle.forEach((w) => parts.push(`<div class="word">${w}</div>`));
  grid.innerHTML = parts.join("");

  const r = residue();
  const residueEl = $("pg-residue");
  residueEl.textContent = `residue ${r === 0 ? "0 (every entry exactly 0 or 1)" : r.toExponential(2)}`;
  residueEl.className = r === 0 ? "ok" : "warn";
  const matches = decoded.every((w, j) => w === oracle[j]);
  const matchEl = $("pg-match");
  matchEl.textContent = matches ? "decoded words match the oracle" : "decoded words differ from the oracle";
  matchEl.className = matches ? "ok" : "bad";
}

function clampInput(id, max) {
  const value = Math.max(0, Math.min(max, Math.trunc(Number($(id).value) || 0)));
  $(id).value = value;
  return value;
}

function write(slot, value) {
  oracle[slot] = value;
  deltaStep(oneHot(slot), bitsOf(value), 1);
}

function reset() {
  S = zeros(BITS).map(() => zeros(SLOTS));
  oracle = zeros(SLOTS);
  $("pg-log").innerHTML = "";
  render();
}

function setupPlayground() {
  if (!$("playground")) return;
  $("pg-write").onclick = () => {
    const slot = clampInput("pg-addr", SLOTS - 1);
    const value = clampInput("pg-value", 255);
    write(slot, value);
    log(`write slot ${slot} ← ${value}`);
    render();
  };
  $("pg-read").onclick = () => {
    const slot = clampInput("pg-addr", SLOTS - 1);
    const value = read(slot);
    log(`read slot ${slot} → ${value} (oracle ${oracle[slot]})`, value !== oracle[slot]);
    render();
  };
  $("pg-idle").onclick = () => {
    for (let n = 0; n < 1000; n++) deltaStep(zeros(SLOTS), zeros(BITS), 0);
    log("idle × 1000");
    render();
  };
  $("pg-random").onclick = () => {
    for (let slot = 0; slot < SLOTS; slot++) write(slot, Math.floor(Math.random() * 256));
    log("random fill of all slots");
    render();
  };
  $("pg-reset").onclick = reset;
  reset();
}

// ---------- Equation copy ----------

function setupEquations() {
  for (const equation of document.querySelectorAll(".equation[data-tex]")) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quiet copy-tex";
    button.textContent = "Copy LaTeX";
    button.setAttribute("aria-label", "Copy this equation as LaTeX");
    button.onclick = async () => {
      try {
        await navigator.clipboard.writeText(equation.dataset.tex);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      setTimeout(() => (button.textContent = "Copy LaTeX"), 1500);
    };
    equation.append(button);
  }
}

// ---------- BibTeX copy ----------

function setupCopy() {
  const button = $("copy-bibtex");
  if (!button) return;
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText($("bibtex").textContent);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select and copy";
    }
    setTimeout(() => (button.textContent = "Copy"), 1500);
  };
}

setupResults();
setupPlayground();
setupCopy();
setupEquations();
