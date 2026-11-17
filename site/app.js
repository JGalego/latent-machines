"use strict";

// ---------- Results, generated from committed run records ----------

const fmtInt = (n) => n.toLocaleString("en-US");
const superscript = (digits) => [...digits].map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]).join("");
const fmtHorizon = (h) => {
  const exp = Math.log10(h);
  return Number.isInteger(exp) ? `10${superscript(String(exp))}` : fmtInt(h);
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function badge(kind, text = kind) {
  const cls = kind === "conforms with residue" ? "residue" : kind;
  return `<span class="badge ${esc(cls)}">${esc(text)}</span>`;
}

function rowKey(variant) {
  return variant.machine ? `${variant.machine} · ${variant.workload}` : variant.workload;
}

function renderExperiment(id, exp, firstTable) {
  const runs = [...exp.runs].sort((a, b) => a.horizon - b.horizon || a.seed - b.seed);
  const summary = runs
    .map(
      (r) => `<tr>
        <td>${fmtHorizon(r.horizon)}</td>
        <td class="num">${r.seed}</td>
        <td class="num">${fmtInt(r.totals.actions_checked)}</td>
        <td class="num">${r.totals.mismatches}</td>
        <td class="num">${r.totals.max_residue}</td>
        <td class="num">${r.snapshots_agree}/${r.totals.variants}</td>
        <td>${badge(r.status)}</td>
        <td>${esc(r.cpu || "")}</td>
        <td><a href="${esc(r.manifest)}">${esc(r.run_id.split("-").pop())}</a></td>
      </tr>`
    )
    .join("");

  const largest = runs[runs.length - 1];
  const dtypes = [...new Set(largest.variants.map((v) => v.dtype))];
  const rows = [...new Set(largest.variants.map(rowKey))];
  const cell = (row, dtype) => {
    const v = largest.variants.find((x) => rowKey(x) === row && x.dtype === dtype);
    if (!v) return "<td></td>";
    const title = `${v.variant}: ${fmtInt(v.actions_checked)} actions, residue ${v.max_residue}, snapshot ${v.snapshot_resume}`;
    return `<td title="${esc(title)}">${badge(v.classification)}</td>`;
  };
  const matrix = rows
    .map((row) => `<tr><th scope="row">${esc(row)}</th>${dtypes.map((d) => cell(row, d)).join("")}</tr>`)
    .join("");
  const total = runs.reduce((sum, r) => sum + r.totals.actions_checked, 0);

  const title = exp.title.charAt(0).toUpperCase() + exp.title.slice(1);
  const revisions = [...new Set(runs.map((r) => r.revision.slice(0, 7)))];
  return `<p><strong>${esc(title)} (${esc(id)}).</strong>
      ${fmtInt(total)} actions were checked in ${runs.length} runs from revision${revisions.length > 1 ? "s" : ""}
      ${revisions.map((s) => `<code>${s}</code>`).join(", ")}.
      ${exp.report ? `The <a href="${esc(exp.report)}">report</a> gives the setup and limits.` : ""}</p>
    <p class="caption"><strong>Table ${firstTable}:</strong> ${esc(id)} runs. Mismatches count nonconforming steps; residue is Equation&nbsp;(2); a snapshot agrees if the restored state equals the live state.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Actions per configuration</th><th>Seed</th><th>Actions checked</th><th>Mismatches</th><th>Max residue</th><th>Snapshots agree</th><th>Status</th><th>CPU</th><th>Record</th></tr></thead>
      <tbody>${summary}</tbody>
    </table></div>
    <p class="caption"><strong>Table ${firstTable + 1}:</strong> Classification of each ${esc(id)} configuration in the run with ${fmtHorizon(largest.horizon)} actions per configuration. Hover over a cell for details.</p>
    <div class="table-wrap"><table class="grid-matrix">
      <thead><tr><th></th>${dtypes.map((d) => `<th>${esc(d)}</th>`).join("")}</tr></thead>
      <tbody>${matrix}</tbody>
    </table></div>`;
}

async function loadResults() {
  const target = document.getElementById("results-tables");
  try {
    const response = await fetch("data/results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const ids = Object.keys(data.experiments).sort();
    target.innerHTML = ids.length
      ? ids.map((id, i) => renderExperiment(id, data.experiments[id], 2 * i + 1)).join("")
      : '<p class="muted">No results recorded yet.</p>';
    const total = ids
      .flatMap((id) => data.experiments[id].runs)
      .reduce((sum, r) => sum + r.totals.actions_checked, 0);
    for (const el of document.querySelectorAll(".total-actions")) el.textContent = fmtInt(total);
  } catch (error) {
    target.innerHTML = `<p class="muted">Results could not be loaded (${esc(error.message)}). They are also in the <a href="https://github.com/JGalego/latent-machines/tree/main/results">repository</a>.</p>`;
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

loadResults();
setupPlayground();
setupCopy();
