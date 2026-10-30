/* drug-agent console — offline vanilla-JS dashboard with inline SVG charts. */
"use strict";

// ---------------------------------------------------------------- helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—"
  : (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(d)) : String(v));
const clamp01 = (x) => Math.max(0, Math.min(1, x));

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}
async function post(path, body, headers) {
  const r = await fetch(path, { method: "POST",
    headers: { "Content-Type": "application/json", ...(headers || {}) },
    body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data?.error?.detail || `${r.status} ${path}`);
  return data;
}
function toast(msg, ms = 2600) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

const TRUST = { OBSERVED: "b-good", CURATED: "b-info", CALCULATED: "b-info",
  PREDICTED: "b-warn", INFERRED: "b-violet", HYPOTHESIZED: "b-warn", PROPOSED: "b-mut" };
const SEV_COLOR = { NONE: "var(--sev-none)", LOW: "var(--sev-low)", MODERATE: "var(--sev-mod)",
  HIGH: "var(--sev-high)", CRITICAL: "var(--sev-crit)" };
const DECISION = { PROMOTE: "b-good", DEFER: "b-warn", REJECT: "b-bad" };
const TASK_COLOR = { SUCCEEDED: "var(--good)", RUNNING: "var(--info)", VALIDATING: "var(--info)",
  AWAITING_APPROVAL: "var(--warn)", FAILED_FINAL: "var(--bad)", FAILED_RETRYABLE: "var(--bad)",
  PENDING: "var(--faint)", READY: "var(--muted)", CANCELLED: "var(--bad)", SUPERSEDED: "var(--faint)",
  BLOCKED: "var(--warn)" };

// ---------------------------------------------------------------- state
const S = { runs: [], runId: null, d: null, view: "overview", paretoX: 0, paretoY: 1,
  sort: { key: "utility", dir: -1 } };

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    wire();
    await loadRuns();
    if (S.runs.length) { await selectRun(S.runs[0].id); }
    else { render(); openNewRun(); }
  } catch (e) {
    $("#view").innerHTML = `<div class="empty">Failed to initialize: ${esc(e.message)}<br>
      <span class="muted">Is the server running? Try reloading.</span></div>`;
    console.error(e);
  }
}
window.addEventListener("error", (e) => toast("UI error: " + e.message, 5000));
function wire() {
  $$("#nav a").forEach((a) => a.addEventListener("click", () => {
    S.view = a.dataset.view; $$("#nav a").forEach((x) => x.classList.toggle("active", x === a)); render();
  }));
  $("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
  $("#btn-refresh").addEventListener("click", () => S.runId && selectRun(S.runId));
  $("#btn-new").addEventListener("click", openNewRun);
  $("#new-close").addEventListener("click", closeDrawers);
  $("#cand-close").addEventListener("click", closeDrawers);
  $("#drawer-backdrop").addEventListener("click", closeDrawers);
  $("#launch").addEventListener("click", launchRun);
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    if (S.d) render();
  });
}
async function loadRuns() {
  try { S.runs = (await api("/runs?limit=100")).items || []; }
  catch { S.runs = []; }
  const sel = $("#run-select");
  sel.innerHTML = S.runs.length
    ? S.runs.map((r) => `<option value="${esc(r.id)}">${esc(r.id.slice(0, 16))} · ${esc(r.state)}</option>`).join("")
    : `<option>no runs yet</option>`;
}
async function selectRun(id) {
  S.runId = id; $("#run-select").value = id;
  try { S.d = await api(`/runs/${id}/dashboard`); }
  catch (e) { toast("Failed to load run: " + e.message); return; }
  render();
}

// ---------------------------------------------------------------- top bar
function topbar() {
  const d = S.d; const badge = $("#run-badge"); const stats = $("#topbar-stats");
  if (!d) { badge.textContent = "no run selected"; stats.innerHTML = ""; return; }
  const st = d.run.state;
  const cls = st === "SUCCEEDED" ? "b-good" : st === "FAILED" ? "b-bad"
    : st === "AWAITING_APPROVAL" ? "b-warn" : "b-info";
  badge.innerHTML = `<span class="badge ${cls}">${esc(st)}</span> <span class="mono">${esc(d.run.id)}</span>`;
  const f = d.funnel; const p = d.portfolio || {};
  stats.innerHTML = [
    ["brief", d.brief.kind || "—"], ["candidates", f.standardized], ["passed", f.passed_filters],
    ["pareto", f.pareto], ["shortlist", f.shortlist],
    ["diversity", fmt(p.diversity, 3)], ["deterministic", d.run.deterministic ? "yes" : "no"],
  ].map(([k, v]) => `${k} <b>${esc(v)}</b>`).join("");
  if (st === "AWAITING_APPROVAL") stats.innerHTML +=
    ` <button class="btn btn-accent" onclick="gateBar()">Review gate →</button>`;
}

// ---------------------------------------------------------------- router
function render() {
  topbar();
  const root = $("#view");
  if (!S.d) { root.innerHTML = `<div class="empty">Select a run or launch a new one.</div>`; return; }
  const fn = { overview: vOverview, workflow: vWorkflow, portfolio: vPortfolio, candidates: vCandidates,
     riskmatrix: vRisk, safety: vSafety, evidence: vEvidence, sensitivity: vSensitivity,
     repro: vRepro }[S.view] || vOverview;
  try { fn(root); }
  catch (e) { root.innerHTML = `<div class="empty">Render error in "${esc(S.view)}": ${esc(e.message)}</div>`; console.error(e); }
}

// ---------------------------------------------------------------- OVERVIEW
function vOverview(root) {
  const d = S.d, f = d.funnel, p = d.portfolio || {}, bl = d.baseline || {};
  const kpis = [
    ["candidates", f.standardized, "standardized identities"],
    ["passed filters", f.passed_filters, "hard constraints"],
    ["Pareto-optimal", f.pareto, `${p.n_clusters || 0} scaffold clusters`],
    ["shortlist", f.shortlist, `${f.reserve} reserve`],
    ["hypervolume", fmt(p.hypervolume, 3), "dominated volume"],
    ["diversity", fmt(p.diversity, 3), "1 − mean Tanimoto"],
    ["evidence coverage", fmt(p.evidence_coverage, 3), "0–1"],
    ["evidence items", (d.evidence || []).length, "trust-labelled"],
  ];
  root.innerHTML = `
    <div class="banner">⚕️ Research-prioritization output — <b>not a clinical decision system</b>.
      No candidate is claimed safe, effective, synthesizable, or clinically viable. Demo activity/safety
      labels are synthetic. Values carry epistemic trust labels and require external validation.</div>
    <div class="cards">${kpis.map(([k, n, s]) =>
      `<div class="kpi"><div class="n">${esc(n)}</div><div class="k">${esc(k)}</div><div class="sub">${esc(s)}</div></div>`).join("")}</div>
    <div class="section-title">Candidate funnel</div>
    <div class="panel funnel">${funnel(f)}</div>
    ${originPanel(d)}
    <div class="two" style="margin-top:16px">
      <div class="panel"><h3>Baseline model (${esc(bl.split || "—")} split)</h3>${baselineCard(bl)}</div>
      <div class="panel"><h3>Capability matrix</h3>${capMatrix(d.capability_matrix)}</div>
    </div>`;
}
function funnel(f) {
  const stages = [["sourced", f.sourced], ["standardized", f.standardized],
    ["passed filters", f.passed_filters], ["Pareto", f.pareto], ["shortlist", f.shortlist]];
  const max = Math.max(...stages.map((s) => s[1]), 1);
  return stages.map(([k, v]) =>
    `<div class="bar"><span class="lbl">${k}</span>
      <span class="track"><span class="fill" style="width:${(100 * v / max).toFixed(1)}%"></span></span>
      <span class="val">${v}</span></div>`).join("");
}
// Where candidates came from, and — for de-novo ones — from which generator.
// A surrogate batch is called out explicitly so a generated molecule is never
// mistaken for a model-generated one.
function originPanel(d) {
  const counts = d.origin_counts || {}, gens = d.generation || [];
  if (!Object.keys(counts).length && !gens.length) return "";
  const chips = Object.entries(counts).map(([k, v]) => {
    const denovo = k === "pfm" || k === "pfm_mock";
    return `<span class="chip">${denovo ? "🧬 " : ""}${esc(k)} <b>${esc(v)}</b></span>`;
  }).join("");
  const cards = gens.map((g) => {
    const m = g.metrics || {};
    const surrogate = !!g.is_surrogate;
    return `<div class="kv" style="margin-top:10px">
      <div>Generator</div><div>${surrogate
        ? `<span class="badge b-warn">SURROGATE — not PFM</span>`
        : `<span class="badge b-good">PFM</span>`} <span class="mono">${esc(g.model_id)}</span> v${esc(g.model_version)}</div>
      <div>Origin tag</div><div><span class="chip">${esc(g.origin_tag)}</span> trust ${esc(g.trust_label)}</div>
      <div>Kept / requested</div><div>${esc(g.n_kept)} / ${esc(m.n_generated_requested ?? "—")}</div>
      <div>Validity / uniqueness</div><div>${fmt(m.generation_validity)} / ${fmt(m.generation_uniqueness)}</div>
      <div>Pocket</div><div>${esc((g.pocket || {}).receptor_id || (g.pocket || {}).pdb_id || "—")}</div>
      <div>Seed</div><div class="mono">${esc(g.seed)}</div>
    </div>
    ${surrogate ? `<div class="banner" style="margin-top:8px">⚠️ PFM was unavailable; these structures come from a
      deterministic BRICS-recombination surrogate and are <b>not structure-conditioned</b>.</div>` : ""}
    ${(g.errors || []).map((e) => `<div class="banner" style="margin-top:8px">Generation error: ${esc(e)}</div>`).join("")}`;
  }).join("");
  return `<div class="section-title">Candidate provenance</div>
    <div class="panel"><div>${chips}</div>${cards}</div>`;
}
function baselineCard(bl) {
  if (!bl || bl.mae === undefined) return `<div class="muted">Not trained (insufficient data).</div>`;
  const beats = bl.beats_constant_baseline;
  return `<div class="kv">
    <div>Endpoint</div><div>${esc(bl.endpoint)}</div>
    <div>MAE</div><div>${fmt(bl.mae)} <span class="muted">vs constant ${fmt(bl.mae_constant_baseline)}</span></div>
    <div>Spearman ρ</div><div>${fmt(bl.spearman)}</div>
    <div>Train / test</div><div>${bl.n_train} / ${bl.n_test}</div>
    <div>Beats baseline</div><div><span class="badge ${beats ? "b-good" : "b-bad"}">${beats ? "yes" : "no"}</span></div>
  </div>`;
}
// Headings for the capability namespaces. Both agents (`agent.*`) and tools
// (`chem.*`, `molecule.*`, …) publish into one matrix, and the id's first segment
// is what says which is which — so the grouping is derived, and a capability added
// to the platform appears under the right heading without this file changing.
const CAP_DOMAINS = {
  agent: "Agents",
  target: "Target evidence",
  chem: "Chemistry",
  model: "Activity model",
  molecule: "De-novo generation",
  pocket: "Binding-site detection",
  structure: "Structure",
  property: "Property prediction",
  synth: "Synthesis",
  safety: "Safety",
  perturb: "Perturbation",
  binding: "Binding free energy",
  benchmark: "Benchmarking",
  platform: "Platform",
};

function capMatrix(m) {
  // MOCKED is still handled here, unlike in the live console. This panel renders a
  // *stored* run's matrix, and runs recorded before ADR-0011 contain that status;
  // dropping the branch would render their history as an error rather than as what
  // the platform then reported.
  const badge = (v) => v === "AVAILABLE" ? "b-good" : v === "MOCKED" ? "b-warn" : "b-bad";
  const dot = (v) => v === "AVAILABLE" ? "var(--good)" : v === "MOCKED" ? "var(--warn)" : "var(--bad)";
  const entries = Object.entries(m || {});
  if (!entries.length) return `<div class="muted">No capabilities recorded for this run.</div>`;

  const groups = new Map();
  entries.forEach(([k, v]) => {
    const domain = String(k).split(".")[0] || "other";
    if (!groups.has(domain)) groups.set(domain, []);
    groups.get(domain).push([k, v]);
  });
  // Largest group first; ties by name, so the order is stable across reloads.
  const order = [...groups.keys()].sort((a, b) =>
    (groups.get(b).length - groups.get(a).length) || a.localeCompare(b));

  const available = entries.filter(([, v]) => v === "AVAILABLE").length;
  const head = `<div class="muted" style="margin-bottom:8px">${available} of ${entries.length} available in this run</div>`;

  return head + order.map((domain) => {
    // Available first: what ran is the question, and a long tail of absent
    // capabilities above the working ones answers it backwards.
    const rows = groups.get(domain).slice().sort((a, b) => {
      const oa = a[1] === "AVAILABLE", ob = b[1] === "AVAILABLE";
      return oa === ob ? a[0].localeCompare(b[0]) : (oa ? -1 : 1);
    });
    const ok = rows.filter(([, v]) => v === "AVAILABLE").length;
    const chips = rows.map(([k, v]) =>
      `<span class="chip"><span class="dot" style="background:${dot(v)}"></span>${esc(k)} <span class="badge ${badge(v)}">${esc(v)}</span></span>`).join("");
    return `<div class="cap-group"><div class="muted cap-group-title">${esc(CAP_DOMAINS[domain] || domain)} <span class="badge">${ok}/${rows.length}</span></div><div>${chips}</div></div>`;
  }).join("");
}

// ---------------------------------------------------------------- WORKFLOW / DAG
function vWorkflow(root) {
  const d = S.d;
  root.innerHTML = `
    <div class="section-title">Task graph (${d.tasks.length} tasks)</div>
    <div class="panel chart" id="dag"></div>
    <div class="legend">${Object.entries(TASK_COLOR).slice(0, 8).map(([k, c]) =>
      `<span><span class="dot" style="background:${c}"></span>${k}</span>`).join("")}</div>
    <div class="section-title">Event log</div>
    <div class="panel timeline">${(d.events || []).slice().reverse().map((e) =>
      `<div class="ev"><span class="ts">${esc((e.ts || "").slice(11, 19))}</span>
        <span>${esc(e.event)} <span class="muted">${esc(e.detail?.name || "")}</span>
        ${e.from_state ? `<span class="mono muted">${esc(e.from_state)}→${esc(e.to_state)}</span>` : ""}
        ${e.detail?.error ? `<span class="badge b-bad">${esc(e.detail.error)}</span>` : ""}
        ${e.detail?.cache_hit ? `<span class="badge b-info">cache</span>` : ""}</span></div>`).join("") || `<div class="empty">no events</div>`}</div>`;
  $("#dag").innerHTML = dagSVG(d.tasks);
}
function dagSVG(tasks) {
  const byId = Object.fromEntries(tasks.map((t) => [t.id, t]));
  const layer = {};
  const depth = (t, seen = new Set()) => {
    if (layer[t.id] !== undefined) return layer[t.id];
    if (seen.has(t.id)) return 0; seen.add(t.id);
    const deps = (t.depends_on || []).map((id) => byId[id]).filter(Boolean);
    const l = deps.length ? Math.max(...deps.map((x) => depth(x, seen))) + 1 : 0;
    return (layer[t.id] = l);
  };
  tasks.forEach((t) => depth(t));
  const layers = {};
  tasks.forEach((t) => (layers[layer[t.id]] = layers[layer[t.id]] || []).push(t));
  const colW = 168, rowH = 46, padX = 20, padY = 24, nodeW = 140, nodeH = 30;
  const nL = Object.keys(layers).length;
  const maxRows = Math.max(...Object.values(layers).map((a) => a.length));
  const W = padX * 2 + nL * colW, H = padY * 2 + maxRows * rowH;
  const pos = {};
  Object.entries(layers).forEach(([l, arr]) => arr.forEach((t, i) => {
    pos[t.id] = { x: padX + (+l) * colW, y: padY + i * rowH + (maxRows - arr.length) * rowH / 2 };
  }));
  let edges = "";
  tasks.forEach((t) => (t.depends_on || []).forEach((dep) => {
    const a = pos[dep], b = pos[t.id]; if (!a || !b) return;
    const x1 = a.x + nodeW, y1 = a.y + nodeH / 2, x2 = b.x, y2 = b.y + nodeH / 2;
    const mx = (x1 + x2) / 2;
    edges += `<path class="dag-edge" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"/>`;
  }));
  let nodes = "";
  tasks.forEach((t) => {
    const p = pos[t.id]; const c = TASK_COLOR[t.state] || "var(--faint)";
    const pulse = t.state === "RUNNING" ? "pulse" : "";
    nodes += `<g class="dag-node ${pulse}">
      <rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="7"
        fill="var(--panel-2)" stroke="${c}" stroke-width="2"/>
      <circle cx="${p.x + 11}" cy="${p.y + nodeH / 2}" r="4" fill="${c}"/>
      <text x="${p.x + 22}" y="${p.y + nodeH / 2 + 3.5}" fill="var(--fg)">${esc(t.name.slice(0, 20))}</text>
    </g>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMinYMin meet" style="min-width:${W}px">${edges}${nodes}</svg>`;
}

// ---------------------------------------------------------------- PORTFOLIO
function vPortfolio(root) {
  const d = S.d; const objs = (d.objectives || []).map((o) => o.name);
  if (S.paretoX >= objs.length) S.paretoX = 0;
  if (S.paretoY >= objs.length) S.paretoY = Math.min(1, objs.length - 1);
  const sel = (id, cur) => `<select id="${id}" style="width:auto;display:inline-block">${objs.map((o, i) =>
    `<option value="${i}" ${i === cur ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
  const short = (d.portfolio || {}).shortlist || [];
  root.innerHTML = `
    <div class="section-title">Pareto front — ${esc(objs[S.paretoX] || "")} vs ${esc(objs[S.paretoY] || "")}</div>
    <div class="panel">
      <div style="margin-bottom:10px">X: ${sel("px", S.paretoX)} &nbsp; Y: ${sel("py", S.paretoY)}
        <span class="muted"> (axes are normalized 0–1, higher is better; ● shortlist, ◇ Pareto)</span></div>
      <div class="chart" id="pareto"></div>
    </div>
    <div class="section-title">Shortlist (${short.length})</div>
    <div class="panel scroll-x">${shortlistTable(short)}</div>`;
  $("#px").addEventListener("change", (e) => { S.paretoX = +e.target.value; vPortfolio(root); });
  $("#py").addEventListener("change", (e) => { S.paretoY = +e.target.value; vPortfolio(root); });
  $("#pareto").innerHTML = paretoSVG(objs);
}
function scByCand() { return Object.fromEntries((S.d.scorecards || []).map((s) => [s.candidate_id, s])); }
function normFor(sc, ep) { const e = (sc.endpoints || []).find((x) => x.endpoint === ep); return e && !e.missing ? e.normalized : null; }
function paretoSVG(objs) {
  const scs = (S.d.scorecards || []).filter((s) => s.hard_constraints_passed);
  const xEp = objs[S.paretoX], yEp = objs[S.paretoY];
  const W = 720, H = 420, m = { l: 48, r: 20, t: 16, b: 44 };
  const px = (v) => m.l + clamp01(v) * (W - m.l - m.r);
  const py = (v) => H - m.b - clamp01(v) * (H - m.t - m.b);
  let grid = "";
  for (let i = 0; i <= 5; i++) {
    const gx = m.l + i / 5 * (W - m.l - m.r), gy = H - m.b - i / 5 * (H - m.t - m.b);
    grid += `<line class="grid-line" x1="${gx}" y1="${m.t}" x2="${gx}" y2="${H - m.b}"/>`;
    grid += `<line class="grid-line" x1="${m.l}" y1="${gy}" x2="${W - m.r}" y2="${gy}"/>`;
    grid += `<text class="axis-label" x="${gx}" y="${H - m.b + 16}" text-anchor="middle">${(i / 5).toFixed(1)}</text>`;
    grid += `<text class="axis-label" x="${m.l - 8}" y="${gy + 3}" text-anchor="end">${(i / 5).toFixed(1)}</text>`;
  }
  const dcol = { PROMOTE: "var(--good)", DEFER: "var(--warn)", REJECT: "var(--bad)" };
  let dots = "";
  scs.forEach((s) => {
    const x = normFor(s, xEp), y = normFor(s, yEp); if (x === null || y === null) return;
    const c = dcol[s.decision] || "var(--muted)";
    const r = s.decision === "PROMOTE" ? 6 : 4.5;
    if (s.pareto_optimal) dots += `<circle cx="${px(x)}" cy="${py(y)}" r="${r + 4}" fill="none" stroke="${c}" stroke-width="1.3" opacity="0.6"/>`;
    dots += `<circle cx="${px(x)}" cy="${py(y)}" r="${r}" fill="${c}" opacity="0.9"
      onclick="openCand('${s.candidate_id}')" style="cursor:pointer"><title>${esc(s.candidate_id)} · ${esc(s.decision)}</title></circle>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="min-width:${W}px">${grid}
    <text class="axis-label" x="${(W) / 2}" y="${H - 6}" text-anchor="middle">${esc(xEp)} (normalized)</text>
    <text class="axis-label" transform="translate(14,${H / 2}) rotate(-90)" text-anchor="middle">${esc(yEp)} (normalized)</text>
    ${dots}</svg>
    <div class="legend"><span><span class="dot" style="background:var(--good)"></span>promote</span>
      <span><span class="dot" style="background:var(--warn)"></span>defer</span>
      <span><span class="dot" style="background:var(--bad)"></span>reject</span>
      <span>◇ Pareto-optimal</span></div>`;
}
function shortlistTable(short) {
  if (!short.length) return `<div class="empty">No shortlist yet (run may be pre-portfolio).</div>`;
  const byC = scByCand(); const objs = (S.d.objectives || []).map((o) => o.name);
  const head = `<tr><th>candidate</th><th>InChIKey</th>${objs.map((o) => `<th class="num">${esc(o)}</th>`).join("")}
    <th class="num">utility</th><th class="num">cluster</th><th>decision</th></tr>`;
  const rows = short.map((m) => {
    const sc = byC[m.candidate_id] || {};
    const cells = objs.map((o) => { const e = (sc.endpoints || []).find((x) => x.endpoint === o);
      const raw = e ? e.raw_value : null, u = e ? e.uncertainty : null;
      return `<td class="num">${fmt(raw)}${u != null ? `<span class="muted"> ±${fmt(u, 2)}</span>` : ""}</td>`; }).join("");
    return `<tr class="clickable" onclick="openCand('${m.candidate_id}')">
      <td class="mono">${esc(m.candidate_id.slice(0, 14))}</td><td class="mono">${esc((m.inchikey || "").slice(0, 14))}</td>
      ${cells}<td class="num">${fmt(sc.utility)}</td><td class="num">${fmt(m.cluster)}</td>
      <td><span class="badge ${DECISION[sc.decision] || "b-mut"}">${esc(sc.decision || "—")}</span></td></tr>`;
  }).join("");
  return `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

// ---------------------------------------------------------------- CANDIDATES
function vCandidates(root) {
  const d = S.d; const byC = scByCand();
  root.innerHTML = `
    <div class="section-title">Candidates (${d.candidates.length})</div>
    <div class="panel">
      <div style="margin-bottom:10px">
        <label class="chip"><input type="checkbox" id="only-pass"> passed filters only</label>
        <input type="text" id="cand-search" placeholder="filter by id / InChIKey / SMILES" style="width:280px;display:inline-block">
      </div>
      <div class="scroll-x" id="cand-table"></div>
    </div>`;
  const draw = () => {
    const onlyPass = $("#only-pass").checked; const q = ($("#cand-search").value || "").toLowerCase();
    let rows = d.candidates.filter((c) =>
      (!onlyPass || c.passed_hard_filters) &&
      (!q || (c.id + (c.inchikey || "") + (c.canonical_smiles || "")).toLowerCase().includes(q)));
    rows = rows.map((c) => ({ c, sc: byC[c.id] || {} }));
    const k = S.sort.key, dir = S.sort.dir;
    rows.sort((a, b) => {
      const va = valOf(a, k), vb = valOf(b, k);
      if (va === vb) return 0; return (va > vb ? 1 : -1) * dir;
    });
    $("#cand-table").innerHTML = candTable(rows);
    $$("#cand-table th[data-k]").forEach((th) => th.addEventListener("click", () => {
      const key = th.dataset.k; S.sort = { key, dir: S.sort.key === key ? -S.sort.dir : -1 }; draw();
    }));
  };
  $("#only-pass").addEventListener("change", draw); $("#cand-search").addEventListener("input", draw);
  draw();
}
function valOf({ c, sc }, k) {
  if (k === "utility") return sc.utility ?? -1; if (k === "pareto") return sc.pareto_rank ?? 999;
  if (k === "decision") return sc.decision ?? ""; if (k === "origin") return c.origin ?? "";
  if (k === "pass") return c.passed_hard_filters ? 1 : 0; return c.id;
}
function candTable(rows) {
  if (!rows.length) return `<div class="empty">No candidates match.</div>`;
  const th = (k, l, num) => `<th data-k="${k}" class="${num ? "num" : ""}">${l}${S.sort.key === k ? (S.sort.dir < 0 ? " ▾" : " ▴") : ""}</th>`;
  const head = `<tr>${th("id", "candidate")}<th>InChIKey</th>${th("origin", "origin")}${th("pass", "filters")}
    ${th("utility", "utility", 1)}${th("pareto", "Pareto", 1)}${th("decision", "decision")}<th>reason</th></tr>`;
  const body = rows.slice(0, 300).map(({ c, sc }) =>
    `<tr class="clickable" onclick="openCand('${c.id}')">
      <td class="mono">${esc(c.id.slice(0, 14))}</td><td class="mono">${esc((c.inchikey || "").slice(0, 14))}</td>
      <td><span class="chip">${esc(c.origin)}</span></td>
      <td>${c.passed_hard_filters ? '<span class="badge b-good">pass</span>' : '<span class="badge b-bad">fail</span>'}</td>
      <td class="num">${fmt(sc.utility)}</td><td class="num">${fmt(sc.pareto_rank)}</td>
      <td><span class="badge ${DECISION[sc.decision] || "b-mut"}">${esc(sc.decision || "—")}</span></td>
      <td class="muted">${esc((sc.decision_reasons || c.filter_failures || []).join("; ").slice(0, 46))}</td></tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// candidate detail drawer
window.openCand = function (cid) {
  const d = S.d; const c = d.candidates.find((x) => x.id === cid) || {};
  const sc = (d.scorecards || []).find((x) => x.candidate_id === cid) || {};
  const saf = (d.safety || []).find((x) => x.candidate_id === cid);
  const syn = (d.synthesis || []).find((x) => x.candidate_id === cid);
  const pert = (d.perturbation || []).find((x) => x.candidate_id === cid);
  const objs = (d.objectives || []).map((o) => o.name);
  $("#cand-title").innerHTML = `${esc(cid.slice(0, 16))}
    <span class="badge ${DECISION[sc.decision] || "b-mut"}">${esc(sc.decision || "—")}</span>`;
  const eps = (sc.endpoints || []);
  $("#cand-body").innerHTML = `
    <div class="kv">
      <div>InChIKey</div><div class="mono">${esc(c.inchikey || "—")}</div>
      <div>SMILES</div><div class="mono">${esc(c.canonical_smiles || "—")}</div>
      <div>Origin</div><div>${esc(c.origin)} ${c.lineage?.length ? `<span class="muted">from ${esc(c.lineage[0])}</span>` : ""}</div>
      <div>Scaffold</div><div class="mono">${esc(c.scaffold || "—")}</div>
      <div>Hard filters</div><div>${c.passed_hard_filters ? '<span class="badge b-good">passed</span>' : `<span class="badge b-bad">failed: ${esc((c.filter_failures || []).join(", "))}</span>`}</div>
      <div>Pareto</div><div>${sc.pareto_optimal ? "optimal" : "dominated"} (rank ${fmt(sc.pareto_rank)}), cluster ${fmt(sc.diversity_cluster)}</div>
      <div>Utility / conflict</div><div>${fmt(sc.utility)} / ${fmt(sc.conflict_score)}</div>
      <div>Evidence quality</div><div>${fmt(sc.evidence_quality)}</div>
    </div>
    <div class="section-title">Endpoint profile</div>
    <div class="chart">${radarSVG(eps, objs)}</div>
    <div class="section-title">Endpoint scorecard</div>
    ${endpointTable(eps)}
    ${saf ? `<div class="section-title">Safety</div>${safetyDetail(saf)}` : ""}
    ${syn ? `<div class="section-title">Synthesizability</div>${synthDetail(syn)}` : ""}
    ${pert ? `<div class="section-title">Perturbation</div>${pertDetail(pert)}` : ""}
    <div class="section-title">Decision rationale</div>
    <ul>${(sc.decision_reasons || []).map((r) => `<li>${esc(r)}</li>`).join("") || "<li class='muted'>—</li>"}</ul>`;
  openDrawer("#cand-detail");
};
function endpointTable(eps) {
  if (!eps.length) return `<div class="muted">no endpoints</div>`;
  return `<table><thead><tr><th>endpoint</th><th class="num">raw</th><th>unit</th><th class="num">norm</th>
    <th class="num">± unc</th><th>applicability</th><th>model</th></tr></thead><tbody>${eps.map((e) => {
    const ad = e.applicability;
    const adc = ad === "IN_DOMAIN" ? "b-good" : ad === "OUT_OF_DOMAIN" ? "b-bad" : "b-warn";
    return `<tr><td>${esc(e.endpoint)}</td><td class="num">${fmt(e.raw_value)}</td><td class="muted">${esc(e.unit || "")}</td>
      <td class="num">${e.normalized == null ? "—" : `<span class="bar-inline" style="width:70px"><i style="width:${(100 * clamp01(e.normalized)).toFixed(0)}%"></i></span> ${fmt(e.normalized, 2)}`}</td>
      <td class="num">${fmt(e.uncertainty, 2)}</td><td><span class="badge ${adc}">${esc(ad || "—")}</span></td>
      <td class="mono muted">${esc(e.model_id || "")}</td></tr>`; }).join("")}</tbody></table>`;
}
function radarSVG(eps, objs) {
  const pts = objs.map((o) => { const e = eps.find((x) => x.endpoint === o); return { o, v: e && !e.missing ? clamp01(e.normalized) : 0 }; });
  if (pts.length < 3) return `<div class="muted">radar needs ≥3 objectives</div>`;
  const cx = 200, cy = 150, R = 110, n = pts.length; let rings = "", axes = "", poly = "";
  for (let r = 1; r <= 4; r++) rings += `<circle cx="${cx}" cy="${cy}" r="${R * r / 4}" fill="none" class="grid-line"/>`;
  const ang = (i) => -Math.PI / 2 + i * 2 * Math.PI / n;
  const coords = pts.map((p, i) => [cx + R * p.v * Math.cos(ang(i)), cy + R * p.v * Math.sin(ang(i))]);
  pts.forEach((p, i) => {
    const ex = cx + R * Math.cos(ang(i)), ey = cy + R * Math.sin(ang(i));
    axes += `<line x1="${cx}" y1="${cy}" x2="${ex}" y2="${ey}" class="grid-line"/>`;
    const lx = cx + (R + 16) * Math.cos(ang(i)), ly = cy + (R + 16) * Math.sin(ang(i));
    axes += `<text class="axis-label" x="${lx}" y="${ly}" text-anchor="middle">${esc(p.o)}</text>`;
  });
  poly = `<polygon points="${coords.map((c) => c.join(",")).join(" ")}" fill="color-mix(in srgb, var(--accent) 30%, transparent)" stroke="var(--accent)" stroke-width="2"/>`;
  const dots = coords.map((c) => `<circle cx="${c[0]}" cy="${c[1]}" r="3" fill="var(--accent)"/>`).join("");
  return `<svg viewBox="0 0 400 300">${rings}${axes}${poly}${dots}</svg>`;
}
function safetyDetail(s) {
  return `<div class="muted" style="margin-bottom:6px">${esc(s.summary)}</div>
    <table><tbody>${(s.endpoints || []).map((e) =>
    `<tr><td>${esc(e.endpoint)}</td><td class="num">${fmt(e.score, 3)}</td>
      <td><span class="badge" style="color:${SEV_COLOR[e.severity]};border-color:${SEV_COLOR[e.severity]}">${esc(e.severity)}</span></td>
      <td class="muted">${esc((e.alerts || []).join(", ") || e.detail || "")}</td>
      <td>${e.calibrated ? '<span class="chip">calibrated</span>' : ""}</td></tr>`).join("")}</tbody></table>
    <div class="muted" style="font-size:11px;margin-top:6px">⚠ ${esc(s.disclaimer)}</div>`;
}
function synthDetail(s) {
  const r = (s.routes || [])[0] || {};
  return `<div class="kv"><div>SA score</div><div>${fmt(s.sa_score, 2)} <span class="chip">${esc(s.sa_label || "")}</span></div>
    <div>BB availability</div><div>${fmt(s.building_block_availability, 2)}</div>
    <div>Route steps</div><div>${fmt(r.n_steps)} (confidence ${fmt(r.overall_confidence, 2)})</div>
    <div>Risks</div><div>${(s.risk_factors || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("") || "—"}</div></div>
    <div class="muted" style="font-size:11px;margin-top:4px">${esc(s.limitation || "")}</div>`;
}
function pertDetail(p) {
  return `<div class="kv"><div>Restoration score</div><div>${fmt(p.restoration_score, 3)}</div>
    <div>Signature score</div><div>${fmt(p.signature_score, 3)}</div>
    <div>Cell context</div><div>${esc(p.cell_context || "—")}</div>
    <div>Split</div><div>${esc(p.split_strategy || "—")}</div></div>`;
}

// ---------------------------------------------------------------- RISK MATRIX
function vRisk(root) {
  const d = S.d;
  root.innerHTML = `<div class="section-title">Endpoint risk matrix (normalized 0–1, higher = better)</div>
    <div class="banner">Rows = shortlist + top-utility candidates; columns = objectives. Hatched cells are out-of-domain.</div>
    <div class="panel chart scroll-x" id="hm"></div>`;
  $("#hm").innerHTML = heatmapSVG();
}
function heatmapSVG() {
  const d = S.d; const objs = (d.objectives || []).map((o) => o.name);
  const scs = (d.scorecards || []).filter((s) => s.hard_constraints_passed)
    .sort((a, b) => (b.utility ?? 0) - (a.utility ?? 0)).slice(0, 30);
  if (!scs.length || !objs.length) return `<div class="empty">no data</div>`;
  const cw = 92, ch = 26, lx = 150, ty = 60;
  const W = lx + objs.length * cw + 20, H = ty + scs.length * ch + 10;
  const color = (v) => { // red→amber→green
    const h = 120 * clamp01(v); return `hsl(${h}, 62%, ${document.documentElement.dataset.theme === "light" ? 55 : 42}%)`;
  };
  let cells = "", cols = "", rows = "";
  objs.forEach((o, j) => cols += `<text class="axis-label" x="${lx + j * cw + cw / 2}" y="${ty - 12}" text-anchor="middle" transform="rotate(-18 ${lx + j * cw + cw / 2} ${ty - 12})">${esc(o)}</text>`);
  scs.forEach((s, i) => {
    rows += `<text class="axis-label" x="${lx - 8}" y="${ty + i * ch + ch / 2 + 3}" text-anchor="end">${esc(s.candidate_id.slice(0, 14))}</text>`;
    objs.forEach((o, j) => {
      const e = (s.endpoints || []).find((x) => x.endpoint === o);
      const v = e && !e.missing ? e.normalized : null;
      const x = lx + j * cw, y = ty + i * ch;
      const ood = e && e.applicability === "OUT_OF_DOMAIN";
      cells += `<rect class="hm-cell" x="${x}" y="${y}" width="${cw}" height="${ch}"
        fill="${v == null ? "var(--panel-2)" : color(v)}" onclick="openCand('${s.candidate_id}')" style="cursor:pointer">
        <title>${esc(s.candidate_id)} · ${esc(o)} = ${fmt(v, 2)}${ood ? " (OOD)" : ""}</title></rect>`;
      if (ood) cells += `<line x1="${x}" y1="${y}" x2="${x + cw}" y2="${y + ch}" stroke="var(--bg)" stroke-width="1" opacity="0.7"/>`;
      if (v != null) cells += `<text x="${x + cw / 2}" y="${y + ch / 2 + 3}" text-anchor="middle" font-size="10" fill="#fff" opacity="0.9">${fmt(v, 2)}</text>`;
    });
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="min-width:${W}px">${cols}${rows}${cells}</svg>`;
}

// ---------------------------------------------------------------- SAFETY
function vSafety(root) {
  const d = S.d;
  root.innerHTML = `<div class="banner">🛡 "No detected signal" is <b>not</b> a safety claim. Endpoint models are
    baselines on synthetic labels (demonstration-only) and require external validation.</div>
    <div class="section-title">Safety register (${(d.safety || []).length})</div>
    <div class="panel scroll-x">${safetyTable(d.safety || [])}</div>`;
}
function safetyTable(safety) {
  if (!safety.length) return `<div class="empty">no safety assessments</div>`;
  const eps = [...new Set(safety.flatMap((s) => (s.endpoints || []).map((e) => e.endpoint)))];
  const head = `<tr><th>candidate</th><th>max severity</th><th class="num">flags</th>${eps.map((e) => `<th class="num">${esc(e)}</th>`).join("")}</tr>`;
  const body = safety.map((s) => {
    const map = Object.fromEntries((s.endpoints || []).map((e) => [e.endpoint, e]));
    const cells = eps.map((ep) => { const e = map[ep];
      if (!e) return `<td class="num muted">—</td>`;
      const col = SEV_COLOR[e.severity];
      const val = e.score != null ? fmt(e.score, 3) : (e.flag ? "⚑" : "○");
      return `<td class="num"><span class="dot" style="background:${col}"></span>${val}</td>`; }).join("");
    return `<tr class="clickable" onclick="openCand('${s.candidate_id}')"><td class="mono">${esc(s.candidate_id.slice(0, 14))}</td>
      <td><span class="badge" style="color:${SEV_COLOR[s.max_severity]};border-color:${SEV_COLOR[s.max_severity]}">${esc(s.max_severity)}</span></td>
      <td class="num">${s.n_flags}</td>${cells}</tr>`;
  }).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------- EVIDENCE
function vEvidence(root) {
  const d = S.d;
  const quarantined = (d.evidence || []).filter((e) => e.value?.quarantined).length;
  root.innerHTML = `
    ${quarantined ? `<div class="banner">🔒 ${quarantined} literature source(s) quarantined for prompt injection —
      treated as untrusted data and excluded from summarization.</div>` : ""}
    <div class="two">
      <div><div class="section-title">Evidence ledger (${(d.evidence || []).length})</div>
        <div class="panel scroll-x" style="max-height:520px;overflow-y:auto">${evidenceTable(d.evidence || [])}</div></div>
      <div><div class="section-title">Decisions (${(d.decisions || []).length})</div>
        <div class="panel scroll-x" style="max-height:520px;overflow-y:auto">${decisionTable(d.decisions || [])}</div></div>
    </div>`;
}
function evidenceTable(ev) {
  if (!ev.length) return `<div class="empty">no evidence</div>`;
  return `<table><tbody>${ev.map((e) =>
    `<tr><td><span class="chip">${esc(e.kind)}</span></td>
      <td><span class="badge ${TRUST[e.trust_label] || "b-mut"}">${esc(e.trust_label)}</span>
        ${e.value?.quarantined ? '<span class="badge b-bad">quarantined</span>' : ""}</td>
      <td>${esc((e.statement || "").slice(0, 90))}<div class="muted mono" style="font-size:10.5px">${esc(e.provenance?.source || "")} · q=${fmt(e.quality_score, 2)}</div></td></tr>`).join("")}</tbody></table>`;
}
function decisionTable(dec) {
  if (!dec.length) return `<div class="empty">no decisions</div>`;
  const order = { PROMOTE: 0, DEFER: 1, REJECT: 2 };
  return `<table><tbody>${dec.slice().sort((a, b) => (order[a.outcome] ?? 3) - (order[b.outcome] ?? 3)).map((x) =>
    `<tr class="clickable" onclick="openCand('${x.subject_id}')"><td class="mono">${esc(x.subject_id.slice(0, 14))}</td>
      <td><span class="badge ${DECISION[x.outcome] || "b-mut"}">${esc(x.outcome)}</span></td>
      <td class="muted">${esc((x.rationale || "").slice(0, 70))}</td></tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- SENSITIVITY & EXPERIMENTS
function vSensitivity(root) {
  const d = S.d; const sens = d.sensitivity;
  root.innerHTML = `
    <div class="section-title">Sensitivity analysis</div>
    <div class="panel scroll-x">${sens ? sensTable(sens) : '<div class="empty">not available</div>'}</div>
    <div class="section-title">Proposed next experiments (${(d.experiments || []).length})</div>
    <div class="panel">${(d.experiments || []).map((e) =>
      `<div style="padding:8px 0;border-bottom:1px solid var(--line)">
        <b>${esc(e.title)}</b> <span class="badge b-mut">${esc(e.trust_label)}</span>
        <span class="chip">${esc(e.assay_type)}</span> <span class="chip">priority ${fmt(e.priority, 2)}</span>
        <div class="muted">${esc(e.rationale)}</div></div>`).join("") || '<div class="empty">none</div>'}</div>
    <div class="section-title">Hypotheses (${(d.hypotheses || []).length})</div>
    <div class="panel">${(d.hypotheses || []).map((h) =>
      `<div style="padding:6px 0"><span class="badge b-warn">${esc(h.trust_label)}</span> ${esc(h.statement)}</div>`).join("") || '<div class="empty">none</div>'}</div>`;
}
function sensTable(sens) {
  return `<table><thead><tr><th>scenario</th><th>description</th><th class="num">Jaccard vs base</th><th>changed</th></tr></thead>
    <tbody>${(sens.rows || []).map((r) =>
      `<tr><td>${esc(r.scenario)}</td><td class="muted">${esc(r.description)}</td>
        <td class="num"><span class="bar-inline" style="width:70px"><i style="width:${(100 * (r.jaccard_vs_base ?? 0)).toFixed(0)}%"></i></span> ${fmt(r.jaccard_vs_base, 2)}</td>
        <td>${r.changed ? '<span class="badge b-warn">yes</span>' : '<span class="badge b-good">no</span>'}</td></tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- REPRODUCIBILITY
function vRepro(root) {
  const d = S.d; const m = d.manifest;
  if (!m) { root.innerHTML = `<div class="empty">Manifest not available (run not yet bundled).</div>`; return; }
  const rows = (o) => Object.entries(o || {}).map(([k, v]) => `<div>${esc(k)}</div><div class="mono">${esc(v)}</div>`).join("");
  root.innerHTML = `
    <div class="section-title">Run manifest</div>
    <div class="two">
      <div class="panel"><h3>Environment</h3><div class="kv">
        <div>repo revision</div><div class="mono">${esc((m.repo_revision || "—")).slice(0, 12)}</div>
        <div>dirty</div><div>${m.repo_dirty}</div>
        <div>python</div><div>${esc(m.python_version)}</div>
        <div>platform</div><div class="mono">${esc(m.platform)}</div>
        <div>lock hash</div><div class="mono">${esc((m.lock_hash || "—")).slice(0, 22)}</div>
        <div>config hash</div><div class="mono">${esc((m.config_hash || "—")).slice(0, 22)}</div>
        <div>seeds</div><div class="mono">${esc(JSON.stringify(m.seeds))}</div>
        <div>deterministic</div><div>${m.deterministic}</div>
      </div></div>
      <div class="panel"><h3>Versions</h3><div class="kv">${rows(m.package_versions)}</div>
        <h3 style="margin-top:14px">Data snapshots</h3><div class="kv">${rows(m.data_snapshots)}</div></div>
    </div>
    <div class="section-title">Artifact hashes (${Object.keys(m.artifact_hashes || {}).length})</div>
    <div class="panel scroll-x" style="max-height:360px;overflow-y:auto"><div class="kv">${rows(m.artifact_hashes)}</div></div>`;
}

// ---------------------------------------------------------------- NEW RUN + GATE
let EXAMPLES = [];
async function openNewRun() {
  openDrawer("#new-run");
  const log = $("#launch-log"); log.textContent = "";
  try { EXAMPLES = (await api("/examples/briefs")).items.filter((e) => e.valid); }
  catch { EXAMPLES = []; }
  $("#brief-select").innerHTML = EXAMPLES.map((e, i) =>
    `<option value="${i}">${esc(e.title)} (${esc(e.kind)})</option>`).join("") || `<option>no valid briefs</option>`;
  const showPrev = () => { const e = EXAMPLES[$("#brief-select").value]; if (!e) return;
    $("#brief-preview").innerHTML = `<b>${esc(e.title)}</b><div class="muted">${esc(e.body.description || "")}</div>
      <div style="margin-top:6px">objectives: ${(e.body.objectives || []).map((o) => `<span class="chip">${esc(o.name)} ${esc(o.direction)}</span>`).join("")}</div>`; };
  $("#brief-select").onchange = showPrev; showPrev();
}
async function launchRun() {
  const e = EXAMPLES[$("#brief-select").value]; if (!e) { toast("no brief"); return; }
  const log = $("#launch-log"); const det = $("#det").checked;
  log.textContent = `→ creating run (${e.name}, deterministic=${det})…\n`;
  $("#launch").disabled = true;
  try {
    const res = await post("/runs", { brief: e.body, deterministic: det, start: true });
    const run = res.run; log.textContent += `→ run ${run.id}\n→ state: ${run.state}\n`;
    await loadRuns(); await selectRun(run.id);
    if (run.state === "AWAITING_APPROVAL") { log.textContent += "→ paused at approval gate.\n"; renderGate(run.id, log); }
    else { log.textContent += "✓ complete.\n"; setTimeout(closeDrawers, 900); }
  } catch (err) { log.textContent += "✗ " + err.message + "\n"; }
  finally { $("#launch").disabled = false; }
}
function renderGate(runId, log) {
  const gate = (S.d?.run?.gates || []).find((g) => g.status === "PENDING")?.name || "portfolio_review";
  const div = document.createElement("div"); div.className = "gate-actions";
  div.innerHTML = `<button class="btn btn-accent" id="g-appr">✓ Approve ${esc(gate)}</button>
    <button class="btn" id="g-rej">✕ Reject</button>`;
  log.after(div);
  $("#g-appr").onclick = async () => {
    div.remove(); log.textContent += `→ approving ${gate}…\n`;
    try { const r = await post(`/runs/${runId}/approvals/${gate}`, { decision: "approve", by: "ui" });
      log.textContent += `✓ run ${r.state}\n`; await loadRuns(); await selectRun(runId); setTimeout(closeDrawers, 900); }
    catch (e) { log.textContent += "✗ " + e.message + "\n"; }
  };
  $("#g-rej").onclick = async () => {
    div.remove(); const reason = prompt("Rejection reason?") || "rejected via UI";
    try { const r = await post(`/runs/${runId}/approvals/${gate}`, { decision: "reject", reason, by: "ui" });
      log.textContent += `✕ run ${r.state}\n`; await loadRuns(); await selectRun(runId); }
    catch (e) { log.textContent += "✗ " + e.message + "\n"; }
  };
}
window.gateBar = function () { openDrawer("#new-run"); const log = $("#launch-log");
  log.textContent = `Run ${S.runId} is awaiting approval.\n`; renderGate(S.runId, log); };

// ---------------------------------------------------------------- drawers
function openDrawer(sel) { $("#drawer-backdrop").classList.remove("hidden"); $(sel).classList.remove("hidden"); }
function closeDrawers() { $("#drawer-backdrop").classList.add("hidden");
  $("#new-run").classList.add("hidden"); $("#cand-detail").classList.add("hidden"); }

boot();
