"use strict";

// ---- tiny DOM helper -------------------------------------------------------
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids) {
    if (kid == null) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (typeof x === "number" ? x.toFixed(d).replace(/\.?0+$/, "") : String(x));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---- state -----------------------------------------------------------------
let es = null;
let strategy = "adaptive";
// The candidate source is NOT a console setting. The master agent chooses it
// during transcription and announces the choice on the event stream; this holds
// what it chose, for the run header only.
let chosenSource = null;
let genPool = null;            // installed de-novo generators, from /api/health
let chosenArms = null;         // optional assessment arms the master agent switched on
let armPool = null;            // optional arms this deployment could run, from /api/health
let currentThinkPre = null;    // where streamed thinking is appended
let currentContentPre = null;
const phaseCards = {};         // key -> element
const cycleEls = {};           // cycle -> container element
let points = [];               // scoreboard points

// ---- init ------------------------------------------------------------------
async function init() {
  // strategy segmented control
  document.querySelectorAll("#strategy button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#strategy button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      strategy = b.dataset.val;
    };
  });
  $("preset").onchange = (e) => {
    const p = presetMap[e.target.value];
    if (p) $("instruction").value = p.text;
  };
  $("launch").onclick = launch;
  $("showcontent").onchange = (e) => { $("contentstream").hidden = !e.target.checked; };

  await loadPresets();
  await loadHealth();
  setInterval(loadHealth, 15000);
}

let presetMap = {};
async function loadPresets() {
  try {
    const r = await fetch("/api/presets");
    const d = await r.json();
    const sel = $("preset");
    d.presets.forEach((p) => {
      presetMap[p.id] = p;
      sel.append(el("option", { value: p.id }, p.label));
    });
    if (d.config) {
      $("maxcycles").value = d.config.max_cycles;
      strategy = d.config.refine_strategy;
      document.querySelectorAll("#strategy button").forEach((b) =>
        b.classList.toggle("on", b.dataset.val === strategy));
      renderGeneratorStatus();
    }
  } catch (e) { /* ignore */ }
}

async function loadHealth() {
  const h = $("health");
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const sel = $("model");
    if (sel.children.length === 0 && d.ollama.ok) {
      d.ollama.models.forEach((m) => sel.append(el("option", { value: m }, m)));
      if (d.ollama.models.includes(d.model)) sel.value = d.model;
    }
    if (!d.ollama.ok) {
      h.innerHTML = "";
      h.append(el("span", { class: "dot dot-bad" }), "Ollama unreachable");
    } else if (!d.model_installed) {
      h.innerHTML = "";
      h.append(el("span", { class: "dot dot-wait" }), `model ${d.model} not pulled`);
    } else {
      h.innerHTML = "";
      h.append(el("span", { class: "dot dot-ok" }), `Ollama · ${d.ollama.models.length} models`);
    }
    if (d.generator) { genPool = d.generator; renderGeneratorStatus(); }
    if (d.arms) { armPool = d.arms; renderGeneratorStatus(); }
  } catch (e) {
    h.innerHTML = "";
    h.append(el("span", { class: "dot dot-bad" }), "API error");
  }
}

// ---- launch ----------------------------------------------------------------
async function launch() {
  const instruction = $("instruction").value.trim();
  if (!instruction) { $("instruction").focus(); return; }
  // reset
  if (es) es.close();
  $("timeline").innerHTML = "";
  $("final").hidden = true; $("final").innerHTML = "";
  $("scoreboard").hidden = true; $("chart").innerHTML = ""; $("scoretable").innerHTML = "";
  for (const k in phaseCards) delete phaseCards[k];
  for (const k in cycleEls) delete cycleEls[k];
  points = [];
  $("launch").disabled = true;
  setPulse("run");

  chosenSource = null;
  chosenArms = null;
  // No `generator` and no `n_generated`: omitting them is what tells the server
  // that the master agent decides. Sending one would pin it.
  const body = {
    instruction,
    strategy,
    max_cycles: parseInt($("maxcycles").value, 10) || 3,
    model: $("model").value || undefined,
  };
  let runId;
  try {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    runId = (await r.json()).run_id;
  } catch (e) {
    renderError("Could not start run: " + e.message);
    $("launch").disabled = false; setPulse("fail");
    return;
  }
  $("runmeta").textContent =
    `run ${runId} · ${strategy} · ≤${body.max_cycles} cycles · source: master agent decides`;
  $("thinkpanel").hidden = false;

  es = new EventSource(`/api/runs/${runId}/events`);
  es.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) { console.error(e, m.data); } };
  es.onerror = () => { /* auto-reconnect; stream closes on DONE */ };
}

// ---- thinking panel --------------------------------------------------------
function setPulse(mode) {
  const p = $("pulse");
  p.className = "pulse" + (mode === "done" ? " done" : mode === "fail" ? " fail" : "");
}
const PHASE_LABEL = { plan: "Planning", transcribe: "Transcribing", critique: "Critiquing", refine: "Adapting", cycle: "Executing" };
function setPhaseChip(phase, title) {
  $("phasechip").textContent = PHASE_LABEL[phase] || phase || "…";
  $("thinktitle").textContent = title || "";
}
function appendThinking(delta) {
  const live = $("thinkstream");
  live.textContent += delta;
  live.scrollTop = live.scrollHeight;
  if (currentThinkPre) { currentThinkPre.textContent += delta; currentThinkPre.scrollTop = currentThinkPre.scrollHeight; }
}
function appendContent(delta) {
  const c = $("contentstream");
  c.textContent += delta; c.scrollTop = c.scrollHeight;
  if (currentContentPre) currentContentPre.textContent += delta;
}

// ---- containers ------------------------------------------------------------
function getCycle(cycle) {
  if (cycleEls[cycle]) return cycleEls[cycle];
  const wrap = el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("span", { class: "tag tag-cycle" }, "Cycle " + cycle),
      el("span", { class: "cyc-title", text: "executing…" })));
  const body = el("div", { class: "card-body cyc-body" });
  wrap.append(body);
  $("timeline").append(wrap);
  cycleEls[cycle] = body;
  return body;
}

function container(phase, cycle) {
  return cycle && cycle > 0 ? getCycle(cycle) : $("timeline");
}

// make a phase card with a live thinking accordion; returns its body element
function makePhaseCard(phase, cycle, title) {
  const key = phase + ":" + (cycle || 0);
  const tagcls = "tag-" + phase;
  const card = el("div", { class: "card" });
  const head = el("div", { class: "card-head" },
    el("span", { class: "tag " + tagcls }, phase),
    el("span", {}, title || phase));
  const body = el("div", { class: "card-body" });
  const think = el("details", { class: "think-accordion", open: "" },
    el("summary", {}, "💭 reasoning (live)"),
    el("pre", {}));
  card.append(head, body, think);
  container(phase, cycle).append(card);
  phaseCards[key] = { card, body, thinkPre: think.querySelector("pre"), think };
  currentThinkPre = think.querySelector("pre");
  currentContentPre = null;
  return phaseCards[key];
}
function phaseBody(phase, cycle) {
  const key = phase + ":" + (cycle || 0);
  return phaseCards[key] || makePhaseCard(phase, cycle, phase);
}

// ---- event router ----------------------------------------------------------
function handle(ev) {
  switch (ev.type) {
    case "run_started": onRunStarted(ev); break;
    case "phase_started": onPhaseStarted(ev); break;
    case "thinking": appendThinking(ev.data.delta || ""); break;
    case "content": appendContent(ev.data.delta || ""); break;
    case "plan_ready": onPlanReady(ev); break;
    case "requirements_ready": onRequirementsReady(ev); break;
    case "cycle_started": onCycleStarted(ev); break;
    case "dispatch": onDispatch(ev); break;
    case "chemical_result": onChemical(ev); break;
    case "evaluation_result": onEvaluation(ev); break;
    case "metrics": onMetrics(ev); break;
    case "critique_ready": onCritique(ev); break;
    case "decision": onDecision(ev); break;
    case "requirements_adjusted": onAdjusted(ev); break;
    case "run_succeeded": onSucceeded(ev); break;
    case "run_failed": onFailed(ev); break;
    case "log": onLog(ev); break;
    case "done": if (es) es.close(); $("launch").disabled = false; break;
  }
}

function onRunStarted(ev) {
  const emptyEl = $("empty"); if (emptyEl) emptyEl.remove();
  $("timeline").append(el("div", { class: "card" },
    el("div", { class: "card-head" }, el("span", { class: "tag tag-cycle" }, "instruction"),
      "Master agent engaged · " + ev.data.model + " · " + ev.data.strategy),
    el("div", { class: "card-body" }, el("p", { text: ev.data.instruction }))));
}

function onPhaseStarted(ev) {
  setPhaseChip(ev.phase, ev.title);
  setPulse("run");
  $("thinkstream").textContent = "";
  makePhaseCard(ev.phase, ev.cycle, ev.title);
}

function onPlanReady(ev) {
  const b = phaseBody("plan", 0).body;
  const plan = ev.data.plan, crits = ev.data.criteria || [];
  b.innerHTML = "";
  b.append(el("h4", {}, "Goal"), el("p", { text: plan.goal_restatement }));
  if (plan.strategy) b.append(el("h4", {}, "Strategy"), el("p", { text: plan.strategy }));
  if (plan.interpretation) b.append(el("h4", {}, "Interpretation"), el("p", { text: plan.interpretation }));
  if (plan.subtasks && plan.subtasks.length) {
    const ul = el("ul", { class: "tight" });
    plan.subtasks.forEach((t) => ul.append(el("li", {}, el("b", { text: (t.agent || "") + ": " }), t.description)));
    b.append(el("h4", {}, "Subtasks"), ul);
  }
  const ul2 = el("ul", { class: "tight" });
  crits.forEach((c) => ul2.append(el("li", {}, el("code", { text: c.metric }), " " + c.target, c.rationale ? el("span", { class: "tgt" }, "  — " + c.rationale) : null)));
  b.append(el("h4", {}, "Success criteria (what the evaluation agent will measure)"), ul2);
  // Criteria naming a metric nothing computes can never be met. They are shown
  // rather than hidden — the planner wanting something unmeasurable is a finding.
  const dropped = ev.data.unscorable || [];
  if (dropped.length) {
    const ul3 = el("ul", { class: "tight" });
    dropped.forEach((c) => ul3.append(el("li", {}, el("code", { text: c.metric }), " " + c.target)));
    b.append(el("h4", {}, `Excluded — this platform cannot measure these (${dropped.length})`), ul3);
  }
  collapse("plan", 0);
}

function onRequirementsReady(ev) {
  const b = phaseBody("transcribe", 0).body;
  const r = ev.data.requirements;
  b.innerHTML = "";
  b.append(el("p", { class: "tgt", text: "This is the machine-readable generation prompt handed to the chemical agent." }));
  b.append(reqView(r));
  collapse("transcribe", 0);
}

function reqView(r) {
  const wrap = el("div", {});
  const chips = el("div", { class: "chips" },
    chip("kind", r.kind),
    r.target_name ? chip("target", r.target_name) : null,
    r.phenotype_description ? chip("phenotype", (r.phenotype_description || "").slice(0, 40)) : null,
    chip("novelty", r.novelty),
    chip("diversity", r.diversity_requirement),
    chip("max_candidates", r.max_candidates),
    chip("shortlist", r.shortlist_size));
  wrap.append(chips);
  if (r.objectives && r.objectives.length) {
    const ul = el("ul", { class: "tight" });
    r.objectives.forEach((o) => ul.append(el("li", {},
      el("code", { text: o.name }), ` ${o.direction}`, o.target_value != null ? ` → ${o.target_value}` : "",
      ` (w=${o.weight}${o.required ? ", required" : ""})`)));
    wrap.append(el("h4", {}, "Objectives"), ul);
  }
  if (r.property_constraints && r.property_constraints.length) {
    const ul = el("ul", { class: "tight" });
    r.property_constraints.forEach((c) => ul.append(el("li", {}, el("code", { text: `${c.property} ${c.op} ${c.value}` }), c.rationale ? "  — " + c.rationale : "")));
    wrap.append(el("h4", {}, "Hard constraints"), ul);
  }
  if (r.seed_smiles && r.seed_smiles.length) {
    const ul = el("ul", { class: "tight" });
    r.seed_smiles.forEach((s) => ul.append(el("li", {}, el("code", { text: s }))));
    wrap.append(el("h4", {}, `Proposed seed scaffolds (${r.seed_smiles.length}, RDKit-validated)`), ul);
  }
  if (r.generation_notes) wrap.append(el("h4", {}, "Notes"), el("p", { class: "tgt", text: r.generation_notes }));
  return wrap;
}
function chip(k, v) { const c = el("span", { class: "chip" }); c.append(k + " ", el("b", { text: String(v) })); return c; }

// How a candidate source reads to a person. The wire token for the de-novo arm
// is "pfm" whichever of the installed models sampled it, so every one of these
// surfaces used to name PFM for work PocketXMol had done. Mirrors
// `candidate_source_label` in llm/schemas.py; keep the two in step.
function sourceLabel(generator, backend) {
  const gen = String(generator ?? "library").toLowerCase();
  const engine = String(backend ?? "").toLowerCase();
  if (!(gen.includes("pfm") || gen.includes("denovo"))) return "library";
  const arm = (engine === "" || engine === "auto") ? "de-novo" : `de-novo (${engine})`;
  return gen.includes("librar") ? `library + ${arm}` : arm;
}

// Describe what the de-novo arm can do HERE, and, once the run has said so,
// what the master agent actually chose. This box never asks for a decision:
// which source suits a goal is a judgement about the goal, and the agent that
// read the goal is the one making it.
function renderGeneratorStatus() {
  const box = $("genstatus");
  if (!box) return;
  box.hidden = false;
  const parts = [];
  if (chosenSource) {
    const denovo = chosenSource.generator !== "library";
    parts.push(`<strong>This run: ${esc(sourceLabel(chosenSource.generator, chosenSource.generator_backend))}</strong>` +
      (denovo ? ` · model <code>${esc(chosenSource.generator_backend || "auto")}</code>` +
                ` · ${esc(String(chosenSource.n_generated ?? ""))} de-novo molecules` : "") +
      (chosenSource.pinned ? ` <span class="dim">(pinned by the operator)</span>`
                           : ` <span class="dim">(chosen by the master agent)</span>`));
  } else {
    parts.push(`<strong>The master agent chooses this.</strong> It reads the goal, then
      picks the compound library, de-novo generation, or both, and says which under
      <em>Transcribing</em>.`);
  }
  if (!chosenArms && armPool && Object.keys(armPool).length) {
    parts.push(`<span class="dim">Optional assessment arms available for the agent to
      switch on: ${Object.keys(armPool).map((k) => `<code>${esc(k)}</code>`).join(", ")}.
      All default to off.</span>`);
  }
  if (chosenArms) {
    const arms = [];
    if (chosenArms.property_backends.length) arms.push(`property panels: ${chosenArms.property_backends.map(esc).join(", ")}`);
    if (chosenArms.route_backend && chosenArms.route_backend !== "none") arms.push(`routes: ${esc(chosenArms.route_backend)}`);
    if (chosenArms.complex_backend && chosenArms.complex_backend !== "none") arms.push(`complex: ${esc(chosenArms.complex_backend)}`);
    if (chosenArms.target_evidence) arms.push("target evidence");
    parts.push(arms.length
      ? `<span class="dim">Assessment arms switched on: ${arms.join(" · ")}.</span>`
      : `<span class="dim">No optional assessment arm was switched on: no learned property
         panel, no retrosynthetic route search, no predicted complex, no target dossier.</span>`);
  }
  const engines = (genPool && genPool.installed) || [];
  if (engines.length) {
    parts.push(`<span class="dim">De-novo generators installed: ${engines.map(esc).join(", ")}.</span>`);
  } else {
    parts.push(`<span class="dim">No de-novo generator is installed here, so every candidate
      comes from the compound library plus deterministic seed analogues.</span>`);
  }
  if (engines.length && genPool && !genPool.pocket) {
    parts.push(`<span class="dim">⚠️ No prepared binding pocket is configured. These
      generators sample a <em>pocket</em>, not a whole protein, so the de-novo arm cannot
      run until one is set. Cut one with
      <code>python scripts/prepare_pocket.py &lt;structure&gt;.pdb --ligand &lt;code&gt; -o pocket.pdb</code>
      and point <code>LLM_AGENT_PFM_POCKET</code> at it.</span>`);
  } else if (engines.length && genPool && genPool.pocket) {
    parts.push(`<span class="dim">Pocket: ${esc(genPool.pocket)}. De-novo molecules are
      screened, filtered and ranked by the same pipeline as library compounds and carry the
      trust label <b>PROPOSED</b>.</span>`);
  }
  box.innerHTML = parts.map((x) => `<div>${x}</div>`).join("");
}

function collapse(phase, cycle) {
  const pc = phaseCards[phase + ":" + (cycle || 0)];
  if (pc && pc.think) pc.think.open = false;
}

function onCycleStarted(ev) {
  getCycle(ev.cycle);
  const head = cycleEls[ev.cycle].parentElement.querySelector(".cyc-title");
  if (head) head.textContent = `${ev.data.strategy} · started`;
}

function onDispatch(ev) {
  const b = getCycle(ev.cycle);
  b.append(el("div", { class: "chips", style: "margin:6px 0" },
    el("span", { class: "chip" }, "▸ dispatch ", el("b", { text: ev.data.agent + " agent" }))));
}

function onChemical(ev) {
  const b = getCycle(ev.cycle);
  const res = ev.data.result;
  const card = el("div", { class: "card", style: "margin:8px 0" },
    el("div", { class: "card-head" }, el("span", { class: "tag tag-chemical" }, "chemical"), "Generation & screening"));
  const body = el("div", { class: "card-body" });
  if (res.error) body.append(el("div", { class: "errbox" }, "✖ " + res.error));
  body.append(el("p", { text: ev.data.summary }));
  const chips = el("div", { class: "chips" },
    chip("candidates", res.n_candidates),
    chip("passed filters", res.n_passed_filters),
    chip("pool best pIC50", fmt(res.pool_max_pIC50)),
    chip("pool median pIC50", fmt(res.pool_median_pIC50)));
  Object.entries(res.origin_counts || {}).forEach(([k, v]) => chips.append(chip(k, v)));
  body.append(chips);
  if (res.generator_backend) {
    // Never the literal "PFM" again unless PFM is what ran: this chip is read
    // straight after a four-minute PocketXMol sample and it used to deny it.
    const engine = res.generator_backend === "auto" ? "the de-novo model" : res.generator_backend;
    const label = res.generator_is_surrogate ? `${engine} SURROGATE (not the real model)` : engine;
    body.append(el("div", { class: "chips", style: "margin-top:6px" },
      chip("de-novo", `${res.n_generated} from ${label}`),
      chip("validity", fmt(res.generation_validity)),
      chip("uniqueness", fmt(res.generation_uniqueness))));
    (res.generation_warnings || []).slice(0, 2).forEach((w) =>
      body.append(el("p", { class: "tgt", text: "⚠ " + w })));
  }
  if (res.seeds_used && res.seeds_used.length) body.append(el("p", { class: "tgt", text: `seeds used: ${res.seeds_used.join(", ")}` }));
  if (res.seeds_dropped && res.seeds_dropped.length) body.append(el("p", { class: "tgt", html: `⚠ RDKit rejected: <code>${res.seeds_dropped.join("</code> <code>")}</code>` }));
  card.append(body); b.append(card);
}

function onEvaluation(ev) {
  const b = getCycle(ev.cycle);
  const rep = ev.data.report;
  const card = el("div", { class: "card", style: "margin:8px 0" },
    el("div", { class: "card-head" }, el("span", { class: "tag tag-evaluation" }, "evaluation"),
      `Judged vs criteria — ${rep.n_criteria_met}/${rep.n_criteria_total} met`));
  const body = el("div", { class: "card-body" });
  if (rep.failure_reason) {
    body.append(el("div", { class: "errbox" },
      "✖ Nothing could be scored: " + rep.failure_reason));
  }
  // criteria rows
  (rep.criteria || []).forEach((c) => {
    const met = c.met === true;
    body.append(el("div", { class: "crit" },
      el("span", { class: "mark " + (c.met == null ? "" : met ? "met" : "miss") }, c.met == null ? "•" : met ? "✓" : "✗"),
      el("span", { class: "lbl" }, c.label || c.key),
      el("span", { class: "val" }, fmt(c.value)),
      el("span", { class: "tgt" }, "target " + (c.target || "—"))));
  });
  // pool stats
  const chips = el("div", { class: "chips", style: "margin-top:10px" },
    chip("pool", rep.n_pool_candidates ?? rep.n_candidates), chip("scored", rep.n_candidates),
    chip("pareto", rep.n_pareto),
    chip("clusters", rep.n_clusters), chip("diversity", fmt(rep.diversity)),
    chip("shortlist", `${rep.shortlist_size}/${rep.requested_shortlist}`));
  body.append(chips);
  if (rep.decision_counts) {
    const dc = el("div", { class: "chips", style: "margin-top:6px" });
    Object.entries(rep.decision_counts).forEach(([k, v]) => dc.append(chip(k.toLowerCase(), v)));
    body.append(dc);
  }
  (rep.notes || []).forEach((n) => body.append(el("p", { class: "tgt", text: "• " + n })));
  // top candidates
  if (rep.top_candidates && rep.top_candidates.length) body.append(candTable(rep.top_candidates));
  card.append(body); b.append(card);
}

function candTable(rows) {
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {},
    ...["#", "SMILES", "pIC50", "QED", "SA", "hERG", "utility", "decision"].map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  rows.forEach((r, i) => {
    const dec = (r.decision || "").toLowerCase();
    tb.append(el("tr", {},
      el("td", {}, String(i + 1)),
      el("td", { class: "smiles" }, r.smiles),
      el("td", {}, fmt(r.pIC50)),
      el("td", {}, fmt(r.qed)),
      el("td", {}, fmt(r.sa_score)),
      el("td", {}, fmt(r.hERG)),
      el("td", {}, fmt(r.utility)),
      el("td", {}, r.decision ? el("span", { class: "badge badge-" + dec }, r.decision) : "—")));
  });
  t.append(tb);
  return t;
}

function onMetrics(ev) {
  points.push(ev.data.point);
  drawScoreboard();
}

function onCritique(ev) {
  const b = phaseBody("critique", ev.cycle).body;
  const c = ev.data.critique;
  b.innerHTML = "";
  const v = (c.overall_verdict || "REFINE").toLowerCase();
  b.append(el("p", {}, el("span", { class: "badge badge-" + v }, c.overall_verdict),
    el("span", { class: "tgt", style: "margin-left:8px" }, "confidence: " + (c.confidence || "—"))));
  if (c.summary) b.append(el("p", { text: c.summary }));
  if (c.diagnosis) b.append(el("h4", {}, "Diagnosis (root cause)"), el("p", { text: c.diagnosis }));
  collapse("critique", ev.cycle);
}

function onDecision(ev) {
  const b = getCycle(ev.cycle);
  const v = (ev.data.verdict || "").toLowerCase();
  b.append(el("div", { style: "margin:10px 0" },
    el("span", { class: "badge badge-" + v }, "Decision: " + ev.data.verdict),
    ev.data.is_final ? el("span", { class: "tgt", style: "margin-left:8px" }, "final") : null,
    ev.data.reason ? el("p", { class: "tgt", style: "margin-top:6px", text: ev.data.reason }) : null));
  const head = cycleEls[ev.cycle].parentElement.querySelector(".cyc-title");
  if (head) head.textContent = ev.data.verdict + (ev.data.is_final ? " · final" : "");
}

function onAdjusted(ev) {
  // adaptive: fill the refine phase card; naive: append a note to the cycle
  if (ev.data.strategy === "naive") {
    getCycle(ev.cycle).append(el("div", { class: "errbox", style: "margin:8px 0;border-color:#5a4a1a;background:#2c2410;color:var(--warn)" },
      "↺ Naive retry — requirements unchanged. Deterministic backend ⇒ identical results next cycle (ablation baseline)."));
    return;
  }
  const b = phaseBody("refine", ev.cycle).body;
  b.innerHTML = "";
  if (ev.data.strategy_note) b.append(el("p", {}, el("b", {}, "Strategy: "), ev.data.strategy_note));
  // LLM's own change list
  if (ev.data.changes && ev.data.changes.length) {
    b.append(el("h4", {}, "Adjustments (diagnosis → change)"));
    ev.data.changes.forEach((c) => {
      b.append(el("div", { class: "change" },
        el("div", {}, el("span", { class: "p" }, c.parameter), " : ",
          el("span", { class: "old" }, c.old_value), el("span", { class: "arrow" }, "→"), el("span", { class: "new" }, c.new_value)),
        el("div", { class: "reason" }, c.reason)));
    });
  }
  // computed field-level diff
  if (ev.data.diff && ev.data.diff.length) {
    const d = el("div", { class: "diff" });
    ev.data.diff.forEach((row) => d.append(el("div", { class: "row" },
      el("span", { class: "field" }, row.field),
      el("span", {}, el("span", { class: "old" }, row.old), el("span", { class: "arrow" }, "→"), el("span", { class: "new" }, row.new)))));
    b.append(el("h4", {}, "Effective diff (old → new)"), d);
  }
  collapse("refine", ev.cycle);
}

// ---- scoreboard chart ------------------------------------------------------
function drawScoreboard() {
  if (!points.length) return;
  $("scoreboard").hidden = false;
  // table
  const t = el("table", {});
  t.append(el("thead", {}, el("tr", {}, ...["cycle", "criteria met", "best pIC50", "hard-pass", "diversity", "pareto", "clusters", "utility"].map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  points.forEach((p) => tb.append(el("tr", {},
    el("td", {}, "#" + p.cycle),
    el("td", {}, `${p.n_criteria_met}/${p.n_criteria_total}`),
    el("td", {}, fmt(p.max_pIC50)),
    el("td", {}, fmt(p.hard_constraint_pass_rate)),
    el("td", {}, fmt(p.diversity)),
    el("td", {}, p.n_pareto),
    el("td", {}, p.n_clusters),
    el("td", {}, fmt(p.headline_utility)))));
  t.append(tb);
  $("scoretable").innerHTML = ""; $("scoretable").append(t);
  // chart
  $("chart").innerHTML = chartSVG(points);
}

function chartSVG(pts) {
  const W = 640, H = 180, pad = 34;
  const n = pts.length;
  const xs = (i) => n <= 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1);
  const total = Math.max(1, ...pts.map((p) => p.n_criteria_total || 1));
  const yMet = (v) => H - pad - ((v || 0) / total) * (H - 2 * pad);
  const pmax = Math.max(8, ...pts.map((p) => p.max_pIC50 || 0));
  const pmin = Math.min(...pts.map((p) => p.max_pIC50 || 0), 4);
  const yP = (v) => v == null ? null : H - pad - ((v - pmin) / Math.max(0.1, pmax - pmin)) * (H - 2 * pad);
  const line = (accessor, color) => {
    let d = "", dots = "";
    pts.forEach((p, i) => {
      const y = accessor(p); if (y == null) return;
      d += (d ? " L" : "M") + xs(i).toFixed(1) + " " + y.toFixed(1);
      dots += `<circle cx="${xs(i).toFixed(1)}" cy="${y.toFixed(1)}" r="3.5" fill="${color}"/>`;
    });
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>${dots}`;
  };
  const xlabels = pts.map((p, i) => `<text x="${xs(i)}" y="${H - 10}" fill="#62748d" font-size="10" text-anchor="middle">cycle ${p.cycle}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#263447"/>
    ${line((p) => yMet(p.n_criteria_met), "#34d399")}
    ${line((p) => yP(p.max_pIC50), "#38bdf8")}
    ${xlabels}
    <g font-size="11">
      <rect x="${pad}" y="8" width="10" height="10" fill="#34d399"/><text x="${pad + 15}" y="17" fill="#8ea3bd">criteria met</text>
      <rect x="${pad + 110}" y="8" width="10" height="10" fill="#38bdf8"/><text x="${pad + 125}" y="17" fill="#8ea3bd">best pIC50</text>
    </g>
  </svg>`;
}

// ---- finalize --------------------------------------------------------------
function onSucceeded(ev) {
  setPulse("done"); setPhaseChip("cycle", "complete"); $("launch").disabled = false;
  const res = ev.data.result;
  const rep = res.portfolio;
  const f = $("final"); f.hidden = false; f.innerHTML = "";
  const card = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("span", { class: "tag tag-refine" }, "result"), ev.title));
  const body = el("div", { class: "card-body" });
  body.append(el("p", {}, el("b", {}, `Best cycle: #${res.best_cycle} · ${rep.n_criteria_met}/${rep.n_criteria_total} criteria met`)));
  const chips = el("div", { class: "chips" },
    chip("strategy", res.strategy), chip("cycles run", res.cycles.length),
    chip("shortlist", rep.shortlist_size), chip("pareto", rep.n_pareto),
    chip("clusters", rep.n_clusters), chip("diversity", fmt(rep.diversity)));
  body.append(chips);
  body.append(el("h4", { style: "margin-top:12px" }, "Prioritized shortlist"));
  if (rep.top_candidates && rep.top_candidates.length) body.append(candTable(rep.top_candidates));
  card.append(body); f.append(card);
  f.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// The harness narrates the things worth knowing about a decode: a reply that ran
// out of budget, a self-repair, criteria nothing can measure. Dropping these was
// how a run could spend three cycles chasing an unscorable target in silence.
function onLog(ev) {
  // The transcription step announces the candidate source it chose. Mirror it
  // into the sidebar so the panel that used to ASK for a source now REPORTS one.
  if (ev.data && ev.data.generator && ev.phase === "transcribe") {
    chosenSource = {
      generator: ev.data.generator,
      generator_backend: ev.data.generator_backend,
      n_generated: ev.data.n_generated,
      pinned: !!ev.data.pinned,
    };
    renderGeneratorStatus();
  }
  // The same step announces which OPTIONAL assessment arms it switched on. They
  // default to off and were, until they had fields, switched on by nothing at
  // all — so "none" is worth showing rather than leaving the operator to infer.
  if (ev.data && ev.phase === "transcribe" && ev.data.route_backend !== undefined) {
    chosenArms = {
      property_backends: ev.data.property_backends || [],
      route_backend: ev.data.route_backend,
      complex_backend: ev.data.complex_backend,
      target_evidence: !!ev.data.target_evidence,
    };
    renderGeneratorStatus();
  }
  const where = ev.cycle && ev.cycle > 0 ? getCycle(ev.cycle) : $("timeline");
  if (!where) return;
  // A pipeline that stopped before scoring is the whole story of its cycle, not
  // a footnote: show the engine's reason in full rather than as a 300-char stub.
  if (ev.data && ev.data.pipeline_error) {
    where.append(el("div", { class: "errbox" },
      "✖ " + (ev.title || "pipeline failed") + " — " + ev.data.pipeline_error));
    return;
  }
  const line = el("div", { class: "logline" }, "· " + (ev.title || "note"));
  if (ev.data.detail) line.append(" ", el("code", { text: String(ev.data.detail).slice(0, 300) }));
  where.append(line);
}

function onFailed(ev) { setPulse("fail"); $("launch").disabled = false; renderError(ev.data.error || ev.title); }
function renderError(msg) {
  const emptyEl = $("empty"); if (emptyEl) emptyEl.remove();
  $("timeline").append(el("div", { class: "errbox" }, "✖ " + msg));
}

// ---- journal ---------------------------------------------------------------
// Read-after-the-fact surface: which requests ran, how the tools performed
// across all of them, and what this deployment is made of.
let jtab = "requests";

function initJournal() {
  document.querySelectorAll("#jtabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#jtabs button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      jtab = b.dataset.val;
      loadJournal();
    };
  });
  const refresh = $("jrefresh");
  if (refresh) refresh.onclick = loadJournal;
  loadJournal();
}

async function loadJournal() {
  const body = $("jbody");
  if (!body) return;
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" }, "loading…"));
  try {
    if (jtab === "requests") return renderRequests(await (await fetch("/api/journal")).json());
    if (jtab === "tools") return renderToolStats(await (await fetch("/api/journal/tools")).json());
    return renderPools(await (await fetch("/api/pools")).json());
  } catch (e) {
    body.innerHTML = "";
    body.append(el("p", { class: "tgt" }, "journal unavailable: " + e.message));
  }
}

function statusPill(status) {
  const cls = status === "SUCCEEDED" ? "pill-ok" : status === "FAILED" ? "pill-bad" : "";
  return el("span", { class: "pill " + cls }, status || "—");
}

function renderRequests(data) {
  const body = $("jbody");
  body.innerHTML = "";
  const rows = data.requests || [];
  body.append(el("p", { class: "tgt", text: `${rows.length} recorded · ${data.root}` }));
  if (!rows.length) {
    body.append(el("p", { class: "tgt" }, "No requests recorded yet. Launch one above."));
    return;
  }
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["request", "status", "model", "source", "sec", "chat", "repairs", "tool calls", "instruction"]
    .map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  rows.forEach((r) => {
    const tr = el("tr", { class: "jrow" },
      el("td", { class: "mono" }, r.request_id),
      el("td", {}, statusPill(r.status)),
      el("td", { class: "mono" }, r.model || "—"),
      el("td", {}, r.source || "—"),
      el("td", {}, fmt(r.duration_s, 1)),
      el("td", {}, String(r.n_exchanges ?? "—")),
      el("td", {}, String(r.n_repairs ?? "—")),
      el("td", {}, String(r.n_tool_calls ?? "—")),
      el("td", {}, (r.instruction || "").slice(0, 70)));
    tr.onclick = () => showRequest(r.request_id);
    tb.append(tr);
  });
  t.append(tb);
  body.append(t);
  body.append(el("div", { id: "jdetail" }));
}

async function showRequest(id) {
  const slot = $("jdetail");
  if (!slot) return;
  slot.innerHTML = "";
  slot.append(el("p", { class: "tgt" }, "loading " + id + "…"));
  try {
    const d = await (await fetch(`/api/journal/${encodeURIComponent(id)}`)).json();
    const m = d.manifest;
    slot.innerHTML = "";
    const wrap = el("div", { class: "jdetail" });
    const head = el("div", { class: "card-head" },
      el("span", { class: "tag tag-cycle" }, "request"), id, " ", statusPill(m.status));
    const inner = el("div", { class: "card-body" });
    inner.append(el("p", { text: m.instruction }));
    inner.append(el("div", { class: "chips" },
      chip("model", (m.config || {}).model || "—"),
      chip("strategy", (m.config || {}).strategy || "—"),
      chip("source", (m.config || {}).generator
        ? sourceLabel((m.config || {}).generator, (m.config || {}).generator_backend)
        : "—"),
      chip("seconds", fmt(m.duration_s, 1)),
      chip("exchanges", m.llm.n_exchanges),
      chip("tokens", m.llm.eval_tokens),
      chip("repairs", m.llm.n_repairs),
      chip("undecodable", m.llm.n_decode_failures)));
    if (m.error) inner.append(el("p", { class: "logline", text: "✖ " + m.error }));
    const tools = (m.tools || {}).by_tool || [];
    if (tools.length) {
      inner.append(el("h4", { style: "margin-top:12px" }, "Tools this request used"));
      inner.append(toolTable(tools, ["tool", "operation", "backend", "calls", "errors", "total ms", "reported"]));
    }
    // The chat log is the bulky part; show what each exchange cost and how it
    // decoded, which is what you look at when a phase misbehaved.
    if (d.chat && d.chat.length) {
      inner.append(el("h4", { style: "margin-top:12px" }, `Chat log (${d.chat.length} exchanges)`));
      const t = el("table", { class: "data" });
      t.append(el("thead", {}, el("tr", {}, ...["phase", "cycle", "try", "mode", "sec", "tokens", "stop", "decode", "ok"]
        .map((h) => el("th", {}, h)))));
      const tb = el("tbody", {});
      d.chat.forEach((c) => tb.append(el("tr", {},
        el("td", {}, c.phase), el("td", {}, String(c.cycle)), el("td", {}, String(c.attempt)),
        el("td", { class: "mono" }, c.mode), el("td", {}, fmt(c.elapsed_s, 1)),
        el("td", {}, String(c.eval_count ?? "—")), el("td", {}, c.done_reason || "—"),
        el("td", { class: "mono" }, c.decode || "—"),
        el("td", {}, el("span", { class: "pill " + (c.ok ? "pill-ok" : "pill-bad") }, c.ok ? "ok" : "fail")))));
      t.append(tb);
      inner.append(t);
    }
    wrap.append(head, inner);
    slot.append(wrap);
  } catch (e) {
    slot.innerHTML = "";
    slot.append(el("p", { class: "tgt" }, "could not load: " + e.message));
  }
}

function toolTable(tools, headers) {
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  tools.forEach((x) => {
    const metrics = Object.entries(x.metrics || {}).map(([k, v]) => `${k}=${fmt(v.mean)}`).join("  ") || "—";
    tb.append(el("tr", {},
      el("td", { class: "mono" }, x.tool_id),
      el("td", {}, x.operation),
      el("td", {}, x.backend || "—"),
      el("td", {}, String(x.n_calls)),
      el("td", {}, String(x.n_errors)),
      el("td", {}, fmt(x.total_ms, 1)),
      el("td", { class: "mono" }, metrics)));
  });
  t.append(tb);
  return t;
}

function renderToolStats(data) {
  const body = $("jbody");
  body.innerHTML = "";
  const tools = data.tools || [];
  body.append(el("p", { class: "tgt", text: `across ${data.n_requests} recorded requests` }));
  if (!tools.length) {
    body.append(el("p", { class: "tgt" }, "No tool usage recorded yet."));
    return;
  }
  body.append(toolTable(tools, ["tool", "operation", "backend", "calls", "errors", "total ms", "reported quality"]));
  body.append(el("p", { class: "tgt", style: "margin-top:8px" },
    "Reported quality is what each tool says about its own output — in-domain fraction for the "
    + "activity model, validity and is_surrogate for de-novo generation. A number here is how you "
    + "notice a tool degrading before it shows up in a shortlist."));
}

// ---- the pools -------------------------------------------------------------
// Both pools are grouped by the taxonomy their ids already encode, rather than by
// a list maintained here. A tool id is `<domain>.<...>` and an agent card carries
// its `kind`, so a capability added to the platform lands in the right section of
// this page without anyone editing this file. That mattered little when the pool
// was nine tools; the external catalogue made it thirty-nine, and a flat table of
// thirty-nine rows is a list nobody reads to the end.

//: Heading for each tool namespace. A namespace with no entry here still groups
//: correctly and shows its bare prefix — the grouping comes from the id, and this
//: map only makes it readable. `test_console_ui.py` asserts every namespace the
//: live pool reports has a label, so a new one is a failing test rather than an
//: unlabelled section.
const TOOL_DOMAINS = {
  chem: "Chemistry",
  model: "Activity model",
  activity: "Activity scoring",
  // Target evidence and pocket detection come BEFORE any chemistry: what is the
  // target, and where on it is the site a generator should be conditioned on.
  target: "Target evidence — association, essentiality",
  pocket: "Binding-site detection",
  molecule: "De-novo generation",
  structure: "Structure — docking, complexes, pose validation",
  property: "Property prediction",
  synth: "Synthesis",
  safety: "Safety",
  perturb: "Perturbation",
  binding: "Binding free energy",
  benchmark: "Benchmarking",
  platform: "Platform",
};
const AGENT_KINDS = ["REASONING", "PLATFORM", "SPECIALIST"];

const toolDomain = (id) => String(id || "").split(".")[0] || "other";
const isAvailable = (t) => t.status === "AVAILABLE";

// Available first, then by id: the question this page answers is "what can I run",
// and twenty-six refusals above the six things that work answers it backwards.
function byAvailabilityThenId(a, b) {
  if (isAvailable(a) !== isAvailable(b)) return isAvailable(a) ? -1 : 1;
  return String(a.tool_id).localeCompare(String(b.tool_id));
}

function groupBy(items, key) {
  const out = new Map();
  items.forEach((it) => {
    const k = key(it);
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(it);
  });
  return out;
}

function toolRow(t, wired) {
  // Two statuses exist and only two: the tool computes the real thing, or it
  // computes nothing and says why (ADR-0011). There is no MOCKED branch here any
  // more because no adapter reports it — the old one was dead code that also
  // implied a stand-in might be running.
  const ok = isAvailable(t);
  const reason = t.unavailable_reason || "";
  // Available and reachable are different claims. A tool no agent calls will not
  // run in a pipeline however healthy its card is, and with twenty generators in
  // the pool and one wired into candidate sourcing, showing only the status would
  // let a green badge read as "this will be used".
  const reachable = wired.has(t.tool_id);
  return el("tr", {},
    el("td", { class: "mono" }, t.tool_id),
    el("td", {}, el("span", { class: "pill " + (ok ? "pill-ok" : "pill-bad"), title: reason }, t.status)),
    el("td", {}, ok ? t.backend : "—"),
    el("td", {},
      reachable
        ? el("span", { class: "pill pill-ok", title: "an agent in the pool calls this tool" }, "wired")
        : el("span", { class: "pill", title: "in the catalogue, but no agent calls it — a run will not reach it" }, "catalogue")),
    el("td", { class: "mono" }, Object.keys(t.operations || {}).join(", ")),
    // The reason is shown, not only hovered. With most of the catalogue absent on
    // any given machine, a hover-only explanation is one nobody reads, and the
    // whole point of the reason is that it names the command that would fix it.
    el("td", { class: "tgt", title: reason }, ok ? "—" : reason));
}

function renderToolPool(body, tools, agents) {
  const available = tools.filter(isAvailable).length;
  // Derived from the agent cards, so wiring an agent to a new tool updates this
  // page by itself — the same rule the grouping follows.
  const wired = new Set([].concat(...agents.map((a) => a.uses_tools || [])));
  body.append(el("h4", { style: "margin-top:14px" },
    `Tool pool (${tools.length}) — ${available} available on this machine`));
  body.append(el("p", { class: "tgt" },
    "A tool computes the real thing or reports UNAVAILABLE and refuses; there is no "
    + "stand-in. Which ones run is a fact about this machine, not about the code, so "
    + "this is read live from the pool every time you open the tab. "
    + "“wired” means an agent in the pool calls it, so a run can reach it; "
    + "“catalogue” means it is implemented and describable but nothing dispatches to it yet."));

  const groups = groupBy(tools, (t) => toolDomain(t.tool_id));
  // Largest section first, so the twenty generators do not sit under a heading of
  // one. Ties fall back to the domain name for a stable order between reloads.
  const order = [...groups.keys()].sort((a, b) =>
    (groups.get(b).length - groups.get(a).length) || a.localeCompare(b));

  order.forEach((domain) => {
    const rows = groups.get(domain).slice().sort(byAvailabilityThenId);
    const ok = rows.filter(isAvailable).length;
    body.append(el("h5", { class: "pool-group" },
      TOOL_DOMAINS[domain] || domain,
      el("span", { class: "pill" }, `${ok}/${rows.length} available`)));
    const table = el("table", { class: "data" });
    table.append(el("thead", {}, el("tr", {},
      ...["tool", "status", "backend", "reach", "operations", "why not"].map((h) => el("th", {}, h)))));
    const tbody = el("tbody", {});
    rows.forEach((t) => tbody.append(toolRow(t, wired)));
    table.append(tbody);
    body.append(table);
  });
}

function renderAgentPool(body, agents) {
  body.append(el("h4", {}, `Agent pool (${agents.length})`));
  const groups = groupBy(agents, (a) => String(a.kind || "OTHER"));
  // The declared taxonomy first, in its documented order, then anything new.
  const order = AGENT_KINDS.filter((k) => groups.has(k))
    .concat([...groups.keys()].filter((k) => !AGENT_KINDS.includes(k)).sort());

  order.forEach((kind) => {
    const rows = groups.get(kind).slice()
      .sort((a, b) => String(a.agent_id).localeCompare(String(b.agent_id)));
    body.append(el("h5", { class: "pool-group" }, kind,
      el("span", { class: "pill" }, `${rows.length}`)));
    const table = el("table", { class: "data" });
    table.append(el("thead", {}, el("tr", {},
      ...["agent", "kind", "title", "llm", "uses tools"].map((h) => el("th", {}, h)))));
    const tbody = el("tbody", {});
    rows.forEach((a) => tbody.append(el("tr", {},
      el("td", { class: "mono" }, a.agent_id),
      el("td", {}, el("span", { class: "pill" }, a.kind)),
      el("td", {}, a.title),
      el("td", {}, a.llm_backed ? el("span", { class: "pill pill-ok" }, "yes") : "—"),
      el("td", { class: "mono" }, (a.uses_tools || []).join(", ") || "—"))));
    table.append(tbody);
    body.append(table);
  });
}

function renderPools(data) {
  const body = $("jbody");
  body.innerHTML = "";
  const agents = data.agents || [];
  renderAgentPool(body, agents);
  renderToolPool(body, data.tools || [], agents);
}

init();
initJournal();
