"use strict";

// ============================================================================
// The self-evolution panel.
//
// A sibling view to the discovery console, not a replacement for it: the
// de-novo generation controls, the thinking stream and the journal stay exactly
// where they were, and this switches in beside them. The two belong in one page
// because they are two views of one thing — what the platform did, and what that
// says about how it should change.
//
// Reuses el / $ / fmt / esc / chip / statusPill from app.js, which loads first.
// ============================================================================

let etab = "overview";
let eloaded = false;          // the panel only fetches once it is first shown
let ejobES = null;            // EventSource for a running job
let elastJob = null;

const SEV_CLASS = { HIGH: "pill-bad", MEDIUM: "pill-mock", LOW: "pill", INFO: "pill" };

// ---- view switch -----------------------------------------------------------
function initViews() {
  const tabs = document.querySelectorAll("#viewtabs button");
  tabs.forEach((b) => {
    b.onclick = () => {
      tabs.forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      const evolve = b.dataset.val === "evolve";
      const d = $("view-discovery"), e = $("view-evolve");
      if (d) d.hidden = evolve;
      if (e) e.hidden = !evolve;
      if (evolve && !eloaded) { eloaded = true; loadEvolve(); }
    };
  });
  document.querySelectorAll("#etabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#etabs button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      etab = b.dataset.val;
      loadEvolve();
    };
  });
  const r = $("erefresh");
  if (r) r.onclick = loadEvolve;
}

async function getJSON(url) {
  const r = await fetch(url);
  const body = await r.json();
  if (!r.ok) throw new Error(body && body.detail ? body.detail : "HTTP " + r.status);
  return body;
}

async function loadEvolve() {
  const body = $("ebody");
  if (!body) return;
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" }, "loading…"));
  try {
    if (etab === "overview") return renderOverview(await getJSON("/api/evolve/status"));
    if (etab === "findings") return renderFindings(await getJSON("/api/evolve/analysis"));
    if (etab === "methods") return renderMethods(await getJSON("/api/evolve/methods"));
    if (etab === "logs") return renderLogs(await getJSON("/api/evolve/logs"));
    if (etab === "benchmark") return renderBenchmark(
      await getJSON("/api/evolve/benchmark"), await getJSON("/api/evolve/records"));
    if (etab === "candidates") return renderCandidates(await getJSON("/api/evolve/archive"));
    if (etab === "run") return renderRun(await getJSON("/api/evolve/status"), await getJSON("/api/evolve/jobs"));
    return renderCycles(await getJSON("/api/evolve/cycles"));
  } catch (e) {
    body.innerHTML = "";
    body.append(el("p", { class: "tgt" }, "self-evolution unavailable: " + e.message));
  }
}

// ---- overview --------------------------------------------------------------
function renderOverview(s) {
  const body = $("ebody");
  body.innerHTML = "";
  if (s.available === false) {
    body.append(el("p", { class: "tgt" }, "the subsystem did not load: " + s.reason));
    return;
  }
  const o = s.backends.ollama, c = s.backends.claude_cli;

  const cards = el("div", { class: "ecards" });
  cards.append(
    statCard("Cheap controller", o.model || "—", o.available ? "ready" : (o.reason || "unavailable"),
      o.available ? "ok" : "bad", "Ollama · routes clusters, compresses context, schedules"),
    statCard("Deep worker", c.binary ? "claude CLI" : "not on PATH",
      c.available ? "ready" : (c.reason || "unavailable"), c.available ? "ok" : "bad",
      "Claude Code · hypotheses, adversarial verification, candidate design"),
    statCard("Evaluation kernel", "sealed", s.kernel.digest.slice(0, 16) + "…", "ok",
      s.kernel.n_guarded_files + " files hashed and verified around every measurement"),
    statCard("Evaluator model", "none", "deterministic", "ok",
      "Nothing in this system asks a model whether it succeeded"),
    statCard("Auto-promote", s.auto_promote ? "ON" : "off", s.auto_promote ? "review this" : "human required",
      s.auto_promote ? "warn" : "ok", "A passing candidate returns NEEDS_HUMAN"),
    statCard("Internet", s.allow_internet ? "ON" : "off", s.allow_internet ? "review this" : "no egress",
      s.allow_internet ? "warn" : "ok", "The loop reaches only Ollama and the claude CLI"),
    statCard("Baseline", s.has_baseline ? "recorded" : "none yet",
      s.has_baseline ? "comparisons possible" : "run one first",
      s.has_baseline ? "ok" : "warn", "Everything is compared against the platform as shipped"),
    statCard("Claude budget", s.budget.claude_calls, "$" + fmt(s.budget.claude_cost_estimate_usd, 2) + " est.",
      "ok", "A client-side list-price estimate, not a bill — the run uses the subscription"));
  body.append(cards);

  // billing containment — the thing an operator most wants to be able to check
  const bill = el("div", { class: "ebox" });
  bill.append(el("h4", {}, "Billing containment"));
  bill.append(el("p", { class: "tgt" }, s.billing.policy));
  bill.append(el("div", { class: "chips" },
    ...s.billing.removed_from_child.map((v) => chip("stripped", v))));
  if ((s.billing.present_in_this_shell || []).length) {
    bill.append(el("p", { class: "logline" },
      "⚠ set in the shell that launched this console (removed from the child, never from you): "
      + s.billing.present_in_this_shell.join(", ")));
  } else {
    bill.append(el("p", { class: "tgt" }, "None of these are set in this shell."));
  }
  body.append(bill);

  // the envelope — the platform's own operating range
  const envBox = el("div", { class: "ebox" });
  envBox.append(el("h4", {}, "Capability envelope "),
    el("span", { class: "tgt" }, "what this deployment has been measured to achieve"));
  envBox.append(el("p", { class: "tgt" }, "loading…"));
  body.append(envBox);
  getJSON("/api/evolve/envelope").then((env) => {
    envBox.innerHTML = "";
    envBox.append(el("h4", {}, "Capability envelope "),
      el("span", { class: "tgt" }, "measured by probe runs of the real pipeline — observations, not targets"));
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, ...["metric", "best achievable", "median", "source"].map((h) => el("th", {}, h)))));
    const tb = el("tbody", {});
    Object.entries(env.metrics || {}).forEach(([k, m]) => {
      const src = (m.backend || "—") + (m.mocked ? "  ⚠ MOCKED" : "");
      tb.append(el("tr", {},
        el("td", { class: "mono" }, k),
        el("td", {}, m.available ? fmt(m.best, 4) : el("span", { class: "pill pill-mock" }, "UNAVAILABLE")),
        el("td", {}, m.available ? fmt(m.median, 4) : "—"),
        el("td", { class: m.mocked ? "mono warnink" : "mono" }, src)));
    });
    t.append(tb);
    envBox.append(t);
    (env.warnings || []).forEach((w) => envBox.append(el("p", { class: "logline" }, "⚠ " + w)));
  }).catch((e) => { envBox.append(el("p", { class: "tgt" }, "envelope unavailable: " + e.message)); });

  body.append(el("p", { class: "tgt", style: "margin-top:10px" },
    "Store: " + s.store + " · journals studied: " + s.config.resolved.journal_dir));
}

function statCard(title, value, sub, tone, help) {
  return el("div", { class: "ecard tone-" + tone, title: help || "" },
    el("div", { class: "ecard-t" }, title),
    el("div", { class: "ecard-v" }, String(value)),
    el("div", { class: "ecard-s" }, String(sub)));
}

// ---- findings --------------------------------------------------------------
function renderFindings(a) {
  const body = $("ebody");
  body.innerHTML = "";
  const findings = a.findings || [], clusters = a.clusters || [];
  body.append(el("p", { class: "tgt" },
    `${findings.length} findings over ${clusters.length} clusters, from ${a.n_episodes} recorded requests`));

  if (!findings.length) {
    body.append(el("p", { class: "tgt" },
      "Nothing measured yet — run some requests through the discovery console first, then press ⟳."));
    return;
  }

  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["severity", "finding", "affected", "inference", "what it says"].map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  findings.forEach((f) => {
    const tr = el("tr", { class: "jrow" },
      el("td", {}, el("span", { class: "pill " + (SEV_CLASS[f.severity] || "pill") }, f.severity)),
      el("td", { class: "mono" }, f.finding_id),
      el("td", {}, `${f.n_affected}/${f.n_applicable}`),
      el("td", { class: "mono" }, (f.inference || "").split(" ")[0]),
      el("td", {}, f.title));
    tr.onclick = () => showFinding(f);
    tb.append(tr);
  });
  t.append(tb);
  body.append(t);
  body.append(el("div", { id: "edetail" }));

  body.append(el("h4", { style: "margin-top:16px" }, "Failure clusters"));
  const ct = el("table", { class: "data" });
  ct.append(el("thead", {}, el("tr", {}, ...["cluster", "severity", "surface", "summary"].map((h) => el("th", {}, h)))));
  const ctb = el("tbody", {});
  clusters.forEach((c) => ctb.append(el("tr", {},
    el("td", { class: "mono" }, c.cluster_id),
    el("td", {}, el("span", { class: "pill " + (SEV_CLASS[c.severity] || "pill") }, c.severity)),
    el("td", { class: "mono" }, c.surface),
    el("td", {}, c.summary))));
  ct.append(ctb);
  body.append(ct);
  body.append(el("p", { class: "tgt", style: "margin-top:8px" },
    "A cluster is indexed by the surface a fix would have to touch, which is also the archive's "
    + "behaviour dimension — so a planner fix and a refiner fix never evict one another."));
}

function showFinding(f) {
  const slot = $("edetail");
  if (!slot) return;
  slot.innerHTML = "";
  const wrap = el("div", { class: "jdetail" });
  wrap.append(el("div", { class: "card-head" },
    el("span", { class: "pill " + (SEV_CLASS[f.severity] || "pill") }, f.severity), " ", f.finding_id));
  const inner = el("div", { class: "card-body" });
  inner.append(el("p", {}, f.title));
  if (f.detail) inner.append(el("p", { class: "tgt" }, f.detail));
  inner.append(el("div", { class: "chips" },
    chip("detector", f.detector),
    chip("affected", `${f.n_affected}/${f.n_applicable}`),
    chip("rate", fmt(f.n_applicable ? f.n_affected / f.n_applicable : null, 3)),
    chip("inference", f.inference)));
  if ((f.evidence || []).length) {
    inner.append(el("h4", { style: "margin-top:10px" }, "Evidence — the episodes this was computed from"));
    inner.append(el("div", { class: "chips" }, ...f.evidence.map((e) => chip("episode", e))));
  }
  inner.append(el("h4", { style: "margin-top:10px" }, "The arithmetic behind it"));
  inner.append(el("pre", { class: "codeblock", text: JSON.stringify(f.stats, null, 1) }));
  wrap.append(inner);
  slot.append(wrap);
}

// ---- methods ---------------------------------------------------------------
// One row per distinct change, folded across cycles: several hypotheses arriving
// at the same method are one entry, because that convergence is a finding.
function renderMethods(s) {
  const body = $("ebody");
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" },
    `${s.n_methods} distinct methods, from ${s.n_hypotheses} hypotheses across ${s.n_cycles} `
    + `research cycles and ${s.n_candidates} candidates · ${s.n_arms} benchmark arms measured`));
  if (!s.n_methods) {
    body.append(el("p", { class: "tgt" }, "Nothing proposed yet. Run a cycle from “Run the loop”."));
    return;
  }
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["#", "method", "class", "status", "outcome"]
    .map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  (s.methods || []).forEach((m, i) => {
    const cls = m.status === "PROMOTED" ? "pill-ok"
      : m.status === "QUARANTINED" ? "pill-bad"
      : m.status === "REJECTED" ? "pill-mock" : "pill";
    const tr = el("tr", { class: "jrow" },
      el("td", {}, String(i + 1)),
      el("td", {}, m.title),
      el("td", { class: "mono" }, m.candidate_class),
      el("td", {}, el("span", { class: "pill " + cls }, m.status)),
      el("td", { class: "tgt" }, m.outcome));
    tr.onclick = () => showMethod(m);
    tb.append(tr);
  });
  t.append(tb);
  body.append(t);
  body.append(el("div", { id: "edetail" }));
  if ((s.unattached_hypotheses || []).length) {
    body.append(el("p", { class: "tgt", style: "margin-top:8px" },
      `${s.unattached_hypotheses.length} hypotheses produced no method — refuted twice, or `
      + "naming no implementable operator. Failure is recorded, not discarded."));
  }
}

function showMethod(m) {
  const slot = $("edetail");
  if (!slot) return;
  slot.innerHTML = "";
  const wrap = el("div", { class: "jdetail" });
  wrap.append(el("div", { class: "card-head" },
    el("span", { class: "tag tag-cycle" }, "method"), " ", m.title));
  const inner = el("div", { class: "card-body" });
  inner.append(el("div", { class: "chips" },
    chip("class", m.candidate_class), chip("cell", m.cell), chip("status", m.status),
    chip("surfaces", (m.surfaces || []).join(", ") || "—"),
    chip("addresses", (m.addresses || []).join(", ") || "—")));
  if (m.expected_benefit) inner.append(el("p", {}, el("b", {}, "Expected. "), m.expected_benefit));

  (m.hypotheses || []).forEach((h) => {
    const via = h.arrived_via ? ` (reached again via ${h.arrived_via})` : "";
    const rev = h.revision_of ? `, revision of ${h.revision_of}` : "";
    inner.append(el("h4", { style: "margin-top:10px" },
      h.hypothesis_id + " — ", el("span", { class: "pill" }, h.verdict || "—"),
      " ", el("span", { class: "tgt" }, h.author + rev + via)));
    inner.append(el("p", {}, h.statement));
    if (h.falsifier) inner.append(el("p", { class: "tgt" }, "How it could be shown wrong: " + h.falsifier));
  });

  if ((m.declared_regressions || []).length) {
    inner.append(el("h4", { style: "margin-top:10px" }, "Regressions it declared in advance"));
    const ul = el("ul", { class: "goldlist" });
    m.declared_regressions.forEach((r) => ul.append(el("li", {}, r)));
    inner.append(ul);
  }
  inner.append(el("h4", { style: "margin-top:10px" }, "Outcome"));
  inner.append(el("p", {}, m.outcome));
  inner.append(el("pre", { class: "codeblock", text: JSON.stringify(m.overlay, null, 1) }));
  wrap.append(inner);
  slot.append(wrap);
}

// ---- logs ------------------------------------------------------------------
function renderLogs(d) {
  const body = $("ebody");
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" },
    `${(d.logs || []).length} log locations · ${(d.total_bytes / 1e6).toFixed(1)} MB total`));
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["log", "files", "size", "what it answers", "path"]
    .map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  (d.logs || []).forEach((r) => {
    const size = r.bytes > 1e6 ? (r.bytes / 1e6).toFixed(1) + " MB" : Math.round(r.bytes / 1e3) + " KB";
    tb.append(el("tr", { class: r.exists ? "" : "dim" },
      el("td", { class: "mono" }, r.log),
      el("td", {}, r.exists ? String(r.n_files) : "—"),
      el("td", {}, r.exists ? size : "—"),
      el("td", { class: "tgt" }, r.what),
      el("td", { class: "mono tgt" }, r.path)));
  });
  t.append(tb);
  body.append(t);

  const box = el("div", { class: "ebox" });
  box.append(el("h4", {}, "Collect them into one place"));
  box.append(el("p", { class: "tgt" },
    "A copy, never a move: the originals stay where the code that wrote them expects them. "
    + "The bundle gets an INDEX.md saying what each log answers, a METHODS.md summarizing every "
    + "method proposed, and a MANIFEST.json hashing every file."));
  const status = el("p", { class: "tgt" });
  const btn = el("button", { class: "launch small" }, "▸ Bundle every log");
  btn.onclick = async () => {
    btn.disabled = true;
    status.textContent = "bundling…";
    try {
      const r = await fetch("/api/evolve/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "bundle" }),
      });
      const out = await r.json();
      if (!r.ok) throw new Error(out.detail || "HTTP " + r.status);
      attachJob(out.job_id);
      status.textContent = "started — see the live log under “Run the loop”.";
    } catch (e) {
      status.textContent = "failed: " + e.message;
    }
    btn.disabled = false;
  };
  box.append(btn, status);
  body.append(box);

  body.append(el("h4", { style: "margin-top:14px" }, "What a bundle contains"));
  const it = el("table", { class: "data" });
  it.append(el("thead", {}, el("tr", {}, ...["file", "what it is", "how to read it"].map((h) => el("th", {}, h)))));
  const itb = el("tbody", {});
  (d.index || []).forEach((e) => itb.append(el("tr", {},
    el("td", { class: "mono" }, e.path),
    el("td", {}, e.what),
    el("td", { class: "tgt" }, e.how))));
  it.append(itb);
  body.append(it);
}

// ---- benchmark -------------------------------------------------------------
function renderBenchmark(b, recs) {
  const body = $("ebody");
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" },
    `${b.benchmark_id}@${b.version} · fingerprint ${b.fingerprint} · ${b.n_tasks} tasks`));

  const arms = recs.records || [];
  if (arms.length) {
    body.append(el("h4", {}, "Arms measured"));
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, ...["arm", "gold_score", "decision_accuracy", "accepts",
      "false accepts", "tokens", "wall s", "kernel", "record"].map((h) => el("th", {}, h)))));
    const tb = el("tbody", {});
    arms.forEach((r) => {
      const a = r.aggregates || {};
      const tr = el("tr", { class: "jrow" },
        el("td", { class: "mono" }, r.arm),
        el("td", {}, fmt(a.gold_score, 4)),
        el("td", {}, el("b", { text: fmt(a.decision_accuracy, 4) })),
        el("td", {}, String(a.n_accept ?? "—")),
        el("td", {}, String(a.n_false_accept ?? "—")),
        el("td", {}, fmt(a.mean_eval_tokens, 0)),
        el("td", {}, fmt(a.mean_wall_s, 0)),
        el("td", {}, el("span", { class: "pill " + (r.kernel_verified ? "pill-ok" : "pill-bad") },
          r.kernel_verified ? "verified" : "VOID")),
        el("td", { class: "mono" }, r.record_id));
      tb.append(tr);
    });
    t.append(tb);
    body.append(t);
    body.append(el("p", { class: "tgt", style: "margin-top:6px" },
      "gold_score is the guard — the rubric below, computed over the molecules produced, by code "
      + "no candidate can reach. decision_accuracy is the primary metric: does the agent's "
      + "ACCEPT/REFINE agree with that rubric?"));
  } else {
    body.append(el("p", { class: "tgt" }, "No arms measured yet. Run a baseline from “Run the loop”."));
  }

  body.append(el("h4", { style: "margin-top:16px" }, "The gold rubric "),
    el("span", { class: "tgt" }, "fixed, hash-pinned, and never shown to a candidate generator"));
  (b.tasks || []).forEach((task) => {
    const card = el("div", { class: "ebox" });
    card.append(el("div", { class: "card-head" },
      el("span", { class: "pill " + (task.split === "holdout" ? "pill-mock" : "pill-ok") }, task.split),
      " ", el("span", { class: "mono" }, task.task_id), " ",
      el("span", { class: "tgt" }, task.origin === "synthetic" ? "synthetic" : "from the journal")));
    card.append(el("p", {}, task.instruction));
    if (task.notes) card.append(el("p", { class: "tgt" }, task.notes));
    const ul = el("ul", { class: "goldlist" });
    (task.gold || []).forEach((g) => ul.append(el("li", {},
      el("span", { class: "mono" }, `${g.metric} ${g.op} ${g.value}`), " — ",
      el("span", { class: "tgt" }, g.why))));
    card.append(ul);
    body.append(card);
  });
}

// ---- candidates ------------------------------------------------------------
function renderCandidates(a) {
  const body = $("ebody");
  body.innerHTML = "";
  const cands = a.candidates || [];
  body.append(el("p", { class: "tgt" },
    `${cands.length} candidates across ${Object.keys(a.summary.cells || {}).length} quality-diversity cells`));
  if (!cands.length) {
    body.append(el("p", { class: "tgt" },
      "No candidates yet. Run a cycle from “Run the loop” — a dry run costs nothing but a hypothesis."));
    return;
  }
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["candidate", "gen", "parent", "cell", "status", "author", "hypothesis"]
    .map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  cands.forEach((c) => {
    const cls = c.status === "PROMOTED" ? "pill-ok"
      : c.status === "QUARANTINED" ? "pill-bad"
      : c.status === "REJECTED" ? "pill-mock" : "pill";
    const tr = el("tr", { class: "jrow" },
      el("td", { class: "mono" }, c.candidate_id),
      el("td", {}, String(c.generation)),
      el("td", { class: "mono" }, c.parent_id || "—"),
      el("td", { class: "mono" }, c.cell),
      el("td", {}, el("span", { class: "pill " + cls }, c.status)),
      el("td", { class: "mono" }, (c.author || "—").split(" ")[0]),
      el("td", {}, (c.hypothesis || "").slice(0, 90)));
    tr.onclick = () => showCandidate(c.candidate_id);
    tb.append(tr);
  });
  t.append(tb);
  body.append(t);
  body.append(el("div", { id: "edetail" }));
  body.append(el("p", { class: "tgt", style: "margin-top:8px" },
    "REJECTED is not deleted — a rejected candidate stays in the archive as a stepping stone and can "
    + "still be sampled as a parent. QUARANTINED is the one status that cannot: it means the "
    + "candidate touched the evaluation kernel."));
}

async function showCandidate(id) {
  const slot = $("edetail");
  if (!slot) return;
  slot.innerHTML = "";
  slot.append(el("p", { class: "tgt" }, "loading " + id + "…"));
  try {
    const d = await getJSON("/api/evolve/archive/" + encodeURIComponent(id));
    const c = d.candidate;
    slot.innerHTML = "";
    const wrap = el("div", { class: "jdetail" });
    wrap.append(el("div", { class: "card-head" },
      el("span", { class: "tag tag-cycle" }, "candidate"), " ", el("span", { class: "mono" }, c.candidate_id),
      " ", el("span", { class: "pill" }, c.status)));
    const inner = el("div", { class: "card-body" });
    inner.append(el("h4", {}, "Hypothesis"), el("p", {}, c.hypothesis || "—"));
    if (c.expected_benefit) inner.append(el("p", { class: "tgt" }, "Expected: " + c.expected_benefit));
    if ((c.possible_regressions || []).length) {
      inner.append(el("h4", { style: "margin-top:8px" }, "Regressions it declared in advance"));
      const ul = el("ul", { class: "goldlist" });
      c.possible_regressions.forEach((r) => ul.append(el("li", {}, r)));
      inner.append(ul);
    }
    inner.append(el("h4", { style: "margin-top:8px" }, "Overlay — the whole change"));
    inner.append(el("pre", { class: "codeblock", text: JSON.stringify(c.overlay, null, 1) }));
    inner.append(el("div", { class: "chips" },
      chip("cell", c.cell || "—"),
      chip("lineage", (d.lineage || []).join(" → ") || "—"),
      chip("children", (d.children || []).length)));

    if (d.gate0) {
      const ok = d.gate0.passed;
      inner.append(el("h4", { style: "margin-top:10px" }, "Gate 0 — invariants, executed by the kernel"));
      (d.gate0.commands || []).forEach((cmd) => {
        inner.append(el("p", { class: ok ? "tgt" : "logline" },
          `${cmd.label} · exit ${cmd.exit_status} · ${fmt(cmd.elapsed_s, 1)} s`));
        if (!cmd.ok) inner.append(el("pre", { class: "codeblock", text: (cmd.stderr_tail || cmd.stdout_tail || "").slice(-1200) }));
      });
    }
    if (d.replay && d.replay.applicable) {
      inner.append(el("h4", { style: "margin-top:10px" }, "Gate 1 — deterministic replay, no model called"));
      const t = el("table", { class: "data" });
      t.append(el("thead", {}, el("tr", {}, ...["suite", "measure", "baseline", "candidate"].map((h) => el("th", {}, h)))));
      const tb = el("tbody", {});
      Object.entries(d.replay.candidate.suites || {}).forEach(([name, s]) => {
        const base = (d.replay.baseline.suites || {})[name] || {};
        Object.keys(s).filter((k) => k.indexOf("n_") === 0).forEach((k) => {
          tb.append(el("tr", {}, el("td", { class: "mono" }, name), el("td", {}, k),
            el("td", {}, String(base[k])), el("td", {}, el("b", { text: String(s[k]) }))));
        });
      });
      t.append(tb);
      inner.append(t);
    }
    if ((d.records || []).length) {
      inner.append(el("h4", { style: "margin-top:10px" }, "Gate 2 — benchmark arms"));
      const t = el("table", { class: "data" });
      t.append(el("thead", {}, el("tr", {}, ...["record", "gold_score", "decision_accuracy", "verified"].map((h) => el("th", {}, h)))));
      const tb = el("tbody", {});
      d.records.forEach((r) => {
        const a = r.aggregates || {};
        tb.append(el("tr", {}, el("td", { class: "mono" }, r.record_id),
          el("td", {}, fmt(a.gold_score, 4)), el("td", {}, fmt(a.decision_accuracy, 4)),
          el("td", {}, el("span", { class: "pill " + (r.kernel_verified ? "pill-ok" : "pill-bad") },
            r.kernel_verified ? "yes" : "VOID"))));
      });
      t.append(tb);
      inner.append(t);
    }

    if (d.decision) {
      inner.append(el("h4", { style: "margin-top:10px" }, "Promotion gate"));
      const dt = el("table", { class: "data" });
      dt.append(el("thead", {}, el("tr", {}, ...["check", "result", "detail"].map((h) => el("th", {}, h)))));
      const dtb = el("tbody", {});
      (d.decision.checks || []).forEach((ch) => dtb.append(el("tr", {},
        el("td", { class: "mono" }, ch.check),
        el("td", {}, el("span", { class: "pill " + (ch.passed ? "pill-ok" : "pill-bad") }, ch.passed ? "PASS" : "FAIL")),
        el("td", { class: "tgt" }, String(ch.detail).slice(0, 160)))));
      dt.append(dtb);
      inner.append(dt);
      inner.append(el("p", {}, el("b", {}, "Decision: " + d.decision.decision)));
      (d.decision.reasons || []).forEach((r) => inner.append(el("p", { class: "tgt" }, "· " + r)));
      inner.append(promoteBox(c, d.decision));
    } else {
      inner.append(el("p", { class: "tgt", style: "margin-top:10px" },
        "No promotion decision yet — this candidate has not been compared against a baseline."));
    }
    wrap.append(inner);
    slot.append(wrap);
  } catch (e) {
    slot.innerHTML = "";
    slot.append(el("p", { class: "tgt" }, "could not load: " + e.message));
  }
}

// The one control that changes what users get. Two refusals stand between a
// click and a live change: the gate, and an explicit confirmation.
function promoteBox(cand, decision) {
  const box = el("div", { class: "ebox promote" });
  const allowed = decision.decision === "PROMOTE" || decision.decision === "NEEDS_HUMAN";
  if (!allowed) {
    box.append(el("p", { class: "logline" },
      "Promotion is refused: the gate says " + decision.decision + "."));
    return box;
  }
  box.append(el("p", {},
    el("b", {}, "This changes how the user-serving system behaves. "),
    "The overlay is written to disk; it becomes active only when ",
    el("code", { class: "mono" }, "DRUGAGENT_EVOLVE_OVERLAY"),
    " points at it and the console is restarted. Unsetting it reverts."));
  const status = el("p", { class: "tgt" });
  const btn = el("button", { class: "launch danger" }, "⚠ Promote " + cand.candidate_id);
  btn.onclick = async () => {
    if (btn.dataset.armed !== "1") {
      btn.dataset.armed = "1";
      btn.textContent = "Click again to confirm — this writes the overlay";
      status.textContent = "Read the gate checks above before confirming.";
      return;
    }
    btn.disabled = true;
    try {
      const r = await fetch("/api/evolve/promote/" + encodeURIComponent(cand.candidate_id), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ i_approve: true }),
      });
      const out = await r.json();
      if (!r.ok) throw new Error(out.detail || "HTTP " + r.status);
      status.innerHTML = "";
      status.append(el("b", {}, "Promoted. "), out.note || "");
      status.append(el("pre", { class: "codeblock", text: out.activate + "\n" + out.revert }));
    } catch (e) {
      status.textContent = "refused: " + e.message;
      btn.disabled = false;
    }
  };
  box.append(btn, status);
  return box;
}

// ---- run the loop ----------------------------------------------------------
const JOB_BUTTONS = [
  ["analyze", "Analyze the journals", "cheap"],
  ["envelope", "Re-measure the envelope", "cheap"],
  ["calibrate", "Calibrate the rubric", "cheap"],
  ["replay", "Gate 1 replay", "cheap"],
  ["bundle", "Bundle every log", "cheap"],
  ["step_dry", "One cycle — dry run", "model"],
  ["step_gates", "One cycle — gates 0 + 1", "model"],
  ["step", "One cycle — full cascade", "expensive"],
  ["baseline", "Measure the baseline", "expensive"],
];

function renderRun(s, jobs) {
  const body = $("ebody");
  body.innerHTML = "";
  const costs = s.job_costs || {};

  body.append(el("p", { class: "tgt" },
    "Cheap operations call no model. The ones marked model-time use the local Ollama and the "
    + "claude CLI; the expensive ones run real master-agent programs and take about half an hour."));

  const grid = el("div", { class: "ejobs" });
  JOB_BUTTONS.forEach(([kind, label, tone]) => {
    const card = el("div", { class: "ejob tone-" + tone });
    card.append(el("div", { class: "ejob-t" }, label));
    card.append(el("div", { class: "ejob-c" }, costs[kind] || ""));
    const b = el("button", { class: "launch small" }, "▸ Run");
    b.onclick = () => startJob(kind, b, tone);
    card.append(b);
    grid.append(card);
  });
  body.append(grid);

  if (!s.has_baseline) {
    body.append(el("p", { class: "logline" },
      "⚠ No baseline recorded. A candidate cannot be compared against anything until one exists — "
      + "run “Measure the baseline” first."));
  }

  body.append(el("h4", { style: "margin-top:14px" }, "Live"));
  body.append(el("pre", { class: "joblog", id: "ejoblog" }, "idle"));

  const rows = jobs.jobs || [];
  if (rows.length) {
    body.append(el("h4", { style: "margin-top:14px" }, "Jobs this session"));
    const t = el("table", { class: "data" });
    t.append(el("thead", {}, el("tr", {}, ...["job", "kind", "status", "elapsed s", "what"].map((h) => el("th", {}, h)))));
    const tb = el("tbody", {});
    rows.forEach((j) => {
      const cls = j.status === "SUCCEEDED" ? "pill-ok" : j.status === "RUNNING" ? "pill" : "pill-bad";
      const tr = el("tr", { class: "jrow" },
        el("td", { class: "mono" }, j.job_id),
        el("td", {}, j.kind),
        el("td", {}, el("span", { class: "pill " + cls }, j.status)),
        el("td", {}, fmt(j.elapsed_s, 0)),
        el("td", {}, j.label));
      tr.onclick = () => attachJob(j.job_id);
      tb.append(tr);
    });
    t.append(tb);
    body.append(t);
  }
}

async function startJob(kind, btn, tone) {
  if (tone === "expensive" && btn.dataset.armed !== "1") {
    btn.dataset.armed = "1";
    btn.textContent = "Click again — this costs ~33 min";
    return;
  }
  btn.disabled = true;
  const log = $("ejoblog");
  if (log) log.textContent = "";
  try {
    const r = await fetch("/api/evolve/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: kind }),
    });
    const out = await r.json();
    if (!r.ok) throw new Error(out.detail || "HTTP " + r.status);
    appendJobLine(`▸ ${out.label}  (${out.cost})`);
    attachJob(out.job_id);
  } catch (e) {
    appendJobLine("✖ " + e.message);
    btn.disabled = false;
  }
}

function attachJob(jobId) {
  if (ejobES) { ejobES.close(); ejobES = null; }
  elastJob = jobId;
  const log = $("ejoblog");
  if (log && log.textContent === "idle") log.textContent = "";
  ejobES = new EventSource("/api/evolve/jobs/" + encodeURIComponent(jobId) + "/events");
  ejobES.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (e) { return; }
    appendJobLine(jobLine(ev));
    if (ev.type === "closed") {
      if (ejobES) { ejobES.close(); ejobES = null; }
      loadEvolve();
    }
  };
  ejobES.onerror = () => { if (ejobES) { ejobES.close(); ejobES = null; } };
}

function jobLine(ev) {
  const mark = { started: "▸", step: "·", progress: "⋯", finished: "✔", failed: "✖",
    cancelled: "⊘", closed: "—" }[ev.type] || "·";
  return `${mark} ${ev.message}`;
}

function appendJobLine(line) {
  const log = $("ejoblog");
  if (!log) return;
  log.textContent = (log.textContent === "idle" ? "" : log.textContent) + line + "\n";
}

// ---- cycles ----------------------------------------------------------------
function renderCycles(data) {
  const body = $("ebody");
  body.innerHTML = "";
  const rows = data.cycles || [];
  body.append(el("p", { class: "tgt" },
    `${rows.length} research cycles — the evolver's own experience, one row each`));
  if (!rows.length) {
    body.append(el("p", { class: "tgt" }, "No cycles yet. Start one from “Run the loop”."));
    return;
  }
  const t = el("table", { class: "data" });
  t.append(el("thead", {}, el("tr", {}, ...["cluster", "hypothesis", "candidate", "decision",
    "wall s", "claude", "ollama", "why"].map((h) => el("th", {}, h)))));
  const tb = el("tbody", {});
  rows.slice().reverse().forEach((r) => {
    const bc = r.backend_calls || {};
    const cls = r.decision === "GATE1_ONLY" || r.decision === "NEEDS_HUMAN" ? "pill-ok"
      : r.decision === "QUARANTINED" ? "pill-bad" : "pill-mock";
    tb.append(el("tr", {},
      el("td", { class: "mono" }, r.cluster_id || "—"),
      el("td", { class: "mono" }, r.hypothesis_id || "—"),
      el("td", { class: "mono" }, r.candidate_id || "—"),
      el("td", {}, el("span", { class: "pill " + cls }, r.decision || "—")),
      el("td", {}, fmt(r.wall_s, 0)),
      el("td", {}, String(bc.claude_cli ?? "—")),
      el("td", {}, String(bc.ollama ?? "—")),
      el("td", { class: "tgt" }, String(r.reason || "").slice(0, 120))));
  });
  t.append(tb);
  body.append(t);
  body.append(el("p", { class: "tgt", style: "margin-top:8px" },
    "SKIPPED_NOT_NOVEL means the archive already held that exact change with a result — the loop "
    + "refusing to spend a benchmark arm twice."));
}

initViews();
