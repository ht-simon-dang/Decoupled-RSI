"use strict";

// ============================================================================
// The agent map — the third view in this console.
//
// One question, answered honestly: what can this deployment actually do, and
// who can call it? The pool holds several dozen capabilities across a dozen-odd
// agents and a large fraction of them are UNAVAILABLE on any given host, so the
// map is not decoration — a tool that computes nothing, and the reason it
// computes nothing, has to be readable at a glance (ADR-0011).
//
// Depth flows left → right: master agent · the agents it dispatches to ·
// the capability groups each agent reaches · the tools themselves. Density is
// handled by that third level: an agent wired to thirty-four tools shows eight
// folded namespace groups, not thirty-four boxes.
//
// The graph itself is *not* computed here. `GET /api/pools` ships a `topology`
// block the server derived from the same two pools the table view renders, and
// this file reads it. That is the whole point: the edges, the unreachable set
// and the headline counts have one implementation, so the map and the table
// cannot describe different machines the way the CLI and the doctor once did.
//
// Reuses el / $ / chip / groupBy / toolDomain / isAvailable / TOOL_DOMAINS /
// byAvailabilityThenId from app.js and statCard from evolve.js, both of which
// load first. The view switch wraps the one in evolve.js rather than adding a
// second router.
// ============================================================================

// ---- the only place the API shape is read ----------------------------------
// Everything below this section works on the normalised model these functions
// return, so pointing the map at a different endpoint is a change to TOPO_URL
// and to the field names here, and to nothing else.
const TOPO_URL = "/api/pools";

//: `topology.tools[]` names the id `id`, the raw tool pool names it `tool_id`,
//: and every helper shared with app.js (`isAvailable`, `byAvailabilityThenId`)
//: plus the panel below read `tool_id`. Renaming it once, here, is why the rest
//: of the file never has to know which of the two shapes it was handed.
function topoTool(t) {
  const id = t.id || t.tool_id || "";
  return {
    tool_id: id,
    title: t.title || id,
    description: t.description || "",
    status: t.status || "UNKNOWN",
    backend: t.backend || "",
    operations: t.operations || {},
    determinism: t.determinism || "",
    cost_class: t.cost_class || "",
    unavailable_reason: t.unavailable_reason || "",
    // The server ships the namespace; deriving it from the id is the fallback
    // for the older payload, and the two are the same split by construction.
    domain: t.domain || toolDomain(id),
  };
}

//: `uses`, `n_ok` and `actions` are passed rather than read off `a`, because the
//: three of them come from somewhere else in each payload: the edges are the
//: graph, the count is the server's, and the action list lives on the pool card.
function topoAgent(a, uses, n_ok, actions) {
  const id = a.id || a.agent_id || "";
  return {
    agent_id: id,
    kind: String(a.kind || "OTHER"),
    title: a.title || id,
    description: a.description || "",
    llm_backed: !!a.llm_backed,
    notes: a.notes || "",
    actions: actions || {},
    uses_tools: uses || [],
    n_ok: n_ok || 0,
  };
}

//: Reads the server's `topology` block. Returns null — not an empty model — when
//: the payload has no such block, so the caller can tell "old server" apart from
//: "no agents", which look identical once flattened.
function readTopology(data) {
  const d = data || {};
  const t = d.topology;
  if (!t || !Array.isArray(t.agents) || !Array.isArray(t.tools)) return null;

  // The actions an agent accepts are its call surface, not part of the graph, so
  // `topology.agents[]` does not carry them — the pool card in the same payload
  // does, and the panel shows them.
  const actions = new Map((d.agents || []).map((a) => [a.agent_id, a.actions || {}]));

  // An agent naming a tool the pool does not hold is a wiring bug: the server
  // reports it in `dangling_edges` instead of dropping it, and it is put back on
  // the agent that named it so the map can say so on the node rather than look
  // tidy by drawing one edge fewer.
  const dangling = new Map();
  (t.dangling_edges || []).forEach((e) => {
    if (!dangling.has(e.agent_id)) dangling.set(e.agent_id, []);
    dangling.get(e.agent_id).push(e.tool_id);
  });

  const tools = t.tools.map(topoTool);
  const byId = new Map(tools.map((x) => [x.tool_id, x]));
  const counts = t.counts || {};
  return {
    legacy: false,
    // Served order is the taxonomy: REASONING (LLM-backed first) · PLATFORM ·
    // SPECIALIST, so agents[0] is the root of the tree. The view re-sorts on one
    // further axis only, in buildTopology, and never re-derives this one.
    agents: t.agents.map((a) => topoAgent(
      a,
      (a.tool_ids || []).concat(dangling.get(a.id) || []),
      a.available_tool_count,
      actions.get(a.id),
    )),
    toolsById: byId,
    unwired: (t.unwired_tool_ids || []).map((id) => byId.get(id)).filter((x) => x),
    counts: {
      agents: counts.agents || t.agents.length,
      tools: counts.tools || tools.length,
      available: counts.available || 0,
      wired: counts.wired_tools || 0,
    },
  };
}

//: Kind order the server sorts its agents by, mirrored here for the one payload
//: that arrives unsorted. Same taxonomy and same order as the pool table.
const TOPO_KIND_RANK = { REASONING: 0, PLATFORM: 1, SPECIALIST: 2 };

// A console page outlives the server it was loaded from — a restart onto an
// older build, or a deployment a release behind — and that server answers
// /api/pools with the two raw pools and no `topology`. The map is still drawn,
// from this adapter, and the page says out loud that it derived the graph
// itself: a second silent derivation is exactly what this view stopped doing,
// so the one that remains is labelled, single, and only ever reached here.
function readLegacyPools(data) {
  const d = data || {};
  const rank = (a) => {
    const k = TOPO_KIND_RANK[String(a.kind)];
    return [k === undefined ? Object.keys(TOPO_KIND_RANK).length : k, a.llm_backed ? 0 : 1,
      String(a.agent_id || "")];
  };
  const tools = (d.tools || []).map(topoTool)
    .sort((a, b) => a.tool_id.localeCompare(b.tool_id));
  const byId = new Map(tools.map((t) => [t.tool_id, t]));
  const wired = new Set();
  (d.agents || []).forEach((a) => (a.uses_tools || []).forEach((id) => wired.add(id)));
  const agents = (d.agents || []).slice()
    .sort((a, b) => {
      const x = rank(a), y = rank(b);
      return (x[0] - y[0]) || (x[1] - y[1]) || x[2].localeCompare(y[2]);
    })
    .map((a) => {
      const uses = a.uses_tools || [];
      // A named tool the pool does not hold is not available, and is not counted.
      const n_ok = uses.filter((id) => byId.has(id) && isAvailable(byId.get(id))).length;
      return topoAgent(a, uses, n_ok, a.actions);
    });
  const unwired = tools.filter((t) => !wired.has(t.tool_id));
  return {
    legacy: true,
    agents: agents,
    toolsById: byId,
    unwired: unwired,
    counts: {
      agents: agents.length,
      tools: tools.length,
      available: tools.filter(isAvailable).length,
      wired: tools.length - unwired.length,
    },
  };
}

// ---- state -----------------------------------------------------------------
let topoLoaded = false;         // the map fetches once, when first shown
let topoModel = null;           // the built tree, kept so filtering never refetches
let topoFilter = "all";         // all | available | unavailable
let topoPinned = false;         // a clicked tooltip stays until dismissed
let topoAnchor = null;          // the node the tooltip is currently pointing at
let topoFocusKey = null;        // the node to re-focus after a toggle re-renders
const topoOpen = new Map();     // node key -> explicitly opened/closed by the user

//: An agent wired to more than this many tools opens with its capability groups
//: folded. Ten is the point where a branch stops being one glance.
const TOPO_FOLD = 10;

const topoIsOpen = (key, fallback) => (topoOpen.has(key) ? topoOpen.get(key) : fallback);

// ---- building the tree -----------------------------------------------------
// Layout only. Every fact this reads — who calls what, how much of a branch runs
// here, which tools nothing reaches — arrives already computed; what is decided
// below is where a box goes, not whether it is true.
function buildTopology(model) {
  // The server excludes a dangling edge from `tools`, so this is where one
  // becomes drawable — and it carries the reason it exists as its own status.
  const resolve = (id) => model.toolsById.get(id) || topoTool({
    id: id, status: "UNKNOWN",
    unavailable_reason: "this agent names a tool the tool pool does not contain",
  });

  // agents[0] is the root by contract; the id lookups keep the tree upright for a
  // pool whose reasoning plane is empty or named something else.
  const master = model.agents.find((a) => a.agent_id === "agent.llm.master")
    || model.agents.find((a) => a.agent_id.indexOf("master") >= 0)
    || topoAgent({ agent_id: "agent.llm.master", kind: "REASONING", title: "Master agent",
      llm_backed: true }, [], 0, {});

  function branch(key, card, self) {
    const list = card.uses_tools.map(resolve).sort(byAvailabilityThenId);
    const grouped = groupBy(list, (t) => t.domain);
    // Largest group first, so the twenty generators never sit under a heading of
    // one — the order the pool table already uses.
    const groups = [...grouped.keys()]
      .sort((a, b) => (grouped.get(b).length - grouped.get(a).length) || a.localeCompare(b))
      .map((domain) => ({
        key: domain,
        label: TOOL_DOMAINS[domain] || domain,
        tools: grouped.get(domain),
        // Per namespace per agent: the one count no endpoint ships, because it
        // exists only once this view has folded the branch into groups.
        n_ok: grouped.get(domain).filter(isAvailable).length,
      }));
    return {
      key: key, card: card, self: !!self, tools: list, groups: groups, n_ok: card.n_ok,
    };
  }

  const branches = [];
  // The master calls a couple of tools itself instead of dispatching; that edge
  // is real and would vanish if the root were drawn as a bare hub.
  if (master.uses_tools.length) {
    branches.push(branch("self", topoAgent({
      agent_id: master.agent_id,
      kind: "DIRECT",
      title: "Called directly by the master",
      description: "Tools the master agent invokes itself rather than dispatching to a specialist agent.",
    }, master.uses_tools, master.n_ok, {}), true));
  }
  // The served order already is the taxonomy, in its documented order, then the
  // id. The view adds one distinction and no more: agents that dispatch to
  // something come first, because the map answers "what can be run", and five
  // agents wired to nothing standing between the reasoning plane and the
  // specialists answers it backwards. A stable partition, so inside each half
  // the server's order survives untouched.
  const rest = model.agents.filter((a) => a.agent_id !== master.agent_id);
  rest.filter((a) => a.uses_tools.length)
    .concat(rest.filter((a) => !a.uses_tools.length))
    .forEach((a) => branches.push(branch(a.agent_id, a, false)));

  // Available, working, and reachable from no run at all — the server's own
  // `unwired_tool_ids`. That is a finding, so it gets its own block with no line
  // back to the master.
  const orphans = model.unwired.slice().sort(byAvailabilityThenId);

  return {
    legacy: !!model.legacy,
    master: master,
    branches: branches,
    orphans: orphans,
    n_agents: model.counts.agents,
    n_tools: model.counts.tools,
    n_available: model.counts.available,
    n_wired: model.counts.wired,
    // Counting a field the model already carries, not walking the edges again.
    n_dispatching: model.agents.filter((a) => a.uses_tools.length).length,
  };
}

// ---- filtering -------------------------------------------------------------
function topoKeep(t) {
  if (topoFilter === "available") return isAvailable(t);
  if (topoFilter === "unavailable") return !isAvailable(t);
  return true;
}

// ---- nodes -----------------------------------------------------------------
// Every node is a button: hover and keyboard focus reveal the same description,
// and a click pins it, which is also how this works on a touch screen.
function topoNode(cls, card, kind, kids) {
  const n = el("button", { type: "button", class: "tnode " + cls, role: "treeitem" });
  (kids || []).forEach((k) => n.append(k));
  const show = () => topoShowTip(n, card, kind);
  n.onmouseenter = show;
  n.onfocus = show;
  n.onmouseleave = () => topoHideTip(false);
  n.onblur = () => topoHideTip(false);
  return n;
}

const topoCaret = (open) => el("span", { class: "tcaret" }, open ? "▾" : "▸");
const topoLines = (a, b) => el("span", { class: "tlines" },
  el("span", { class: "tid" }, a), el("span", { class: "tsub" }, b));

function toolNode(t) {
  const ok = isAvailable(t);
  const ops = Object.keys(t.operations || {});
  const sub = ok
    ? (t.backend || "—") + (ops.length ? " · " + ops.length + " op" + (ops.length === 1 ? "" : "s") : "")
    : (t.unavailable_reason || "no reason reported");
  const n = topoNode("tool " + (ok ? "ok" : "na"), t, "tool", [
    el("span", { class: "tmark" }, ok ? "✓" : "✕"),
    topoLines(t.tool_id, sub),
  ]);
  // A leaf has nothing to expand, so its click pins the description instead —
  // the full paragraph and the full reason, neither of which fits on the node.
  n.onclick = () => topoToggleTip(n, t, "tool");
  return n;
}

function topoItem(node, kids, extraClass) {
  const item = el("div", { class: "titem" + (extraClass ? " " + extraClass : ""), role: "none" });
  item.append(node);
  if (kids) item.append(kids);
  return item;
}

function groupItem(b, g) {
  const tools = g.tools.filter(topoKeep);
  if (!tools.length) return null;
  const key = "g:" + b.key + "/" + g.key;
  // While a filter is on, folding would hide the very rows the filter was asked
  // for, so everything that matched is open.
  const open = topoFilter === "all" ? topoIsOpen(key, b.tools.length <= TOPO_FOLD) : true;
  const n_ok = tools.filter(isAvailable).length;
  const node = topoNode("group", { label: g.label, tools: tools, n_ok: n_ok, agent: b.card }, "group", [
    topoCaret(open),
    topoLines(g.label, n_ok + " of " + tools.length + " run here"),
  ]);
  node.setAttribute("aria-expanded", String(open));
  node.setAttribute("data-key", key);
  node.onclick = () => { topoOpen.set(key, !open); topoFocusKey = key; renderTopology(); };
  const kids = open
    ? el("div", { class: "tkids", role: "group" }, ...tools.map((t) => topoItem(toolNode(t), null, isAvailable(t) ? "" : "na")))
    : null;
  return topoItem(node, kids);
}

function branchItem(b) {
  const key = "b:" + b.key;
  const open = topoFilter === "all" ? topoIsOpen(key, true) : true;
  const groups = open ? b.groups.map((g) => groupItem(b, g)).filter((x) => x) : [];
  const shown = b.tools.filter(topoKeep);
  const idle = !b.tools.length;

  const label = b.self ? b.card.title : b.card.title || b.card.agent_id;
  const sub = idle
    ? "no tools wired — nothing to dispatch"
    : b.n_ok + " of " + b.tools.length + " tools run here";
  const node = topoNode("agent" + (idle ? " idle" : "") + (b.self ? " self" : ""), b.card, "agent", [
    idle ? el("span", { class: "tcaret" }, "·") : topoCaret(open),
    topoLines(label, sub),
    b.card.llm_backed ? el("span", { class: "pill" }, "LLM") : null,
  ].filter((x) => x));
  node.setAttribute("aria-expanded", String(open && !idle));
  node.setAttribute("data-key", key);
  if (!idle) node.onclick = () => { topoOpen.set(key, !open); topoFocusKey = key; renderTopology(); };
  else node.onclick = () => topoToggleTip(node, b.card, "agent");

  let kids = null;
  if (open && groups.length) kids = el("div", { class: "tkids", role: "group" }, ...groups);
  else if (open && !idle && !shown.length) {
    kids = el("div", { class: "tkids", role: "group" },
      topoItem(el("span", { class: "tnode none" }, "no tool here matches the filter")));
  }
  return topoItem(node, kids, idle ? "idle" : "");
}

// ---- the view --------------------------------------------------------------
function renderTopology() {
  const body = $("topobody");
  if (!body || !topoModel) return;
  const m = topoModel;
  topoHideTip(true);              // the nodes below are about to be replaced
  body.innerHTML = "";

  // Available *and* reachable by nobody: the one orphan count that means a
  // working capability is being wasted. Four unreachable tools that do not run
  // here either is a smaller problem than one that does, and the page said the
  // same amber thing for both until this was split out.
  const stranded = m.orphans.filter(isAvailable).length;

  const cards = el("div", { class: "ecards" });
  cards.append(
    statCard("Agents", m.n_agents, m.n_dispatching + " dispatch to tools", "ok",
      "Every agent in the pool. One with no tools wired cannot reach the catalogue at all."),
    statCard("Tools that run here", m.n_available + " / " + m.n_tools,
      "the rest report UNAVAILABLE", m.n_available ? "ok" : "bad",
      "A tool computes the real thing or produces nothing and says why — there is no stand-in."),
    statCard("Reachable", m.n_wired + " / " + m.n_tools, "an agent calls them", "ok",
      "Reachable is not the same claim as available: a run only ever dispatches to these."),
    statCard("Unreachable", m.orphans.length,
      m.orphans.length
        ? (stranded
            ? stranded + (stranded === 1 ? " of them runs" : " of them run") + " here — stranded"
            : "and none of them runs here either")
        : "every tool has a caller",
      stranded ? "warn" : "ok",
      "Implemented and callable by no agent. The number that costs this deployment something is "
      + "how many of them run on this machine: working capability no run can reach."));
  body.append(cards);

  // An older server answers with the two raw pools and no `topology`, and the
  // map above was then derived in this page instead of read from the server's
  // own. That is worth one line on screen: a page quietly computing its own
  // version of a number the server also computes is how two views of one
  // machine start disagreeing, and the operator is the one who can end it.
  if (m.legacy) {
    body.append(el("p", { class: "tgt tlegacy" },
      "⚠ This server answered without the topology block, so the graph below was derived in the "
      + "page from the raw agent and tool pools rather than read from the server's own. Upgrade "
      + "the console server so the map and the pool table share one computed answer."));
  }

  if (!m.n_agents && !m.n_tools) {
    body.append(el("p", { class: "tgt" },
      "The pool came back empty: the server answered, but reports no agents and no tools. "
      + "That is a configuration this console cannot draw — check the profile the server was started with."));
    return;
  }

  const canvas = el("div", { class: "topocanvas" });
  const tree = el("div", { class: "tree", role: "tree", "aria-label": "master agent, agents and tools" });
  const root = topoNode("master", m.master, "agent", [
    el("span", { class: "tglyph" }, "◇"),
    topoLines(m.master.title || m.master.agent_id, m.branches.length + " branches"),
  ]);
  root.onclick = () => topoToggleTip(root, m.master, "agent");
  tree.append(el("div", { class: "troot", role: "none" }, root));
  tree.append(m.branches.length
    ? el("div", { class: "tkids", role: "group" }, ...m.branches.map(branchItem))
    : el("div", { class: "tkids", role: "group" },
        topoItem(el("span", { class: "tnode none" }, "no agent in this pool dispatches to a tool"))));
  canvas.append(tree);
  body.append(canvas);

  // The unreachable tools, deliberately drawn with no line back to the master.
  const orphans = m.orphans.filter(topoKeep);
  const obox = el("div", { class: "torphans" });
  obox.append(el("h4", {}, "Reachable by nobody (" + m.orphans.length + ")"));
  // Whether any of them works is a fact about this host, not a constant: on a
  // machine where every orphan is also UNAVAILABLE, "some of them work" is a
  // sentence the page cannot support.
  obox.append(el("p", { class: "tgt" }, m.orphans.length
    ? ("These are in the tool catalogue, but no agent above names them, so no run can dispatch "
       + "to one. They are drawn detached because that is what they are. "
       + (stranded
          ? stranded + (stranded === 1 ? " of them runs" : " of them run")
            + " on this machine: working capability nothing can reach."
          : "None of them runs on this machine either, so nothing that works is stranded."))
    : "Every tool in the pool has at least one agent that can call it."));
  if (orphans.length) {
    // The same node, outside the tree: a list here, not a treeitem. The listitem
    // role goes on a wrapper rather than on the node, because `role` replaces an
    // element's implicit one — stamping it on the button would announce a
    // clickable node as an inert list entry.
    obox.append(el("div", { class: "torphan-row", role: "list" },
      ...orphans.map((t) => el("div", { class: "torphan", role: "listitem" }, toolNode(t)))));
  } else if (m.orphans.length) {
    obox.append(el("p", { class: "tgt" }, "none under this filter"));
  }
  body.append(obox);

  body.append(el("p", { class: "tgt topolegend" },
    "✓ solid box: the tool runs on this machine.  ✕ dashed box and dashed connector: it reports "
    + "UNAVAILABLE and computes nothing — the line under its id is the reason, in full on hover. "
    + "Hover or tab to any node for what it does. Clicking an agent or a capability group opens "
    + "or folds it; clicking a tool pins its panel, and Esc dismisses."));

  topoRestoreFocus();
}

// Expanding a branch rebuilds the tree, which would otherwise drop the caller
// back to the top of the page — unusable from the keyboard. The node that was
// toggled is found again by its key and re-focused.
function topoRestoreFocus() {
  const key = topoFocusKey;
  topoFocusKey = null;
  if (!key || typeof document.querySelector !== "function") return;
  const n = document.querySelector('#topobody [data-key="' + key + '"]');
  if (n && typeof n.focus === "function") n.focus();
}

// ---- the tooltip -----------------------------------------------------------
// A real panel rather than a `title` attribute: these descriptions are
// paragraphs, and the reason an operator needs most — the missing file or
// command — is the last thing a native tooltip shows well.
function topoTipBody(card, kind) {
  const box = el("div", {});
  if (kind === "group") {
    box.append(el("h4", {}, card.label));
    box.append(el("p", { class: "tgt" },
      card.n_ok + " of " + card.tools.length + " run here · called by "
      + (card.agent.title || card.agent.agent_id)));
    box.append(el("ul", { class: "tight" },
      ...card.tools.slice(0, 12).map((t) => el("li", {},
        (isAvailable(t) ? "✓ " : "✕ ") + t.tool_id))));
    if (card.tools.length > 12) box.append(el("p", { class: "tgt" }, "…and " + (card.tools.length - 12) + " more"));
    return box;
  }
  if (kind === "agent") {
    box.append(el("h4", {}, card.title || card.agent_id));
    box.append(el("p", { class: "tid mono" }, card.agent_id));
    box.append(el("div", { class: "chips" },
      chip("kind", card.kind),
      chip("llm", card.llm_backed ? "yes" : "no"),
      chip("tools", (card.uses_tools || []).length)));
    box.append(el("p", {}, card.description || "no description in the agent pool."));
    const actions = Object.entries(card.actions || {});
    if (actions.length) {
      box.append(el("h5", { class: "tiph" }, "Actions it accepts"));
      box.append(el("ul", { class: "tight" }, ...actions.map(([k, v]) => el("li", {}, k + " — " + v))));
    }
    if (card.notes) box.append(el("p", { class: "tgt" }, card.notes));
    return box;
  }
  const ok = isAvailable(card);
  box.append(el("h4", {}, card.title || card.tool_id));
  box.append(el("p", { class: "tid mono" }, card.tool_id));
  box.append(el("div", { class: "chips" },
    el("span", { class: "pill " + (ok ? "pill-ok" : "pill-bad") }, card.status),
    chip("backend", ok ? (card.backend || "—") : "none"),
    card.cost_class ? chip("cost", card.cost_class) : null,
    card.determinism ? chip("determinism", card.determinism) : null));
  box.append(el("p", {}, card.description || "no description in the tool pool."));
  const ops = Object.entries(card.operations || {});
  if (ops.length) {
    box.append(el("h5", { class: "tiph" }, "Operations"));
    box.append(el("ul", { class: "tight" }, ...ops.map(([k, v]) => el("li", {}, k + " — " + v))));
  }
  if (!ok) box.append(el("p", { class: "why" }, "✕ " + (card.unavailable_reason || "no reason reported")));
  return box;
}

function topoToggleTip(anchor, card, kind) {
  if (topoPinned && topoAnchor === anchor) { topoHideTip(true); return; }
  topoShowTip(anchor, card, kind, true);
}

function topoShowTip(anchor, card, kind, pin) {
  const tip = $("topotip");
  if (!tip) return;
  tip.innerHTML = "";
  tip.append(topoTipBody(card, kind));
  if (pin) tip.append(el("p", { class: "tgt tipdismiss" }, "click again or press Esc to dismiss"));
  tip.hidden = false;
  topoPinned = !!pin;
  topoAnchor = anchor;
  topoPlaceTip(anchor);
}

// Anchored beside the node, flipped to whichever side has room and clamped into
// the viewport, so a node at the right edge of a scrolled map still shows its
// whole description instead of half of one.
function topoPlaceTip(anchor) {
  const tip = $("topotip");
  if (!tip || typeof window === "undefined") return;
  if (typeof anchor.getBoundingClientRect !== "function") return;
  const r = anchor.getBoundingClientRect();
  const w = tip.offsetWidth || 360, h = tip.offsetHeight || 220;
  const vw = window.innerWidth, vh = window.innerHeight, gap = 12, pad = 8;
  let left = r.right + gap;
  if (left + w > vw - pad) left = r.left - gap - w;
  if (left < pad) left = Math.max(pad, Math.min(vw - w - pad, r.left));
  let top = r.top + r.height / 2 - h / 2;
  top = Math.max(pad, Math.min(top, vh - h - pad));
  tip.style.left = Math.round(left) + "px";
  tip.style.top = Math.round(top) + "px";
}

function topoHideTip(force) {
  const tip = $("topotip");
  if (!tip || (topoPinned && !force)) return;
  tip.hidden = true;
  topoPinned = false;
  topoAnchor = null;
}

// ---- loading ---------------------------------------------------------------
async function loadTopology() {
  const body = $("topobody");
  if (!body) return;
  body.innerHTML = "";
  body.append(el("p", { class: "tgt" }, "reading the agent and tool pools…"));
  try {
    const r = await fetch(TOPO_URL);
    const data = await r.json();
    if (r.ok === false) throw new Error((data && data.detail) || ("HTTP " + r.status));
    // The server's own graph when it sends one, the labelled in-page
    // derivation when it does not. One consumer either way.
    topoModel = buildTopology(readTopology(data) || readLegacyPools(data));
    renderTopology();
  } catch (e) {
    body.innerHTML = "";
    const box = el("div", { class: "errbox" });
    box.append(el("p", {}, "The map could not be drawn: this page asked "
      + TOPO_URL + " which agents and tools exist, and got no usable answer ("
      + (e && e.message ? e.message : String(e)) + ")."));
    box.append(el("p", {}, "The console server that served this page is the one that answers it, so "
      + "it is either no longer running or was restarted mid-request. Start it with "
      + "`drugagent console --host 127.0.0.1 --port 2003` and press Retry."));
    const retry = el("button", { class: "launch small" }, "Retry");
    retry.onclick = loadTopology;
    box.append(retry);
    body.append(box);
  }
}

// ---- wiring ----------------------------------------------------------------
// The view switch in evolve.js knows about two views. Rather than a second
// router, each tab's handler is wrapped: the original still runs (it moves the
// `on` class and lazy-loads the evolution panel), then all three containers are
// set from the tab that was actually clicked.
function initTopology() {
  document.querySelectorAll("#viewtabs button").forEach((b) => {
    const prior = b.onclick;
    b.onclick = () => {
      if (prior) prior();
      const val = b.dataset.val;
      const d = $("view-discovery"), e = $("view-evolve"), t = $("view-topology");
      if (d) d.hidden = val !== "discovery";
      if (e) e.hidden = val !== "evolve";
      if (t) t.hidden = val !== "topology";
      topoHideTip(true);
      if (val === "topology" && !topoLoaded) { topoLoaded = true; loadTopology(); }
    };
  });
  document.querySelectorAll("#topofilter button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#topofilter button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      topoFilter = b.dataset.val;
      renderTopology();
    };
  });
  const setAll = (open) => {
    if (!topoModel) return;
    topoModel.branches.forEach((b) => {
      topoOpen.set("b:" + b.key, open);
      b.groups.forEach((g) => topoOpen.set("g:" + b.key + "/" + g.key, open));
    });
    renderTopology();
  };
  const ex = $("topoexpand"); if (ex) ex.onclick = () => setAll(true);
  const co = $("topocollapse"); if (co) co.onclick = () => setAll(false);
  const rf = $("toporefresh"); if (rf) rf.onclick = loadTopology;

  // Window-level listeners are added rather than assigned: they are shared with
  // whatever else the page grows, unlike the per-node handlers above.
  if (typeof document.addEventListener === "function") {
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") topoHideTip(true); });
    document.addEventListener("scroll", () => topoHideTip(true), true);
  }
  if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("resize", () => topoHideTip(true));
  }
}

initTopology();
