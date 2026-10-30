"""Human-readable report generation (Markdown + self-contained HTML)."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ..util import utcnow_iso

if TYPE_CHECKING:
    from ..orchestration.context import RunContext

DISCLAIMER = (
    "This is a research-prioritization output, not a clinical decision system. "
    "No candidate is claimed safe, effective, synthesizable, or clinically viable. "
    "Values are labelled by trust level and require external validation."
)


def gather_report_data(ctx: RunContext) -> dict[str, Any]:
    S = ctx.store
    rid = ctx.run.id
    portfolios = S.list_objects("portfolio", run_id=rid)
    portfolio = portfolios[0] if portfolios else {}
    scorecards = S.list_objects("scorecard", run_id=rid)
    candidates = S.list_objects("candidate", run_id=rid)
    sc_by_id = {s["candidate_id"]: s for s in scorecards}

    n_sourced = len(S.list_objects("raw_candidate", run_id=rid))
    n_std = len(candidates)
    n_pass = sum(1 for c in candidates if c.get("passed_hard_filters"))
    shortlist = portfolio.get("shortlist", [])
    reserve = portfolio.get("reserve", [])

    baseline = S.list_objects("baseline_metrics", run_id=rid)
    sens = S.list_objects("sensitivity", run_id=rid)
    evidence = S.list_objects("evidence", run_id=rid)
    decisions = S.list_objects("decision", run_id=rid)
    experiments = S.list_objects("experiment", run_id=rid)
    hypotheses = S.list_objects("hypothesis", run_id=rid)
    safety = {s["candidate_id"]: s for s in S.list_objects("safety", run_id=rid)}
    synthesis = {s["candidate_id"]: s for s in S.list_objects("synthesis", run_id=rid)}
    data_cards = S.list_objects("data_card", run_id=rid)
    generation = S.list_objects("generation_batch", run_id=rid)
    origin_counts: dict[str, int] = {}
    for c in candidates:
        origin = str(c.get("origin") or "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    brief = S.get_object("brief", ctx.run.brief_id) or {}
    cap_matrix = ctx.registry.capability_matrix()

    return {
        "run_id": rid,
        "generated_at": utcnow_iso(),
        "brief": brief,
        "funnel": {
            "sourced": n_sourced,
            "standardized": n_std,
            "passed_filters": n_pass,
            "pareto": portfolio.get("n_pareto", 0),
            "shortlist": len(shortlist),
            "reserve": len(reserve),
        },
        "portfolio": portfolio,
        "shortlist": shortlist,
        "reserve": reserve,
        "scorecards": sc_by_id,
        "candidates": candidates,
        "baseline": baseline[0] if baseline else {},
        "sensitivity": sens[0] if sens else {},
        "evidence": evidence,
        "decisions": decisions,
        "experiments": experiments,
        "hypotheses": hypotheses,
        "safety": safety,
        "synthesis": synthesis,
        "data_cards": data_cards,
        "generation": generation,
        "origin_counts": origin_counts,
        "capability_matrix": cap_matrix,
        "objectives": portfolio.get("objectives", []),
    }


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(d: dict[str, Any]) -> str:
    b = d["brief"]
    L: list[str] = []
    L.append(f"# Candidate Prioritization Report — {b.get('title', d['run_id'])}")
    L.append("")
    L.append(f"> {DISCLAIMER}")
    L.append("")
    L.append(f"- **Run:** `{d['run_id']}`  ")
    L.append(f"- **Generated:** {d['generated_at']}  ")
    L.append(f"- **Brief kind:** {b.get('kind')}  ")
    L.append("")
    L.append("## Scope & assumptions")
    L.append(b.get("description", "").strip() or "_No description provided._")
    L.append("")

    # Evidence
    L.append("## Evidence summary")
    L.append(
        f"{len(d['evidence'])} evidence items assembled "
        f"(trust label CURATED). Prompt-injection sources are quarantined."
    )
    for e in d["evidence"][:6]:
        L.append(f"- _{e.get('kind')}_: {e.get('statement', '')[:120]} (quality {e.get('quality_score')})")
    L.append("")

    # Data & license
    L.append("## Data & license summary")
    for c in d["data_cards"]:
        L.append(
            f"- **{c.get('name')}** ({c.get('kind')}): {c.get('n_records')} records, "
            f"license `{c.get('license')}`, snapshot `{c.get('snapshot_id', '')[:24]}`"
        )
    L.append("")

    # Baseline
    bl = d["baseline"]
    L.append("## Methods & baselines")
    if bl.get("mae") is not None:
        L.append(
            f"Baseline activity model ({bl.get('model_version')}), **{bl.get('split')} split**: "
            f"MAE={_fmt(bl.get('mae'))} vs constant-baseline MAE={_fmt(bl.get('mae_constant_baseline'))} "
            f"(Spearman ρ={_fmt(bl.get('spearman'))}). "
            f"Beats constant baseline: **{bl.get('beats_constant_baseline')}**."
        )
    else:
        L.append("_Baseline not trained (insufficient data)._")
    L.append("")

    # Funnel
    f = d["funnel"]
    L.append("## Candidate funnel")
    L.append("| Stage | Count |")
    L.append("| ----- | ----- |")
    for k in ("sourced", "standardized", "passed_filters", "pareto", "shortlist", "reserve"):
        L.append(f"| {k.replace('_', ' ')} | {f[k]} |")
    L.append("")

    # Where the candidates came from — and, for generated ones, from which model.
    if d.get("origin_counts"):
        L.append("### Candidate provenance")
        L.append("| Origin | Count |")
        L.append("| ------ | ----- |")
        for origin, n in sorted(d["origin_counts"].items()):
            L.append(f"| {origin} | {n} |")
        L.append("")
    for gen in d.get("generation", []):
        m = gen.get("metrics", {})
        label = "SURROGATE (not PFM)" if gen.get("is_surrogate") else "PFM"
        L.append(
            f"**De-novo generation — {label}.** backend=`{gen.get('backend')}`, "
            f"model=`{gen.get('model_id')}` v{gen.get('model_version')}, seed={gen.get('seed')}; "
            f"kept {gen.get('n_kept', 0)} of {int(m.get('n_generated_requested', 0))} requested "
            f"(validity {_fmt(m.get('generation_validity'))}, "
            f"uniqueness {_fmt(m.get('generation_uniqueness'))}); "
            f"candidates tagged `origin={gen.get('origin_tag')}`, trust label "
            f"**{gen.get('trust_label')}**."
        )
        # Only reachable for runs recorded before the surrogate was removed
        # (ADR-0010); nothing sets is_surrogate now. Kept so an old run still
        # renders with its warning intact rather than silently losing it.
        if gen.get("is_surrogate"):
            L.append(
                "> ⚠️ PFM was not available for this run. Generated structures come from a "
                "deterministic BRICS-recombination surrogate and are **not** "
                "structure-conditioned; they carry no binding-mode information."
            )
        for w in gen.get("warnings", [])[:3]:
            L.append(f"- _{w}_")
        for e in gen.get("errors", [])[:3]:
            L.append(f"- **generation error:** {e}")
        L.append("")

    # Shortlist scorecards
    L.append("## Shortlist (promoted)")
    objs = [o["name"] for o in d["objectives"]]
    header = "| Candidate | InChIKey | " + " | ".join(objs) + " | Utility | Pareto | Cluster | Decision |"
    L.append(header)
    L.append("|" + "---|" * (len(objs) + 6))
    for m in d["shortlist"]:
        sc = d["scorecards"].get(m["candidate_id"], {})
        eps = {e["endpoint"]: e for e in sc.get("endpoints", [])}
        cells = []
        for o in objs:
            e = eps.get(o, {})
            raw = e.get("raw_value")
            unc = e.get("uncertainty")
            cell = "—" if raw is None else (f"{_fmt(raw)}±{_fmt(unc)}" if unc else _fmt(raw))
            cells.append(cell)
        L.append(
            f"| {m['candidate_id'][:14]} | {(m.get('inchikey') or '')[:14]} | "
            + " | ".join(cells)
            + f" | {_fmt(sc.get('utility'))} | {sc.get('pareto_optimal')} "
            f"| {sc.get('diversity_cluster')} | {sc.get('decision')} |"
        )
    L.append("")

    # Safety register
    L.append("## Safety risk register")
    L.append("_'No detected signal' is not a safety claim._")
    L.append("| Candidate | Max severity | Flags | Endpoints |")
    L.append("| --- | --- | --- | --- |")
    for m in d["shortlist"]:
        s = d["safety"].get(m["candidate_id"], {})
        eps_str = ", ".join(
            f"{e['endpoint']}={_fmt(e.get('score'))}"
            for e in s.get("endpoints", [])
            if e.get("score") is not None
        )
        L.append(f"| {m['candidate_id'][:14]} | {s.get('max_severity')} | {s.get('n_flags')} | {eps_str} |")
    L.append("")

    # Sensitivity
    L.append("## Sensitivity analysis")
    L.append("How the shortlist changes under alternate objective weights:")
    L.append("| Scenario | Jaccard vs base | Changed |")
    L.append("| --- | --- | --- |")
    for r in d["sensitivity"].get("rows", []):
        L.append(f"| {r['scenario']} | {_fmt(r.get('jaccard_vs_base'))} | {r.get('changed')} |")
    L.append("")

    # Experiments
    L.append("## Proposed next experiments (PROPOSED)")
    for e in d["experiments"]:
        L.append(
            f"- **{e.get('title')}** ({e.get('assay_type')}, priority {e.get('priority')}): "
            f"{e.get('rationale')}"
        )
    L.append("")

    # Limitations. The per-capability lines are derived from the live tool pool
    # rather than hardcoded: a static list is how a report ends up describing
    # mocks that were deleted, or staying silent about a capability that has
    # since stopped producing anything (ADR-0011).
    L.append("## Limitations")
    L.append("- Activity labels in the demo are synthetic; metrics are demonstration-only.")
    L.append("- Safety endpoint models are baselines on synthetic labels — not validated toxicology.")
    L.append("- Conformer minimisation (MMFF94s) reports ligand strain, not binding affinity.")
    used_by_an_agent, n_catalogue = _unavailable_tools()
    for card in used_by_an_agent:
        L.append(f"- **{card.tool_id} produced NOTHING in this run.** {card.unavailable_reason}")
    if n_catalogue:
        L.append(
            f"- {n_catalogue} further tools in the catalogue are UNAVAILABLE on this machine. "
            "No agent in this run reaches for them, so their absence cost this run nothing — "
            "run `drugagent tools` for the list and `drugagent doctor` for each reason."
        )
    L.append("")
    L.append("## Reproducibility")
    L.append(
        f"Re-run: `drugagent run <brief> --profile {b.get('kind') and 'local'} --deterministic`. "
        "See `manifest.json` in the run bundle for exact versions and hashes."
    )
    L.append("")
    L.append("## Capability matrix")
    for cid, status in sorted(d["capability_matrix"].items()):
        L.append(f"- `{cid}`: {status}")
    L.append("")
    return "\n".join(L)


def render_html(d: dict[str, Any], markdown_body: str) -> str:
    funnel = d["funnel"]
    max_v = max(funnel.values()) or 1
    bars = "".join(
        f'<div class="bar"><span class="lbl">{k.replace("_", " ")}</span>'
        f'<span class="track"><span class="fill" style="width:{100 * v / max_v:.1f}%"></span></span>'
        f'<span class="val">{v}</span></div>'
        for k, v in funnel.items()
    )
    rows = ""
    objs = [o["name"] for o in d["objectives"]]
    for m in d["shortlist"]:
        sc = d["scorecards"].get(m["candidate_id"], {})
        eps = {e["endpoint"]: e for e in sc.get("endpoints", [])}
        tds = "".join(f"<td>{_fmt(eps.get(o, {}).get('raw_value'))}</td>" for o in objs)
        rows += (
            f"<tr><td>{html.escape(m['candidate_id'][:14])}</td>"
            f"<td class='mono'>{html.escape((m.get('inchikey') or '')[:14])}</td>{tds}"
            f"<td>{_fmt(sc.get('utility'))}</td><td>{sc.get('decision')}</td></tr>"
        )
    obj_headers = "".join(f"<th>{html.escape(o)}</th>" for o in objs)
    cap_rows = "".join(
        f"<tr><td class='mono'>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(d["capability_matrix"].items())
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>drug-agent report {html.escape(d["run_id"])}</title>
<style>
:root {{ --bg:#ffffff; --fg:#1a1a2e; --muted:#5b6472; --accent:#3b6fb0; --card:#f4f6fb; --line:#dbe0ea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12141c; --fg:#e6e8ee; --muted:#9aa4b2; --accent:#6fa8dc; --card:#1b1e2a; --line:#2a2e3c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
h1 {{ font-size:1.7rem; margin:0 0 .3rem; }} h2 {{ font-size:1.2rem; margin:2rem 0 .6rem; border-bottom:1px solid var(--line); padding-bottom:.3rem; }}
.disclaimer {{ background:var(--card); border-left:4px solid var(--accent); padding:.8rem 1rem; border-radius:6px; color:var(--muted); }}
.grid {{ display:flex; flex-wrap:wrap; gap:.75rem; margin:1rem 0; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.7rem 1rem; min-width:120px; }}
.stat .n {{ font-size:1.5rem; font-weight:700; color:var(--accent); }} .stat .k {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }}
.bar {{ display:flex; align-items:center; gap:.6rem; margin:.25rem 0; }} .bar .lbl {{ width:130px; color:var(--muted); font-size:.85rem; }}
.bar .track {{ flex:1; background:var(--line); border-radius:6px; height:16px; overflow:hidden; }} .bar .fill {{ display:block; height:100%; background:var(--accent); }}
.bar .val {{ width:44px; text-align:right; font-variant-numeric:tabular-nums; }}
table {{ border-collapse:collapse; width:100%; margin:.5rem 0; font-size:.9rem; }}
th,td {{ text-align:left; padding:.4rem .5rem; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }} .mono {{ font-family:ui-monospace,Menlo,monospace; font-size:.82rem; }}
.scroll {{ overflow-x:auto; }} .muted {{ color:var(--muted); }}
</style></head>
<body><div class="wrap">
<h1>Candidate Prioritization Report</h1>
<p class="muted mono">{html.escape(d["run_id"])} · {html.escape(d["generated_at"])}</p>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<h2>Candidate funnel</h2>
<div class="grid">
  <div class="stat"><div class="n">{funnel["standardized"]}</div><div class="k">standardized</div></div>
  <div class="stat"><div class="n">{funnel["passed_filters"]}</div><div class="k">passed filters</div></div>
  <div class="stat"><div class="n">{funnel["pareto"]}</div><div class="k">pareto</div></div>
  <div class="stat"><div class="n">{funnel["shortlist"]}</div><div class="k">shortlist</div></div>
</div>
{bars}
<h2>Shortlist</h2>
<div class="scroll"><table><thead><tr><th>Candidate</th><th>InChIKey</th>{obj_headers}<th>Utility</th><th>Decision</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>Capability matrix</h2>
<div class="scroll"><table><thead><tr><th>Capability</th><th>Status</th></tr></thead><tbody>{cap_rows}</tbody></table></div>
<h2>Full report</h2>
<pre style="white-space:pre-wrap; background:var(--card); padding:1rem; border-radius:8px; overflow-x:auto;">{html.escape(markdown_body)}</pre>
</div></body></html>"""


def _unavailable_tools() -> tuple[list, int]:
    """Absent tools this run would have used, and a count of the rest.

    Reading this from the live pool keeps the Limitations section honest as
    capabilities come and go; under make-it-or-fail an UNAVAILABLE tool
    contributes no objects at all, and a reader deserves to be told which parts of
    the funnel are simply absent.

    The split exists because the pool stopped being small. It was nine tools, and
    every absent one really was a hole in this run. It is now thirty-nine, thirty
    of them adapters over upstream projects, and most of those are not wired to any
    agent — so reporting each with its full install paragraph buried the three real
    limitations under a page of text about tools the run was never going to call.
    That is not more honest, it is less readable, and a limitation nobody finishes
    reading is not a limitation anyone acts on.

    So: a tool some agent actually reaches for gets its whole reason, because that
    is a hole in *this* run. The rest are counted, named nowhere, and pointed at
    ``drugagent tools`` — they are the catalogue, not this run's shortfall.
    """
    from ..agents.registry import AGENT_TOOLS
    from ..tools.registry import tool_pool

    reachable = {tool_id for tools in AGENT_TOOLS.values() for tool_id in tools}
    absent = [c for c in tool_pool() if c.status != "AVAILABLE"]
    used_by_an_agent = [c for c in absent if c.tool_id in reachable]
    return used_by_an_agent, len(absent) - len(used_by_an_agent)


def generate_report(ctx: RunContext) -> dict[str, Any]:
    data = gather_report_data(ctx)
    md = render_markdown(data)
    html_doc = render_html(data, md)
    md_ref = ctx.artifacts.put_text(md, kind="markdown", produced_by="generate_report", run_id=ctx.run.id)
    html_ref = ctx.artifacts.put_text(
        html_doc, kind="html", media_type="text/html", produced_by="generate_report", run_id=ctx.run.id
    )
    json_ref = ctx.artifacts.put_json(data, kind="json", produced_by="generate_report", run_id=ctx.run.id)
    ctx.store.put_object(
        "report",
        {"id": ctx.run.id, "md_ref": md_ref.uri, "html_ref": html_ref.uri, "json_ref": json_ref.uri},
        run_id=ctx.run.id,
    )
    return {
        "report_md": md_ref.uri,
        "report_html": html_ref.uri,
        "report_json": json_ref.uri,
        "n_shortlist": len(data["shortlist"]),
    }
