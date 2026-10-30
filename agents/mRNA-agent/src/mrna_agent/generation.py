"""Reference controls, region specialists, fixed-boundary assembly and explicit fallbacks."""
from __future__ import annotations
import csv
import json
import random
from Bio.Data import CodonTable
from .contracts import Construct, Regions, translate, identity, rna
from .storage import data_root

TABLE = CodonTable.unambiguous_rna_by_id[1].forward_table
CODONS = {aa: sorted(c for c, a in TABLE.items() if a == aa) for aa in sorted(set(TABLE.values()))}


def backtranslate(protein, seed=0, preferred=False):
    rng = random.Random(seed)
    usage = {}
    path = data_root() / "external/LinearDesign/codon_usage_freq_table_human.csv"
    if preferred and path.is_file():
        for row in csv.reader(path.open()):
            if len(row) >= 3 and row[0].upper().replace("T", "U") in TABLE:
                try: usage[row[0].upper().replace("T", "U")] = float(row[-1])
                except ValueError: pass
    return "".join(max(CODONS[a], key=lambda c: (usage.get(c, 0), c)) if preferred else rng.choice(CODONS[a]) for a in protein) + "UAA"


def reference_elements():
    path = data_root() / "datasets/references.json"
    if path.is_file():
        return json.loads(path.read_text())
    return []


def make_baseline(brief):
    if brief.reference:
        regions = brief.reference
        source = {"baseline": True, "reference": "user supplied", "evidence": "sequence only; measurements must be imported separately"}
    else:
        refs = reference_elements()
        item = next((r for r in refs if r.get("regions") and r.get("kind") == "natural_transcript"), None)
        if item:
            u5, u3 = item["regions"]["utr5"], item["regions"]["utr3"]
            source = {"baseline": True, "utr_reference": item["id"], "reference_source": item["source"], "evidence": "Natural transcript prior; no claim of engineered expression benefit"}
        else:
            u5, u3 = "GGGAAACCGCCACC", "UCUAGAGCUAGC"
            source = {"baseline": True, "utr_reference": "synthetic minimal control v1", "evidence": "Unvalidated synthetic reference, not a measured high-expression UTR"}
        regions = Regions(utr5=u5, cds=backtranslate(brief.protein, preferred=True), utr3=u3)
    return Construct(regions=regions, molecule=brief.reference_config or brief.molecule, protein_version=brief.protein_version,
                     sources=source, hypothesis="Baseline control; establish matched expression and quality before claiming improvement")


def apply_locks(regions, brief):
    out = regions.model_dump()
    for name in brief.locks:
        out[name] = getattr(brief.reference, name)
    for span in brief.locked_spans:
        # Fixed offsets cannot be silently moved if a generator changes region length.
        if len(out[span.region]) < span.end:
            raise ValueError("Generated region is shorter than a locked span")
        ref = getattr(brief.reference, span.region)
        out[span.region] = out[span.region][:span.start] + ref[span.start:span.end] + out[span.region][span.end:]
    return Regions(**out)


def normalized_cds(seq, protein):
    seq = rna(seq)
    # Author CDS generators omit terminal stop when protein lacks '*'. Record this assembly step.
    if len(seq) == 3 * len(protein):
        seq += "UAA"
    if translate(seq) != protein:
        raise ValueError("Generated CDS changes the requested protein")
    return seq


def generate_round(brief, baseline, gateway, round_index=0, plan=None):
    budget = brief.budget.candidates
    seed = brief.seed + 1000 * round_index
    pools = {r: [(getattr(baseline.regions, r), {"source": "baseline"}, [])] for r in ("utr5", "cds", "utr3")}
    failures = []
    routes = list(brief.generators)
    if brief.mode == "diagnose":
        return [baseline], failures
    per_route = max(1, min(8, (budget + len(routes) - 1) // len(routes)))
    focus = (plan or {}).get("focus", "balanced")
    if "gemorna" in routes:
        for region, mode in [("cds", "cds"), ("utr5", "5utr"), ("utr3", "3utr")]:
            if region in brief.locks: continue
            if region == "cds" and len(brief.protein) > 184:
                failures.append({"tool": "gemorna", "region": region, "status": "out_of_scope", "reason": "Installed CDS positional embedding limit; no protein truncation"})
                continue
            count = min(16,per_route*2) if (focus == "cds" and region == "cds") or (focus == "utr" and region != "cds") else 1 if focus in {"cds","utr"} else per_route
            result = gateway.call("gemorna", {"region": mode, "protein": brief.protein if region == "cds" else None, "count": count, "seed": seed, "utr_length": brief.utr_length})
            if result["status"] != "ok":
                failures.append(result); continue
            for item in result["data"]["sequences"]:
                try:
                    sequence = normalized_cds(item["sequence"], brief.protein) if region == "cds" else rna(item["sequence"])
                    pools[region].append((sequence, {"source": "gemorna", "seed": item["seed"], "provenance": result.get("provenance", {}), "assembly": "append terminal UAA if absent" if region == "cds" else None}, []))
                except ValueError as exc:
                    failures.append({"tool": "gemorna", "region": region, "status": "rejected", "reason": str(exc), "sequence": item["sequence"]})
    if "lineardesign" in routes and "cds" not in brief.locks:
        if len(brief.protein) > 2000 or brief.context.species != "human":
            failures.append({"tool": "lineardesign", "status": "out_of_scope", "reason": "Human codon table and <=2000 aa application limit"})
        else:
            result = gateway.call("lineardesign", {"protein": brief.protein, "lambda": brief.linear_lambda})
            if result["status"] == "ok":
                for item in result["data"]["sequences"]:
                    try:
                        seq = normalized_cds(item["sequence"], brief.protein)
                        pools["cds"].append((seq, {"source": "lineardesign", "lambda": brief.linear_lambda, "provenance": result.get("provenance", {}), "assembly": "append terminal UAA if absent"}, ["mfe", "codon_usage"]))
                    except ValueError as exc:
                        failures.append({"tool": "lineardesign", "status": "rejected", "reason": str(exc)})
            else: failures.append(result)
    if "smart5utr" in routes and "utr5" not in brief.locks:
        if brief.molecule.modification != "m1Psi" or len(baseline.regions.utr5) != 50:
            failures.append({"tool": "smart5utr", "status": "out_of_scope", "reason": "Requires m1Psi and an exact 50 nt reference UTR; no implicit trim"})
        else:
            result = gateway.call("smart", {"kind": "smart5utr", "generate": True, "sequences": [baseline.regions.utr5], "count": per_route, "seed": seed})
            if result["status"] == "ok":
                for seq in result["data"]["sequences"]:
                    pools["utr5"].append((rna(seq), {"source": "smart5utr", "seed": seed, "provenance": result.get("provenance", {})}, ["ribosome_load"]))
            else: failures.append(result)
    if "parade" in routes:
        region = "utr5" if "utr5" not in brief.locks else "utr3"
        n = 50 if region == "utr5" else 240
        cell = brief.context.target_cell
        supported = {"c1", "c2", "c4", "c6", "c17"} | ({"c13"} if region == "utr3" else set())
        if region in brief.locks or len(getattr(baseline.regions, region)) != n or cell not in supported or not set(brief.context.non_target_cells) <= supported or brief.molecule.modification not in {"unmodified", "unspecified"}:
            failures.append({"tool": "parade", "status": "out_of_scope", "reason": "Requires exact 50/240 nt reference, matching unmodified chemistry and author cell codes"})
        else:
            rng = random.Random(seed)
            seqs = [getattr(baseline.regions, region)]
            for i in range(max(8, per_route * 4)):
                s = list(seqs[0])
                for pos in rng.sample(range(n), min(4, n)):
                    s[pos] = rng.choice("ACGU")
                seqs.append("".join(s))
            result = gateway.call("parade", {"region": region, "sequences": seqs, "cells": list(dict.fromkeys([cell] + brief.context.non_target_cells))})
            if result["status"] == "ok":
                scored = [(r["scores"][cell], max([r["scores"][c] for c in brief.context.non_target_cells], default=0), r["index"]) for r in result["data"]["predictions"]]
                # Target activity floor is the matched reference; a ratio alone cannot select a low/low candidate.
                floor = scored[0][0]
                selected = sorted([r for r in scored[1:] if r[0] >= floor], key=lambda r: (r[0]-r[1], r[0]), reverse=True)[:per_route]
                for _, _, i in selected:
                    pools[region].append((seqs[i], {"source": "parade_guided_mutation", "seed": seed, "sampler": "bounded one-generation mutation search using official predictor", "target_activity_floor": floor, "provenance": result.get("provenance", {})}, ["parade_activity"]))
            else: failures.append(result)
    for name,region in [("codongpt","cds"),("utrgan","utr5")]:
        if name not in routes or region in brief.locks:continue
        result=gateway.call(name,{"protein":brief.protein,"count":per_route,"seed":seed})
        if result['status']!='ok':failures.append(result);continue
        for item in result['data']['sequences']:
            try:
                seq=normalized_cds(item['sequence'],brief.protein) if region=='cds' else rna(item['sequence'])
                pools[region].append((seq,{'source':name,'seed':seed,'provenance':result.get('provenance',{}),'sampler':result['data'].get('sampler')},[]))
            except ValueError as exc:failures.append({'tool':name,'status':'rejected','reason':str(exc)})
    for name in ('promorna','mrna_gpt'):
        if name in routes:
            failures.append({'tool':name,'status':'unavailable','reason':'Official public deployable checkpoint/API not found; paper-only release, see model availability audit'})
    if "joint" in routes and not brief.joint_proposals:
        failures.append({"tool": "joint", "status": "unavailable", "reason": "ProMORNA/mRNAutilus official deployable service or weights not verified"})
    need_fallback = all(len(pools[r]) == 1 for r in pools) and not brief.joint_proposals
    if "synonymous" in routes or (need_fallback and brief.allow_fallback):
        if "cds" not in brief.locks:
            for i in range(per_route * 2):
                pools["cds"].append((backtranslate(brief.protein, seed+i), {"source": "uniform_synonymous", "seed": seed+i, "fallback": need_fallback, "evidence": "No learned efficacy"}, []))
            if need_fallback:
                failures.append({"tool": "fallback", "status": "activated", "reason": "Explicit uniform synonymous search replaced unavailable learned generation; evidence scope changed"})
    candidates = [baseline]
    seen = {baseline.id}
    if "joint" in routes:
        for proposal in brief.joint_proposals:
            if len(candidates) >= budget:break
            candidate = Construct(regions=proposal.regions,molecule=brief.molecule,protein_version=brief.protein_version,
                                  sources={"joint":proposal.model_dump(exclude={"regions"}),"verification":"user-supplied full-length output; provider provenance asserted, not independently authenticated"},parent_ids=[baseline.id])
            if candidate.id in seen:
                failures.append({"status":"duplicate","construct_id":candidate.id})
                continue
            # Keep even constraint-violating imports for the same independent evaluator/rejection list.
            candidates.append(candidate);seen.add(candidate.id)
    if "mrnautilus" in routes and len(candidates)<budget:
        tokens=len(baseline.regions.utr5)+len(baseline.regions.cds)//3+len(baseline.regions.utr3)+2
        if tokens>1024:
            failures.append({'tool':'mrnautilus','status':'out_of_scope','reason':'1024 mixed-token application context; no truncation'})
        else:
            result=gateway.call('mrnautilus',{'regions':baseline.regions.model_dump(),'count':min(per_route,budget-len(candidates)),'seed':seed,'fixed_regions':sorted(brief.locks)})
            if result['status']!='ok':failures.append(result)
            else:
                for parts in result['data']['constructs']:
                    try:
                        regions=apply_locks(Regions(**parts),brief)
                        if translate(regions.cds)!=brief.protein:raise ValueError('Joint generator changed protein')
                        c=Construct(regions=regions,molecule=brief.molecule,protein_version=brief.protein_version,
                          sources={'joint':{'source':'mrnautilus','sampler':result['data']['sampler'],'provenance':result.get('provenance',{})}},parent_ids=[baseline.id])
                        if c.id not in seen:candidates.append(c);seen.add(c.id)
                    except ValueError as exc:failures.append({'tool':'mrnautilus','status':'rejected','reason':str(exc)})
    combinations = []
    # Always create paired single-region hypotheses before factorial combinations.
    for region in ("cds", "utr5", "utr3"):
        for item in pools[region][1:]:
            combination = {r: pools[r][0] for r in pools}; combination[region] = item
            combinations.append(combination)
    rng = random.Random(seed)
    rng.shuffle(combinations)
    for _ in range(budget * 3):
        combinations.append({r: rng.choice(pools[r]) for r in pools})
    # v2 spreads route/region contributions before paired/factorial sampling.
    if brief.workflow_version == "v2":
        combinations.sort(key=lambda x: (sum(v[1].get("source") != "baseline" for v in x.values()), identity({r: x[r][1].get("source") for r in x})))
    for combination in combinations:
        if len(candidates) >= budget: break
        try:
            regions = apply_locks(Regions(**{r: combination[r][0] for r in pools}), brief)
            if translate(regions.cds) != brief.protein:
                raise ValueError("Locked span restoration changes translated protein")
            candidate = Construct(regions=regions, molecule=brief.molecule, protein_version=brief.protein_version,
                                  sources={r: combination[r][1] for r in pools}, parent_ids=[baseline.id],
                                  optimized_metrics=sorted(set(k for r in pools for k in combination[r][2])))
            if candidate.id in seen:
                failures.append({"status": "duplicate", "construct_id": candidate.id}); continue
            seen.add(candidate.id); candidates.append(candidate)
        except ValueError as exc:
            failures.append({"status": "rejected", "reason": str(exc)})
    return candidates, failures
