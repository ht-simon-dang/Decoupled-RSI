"""Independent, version-bound evaluation and diverse computational nominations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import digest
from .contracts import Binder, Brief, Candidate

POLICY_ID = "antibody-independent-review-v1"


def evaluator_hash():
    root = Path(__file__).parent
    names = ("contracts.py", "evaluation.py", "structure.py", "delivery.py")
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


def identities(binder: Binder):
    chains = sorted(
        ({"role": c.role, "sequence": c.sequence} for c in binder.chains), key=lambda x: x["role"]
    )
    return digest({"chains": chains, "product": binder.product.model_dump()}), digest(chains)


def make_candidate(binder, origin, provenance, parent=None):
    binder = Binder.model_validate(binder)
    identity, variable = identities(binder)
    mutations = []
    if parent:
        old = {c.role: c.sequence for c in parent.binder.chains}
        for chain in binder.chains:
            before = old.get(chain.role, "")
            if len(before) != len(chain.sequence):
                mutations.append(f"{chain.role}:length_or_pairing_change")
            else:
                mutations.extend(
                    f"{chain.role}:{a}{i}{b}"
                    for i, (a, b) in enumerate(zip(before, chain.sequence, strict=True), 1)
                    if a != b
                )
    return Candidate(
        id=identity,
        variable_id=variable,
        binder=binder,
        origin=origin,
        provenance=provenance,
        parent_id=parent.id if parent else None,
        mutations=mutations,
    )


def context_id(candidate: Candidate, state):
    return digest({"candidate_id": candidate.id, "antigen": state.model_dump()})


def assessment_key(candidate, brief):
    return digest({"candidate_id": candidate.id, "brief": brief.model_dump(), "evaluator": evaluator_hash()})


def evaluation_binder(candidate):
    binding = candidate.binder.model_dump()
    binding["structure_pdb"] = None
    for chain in binding["chains"]:
        for row in chain["numbering"]:
            row["structure_residue"] = None
    return binding


def authenticated_result(result, tool, arguments):
    source = result.get("provenance", {})
    return (
        result.get("tool") == tool
        and source.get("model") == tool.split(".")[1]
        and bool(source.get("version"))
        and bool(source.get("weights_version"))
        and source.get("request_hash") == digest(arguments)
        and source.get("output_sha256") == digest(result.get("data", {}))
    )


def validate_edits(candidate: Candidate, brief: Brief):
    """Verify exact paired edits; unsupported length changes cannot be silently accepted."""
    errors = []
    if candidate.binder.product != brief.product:
        errors.append("Candidate product differs from the declared final format")
    for chain in candidate.binder.chains:
        if not chain.numbering:
            errors.append(f"{chain.role}: numbering unavailable")
        elif chain.scheme != brief.constraints.scheme:
            errors.append(f"{chain.role}: numbering scheme mismatch")
    if candidate.origin in {"library", "reference"}:
        return errors
    if not brief.reference:
        return errors + ["Generation requires a framework/reference and declared editable positions"]
    parents = {c.role: c for c in brief.reference.chains}
    changed = 0
    for chain in candidate.binder.chains:
        parent = parents.get(chain.role)
        if parent is None or len(chain.sequence) != len(parent.sequence):
            errors.append("Length/pairing changes require a separately validated mapping adapter")
            continue
        if not parent.numbering:
            errors.append("Parent numbering is unavailable")
            continue
        labels = {r.index: r.label for r in parent.numbering}
        allowed = set(brief.constraints.editable.get(chain.role, []))
        protected = set(brief.constraints.protected.get(chain.role, []))
        for i, (a, b) in enumerate(zip(parent.sequence, chain.sequence, strict=True), 1):
            if a != b:
                changed += 1
                if labels[i] not in allowed or labels[i] in protected:
                    errors.append(f"Mutation outside allowed region: {chain.role}:{labels[i]}")
    if changed > brief.constraints.max_mutations:
        errors.append("Mutation budget exceeded")
    return errors


def sequence_screen(candidate):
    risks = []
    for c in candidate.binder.chains:
        for match in re.finditer(r"N[^P][ST]", c.sequence):
            risks.append(
                {
                    "chain": c.role,
                    "index": match.start() + 1,
                    "risk": "potential N-linked glycosylation motif",
                }
            )
        for match in re.finditer(r"[MW]|N[GS]|D[PG]", c.sequence):
            risks.append(
                {
                    "chain": c.role,
                    "index": match.start() + 1,
                    "risk": "sequence liability to assess in context",
                }
            )
    return {
        "status": "computed",
        "method": "Exact sequence motifs; not TAP/TNP, aggregation, or immunogenicity predictions",
        "flags": risks,
    }


def evaluate(candidate, brief, calls):
    failures = validate_edits(candidate, brief)
    unknowns = [
        "Binding kinetics",
        "Disease function",
        "Clinical immunogenicity",
        "Final-format expression and pharmacology",
    ]
    evidence = []
    metrics = {}
    policy = brief.evaluation_policy
    binding = evaluation_binder(candidate)
    positives = [s for s in brief.states if s.role == "positive"]
    if not positives:
        failures.append("No positive antigen state")
    for gate in (brief.direction_gate, brief.access_gate):
        if gate.status in {"unknown", "contradicted"}:
            failures.append("Unresolved or contradicted mechanism/access prerequisite")
    if brief.direction == "unresolved":
        failures.append("Mechanism direction is unresolved")
    if brief.product.format in {"bispecific", "ADC", "conditional", "fusion"}:
        failures.append(
            "Advanced format requires whole-assembly validation; single-arm results cannot qualify it"
        )

    fold = calls.get("fold", {})
    if fold.get("status") != "ok":
        failures.append("Independent antibody fold unavailable or failed")
    elif not authenticated_result(fold, "fold.immunebuilder", {"binder": binding, "seed": brief.seed}):
        failures.append("Independent fold provenance/identity mismatch")
    else:
        data = fold.get("data", {})
        if data.get("sequence_verified") is not True or not isinstance(data.get("ca_clashes"), int):
            failures.append("Fold lacks independently checked sequence/geometry")
        elif data["ca_clashes"] > policy.max_ca_clashes:
            failures.append("Antibody fold has backbone clashes")
        evidence.append(
            {
                "endpoint": "unbound_fold",
                "status": "predicted",
                "source": fold.get("provenance"),
                "metrics": {k: v for k, v in data.items() if k != "pdb"},
            }
        )

    profile = calls.get("profile", {})
    profile_name = "tnp" if candidate.binder.chains[0].role == "VHH" else "tap"
    if profile.get("status") != "ok":
        failures.append("Format-specific TAP/TNP profile unavailable or failed")
    elif not authenticated_result(
        profile,
        "profile." + profile_name,
        {"binder": binding, "pdb": fold.get("data", {}).get("pdb"), "seed": brief.seed},
    ):
        failures.append("Developability profile provenance/identity mismatch")
    else:
        data = profile.get("data", {})
        expected = "tnp" if candidate.binder.chains[0].role == "VHH" else "tap"
        if (
            data.get("profile") != expected
            or data.get("applicable") is not True
            or not isinstance(data.get("severe_flags"), list)
            or not data.get("metrics")
        ):
            failures.append("Developability profile is incomplete or out of domain")
        elif data["severe_flags"]:
            failures.append("Severe developability flags")
        evidence.append(
            {
                "endpoint": "developability",
                "status": "predicted",
                "source": profile.get("provenance"),
                "metrics": data,
            }
        )

    positive_scores = []
    for state in brief.states:
        result = calls.get("complex:" + state.id, {})
        context = context_id(candidate, state)
        if result.get("status") != "ok":
            (failures if state.role == "positive" else unknowns).append(
                f"{state.id}: independent complex unavailable"
            )
            continue
        data = result.get("data", {})
        model = "chai1" if candidate.origin == "boltzgen" else "boltz2"
        arguments = {
            "binder": binding,
            "state": state.model_dump(),
            "context_id": context,
            "seed": brief.seed,
        }
        if not authenticated_result(result, "complex." + model, arguments):
            failures.append(f"{state.id}: independent complex provenance/identity mismatch")
            continue
        if data.get("context_id") != context or data.get("sequence_verified") is not True:
            failures.append(f"{state.id}: wrong candidate/antigen identity")
            continue
        if data.get("templates_used") is not False or data.get("forced_contacts") is not False:
            failures.append(f"{state.id}: constrained prediction is not independent evaluation")
        source = result.get("provenance", {}).get("model")
        if (
            not source
            or source == candidate.origin
            or (candidate.origin == "boltzgen" and source == "boltz2")
        ):
            failures.append(f"{state.id}: use an evaluator from a different model family")
        score, confidence = data.get("antigen_iptm"), data.get("complex_plddt")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1
            for v in (score, confidence)
        ):
            failures.append(f"{state.id}: missing or invalid structural confidence")
            continue
        metrics[state.id] = {"antigen_iptm": score, "complex_plddt": confidence, "role": state.role}
        evidence.append(
            {
                "endpoint": "pose",
                "status": "predicted",
                "context_id": context,
                "source": result.get("provenance"),
                "metrics": metrics[state.id],
                "limitation": "Structural confidence is not binding probability or KD",
            }
        )
        if state.role == "negative":
            unknowns.append(f"{state.id}: negative-state prediction cannot establish experimental nonbinding")
            continue
        positive_scores.append(score)
        if score < policy.min_iptm or confidence < policy.min_plddt:
            failures.append(f"{state.id}: structural confidence below frozen policy")
        if not isinstance(data.get("ca_clashes"), int) or data["ca_clashes"] > policy.max_ca_clashes:
            failures.append(f"{state.id}: backbone clashes or missing geometry audit")
        epitopes = [e for e in brief.epitopes if e.state_id == state.id]
        if not epitopes or not any(e.residues for e in epitopes):
            failures.append(f"{state.id}: functional epitope not specified")
        contacts = set(data.get("antigen_contacts", []))
        for epitope in epitopes:
            if epitope.accessibility.status == "contradicted":
                failures.append(f"{epitope.id}: inaccessible epitope")
            if contacts & set(epitope.forbidden_residues):
                failures.append(f"{epitope.id}: forbidden contact")
            coverage = len(contacts & set(epitope.residues)) / max(1, len(set(epitope.residues)))
            metrics[state.id]["epitope_coverage:" + epitope.id] = coverage
            if coverage < policy.min_epitope_coverage:
                failures.append(f"{epitope.id}: insufficient predicted epitope coverage")
        if not state.full_context:
            failures.append(f"{state.id}: native/full antigen context not evaluated")
        if state.glycan_context == "unknown":
            unknowns.append(f"{state.id}: glycan occlusion unknown")
        if state.membrane_context == "unknown":
            unknowns.append(f"{state.id}: membrane geometry unknown")
        # A protein-only fold cannot establish known membrane/glycan compatibility.
        if state.glycan_context == "known" or state.membrane_context == "known":
            if data.get("native_context_checked") is not True:
                failures.append(f"{state.id}: supplied native context requires an explicit geometry adapter")
    unknowns += [
        "Off-target selectivity requires a sufficiently broad panel",
        "On-target/off-tissue effects require normal-tissue evidence",
    ]
    return {
        "candidate_id": candidate.id,
        "key": assessment_key(candidate, brief),
        "policy_id": POLICY_ID,
        "evaluator_hash": evaluator_hash(),
        "status": "pass" if not failures else "withheld",
        "reasons": sorted(set(failures)),
        "unknowns": sorted(set(unknowns)),
        "evidence": evidence,
        "metrics": metrics,
        "ranking_value": min(positive_scores) if positive_scores else None,
        "sequence_screen": sequence_screen(candidate),
        "contexts": {s.id: context_id(candidate, s) for s in brief.states},
    }


def similarity(a, b):
    left = {c.role: c.sequence for c in a.binder.chains}
    right = {c.role: c.sequence for c in b.binder.chains}
    if left.keys() != right.keys():
        return 0.0
    matches, total = 0, 0
    for role, seq in left.items():
        other = right[role]
        matches += sum(x == y for x, y in zip(seq, other, strict=False))
        total += max(len(seq), len(other))
    return matches / total


def select(candidates, assessments, brief):
    """Stable greedy family coverage; no synthetic aggregate therapeutic score."""
    passed = [
        c
        for c in candidates
        if assessments.get(c.id, {}).get("status") == "pass"
        and assessments[c.id].get("key") == assessment_key(c, brief)
    ]
    passed.sort(key=lambda c: (-(assessments[c.id]["ranking_value"] or 0), c.id))
    representatives, assignments = [], {}
    for c in passed:
        related = next(
            (p for p in representatives if similarity(c, p) >= brief.evaluation_policy.family_identity), None
        )
        if related is None:
            representatives.append(c)
            related = c
        assignments[c.id] = related.id[:12]
    count = min(brief.budget.shortlist, brief.budget.experiment_slots)
    chosen = representatives[:count]
    return [
        {
            "candidate_id": c.id,
            "family": assignments[c.id],
            "role": "primary" if i == 0 else "alternative",
            "reason": "Independent computational gates passed; distinct paired-sequence family",
            "next_validation": "Measure binding, epitope and disease function separately in the declared final format",
        }
        for i, c in enumerate(chosen)
    ]
