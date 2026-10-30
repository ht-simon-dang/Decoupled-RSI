"""Explicit, read-only inspection of past evidence, separate from delivery."""
from copy import deepcopy

from .delivery import public_run
from .filtering import POLICY_ID


def preview_run(run):
    view = public_run(run)
    raw, payload = run.get("payload", {}), view["payload"]
    delivered = {c["id"] for c in payload["candidates"]}
    legacy = raw.get("portfolio", {}).get("filter_policy_id") != POLICY_ID
    evaluations = {e["construct_id"]: e for e in raw.get("evaluations", [])}
    filters = {r["id"]: r for r in payload["portfolio"].get("filter_results", [])}
    candidates = deepcopy(raw.get("candidates", []))
    for c in candidates:
        key = c["id"]
        status = ("high_potential" if key in delivered else
                  "control" if c.get("sources", {}).get("baseline") else
                  "legacy_unverified" if legacy else
                  "not_evaluated" if key not in evaluations else "not_delivered")
        c["preview"] = {"status": status, "recommended": key in delivered,
                        "filter_result": deepcopy(filters.get(key))}
    payload.update(candidates=candidates, evaluations=deepcopy(raw.get("evaluations", [])))
    view["preview"] = {"read_only": True, "legacy": legacy,
                       "notice": "Historical evidence inspection only. Only current high-potential candidates are recommendations; exports remain filtered."}
    return view


def candidate_record(run, candidate_id):
    """Membership is checked in this run, not in the global construct store."""
    payload = run.get("payload", {})
    candidate = next((c for c in payload.get("candidates", []) if c["id"] == candidate_id), None)
    if candidate is None:
        raise KeyError("Construct not found in this run")
    evaluation = next((e for e in reversed(payload.get("evaluations", []))
                       if e["construct_id"] == candidate_id), {})
    return candidate, evaluation


def rna_topology(candidate, evaluation):
    """Stored MFE pairing with ViennaRNA's planar layout, never atomic coordinates."""
    sequence = candidate["sequence"]
    dot = evaluation.get("structure", {}).get("mfe_dot_bracket")
    result = {"kind": "rna_secondary_topology", "atomic_structure": False,
              "sequence": sequence, "boundaries": candidate["boundaries"],
              "poly_a_boundary": candidate.get("poly_a_boundary"),
              "coordinate_units": "arbitrary layout units", "evaluation_id": evaluation.get("evaluation_id"),
              "scope": "Stored RNA secondary structure; planar topology in a rotatable viewer, not an RNA tertiary prediction."}
    if not dot:
        return {**result, "status": "unavailable", "reason": "This evaluation has no stored full-molecule MFE structure."}
    stack, pairs = [], []
    if len(dot) != len(sequence) or set(dot) - set(".()"):
        return {**result, "status": "unavailable", "reason": "Stored structure does not match the complete sequence."}
    for i, char in enumerate(dot):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if not stack:
                return {**result, "status": "unavailable", "reason": "Unbalanced stored structure."}
            pairs.append([stack.pop(), i])
    if stack:
        return {**result, "status": "unavailable", "reason": "Unbalanced stored structure."}
    import RNA
    coords = RNA.get_xy_coordinates(dot)
    points = [[round(coords.get(i).X, 4), round(coords.get(i).Y, 4), 0.] for i in range(len(dot))]
    return {**result, "status": "ready", "dot_bracket": dot, "pairs": pairs, "points": points,
            "source": "ViennaRNA " + RNA.__version__ + " layout of stored MFE pairing"}
