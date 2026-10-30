import csv
import json
from io import StringIO

from .delivery import allowed_candidate, public_run


def export(run, format):
    view = public_run(run)
    if format == "json":
        return json.dumps(view, ensure_ascii=False, indent=2, allow_nan=False), "application/json"
    if format == "fasta":
        lines = []
        for c in view["candidates"]:
            for chain in c["binder"]["chains"]:
                lines += [
                    f">{c['id']}|{chain['role']}|variable_domain|format={c['binder']['product']['format']}",
                    chain["sequence"],
                ]
        return "\n".join(lines) + ("\n" if lines else ""), "text/plain"
    if format == "csv":
        result = StringIO()
        writer = csv.writer(result)
        writer.writerow(["candidate_id", "variable_id", "format", "origin", "evaluation", "human_review"])
        for c in view["candidates"]:
            writer.writerow(
                [
                    c["id"],
                    c["variable_id"],
                    c["binder"]["product"]["format"],
                    c["origin"],
                    "computational_pass",
                    "required",
                ]
            )
        return result.getvalue(), "text/csv"
    if format == "report":
        lines = [
            f"# {view['name']}",
            "",
            f"Run: {view['id']} · {view['state']}",
            "",
            view.get("diagnosis") or "任务进行中",
            "",
            f"交付状态：{view['delivery']['state']}",
            f"生成 {view['delivery']['generated_count']} · 评估 {view['delivery']['evaluated_count']} · 交付 {len(view['candidates'])}",
            "",
            "候选序列仅为可变区，不能作为全长 IgG 产物。计算通过不等于实测结合、机制或开发成功。",
            "",
            "项目证据与缺口：",
            json.dumps(view["research"], ensure_ascii=False, indent=2),
            "",
            "候选分项证据：",
            json.dumps(view["evaluations"], ensure_ascii=False, indent=2),
            "",
            "未通过原因：",
            json.dumps(view["diagnostics"], ensure_ascii=False, indent=2),
        ]
        return "\n".join(lines), "text/markdown"
    raise ValueError("Unknown export format")


def structure_export(run, candidate_id, state_id=None):
    allowed_candidate(run, candidate_id)
    key = "complex:" + state_id if state_id else "fold"
    result = run["payload"]["evaluation_calls"][candidate_id].get(key, {})
    if result.get("status") != "ok" or not result.get("data", {}).get("pdb"):
        raise ValueError("No independently evaluated structure for this context")
    return result["data"]["pdb"]
