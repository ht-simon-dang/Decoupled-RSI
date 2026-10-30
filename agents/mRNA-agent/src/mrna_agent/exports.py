import csv
import io
import json


def export(run, format):
    from .delivery import public_run
    run=public_run(run)
    p=run["payload"]
    if format=="json": return json.dumps(run,indent=2,ensure_ascii=False), "application/json"
    if format=="fasta":
        return "".join(f">{c['id']} protein={c['protein_version']} cap={c['molecule']['cap']} modification={c['molecule']['modification']}\n" + "\n".join(c["sequence"][i:i+80] for i in range(0,len(c["sequence"]),80)) + "\n" for c in p.get("candidates",[])), "text/plain"
    if format=="csv":
        buf=io.StringIO();writer=csv.writer(buf)
        writer.writerow(["construct_id","status","metric","value","unit","source","version","evidence","independent"])
        for e in p.get("evaluations",[]):
            for m in e["metrics"]:
                writer.writerow([e["construct_id"],m["status"],m["metric"],m["value"],m["unit"],m["source"],m["version"],m["evidence"],m["independent"]])
        return buf.getvalue(), "text/csv"
    raise ValueError("Export format must be json, fasta or csv")
