"""Build CSV and Markdown benchmark tables from immutable raw JSON results."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

COLUMNS=["dataset","model","input_representation","random_seed","threshold_method","precision","recall","f1","auroc","auprc","total_runtime_seconds","status","run_id"]

def build_tables(results_dir: str | Path="results/raw", output_dir: str | Path="results/aggregated") -> dict[str, Path]:
    rows=[]
    for p in Path(results_dir).glob("**/*.json"):
        with p.open() as f: r=json.load(f)
        rows.append({c:r.get(c) for c in COLUMNS})
    df=pd.DataFrame(rows, columns=COLUMNS)
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    csv=out/"benchmark_results.csv"; md=out/"benchmark_results.md"
    df.to_csv(csv,index=False)
    parts=["# Benchmark Results\n", df.to_markdown(index=False) if not df.empty else "No raw results found."]
    ok=df[df["status"]=="success"] if not df.empty else df
    if not ok.empty:
        agg=ok.groupby(["dataset","model"])[["precision","recall","f1","auroc","auprc"]].agg(["mean","std","count"])
        parts += ["\n\n## Successful-run Aggregates\n", agg.to_markdown()]
    failed=df[df["status"]!="success"] if not df.empty else df
    if not failed.empty: parts += ["\n\n## Failed Runs\n", failed.to_markdown(index=False)]
    md.write_text("\n".join(parts))
    return {"csv":csv,"markdown":md}
