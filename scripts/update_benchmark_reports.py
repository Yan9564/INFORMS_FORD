#!/usr/bin/env python
"""Update README benchmark reporting tables from committed CSV sources."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PAPERS_START = "<!-- BENCHMARK-PAPERS:START -->"
PAPERS_END = "<!-- BENCHMARK-PAPERS:END -->"
RESULTS_START = "<!-- BENCHMARK-RESULTS:START -->"
RESULTS_END = "<!-- BENCHMARK-RESULTS:END -->"

RESULT_COLUMNS = [
    "dataset",
    "model",
    "run_mode",
    "seed",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "status",
]
PAPER_COLUMNS = [
    "title",
    "year",
    "venue",
    "method_family",
    "datasets",
    "official_code",
    "repository_url",
    "license",
    "integration_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display_name(value: str) -> str:
    names = {"swat": "SWaT", "gdn": "GDN"}
    return names.get(value.strip().lower(), value.strip())


def format_metric(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    table.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(table)


def build_results_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        status = row.get("status", "")
        run_mode = row.get("run_mode", "")
        if run_mode == "smoke_test" or status == "smoke_test":
            status = f"{status} (execution verification; not final paper results)".strip()
        body.append(
            [
                display_name(row.get("dataset", "")),
                display_name(row.get("model", "")),
                run_mode,
                row.get("seed", ""),
                format_metric(row.get("precision", "")),
                format_metric(row.get("recall", "")),
                format_metric(row.get("f1", "")),
                format_metric(row.get("auroc", "")),
                format_metric(row.get("auprc", "")),
                status,
            ]
        )
    return markdown_table(["Dataset", "Model", "Run mode", "Seed", "Precision", "Recall", "F1", "AUROC", "AUPRC", "Status"], body)


def build_papers_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row.get("selected_for_experiment", "").strip().lower() in {"1", "true", "yes", "y"}]
    if not selected:
        selected = rows
    body = [[row.get(column, "") for column in PAPER_COLUMNS] for row in selected]
    return markdown_table(["Title", "Year", "Venue", "Method family", "Datasets", "Official code", "Repository URL", "License", "Integration status"], body)


def replace_section(text: str, start: str, end: str, content: str) -> str:
    section = f"{start}\n{content}\n{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return before + section + after
    return text.rstrip() + "\n\n" + section + "\n"


def build_readme_content(readme_text: str, results_rows: list[dict[str, str]], paper_rows: list[dict[str, str]], results_csv: Path, results_xlsx: Path) -> str:
    links = [f"Source CSV: [`{results_csv.as_posix()}`]({results_csv.as_posix()})"]
    if results_xlsx.exists():
        links.append(f"XLSX export: [`{results_xlsx.as_posix()}`]({results_xlsx.as_posix()})")
    results_content = "\n".join([
        "## Benchmark results",
        "",
        "Smoke-test rows, when present, are execution verification only and are not final paper results.",
        "",
        " | ".join(links),
        "",
        build_results_table(results_rows),
    ])
    papers_content = "\n".join([
        "## Selected papers",
        "",
        "Generated from [`registry/papers.csv`](registry/papers.csv).",
        "",
        build_papers_table(paper_rows),
    ])
    text = replace_section(readme_text, PAPERS_START, PAPERS_END, papers_content)
    return replace_section(text, RESULTS_START, RESULTS_END, results_content)


def update_readme(repo_root: Path) -> str:
    readme = repo_root / "README.md"
    results_csv = repo_root / "results" / "benchmark_results.csv"
    papers_csv = repo_root / "registry" / "papers.csv"
    results_xlsx = repo_root / "results" / "benchmark_results.xlsx"
    content = build_readme_content(readme.read_text(encoding="utf-8"), read_csv(results_csv), read_csv(papers_csv), results_csv.relative_to(repo_root), results_xlsx.relative_to(repo_root))
    readme.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    update_readme(args.repo_root)


if __name__ == "__main__":
    main()
