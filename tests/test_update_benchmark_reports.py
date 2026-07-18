from pathlib import Path

from scripts.update_benchmark_reports import (
    RESULTS_END,
    RESULTS_START,
    build_papers_table,
    build_readme_content,
    build_results_table,
)


def test_build_results_table_formats_required_columns_and_smoke_note():
    table = build_results_table([
        {
            "dataset": "swat",
            "model": "gdn",
            "run_mode": "smoke_test",
            "seed": "42",
            "precision": "0.123456",
            "recall": "0.5",
            "f1": "0.25",
            "auroc": "0.75",
            "auprc": "0.8",
            "status": "smoke_test",
        }
    ])

    assert "| Dataset | Model | Run mode | Seed | Precision | Recall | F1 | AUROC | AUPRC | Status |" in table
    assert "| SWaT | GDN | smoke_test | 42 | 0.1235 | 0.5000 | 0.2500 | 0.7500 | 0.8000 | smoke_test (execution verification; not final paper results) |" in table


def test_build_papers_table_uses_only_selected_rows_when_available():
    table = build_papers_table([
        {"title": "Unselected", "selected_for_experiment": "false"},
        {
            "title": "Selected Paper",
            "year": "2024",
            "venue": "TestConf",
            "method_family": "graph",
            "datasets": "SWaT",
            "official_code": "yes",
            "repository_url": "https://example.test/repo",
            "license": "MIT",
            "integration_status": "planned",
            "selected_for_experiment": "true",
        },
    ])

    assert "Selected Paper" in table
    assert "Unselected" not in table
    assert "Repository URL" in table


def test_build_readme_content_replaces_existing_markers_and_links_existing_xlsx(tmp_path):
    readme = f"before\n{RESULTS_START}\nold\n{RESULTS_END}\nafter\n"
    content = build_readme_content(
        readme,
        [{"dataset": "swat", "model": "gdn", "run_mode": "full", "seed": "43"}],
        [],
        Path("results/benchmark_results.csv"),
        tmp_path / "benchmark_results.xlsx",
    )

    assert "old" not in content
    assert "before" in content and "after" in content
    assert "results/benchmark_results.csv" in content
    assert "benchmark_results.xlsx" not in content
    assert "| SWaT | GDN | full | 43 |" in content
