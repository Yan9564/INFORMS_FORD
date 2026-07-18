from __future__ import annotations

import json
from pathlib import Path


def test_ford_smoke_test_notebook_contains_required_workflow() -> None:
    notebook = json.loads(Path("colab/01_ford_smoke_test.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "/content/drive/MyDrive/anomaly_benchmark_data" in source
    assert "/content/drive/MyDrive/anomaly_benchmark_results/ford_smoke_test" in source
    assert "FordDatasetLoader" in source
    assert "IsolationForestDetector" in source
    assert "evaluate_binary" in source
    assert "select_threshold(validation_scores" in source
    assert "precision" in source and "recall" in source and "f1" in source
    assert "google.colab" in source
