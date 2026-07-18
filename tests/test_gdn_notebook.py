from __future__ import annotations

import json
from pathlib import Path


def test_gdn_colab_notebook_contains_required_workflow() -> None:
    notebook = json.loads(Path("colab/02_gdn_benchmark.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "https://github.com/Yan9564/INFORMS_FORD.git" in source
    assert "/content/drive/MyDrive/anomaly_benchmark_data" in source
    assert "/content/drive/MyDrive/anomaly_benchmark_results/gdn" in source
    assert "MODE = 'smoke'" in source and "'full'" in source
    assert "DATASET = 'swat'" in source
    assert "torch.cuda.is_available" in source
    assert "GDNDetector" in source
    assert "select_threshold(val_scores" in source
    assert "evaluate_binary" in source
    assert "checkpoint.pt" in source
    assert "scores_predictions.npz" in source
    assert "failure.log" in source
