from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from benchmark.models.gdn.adapter import _build_graph


def test_gdn_learned_graph_is_deterministic_and_normalized() -> None:
    X = np.arange(4 * 3 * 5, dtype=float).reshape(4, 3, 5)
    graph_a = _build_graph(X, top_k=2)
    graph_b = _build_graph(X, top_k=2)
    assert graph_a.shape == (5, 5)
    assert np.allclose(graph_a, graph_b)
    assert np.allclose(graph_a.sum(axis=1), np.ones(5))
    assert np.all(np.diag(graph_a) > 0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is required for GDN CPU smoke test")
def test_gdn_cpu_fit_score_save_load(tmp_path) -> None:
    from benchmark.models.gdn import GDNDetector

    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 4, 3)).astype(np.float32)
    model = GDNDetector(epochs=1, hidden_dim=8, batch_size=4, top_k=1, device="cpu", random_state=7)
    model.fit(X)
    scores = model.score_samples(X)
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()
    exported = model.export_configuration()
    assert exported["learned_graph_adjacency"] is not None
    checkpoint = tmp_path / "gdn.pt"
    model.save(checkpoint)
    loaded = GDNDetector.load(checkpoint)
    assert loaded.score_samples(X[:2]).shape == (2,)
