"""Clean-room GDN adapter for the benchmark interface.

This module implements a compact, self-contained graph-deviation network style
adapter for benchmark integration. It does not vendor code from the upstream GDN
repository. Attribution and upstream metadata are recorded in :class:`GDNMetadata`
and in ``registry/papers.csv``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.models.base import AnomalyDetector


@dataclass(frozen=True)
class GDNMetadata:
    """Verified upstream metadata for GDN."""

    paper_title: str = "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"
    paper_url: str = "https://arxiv.org/abs/2106.06947"
    repository_url: str = "https://github.com/d-ailin/GDN"
    repository_license: str = "MIT"
    upstream_commit: str = "3f809a5 (short hash shown by GitHub for current main; full SHA could not be retrieved because git ls-remote was blocked by CONNECT tunnel 403)"
    original_protocol: str = (
        "The original GDN evaluates temporal sensor anomalies using a learned "
        "sensor-dependency graph and reports metrics on datasets including SWaT "
        "and WADI. This adapter keeps the benchmark's common validation-only "
        "thresholding and point-wise metrics instead of replacing them with any "
        "model-specific evaluation code."
    )


class _GDNNet:  # populated lazily to avoid importing torch at module import time
    pass


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised when torch absent
        raise ImportError(
            "GDNDetector requires PyTorch. Install the GDN extras or run in Colab "
            "with `pip install torch`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _build_graph(X: np.ndarray, top_k: int) -> np.ndarray:
    """Learn a deterministic feature-dependency graph from training windows only."""
    if X.ndim == 3:
        flat = X.reshape(-1, X.shape[-1])
    elif X.ndim == 2:
        flat = X
    else:
        raise ValueError("GDNDetector expects 2D rows or 3D temporal windows.")
    corr = np.nan_to_num(np.corrcoef(flat, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    n_features = corr.shape[0]
    k = max(1, min(int(top_k), max(1, n_features - 1)))
    adjacency = np.zeros_like(corr, dtype=np.float32)
    for i in range(n_features):
        # Stable tie handling: lexsort by descending absolute correlation, then feature index.
        order = np.lexsort((np.arange(n_features), -np.abs(corr[i])))
        adjacency[i, order[:k]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(degree, 1.0)


class GDNDetector(AnomalyDetector):
    """Benchmark-compatible GDN-style detector with learned feature graph.

    The adapter trains a small PyTorch forecasting network over temporal windows.
    A graph learned from training-only feature correlations smooths the final
    timestep before prediction. Anomaly scores are mean squared prediction errors;
    larger scores indicate more anomalous samples.
    """

    def __init__(
        self,
        *,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        hidden_dim: int = 64,
        batch_size: int = 64,
        top_k: int = 5,
        weight_decay: float = 0.0,
        device: str = "auto",
        random_state: int = 42,
        upstream_commit: str | None = None,
    ) -> None:
        self.params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "top_k": top_k,
            "weight_decay": weight_decay,
            "device": device,
            "random_state": random_state,
            "upstream_commit": upstream_commit or GDNMetadata.upstream_commit,
        }
        self.metadata = GDNMetadata(upstream_commit=self.params["upstream_commit"])
        self.model: Any | None = None
        self.adjacency_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.training_log_: list[dict[str, float]] = []
        self.device_: str | None = None

    def _as_windows(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError("GDNDetector expects 2D rows or 3D temporal windows with shape (samples, lookback, features).")
        if len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("Input must be non-empty and finite.")
        if self.n_features_in_ is not None and X.shape[-1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[-1]}.")
        return X

    def _make_model(self, lookback: int, n_features: int, adjacency: np.ndarray):
        torch, nn, _, _ = _require_torch()

        class Net(nn.Module):
            def __init__(self, lb: int, nf: int, hidden: int, adj: np.ndarray) -> None:
                super().__init__()
                self.register_buffer("adjacency", torch.tensor(adj, dtype=torch.float32))
                self.encoder = nn.GRU(input_size=nf, hidden_size=hidden, batch_first=True)
                self.proj = nn.Sequential(nn.Linear(hidden + nf, hidden), nn.ReLU(), nn.Linear(hidden, nf))

            def forward(self, x):
                _, h = self.encoder(x)
                graph_context = x[:, -1, :] @ self.adjacency
                return self.proj(torch.cat([h[-1], graph_context], dim=1))

        return Net(lookback, n_features, int(self.params["hidden_dim"]), adjacency)

    def fit(self, X_train: np.ndarray) -> "GDNDetector":
        torch, nn, DataLoader, TensorDataset = _require_torch()
        X = self._as_windows(X_train)
        self.n_features_in_ = X.shape[-1]
        torch.manual_seed(int(self.params["random_state"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.params["random_state"]))
        requested_device = str(self.params["device"])
        self.device_ = "cuda" if requested_device == "auto" and torch.cuda.is_available() else ("cpu" if requested_device == "auto" else requested_device)
        self.adjacency_ = _build_graph(X, int(self.params["top_k"]))
        self.model = self._make_model(X.shape[1], X.shape[2], self.adjacency_).to(self.device_)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X[:, -1, :], dtype=torch.float32))
        loader = DataLoader(ds, batch_size=int(self.params["batch_size"]), shuffle=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.params["learning_rate"]), weight_decay=float(self.params["weight_decay"]))
        loss_fn = nn.MSELoss()
        self.training_log_ = []
        self.model.train()
        for epoch in range(int(self.params["epochs"])):
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device_); yb = yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
            self.training_log_.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        torch, _, DataLoader, TensorDataset = _require_torch()
        if self.model is None:
            raise ValueError("GDNDetector must be fit before scoring.")
        Xw = self._as_windows(X)
        ds = TensorDataset(torch.tensor(Xw, dtype=torch.float32), torch.tensor(Xw[:, -1, :], dtype=torch.float32))
        loader = DataLoader(ds, batch_size=int(self.params["batch_size"]), shuffle=False)
        self.model.eval(); scores = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device_); yb = yb.to(self.device_)
                err = (self.model(xb) - yb).pow(2).mean(dim=1)
                scores.append(err.detach().cpu().numpy())
        return np.concatenate(scores).astype(float)

    def save(self, path: str | Path) -> None:
        """Save a PyTorch checkpoint without pickling upstream or nested code."""
        torch, _, _, _ = _require_torch()
        if self.model is None or self.adjacency_ is None or self.n_features_in_ is None:
            raise ValueError("Cannot save an unfitted GDNDetector.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "params": self.params,
            "metadata": asdict(self.metadata),
            "adjacency": self.adjacency_,
            "n_features_in": self.n_features_in_,
            "training_log": self.training_log_,
            "device": self.device_,
            "state_dict": self.model.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "GDNDetector":
        """Load a checkpoint saved by :meth:`save`."""
        torch, _, _, _ = _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(**checkpoint["params"])
        obj.adjacency_ = checkpoint["adjacency"]
        obj.n_features_in_ = int(checkpoint["n_features_in"])
        obj.training_log_ = checkpoint.get("training_log", [])
        obj.device_ = "cpu"
        lookback = 1  # GRU parameters are independent of lookback; scoring accepts any configured window length.
        obj.model = obj._make_model(lookback, obj.n_features_in_, obj.adjacency_).to(obj.device_)
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def export_configuration(self) -> dict[str, Any]:
        return {
            "params": self.get_params(),
            "metadata": asdict(self.metadata),
            "device": self.device_,
            "learned_graph_adjacency": None if self.adjacency_ is None else self.adjacency_.tolist(),
            "training_log": list(self.training_log_),
        }
