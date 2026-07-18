"""Isolation Forest baseline adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from .base import AnomalyDetector

class IsolationForestDetector(AnomalyDetector):
    def __init__(self, **params: Any):
        self.params = {"n_estimators": 100, "contamination": "auto", "random_state": 42, **params}
        self.model = IsolationForest(**self.params)
        self.n_features_in_: int | None = None
    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2: raise ValueError("IsolationForestDetector requires 2D tabular input; flatten sequence windows first.")
        if len(X) == 0 or not np.isfinite(X).all(): raise ValueError("Input must be non-empty and finite")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_: raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X
    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        X = self._validate(X_train); self.model.fit(X); self.n_features_in_ = X.shape[1]; return self
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(self._validate(X))
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)
    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        return joblib.load(path)
    def get_params(self) -> dict[str, Any]: return dict(self.params)
