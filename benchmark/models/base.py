"""Anomaly detector interface. Scores must increase with anomalousness."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np

class AnomalyDetector(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "AnomalyDetector": ...
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score_samples(X) >= threshold).astype(int)
    @abstractmethod
    def save(self, path: str | Path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "AnomalyDetector": ...
    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...
