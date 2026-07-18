"""Dataset interfaces and shared result containers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class DatasetBundle:
    """Preprocessed arrays, labels, metadata, and fitted preprocessing state."""
    X_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_metadata: pd.DataFrame
    validation_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    preprocessing_state: dict[str, Any]
    feature_names: list[str]
    warnings: list[str] = field(default_factory=list)

class DatasetLoader(ABC):
    """Abstract dataset loader interface."""
    @abstractmethod
    def load(self) -> DatasetBundle:
        """Load, split, fit preprocessing on training data, and transform all splits."""
