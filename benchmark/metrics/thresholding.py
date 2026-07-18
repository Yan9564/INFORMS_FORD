"""Validation-only threshold selection utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score

@dataclass(frozen=True)
class ThresholdResult:
    method: str
    threshold: float

def select_threshold(scores, y_validation=None, method="percentile", *, fixed_threshold=None, percentile=95.0, contamination=0.05) -> ThresholdResult:
    s=np.asarray(scores,float)
    if len(s)==0 or not np.isfinite(s).all(): raise ValueError("Validation scores must be non-empty and finite")
    if method=="fixed":
        if fixed_threshold is None: raise ValueError("fixed_threshold is required")
        return ThresholdResult(method, float(fixed_threshold))
    if method=="percentile": return ThresholdResult(method, float(np.percentile(s, percentile)))
    if method=="contamination": return ThresholdResult(method, float(np.percentile(s, 100*(1-contamination))))
    if method=="best_f1":
        if y_validation is None: raise ValueError("Validation labels are required for best_f1")
        y=np.asarray(y_validation,int); best=(-1.0, float("inf"))
        for t in np.unique(s):
            f=float(f1_score(y, (s>=t).astype(int), zero_division=0))
            if f > best[0] or (f == best[0] and t < best[1]): best=(f,float(t))
        return ThresholdResult(method, best[1])
    raise ValueError(f"Unsupported threshold method: {method}")
