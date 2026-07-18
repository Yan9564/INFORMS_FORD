"""Ford manufacturing CSV loader with leakage-safe preprocessing."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from .base import DatasetBundle, DatasetLoader

@dataclass
class FordDatasetConfig:
    accepted_dir: str | Path | None
    rejected_dir: str | Path | None
    accepted_file_pattern: str = "Sweep_trans_{index}_accept.csv"
    rejected_file_pattern: str = "Sweep_trans_{index}_reject.csv"
    feature_columns: list[str] = field(default_factory=lambda: [f"feature_{i}" for i in range(48)])
    label_column: str = "feature_55"
    normal_label: int | str = 0
    anomaly_label: int | str = 1
    train_indices: list[int] | None = None
    test_indices: list[int] | None = None
    validation_fraction: float = 0.2
    split_by_file: bool = True
    random_seed: int = 42
    missing_strategy: str = "median"
    clipping_enabled: bool = True
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    scaling_method: str = "standard"
    lookback: int = 10
    stride: int = 1
    representation: str = "full_window_flattened"
    every_n_rows: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FordDatasetConfig":
        fc = data.get("feature_columns", {})
        features = fc if isinstance(fc, list) else [f"{fc.get('prefix','feature_')}{i}" for i in range(int(fc.get('start',0)), int(fc.get('end',47))+1)]
        return cls(
            accepted_dir=data.get("accepted_dir"), rejected_dir=data.get("rejected_dir"),
            accepted_file_pattern=data.get("accepted_file_pattern", cls.accepted_file_pattern),
            rejected_file_pattern=data.get("rejected_file_pattern", cls.rejected_file_pattern),
            feature_columns=features, label_column=data.get("label_column", "feature_55"),
            normal_label=data.get("normal_label", 0), anomaly_label=data.get("anomaly_label", 1),
            train_indices=data.get("train_indices"), test_indices=data.get("test_indices"),
            validation_fraction=float(data.get("validation_fraction", 0.2)), split_by_file=bool(data.get("split_by_file", True)),
            random_seed=int(data.get("random_seed", 42)), missing_strategy=data.get("missing_values", {}).get("strategy", "median"),
            clipping_enabled=bool(data.get("clipping", {}).get("enabled", True)),
            lower_quantile=float(data.get("clipping", {}).get("lower_quantile", 0.001)), upper_quantile=float(data.get("clipping", {}).get("upper_quantile", 0.999)),
            scaling_method=data.get("scaling", {}).get("method", "standard"), lookback=int(data.get("windowing", {}).get("lookback", 10)),
            stride=int(data.get("windowing", {}).get("stride", 1)), representation=data.get("windowing", {}).get("representation", "full_window_flattened"),
            every_n_rows=int(data.get("downsampling", {}).get("every_n_rows", 1)),
        )

class FordDatasetLoader(DatasetLoader):
    def __init__(self, config: FordDatasetConfig): self.config = config

    def discover_files(self, kind: str) -> list[tuple[int, Path]]:
        cfg = self.config; directory = Path(cfg.accepted_dir if kind == "accept" else cfg.rejected_dir) if (cfg.accepted_dir if kind == "accept" else cfg.rejected_dir) else None
        if directory is None or not directory.is_dir(): raise FileNotFoundError(f"Missing Ford {kind} directory: {directory}")
        pattern = cfg.accepted_file_pattern if kind == "accept" else cfg.rejected_file_pattern
        indices = cfg.train_indices if kind == "accept" else cfg.test_indices
        if indices is not None:
            files = [(int(i), directory / pattern.format(index=i)) for i in indices]
            missing = [str(p) for _, p in files if not p.is_file()]
            if missing: raise FileNotFoundError(f"Missing selected Ford files: {missing}")
            return files
        rx = re.compile(re.escape(pattern).replace(re.escape("{index}"), r"(\d+)"))
        files = []
        for p in directory.glob("*.csv"):
            m = rx.fullmatch(p.name)
            if m: files.append((int(m.group(1)), p))
        if not files: raise FileNotFoundError(f"No Ford {kind} CSV files matching {pattern} in {directory}")
        return sorted(files)

    def _read_file(self, item: tuple[int, Path], labelled: bool) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame]:
        idx, path = item; df = pd.read_csv(path)
        missing = [c for c in self.config.feature_columns if c not in df.columns]
        if missing: raise ValueError(f"Missing feature columns in {path.name}: {missing}")
        X = df[self.config.feature_columns].apply(pd.to_numeric, errors="coerce")
        y = None
        if labelled:
            if self.config.label_column not in df.columns: raise ValueError(f"Missing label column {self.config.label_column} in {path.name}")
            raw = df[self.config.label_column]
            bad = set(raw.dropna().unique()) - {self.config.normal_label, self.config.anomaly_label}
            if bad: raise ValueError(f"Invalid labels in {path.name}: {bad}")
            y = raw.map({self.config.normal_label: 0, self.config.anomaly_label: 1}).astype(int).to_numpy()
        if self.config.every_n_rows < 1: raise ValueError("downsampling.every_n_rows must be >= 1")
        X = X.iloc[::self.config.every_n_rows].reset_index(drop=True)
        if y is not None: y = y[::self.config.every_n_rows]
        meta = pd.DataFrame({"source_filename": path.name, "source_file_index": idx, "row_number": np.arange(len(X))*self.config.every_n_rows, "source_file_rows": len(df)})
        return X, y, meta

    def _handle_missing(self, frames: list[pd.DataFrame], med: pd.Series | None = None) -> tuple[list[pd.DataFrame], pd.Series | None]:
        s = self.config.missing_strategy
        if s == "error":
            for f in frames:
                if f.isna().any().any(): raise ValueError("NaN values found and missing strategy is error")
            return frames, med
        if s == "forward_fill": return [f.ffill().bfill() for f in frames], med
        if s == "backward_fill": return [f.bfill().ffill() for f in frames], med
        if s == "median":
            med = med if med is not None else pd.concat(frames).median(numeric_only=True)
            return [f.fillna(med) for f in frames], med
        raise ValueError(f"Unsupported missing-value strategy: {s}")

    def _window(self, X: np.ndarray, y: np.ndarray | None, meta: pd.DataFrame):
        lb, stride, rep = self.config.lookback, self.config.stride, self.config.representation
        if rep == "row": lb = 1
        if lb < 1 or stride < 1: raise ValueError("lookback and stride must be >= 1")
        Xs=[]; ys=[]; ms=[]
        for start in range(0, len(X)-lb+1, stride):
            win = X[start:start+lb]
            Xs.append(win[-1] if rep in {"row","last_timestep"} else win.reshape(-1) if rep=="full_window_flattened" else win)
            if y is not None: ys.append(int(y[start+lb-1]))
            ms.append(meta.iloc[start+lb-1].to_dict() | {"window_start_row": int(meta.iloc[start]["row_number"]), "window_end_row": int(meta.iloc[start+lb-1]["row_number"])})
        if not Xs: raise ValueError("Empty dataset after windowing; reduce lookback or check file lengths")
        return np.asarray(Xs, float), (np.asarray(ys, int) if y is not None else None), pd.DataFrame(ms)

    def load(self) -> DatasetBundle:
        cfg = self.config
        acc = [self._read_file(x, False) for x in self.discover_files("accept")]
        rej = [self._read_file(x, True) for x in self.discover_files("reject")]
        rng = np.random.default_rng(cfg.random_seed); order = np.arange(len(acc)); rng.shuffle(order)
        n_val = max(1, int(round(len(acc)*cfg.validation_fraction))) if len(acc) > 1 else 1
        val_ids = set(order[:n_val]); train_ids = [i for i in range(len(acc)) if i not in val_ids] or [int(order[-1])]
        val_ids = [i for i in range(len(acc)) if i in val_ids and i not in train_ids]
        train_frames=[acc[i][0] for i in train_ids]; val_frames=[acc[i][0] for i in val_ids] or [acc[train_ids[0]][0].iloc[:0].copy()]
        train_frames, med = self._handle_missing(train_frames); val_frames, _ = self._handle_missing(val_frames, med); rej_frames, _ = self._handle_missing([r[0] for r in rej], med)
        train_concat = pd.concat(train_frames)
        lower = train_concat.quantile(cfg.lower_quantile) if cfg.clipping_enabled else None; upper = train_concat.quantile(cfg.upper_quantile) if cfg.clipping_enabled else None
        def clip(frames): return [f.clip(lower=lower, upper=upper, axis=1) if cfg.clipping_enabled else f for f in frames]
        train_frames, val_frames, rej_frames = clip(train_frames), clip(val_frames), clip(rej_frames)
        scaler = None
        if cfg.scaling_method == "standard": scaler = StandardScaler().fit(pd.concat(train_frames))
        elif cfg.scaling_method == "robust": scaler = RobustScaler().fit(pd.concat(train_frames))
        elif cfg.scaling_method not in {"none", None}: raise ValueError(f"Unsupported scaling method: {cfg.scaling_method}")
        def transform(frames): return [pd.DataFrame(scaler.transform(f), columns=f.columns) if scaler else f for f in frames]
        train_frames, val_frames, rej_frames = transform(train_frames), transform(val_frames), transform(rej_frames)
        train_parts=[self._window(f.to_numpy(), None, acc[i][2]) for f,i in zip(train_frames, train_ids)]
        val_parts=[self._window(f.to_numpy(), np.zeros(len(f), int), acc[i][2]) for f,i in zip(val_frames, val_ids)] if val_ids else []
        test_parts=[self._window(f.to_numpy(), r[1], r[2]) for f,r in zip(rej_frames, rej)]
        def cat(parts, label):
            X=np.concatenate([p[0] for p in parts]); y=np.concatenate([p[1] for p in parts]) if parts[0][1] is not None else None; m=pd.concat([p[2] for p in parts], ignore_index=True)
            if len(X)==0 or not np.isfinite(X).all(): raise ValueError(f"Invalid {label} array: empty or non-finite")
            return X,y,m
        Xtr,_,mtr=cat(train_parts,"train"); Xv,yv,mv=cat(val_parts,"validation") if val_parts else (Xtr[:1], np.zeros(1,int), mtr.iloc[:1].copy()); Xt,yt,mt=cat(test_parts,"test")
        if set(np.unique(yv))- {0,1} or set(np.unique(yt))- {0,1}: raise ValueError("Labels must be binary after mapping")
        return DatasetBundle(Xtr,Xv,yv,Xt,yt,mtr,mv,mt,{"median": None if med is None else med.to_dict(), "clip_lower": None if lower is None else lower.to_dict(), "clip_upper": None if upper is None else upper.to_dict(), "scaler": scaler, "train_files":[acc[i][2].source_filename.iloc[0] for i in train_ids]}, cfg.feature_columns, [])
