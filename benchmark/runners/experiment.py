"""Experiment runner for reproducible benchmark runs."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, platform, random, subprocess, sys, time, uuid
from importlib.metadata import version, PackageNotFoundError
from typing import Any
import numpy as np, yaml
from benchmark.datasets.ford import FordDatasetConfig, FordDatasetLoader
from benchmark.metrics.classification import evaluate_binary
from benchmark.metrics.thresholding import select_threshold
from benchmark.models.isolation_forest import IsolationForestDetector

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f: return yaml.safe_load(f)

def _git_hash() -> str | None:
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return None

def _versions() -> dict[str, str | None]:
    out={}
    for p in ["numpy","pandas","scikit-learn","PyYAML","joblib"]:
        try: out[p]=version(p)
        except PackageNotFoundError: out[p]=None
    return out

def run_experiment(config_path: str | Path, output_dir: str | Path = "results/raw", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg=load_yaml(config_path); overrides=overrides or {}
    dcfg=cfg["dataset"].copy(); dcfg.update({k:v for k,v in overrides.items() if k in {"accepted_dir","rejected_dir"} and v is not None})
    seed=int(cfg.get("random_seed", dcfg.get("random_seed", 42))); random.seed(seed); np.random.seed(seed)
    t0=time.perf_counter(); bundle=FordDatasetLoader(FordDatasetConfig.from_dict(dcfg)).load()
    model_cfg=cfg.get("model",{}); model=IsolationForestDetector(**model_cfg.get("parameters",{}))
    t_train=time.perf_counter(); model.fit(bundle.X_train); train_runtime=time.perf_counter()-t_train
    t_score=time.perf_counter(); val_scores=model.score_samples(bundle.X_validation); test_scores=model.score_samples(bundle.X_test); scoring_runtime=time.perf_counter()-t_score
    th_cfg=cfg.get("threshold", {"method":"percentile"}); th=select_threshold(val_scores, bundle.y_validation, **th_cfg)
    metrics=evaluate_binary(bundle.y_test, test_scores, th.threshold)
    run_id=cfg.get("run_id") or f"{cfg['dataset']['name']}_{model_cfg.get('name','isolation_forest')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    result={"schema_version":"1.0","run_id":run_id,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"git_commit_hash":_git_hash(),"dataset":dcfg.get("name","ford"),"dataset_configuration":dcfg,"model":model_cfg.get("name","isolation_forest"),"model_configuration":model_cfg,"experiment_configuration":cfg,"random_seed":seed,"feature_names":bundle.feature_names,"number_of_input_features":int(bundle.X_train.shape[-1] if bundle.X_train.ndim==2 else bundle.X_train.shape[-1]),"input_representation":dcfg.get("windowing",{}).get("representation"),"lookback":dcfg.get("windowing",{}).get("lookback"),"training_sample_count":len(bundle.X_train),"validation_sample_count":len(bundle.X_validation),"test_sample_count":len(bundle.X_test),"validation_normal_count":int((bundle.y_validation==0).sum()),"validation_anomaly_count":int((bundle.y_validation==1).sum()),"test_normal_count":int((bundle.y_test==0).sum()),"test_anomaly_count":int((bundle.y_test==1).sum()),"threshold_method":th.method,"threshold_value":th.threshold,**metrics,"training_runtime_seconds":train_runtime,"scoring_runtime_seconds":scoring_runtime,"total_runtime_seconds":time.perf_counter()-t0,"python_version":sys.version,"package_versions":_versions(),"platform":platform.platform(),"source_data_filenames":sorted(set(bundle.train_metadata.source_filename)|set(bundle.validation_metadata.source_filename)|set(bundle.test_metadata.source_filename)),"warnings":bundle.warnings+metrics.get("warnings",[]),"status":"success"}
    out=Path(output_dir)/result["dataset"]/result["model"]/f"{run_id}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): raise FileExistsError(f"Raw result already exists and will not be overwritten: {out}")
    out.write_text(json.dumps(result, indent=2, default=str))
    if cfg.get("save_model", {}).get("enabled", False): model.save(Path(cfg["save_model"].get("dir","models"))/f"{run_id}.joblib")
    return result
