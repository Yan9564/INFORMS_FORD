# Industrial Anomaly Detection Benchmark

Initial reproducible benchmark scaffold for an acadeimc paper. The current milestone implements the Ford manufacturing dataset pipeline and an Isolation Forest baseline with common preprocessing, thresholding, evaluation, immutable raw results, and generated summary tables.

## Scope

Implemented now: Ford CSV loader, leakage-safe preprocessing, Isolation Forest, validation-only thresholding, point-wise/sample-wise metrics, raw JSON results, CSV/Markdown reporting, CLI scripts, Colab notebook scaffold, and synthetic tests. Planned datasets: SWaT, PaySim, and Arrhythmia. Planned model families: forecasting, graph-based, diffusion-based, physics-informed, and knowledge-enhanced anomaly detection.

## Repository structure

`benchmark/` contains reusable dataset, model, metric, runner, and reporting modules. `configs/` stores path-free YAML defaults. `scripts/` provides command-line entry points. `registry/` tracks datasets, models, and paper metadata. `results/` stores ignored generated outputs. `docs/` records preprocessing notes. `tests/` uses synthetic Ford-style CSVs only.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
```

## Ford dataset layout

Real data must not be committed. Use external paths or environment-specific CLI overrides:

```text
ford/Train/accept/Sweep_trans_0_accept.csv
ford/Train/reject/Sweep_trans_0_reject.csv
```

Default features are `feature_0` through `feature_47`; default label column is `feature_55` with provisional mapping 0=normal and 1=anomalous.

## Google Drive and Colab

Recommended data structure:

```text
MyDrive/industrial-anomaly-benchmark-data/ford/Train/accept/
MyDrive/industrial-anomaly-benchmark-data/ford/Train/reject/
MyDrive/industrial-anomaly-benchmark-data/swat/
MyDrive/industrial-anomaly-benchmark-data/paysim/
MyDrive/industrial-anomaly-benchmark-data/arrhythmia/
```

Recommended output structure:

```text
MyDrive/industrial-anomaly-benchmark-results/raw/
MyDrive/industrial-anomaly-benchmark-results/aggregated/
MyDrive/industrial-anomaly-benchmark-results/models/
MyDrive/industrial-anomaly-benchmark-results/logs/
```

Open `colab/run_benchmark.ipynb`, set the repository URL, branch, dataset root, and result root, then run the thin notebook cells.

## Commands

Validate Ford data:

```bash
python scripts/validate_dataset.py --config configs/datasets/ford.yaml --accepted-dir /path/to/ford/Train/accept --rejected-dir /path/to/ford/Train/reject
```

Run the baseline:

```bash
python scripts/run_experiment.py --config configs/experiments/ford_isolation_forest.yaml --accepted-dir /path/to/ford/Train/accept --rejected-dir /path/to/ford/Train/reject --output-dir results/raw
```

Build tables:

```bash
python scripts/build_results_table.py --results-dir results/raw --output-dir results/aggregated
```

## Reproducibility and leakage prevention

The Ford loader splits accepted normal data into train and validation by source file where possible, fits missing-value medians, clipping bounds, and scalers on training data only, transforms validation/test data with that fitted state, preserves file boundaries, and never uses test labels for preprocessing, tuning, model selection, or threshold selection. Raw results are immutable and tables are regenerated from raw JSON files.

## Extending

Add datasets by implementing `DatasetLoader`, adding path-free YAML, registry metadata, and synthetic tests. Add models by implementing `AnomalyDetector`; scores must increase with anomalousness and model adapters must not bypass common thresholding or metrics. External SOTA repositories should be isolated under ignored locations until license, commit, environment, smoke-test, and deviation metadata are recorded in the registry.

## Current limitations and assumptions

No external SOTA repositories or deep-learning models are integrated yet. Metrics are point-wise/sample-wise and do not apply point adjustment. Researchers must confirm the Ford label mapping, evaluation unit, official file indices, rejected-file label composition, downsampling, temporal representation, file-level validation split, Google Drive layout, and point-wise protocol.

<!-- BENCHMARK-PAPERS:START -->
## Selected papers

Generated from [`registry/papers.csv`](registry/papers.csv).

| Title | Year | Venue | Method family | Datasets | Official code | Repository URL | License | Integration status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- BENCHMARK-PAPERS:END -->

<!-- BENCHMARK-RESULTS:START -->
## Benchmark results

Smoke-test rows, when present, are execution verification only and are not final paper results.

Source CSV: [`results/benchmark_results.csv`](results/benchmark_results.csv) | XLSX export: [`results/benchmark_results.xlsx`](results/benchmark_results.xlsx)

| Dataset | Model | Run mode | Seed | Precision | Recall | F1 | AUROC | AUPRC | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWaT | GDN | full | 42 | 0.1806 | 0.9036 | 0.3010 | 0.8765 | 0.7626 | full_experiment |
| SWaT | GDN | full | 43 | 0.2597 | 0.8276 | 0.3954 | 0.8721 | 0.7631 | full_experiment |
| SWaT | GDN | full | 44 | 0.1824 | 0.9024 | 0.3035 | 0.8773 | 0.7622 | full_experiment |
<!-- BENCHMARK-RESULTS:END -->
