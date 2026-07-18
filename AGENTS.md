# Mandatory Rules for Future Codex Tasks

1. Never commit datasets, credentials, tokens, checkpoints, or sensitive paths.
2. Never modify legacy code unless explicitly instructed.
3. Preserve common preprocessing, thresholding, and evaluation protocols.
4. Do not let individual model adapters silently replace common evaluation logic.
5. Never use test labels for preprocessing, hyperparameter tuning, model selection, or threshold selection.
6. Add tests for every new dataset adapter and model adapter.
7. Record external-model paper title, year, venue, repository URL, official/unofficial status, licence, upstream commit hash, required environment, integration status, and deviations from the paper.
8. Never claim successful reproduction unless installation succeeds, a smoke test succeeds, a complete experiment succeeds, and metrics are written to an immutable raw result.
9. Keep raw results immutable.
10. Generate all summary tables from raw results.
11. Do not fabricate metrics, repository URLs, citations, paper metadata, or successful run status.
12. Preserve dataset file boundaries.
13. Prefer configuration over hard-coded values.
14. Keep model-specific dependencies isolated when external models are added later.
15. Run the complete test suite before creating a pull request.
