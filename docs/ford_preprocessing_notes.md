# Ford Preprocessing Notes

The requested legacy file `legacy/data_preprocessing.py` was not present in the repository at implementation time, so these notes document the behavioural issues called out in the milestone and the corresponding safeguards in the new loader. If the legacy script is restored later, these notes should be reconciled line-by-line without modifying the legacy file unless explicitly instructed.

## Legacy behaviours to review

The legacy preprocessing was described as using quantile clipping, backward filling, temporalisation, `iloc[::10]` downsampling, concatenation of several files, possible temporal windows across file boundaries, final-timestep flattening, separation of accepted and rejected test arrays, and unused function arguments.

## Retained

- Configurable quantile clipping is supported.
- Backward fill is available as an explicit missing-value strategy.
- Temporal lookback windows are supported.
- Downsampling is available through `downsampling.every_n_rows`.
- Final-timestep and flattened-window representations are both supported.

## Changed for benchmark safety

- Clipping thresholds and scaling parameters are fit on training accepted data only.
- Validation and test data are transformed only with training-fitted preprocessing state.
- Windows are built per source file and never cross CSV file boundaries.
- File-level metadata is preserved for every output sample.
- Downsampling defaults to 1, not 10, until researchers confirm the intended protocol.
- Label mapping is configurable rather than hard-coded.
- Accepted files are split into training and validation by complete file where possible.

## Assumptions requiring confirmation

1. `feature_55 = 0` is normal and `feature_55 = 1` is anomalous.
2. The primary evaluation unit is row/window, not sweep file or manufactured component.
3. The accepted and rejected file indices for official experiments.
4. Whether every rejected CSV contains both normal and anomalous rows.
5. Whether every-tenth-row downsampling should be retained.
6. Whether flattened windows or final-timestep windows should be primary.
7. Whether accepted files may be split by complete file for validation.
8. Whether Google Drive can store the Ford data in the documented structure.
9. Whether point-wise evaluation without point adjustment is the primary IISE protocol.
