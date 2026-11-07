# Progress log

- [x] Setup, data download, EDA (01), baselines (02)
- [x] Data check vs official statistics
- [x] Leakage-safe features, models, backtest, tuning, pipeline
- [x] Tuning on 2018 (`reports/results/best_params.json`, log in `tuning_log.txt`)
- [x] Full backtest (11 runs) → `data/processed/predictions.parquet`, `reports/results/*.csv`
- [x] Conformal calibration of intervals (raw quantile intervals covered only 44 %)
- [x] End-to-end notebook `notebooks/03_end_to_end.ipynb` executed
- [x] `reports/REPORT.md`, README
- [x] 15 tests passing, lint clean

Run notes: the first overnight run stalled because the Mac slept on battery / with the lid
closed; the backtest was re-run detached under `caffeinate`. Future runs: see report §8.5.
