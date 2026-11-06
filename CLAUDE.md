# Energy Demand Forecasting

Time-series forecasting project built on real public data (electricity demand + weather).

## About the user
- Experienced in data science fundamentals — skip basic explanations.
- Do briefly justify modeling choices (features, validation scheme, metrics).

## Environment
- Python 3.12 managed by **uv**. Never use system `python3` or `pip`.
- Run code: `uv run python ...`, `uv run pytest`, `uv run jupyter lab`.
- Add packages: `uv add <pkg>` (dev tools: `uv add --dev <pkg>`).
- Lint/format: `uv run ruff check . --fix && uv run ruff format .`

## Layout
- `data/raw/` — downloaded source data. **Immutable: never edit or overwrite by hand.** Not committed.
- `data/interim/`, `data/processed/` — derived data, reproducible from code. Not committed.
- `src/energy_demand_forecasting/` — reusable code (data loading, features, models, evaluation).
- `notebooks/` — exploration only; name as `NN_topic.ipynb`. Move reusable logic into `src/`.
- `reports/figures/` — saved plots. `models/` — saved model artifacts (not committed).
- `tests/` — pytest tests for `src/`.

## Forecasting conventions
- Store timestamps in UTC; convert to local time only for features/plots.
- No leakage: features at time t may only use data available at forecast time.
- Validate with time-based splits / rolling-origin backtests, never random splits.
- Always compare against baselines (naive, seasonal naive) before fancier models.
- Report MAE, RMSE and MAPE (or sMAPE) on the same held-out horizon.

## Data sources
- Document every source (URL, license, retrieval date) in README.md under "Data".
- Download scripts live in `src/`, so raw data can be re-fetched reproducibly.
- API keys go in `.env` (git-ignored), never in code or notebooks.
