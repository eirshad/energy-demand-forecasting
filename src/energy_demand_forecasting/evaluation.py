"""Forecast metrics and period-based evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed evaluation windows (UTC). 2019 is a "normal" year; 2020 includes the COVID-19 shock.
PERIODS = {
    "test_2019": ("2019-01-01", "2019-12-31"),
    "stress_2020": ("2020-01-01", "2020-09-30"),
}


def metrics(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    mask = y.notna() & yhat.notna()
    e = (y[mask] - yhat[mask]).to_numpy()
    yt = y[mask].to_numpy()
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAPE_%": float(np.mean(np.abs(e) / np.abs(yt)) * 100),
        "n": int(mask.sum()),
    }


def evaluate(
    y: pd.Series, preds: pd.DataFrame, periods: dict[str, tuple[str, str]] = PERIODS
) -> pd.DataFrame:
    """Metrics for every prediction column in every period -> tidy table."""
    rows = []
    for period, (start, end) in periods.items():
        ys = y.loc[start:end]
        for col in preds.columns:
            m = metrics(ys, preds[col].loc[start:end])
            rows.append({"period": period, "model": col, **m})
    return pd.DataFrame(rows).set_index(["period", "model"]).round(2)
