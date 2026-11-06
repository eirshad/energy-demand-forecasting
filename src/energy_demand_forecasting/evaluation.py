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


def daily_peak_mae(y: pd.Series, yhat: pd.Series) -> float:
    """MAE of the daily maximum (what matters for grid capacity planning)."""
    mask = y.notna() & yhat.notna()
    days = y[mask].index.floor("D")
    peak_y = y[mask].groupby(days).max()
    peak_f = yhat[mask].groupby(days).max()
    return float((peak_y - peak_f).abs().mean())


def diebold_mariano(
    y: pd.Series, f1: pd.Series, f2: pd.Series, max_lag: int = 7
) -> dict[str, float]:
    """Diebold-Mariano test of equal MAE, on daily-mean absolute errors.

    Hourly errors are strongly autocorrelated, so we aggregate to days and use a Newey-West
    variance (Bartlett kernel, `max_lag` days), with the Harvey-Leybourne-Newbold small-sample
    correction. Negative statistic => f1 is more accurate than f2.
    """
    from scipy import stats

    mask = y.notna() & f1.notna() & f2.notna()
    d_hourly = (y - f1).abs()[mask] - (y - f2).abs()[mask]
    d = d_hourly.groupby(d_hourly.index.floor("D")).mean().to_numpy()
    n, dbar = len(d), d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(max_lag + 1)]
    var = gamma[0] + 2 * sum(
        (1 - k / (max_lag + 1)) * gamma[k] for k in range(1, max_lag + 1)
    )
    dm = dbar / np.sqrt(var / n)
    h = 1  # one-step (daily) horizon for the HLN correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {
        "mean_diff_MW": float(dbar),
        "DM_stat": float(stat),
        "p_value": float(p),
        "days": n,
    }


def pinball_loss(y: pd.Series, q_pred: pd.Series, q: float) -> float:
    mask = y.notna() & q_pred.notna()
    e = (y[mask] - q_pred[mask]).to_numpy()
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def interval_metrics(y: pd.Series, lo: pd.Series, hi: pd.Series) -> dict[str, float]:
    mask = y.notna() & lo.notna() & hi.notna()
    inside = (y[mask] >= lo[mask]) & (y[mask] <= hi[mask])
    return {
        "coverage_%": float(inside.mean() * 100),
        "mean_width_MW": float((hi - lo)[mask].mean()),
    }
