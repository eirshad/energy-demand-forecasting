"""Rolling-origin (expanding-window) backtest with monthly refits.

For each test month M the model is trained on every hour up to 2 days before M starts (all of
which is known at the first issue time of M, 09:00 UTC on the last day before M), then predicts
all hours of M. Forecast features for each target day are already restricted to information
available at that day's issue time (see forecast_features), so the month-long horizon of a
single fitted model does not leak: it only means the model parameters are up to ~1 month old.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

TRAIN_START = (
    "2015-02-01"  # first ~4 weeks are consumed by the 28-day level / 2-week lags
)
TRAIN_GAP = pd.Timedelta(days=2)


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS", tz="UTC"))


def run_backtest(
    make_model: Callable[[], object],
    X_train: pd.DataFrame,
    y: pd.Series,
    months: list[pd.Timestamp],
    X_test: pd.DataFrame | None = None,
    train_start: str = TRAIN_START,
    return_models: bool = False,
):
    """Return out-of-sample predictions over `months` (and optionally the fitted models).

    X_train is used for fitting; X_test (default: X_train) for predicting. They differ when we
    train on actual weather but test with simulated weather-forecast errors.
    """
    X_test = X_train if X_test is None else X_test
    train_start = pd.Timestamp(train_start, tz="UTC")
    preds, models = [], {}
    for m in months:
        m_end = m + pd.offsets.MonthBegin(1)
        tr = X_train.loc[train_start : m - TRAIN_GAP - pd.Timedelta(hours=1)]
        y_tr = y.loc[tr.index]
        ok = tr.notna().all(axis=1) & y_tr.notna()
        model = make_model().fit(tr[ok], y_tr[ok])
        te = X_test.loc[m : m_end - pd.Timedelta(hours=1)]
        preds.append(pd.Series(model.predict(te), index=te.index))
        if return_models:
            models[m] = model
    out = pd.concat(preds)
    return (out, models) if return_models else out
