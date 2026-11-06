"""Naive baselines for day-ahead forecasting.

Forecast setup used throughout the project: forecasts for local/UTC day D are issued at
00:00 UTC on D, so only values from D-1 or earlier are allowed. Every baseline below uses
lags >= 24 h, which satisfies that for all 24 target hours. Assumes a gap-free hourly index.
"""

from __future__ import annotations

import pandas as pd


def naive_day(y: pd.Series) -> pd.Series:
    """Same hour yesterday."""
    return y.shift(24).rename("naive_day")


def seasonal_naive_week(y: pd.Series) -> pd.Series:
    """Same hour, same weekday, last week."""
    return y.shift(168).rename("snaive_week")


def weekday_mean(y: pd.Series, weeks: int = 4) -> pd.Series:
    """Mean of the same hour and weekday over the last `weeks` weeks."""
    lags = [y.shift(168 * k) for k in range(1, weeks + 1)]
    return (
        pd.concat(lags, axis=1)
        .mean(axis=1, skipna=False)
        .rename(f"weekday_mean_{weeks}w")
    )
