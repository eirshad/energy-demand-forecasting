"""Naive baselines and the TSO benchmark.

Main forecast timing (see forecast_features): issued 09:00 UTC on D-1, so load lags must be
>= 40 h. `naive_day` (lag 24) is only valid for the simpler midnight-issue setup used in
notebook 02 and is kept for reference; the main backtest uses `naive_2day` instead.
Assumes a gap-free hourly index.
"""

from __future__ import annotations

import pandas as pd


def naive_day(y: pd.Series) -> pd.Series:
    """Same hour yesterday (midnight-issue setup only)."""
    return y.shift(24).rename("naive_day")


def naive_2day(y: pd.Series) -> pd.Series:
    """Same hour two days ago (latest same-hour value known at 09:00 UTC on D-1)."""
    return y.shift(48).rename("naive_2day")


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


def tso_bias_corrected(y: pd.Series, tso: pd.Series, days: int = 28) -> pd.Series:
    """TSO forecast plus its trailing mean error, using only errors known at issue time.

    The ENTSO-E 'actual load' series and the TSO forecast drift apart in level (data-coverage
    changes, see the report), so a level-adjusted TSO forecast is the fairer benchmark.
    Error at target t uses errors up to t-48 h (>= the 40 h availability limit).
    """
    err = (y - tso).shift(48)
    return (tso + err.rolling(days * 24, min_periods=24 * 7).mean()).rename(
        "tso_bias_corrected"
    )
