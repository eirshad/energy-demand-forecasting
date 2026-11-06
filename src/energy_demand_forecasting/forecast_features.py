"""Leakage-safe feature matrix for day-ahead forecasting.

Forecast timing (matches the TSOs' day-ahead process):
    * Forecasts for UTC day D (hours 00..23) are issued at 09:00 UTC on D-1.
    * The latest load value known at issue time is the hour starting 07:00 UTC on D-1
      (it completes at 08:00; we allow one hour of data latency).
    * So for target hour h of day D, load lag k is allowed only if k >= h + 17.
      The worst case (h = 23) needs k >= 40, so every load lag used here is >= 48 h, and
      every "recent state" feature is read at the issue-time origin, D 00:00 - 17 h.

Weather features use reanalysis *actuals* at the target time. That stands in for a weather
forecast; the "noisy" variant adds forecast-like errors to measure how much that matters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from energy_demand_forecasting.data import CITIES, WEATHER_VARS
from energy_demand_forecasting.features import TARGET, calendar_features

ORIGIN_OFFSET = pd.Timedelta(
    hours=17
)  # last known load hour relative to target-day midnight
MIN_LOAD_LAG_H = 40

LOAD_LAGS = [48, 72, 168, 336]
WeatherMode = Literal["none", "actual", "noisy"]

# Load-valued columns (MW). The normalised models divide these by the 28-day level.
LOAD_COLS = [f"lag{k}" for k in LOAD_LAGS] + [
    "lag168_clean",
    "last_load",
    "mean_last24",
    "mean_last168",
]
CALENDAR_COLS = [
    "hour", "dow", "month", "doy_sin", "doy_cos", "is_weekend", "holiday_share",
    "is_xmas_period", "bridge_share", "days_to_hol", "days_since_hol",
    "hol_lag48", "hol_lag168",
]  # fmt: skip
LEVEL_COL = "level28"


def _abnormal(cal: pd.DataFrame) -> pd.Series:
    """Hours on (mostly) holiday or Christmas-period days: bad references for lag features."""
    return (cal.holiday_share >= 0.5) | (cal.is_xmas_period == 1)


def _day_level_holiday_features(cal: pd.DataFrame) -> pd.DataFrame:
    """Bridge days and distance to/from the nearest major holiday, on local calendar days."""
    local_day = cal.index.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    days = (
        pd.DataFrame(
            {"hol": cal.holiday_share.to_numpy(), "dow": cal.dow.to_numpy()},
            index=local_day,
        )
        .groupby(level=0)
        .first()
    )
    days = days.asfreq("D")  # ensure contiguous days (DST days are still single days)
    hol_prev = days.hol.shift(1).fillna(0)
    hol_next = days.hol.shift(-1).fillna(0)
    workday = (days.dow < 5) & (days.hol < 0.5)
    # Mon before a Tue holiday, Fri after a Thu holiday: weighted by how many people get the bridge.
    bridge = np.where(workday & (days.dow == 0), hol_next, 0.0) + np.where(
        workday & (days.dow == 4), hol_prev, 0.0
    )
    major = days.hol >= 0.5
    idx = np.arange(len(days))
    last_major = pd.Series(np.where(major, idx, np.nan), index=days.index).ffill()
    next_major = pd.Series(np.where(major, idx, np.nan), index=days.index).bfill()
    out = pd.DataFrame(
        {
            "bridge_share": bridge,
            "days_since_hol": np.clip(idx - last_major.to_numpy(), 0, 10),
            "days_to_hol": np.clip(next_major.to_numpy() - idx, 0, 10),
        },
        index=days.index,
    ).fillna(10)
    return out.reindex(local_day).set_axis(cal.index)


def add_forecast_noise(weather: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Simulate day-ahead weather-forecast errors with smooth AR(1) noise.

    Error sizes are typical for 15–38 h ahead forecasts: temperature σ≈1.3 °C (shared across
    cities, plus 0.5 °C per city), radiation ±25 % (multiplicative), cloud cover σ≈15 pts,
    wind σ≈3 km/h, humidity σ≈6 pts. Errors are autocorrelated (φ=0.97 per hour), as real
    forecast errors are.
    """
    rng = np.random.default_rng(seed)
    n, phi = len(weather), 0.97

    def ar1(sigma: float) -> np.ndarray:
        e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
        x = np.empty(n)
        x[0] = rng.normal(0, sigma)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    w = weather.copy()
    common_t = ar1(1.3)
    w["w_temperature_2m"] += common_t
    for city in CITIES:
        w[f"temp_{city}"] += common_t + ar1(0.5)
    w["w_shortwave_radiation"] = (w["w_shortwave_radiation"] * np.exp(ar1(0.25))).clip(
        lower=0
    )
    w["w_cloud_cover"] = (w["w_cloud_cover"] + ar1(15)).clip(0, 100)
    w["w_wind_speed_10m"] = (w["w_wind_speed_10m"] + ar1(3)).clip(lower=0)
    w["w_relative_humidity_2m"] = (w["w_relative_humidity_2m"] + ar1(6)).clip(0, 100)
    return w


def _weather_features(w: pd.DataFrame) -> pd.DataFrame:
    t = w["w_temperature_2m"]
    f = w[[f"w_{v}" for v in WEATHER_VARS] + [f"temp_{c}" for c in CITIES]].copy()
    f["hdd"] = (15.5 - t).clip(lower=0)  # heating degree-hours
    f["cdd"] = (t - 22.0).clip(lower=0)  # cooling degree-hours
    f["temp_ewm24"] = t.ewm(span=24).mean()  # thermal inertia of buildings
    f["temp_ewm72"] = t.ewm(span=72).mean()
    f["temp_daymean"] = t.groupby(t.index.floor("D")).transform("mean")
    f["temp_lag24"] = t.shift(24)
    return f


def weather_columns() -> list[str]:
    dummy = pd.DataFrame(
        0.0,
        index=pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"),
        columns=[f"w_{v}" for v in WEATHER_VARS] + [f"temp_{c}" for c in CITIES],
    )
    return list(_weather_features(dummy).columns)


def build_features(
    df: pd.DataFrame, weather: WeatherMode = "actual", seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X includes LEVEL_COL (needed by the normalised models)."""
    y = df[TARGET].astype(float)
    cal = calendar_features(df.index)
    X = pd.DataFrame(index=df.index)

    # --- calendar & holidays -------------------------------------------------------------
    for c in ["hour", "dow", "month", "is_weekend", "holiday_share", "is_xmas_period"]:
        X[c] = cal[c]
    X["doy_sin"] = np.sin(2 * np.pi * cal.doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * cal.doy / 365.25)
    X = X.join(_day_level_holiday_features(cal))
    X["hol_lag48"] = cal.holiday_share.shift(48)
    X["hol_lag168"] = cal.holiday_share.shift(168)

    # --- load lags (all >= MIN_LOAD_LAG_H) -------------------------------------------------
    assert min(LOAD_LAGS) >= MIN_LOAD_LAG_H
    for k in LOAD_LAGS:
        X[f"lag{k}"] = y.shift(k)
    abn = _abnormal(cal)
    # Same hour last week, but skip back a week (or two) when that day was a holiday.
    X["lag168_clean"] = np.where(
        ~abn.shift(168, fill_value=True),
        y.shift(168),
        np.where(~abn.shift(336, fill_value=True), y.shift(336), y.shift(504)),
    )

    # --- state at the issue-time origin (same value for all 24 hours of a target day) -------
    origin = df.index.floor("D") - ORIGIN_OFFSET
    state = pd.DataFrame(
        {
            "last_load": y,
            "mean_last24": y.rolling(24).mean(),
            "mean_last168": y.rolling(168).mean(),
            LEVEL_COL: y.rolling(28 * 24).mean(),
        }
    )
    X = X.join(state.reindex(origin).set_axis(df.index))

    # --- weather ---------------------------------------------------------------------------
    if weather != "none":
        wcols = [f"w_{v}" for v in WEATHER_VARS] + [f"temp_{c}" for c in CITIES]
        w = df[wcols]
        if weather == "noisy":
            w = add_forecast_noise(w, seed=seed)
        X = X.join(_weather_features(w))

    return X, y
