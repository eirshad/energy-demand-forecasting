"""Loading the interim data and building calendar / holiday features."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from energy_demand_forecasting.data import INTERIM_DIR

LOCAL_TZ = "Europe/Berlin"
TARGET = "DE_load_actual_entsoe_transparency"
TSO_FORECAST = "DE_load_forecast_entsoe_transparency"

# State populations (millions, ~2019) to weight regional holidays by how much
# of the country (and so of the demand) is actually off work.
STATE_POP = {
    "BW": 11.1, "BY": 13.1, "BE": 3.67, "BB": 2.52, "HB": 0.68, "HH": 1.85,
    "HE": 6.29, "MV": 1.61, "NI": 8.00, "NW": 17.9, "RP": 4.09, "SL": 0.99,
    "SN": 4.07, "ST": 2.19, "SH": 2.90, "TH": 2.13,
}  # fmt: skip


def load_dataset() -> pd.DataFrame:
    """Load + weather on one hourly UTC index."""
    load = pd.read_parquet(INTERIM_DIR / "de_load.parquet")
    weather = pd.read_parquet(INTERIM_DIR / "de_weather.parquet")
    return load.join(weather, how="inner")


def holiday_share(dates: pd.DatetimeIndex) -> pd.Series:
    """Population share of Germany observing a public holiday on each (local) date.

    1.0 = national holiday, 0 < x < 1 = regional holiday (e.g. Corpus Christi), 0 = normal day.
    """
    days = pd.DatetimeIndex(dates.normalize().unique())
    years = sorted(set(days.year))
    total = sum(STATE_POP.values())
    share = pd.Series(0.0, index=days)
    for state, pop in STATE_POP.items():
        cal = holidays.Germany(subdiv=state, years=years)
        is_hol = np.array([d.date() in cal for d in days])
        share += is_hol * pop / total
    return share.round(4)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in *local* time (demand follows the clock people live by, incl. DST)."""
    local = index.tz_convert(LOCAL_TZ)
    local_dates = local.tz_localize(None).normalize()
    hol = holiday_share(local_dates)
    md = local.month * 100 + local.day
    f = pd.DataFrame(
        {
            "hour": local.hour,
            "dow": local.dayofweek,  # 0 = Monday
            "month": local.month,
            "doy": local.dayofyear,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "holiday_share": hol.reindex(local_dates).to_numpy(),
            # Dec 24 - Jan 1: many businesses closed even on non-holiday days.
            "is_xmas_period": ((md >= 1224) | (md <= 101)).astype(int),
        },
        index=index,
    )
    return f
