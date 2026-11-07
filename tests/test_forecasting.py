import numpy as np
import pandas as pd
import pytest

from energy_demand_forecasting.backtest import TRAIN_GAP, run_backtest
from energy_demand_forecasting.baselines import tso_bias_corrected
from energy_demand_forecasting.data import CITIES, WEATHER_VARS
from energy_demand_forecasting.evaluation import (
    diebold_mariano,
    interval_metrics,
    pinball_loss,
)
from energy_demand_forecasting.features import TARGET
from energy_demand_forecasting.forecast_features import (
    LEVEL_COL,
    MIN_LOAD_LAG_H,
    ORIGIN_OFFSET,
    add_forecast_noise,
    build_features,
)
from energy_demand_forecasting.models import LGBMForecaster, RidgeForecaster


@pytest.fixture(scope="module")
def synthetic():
    """Two years of hourly load with daily/weekly cycles + temperature dependence."""
    idx = pd.date_range("2017-01-01", "2018-12-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    h = idx.hour.to_numpy()
    temp = (
        10
        + 10 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 110) / 365)
        + rng.normal(0, 2, len(idx))
    )
    load = (
        50_000
        + 8_000 * np.sin(np.pi * h / 24)
        - 6_000 * (idx.dayofweek.to_numpy() >= 5)
        - 300 * temp
    )
    load = load + rng.normal(0, 500, len(idx))
    df = pd.DataFrame({TARGET: load}, index=idx)
    for v in WEATHER_VARS:
        df[f"w_{v}"] = temp if v == "temperature_2m" else rng.uniform(0, 100, len(idx))
    for c in CITIES:
        df[f"temp_{c}"] = temp
    return df


def test_features_do_not_leak_future_load(synthetic):
    X, _ = build_features(synthetic, "actual")
    day = pd.Timestamp("2018-06-14", tz="UTC")
    changed = synthetic.copy()
    changed.loc[changed.index > day - ORIGIN_OFFSET, TARGET] *= 5
    X2, _ = build_features(changed, "actual")
    sl = slice(day, day + pd.Timedelta(hours=23))
    pd.testing.assert_frame_equal(X.loc[sl], X2.loc[sl])
    assert MIN_LOAD_LAG_H == 17 + 23


def test_origin_state_constant_within_target_day(synthetic):
    X, _ = build_features(synthetic, "none")
    day = X.loc["2018-03-10"]
    for c in ["last_load", "mean_last24", LEVEL_COL]:
        assert day[c].nunique() == 1
    assert not any(c.startswith("w_") for c in X.columns)


def test_noise_changes_weather_but_keeps_ranges(synthetic):
    w = synthetic[[c for c in synthetic.columns if c != TARGET]]
    n = add_forecast_noise(w, seed=1)
    assert 0.8 < (n.w_temperature_2m - w.w_temperature_2m).std() < 2.0
    assert (
        n.w_cloud_cover.between(0, 100).all() and (n.w_shortwave_radiation >= 0).all()
    )


def test_backtest_trains_only_on_past(synthetic):
    X, y = build_features(synthetic, "actual")
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X.index.max())
            return self

        def predict(self, X):
            return np.zeros(len(X))

    months = [
        pd.Timestamp("2018-05-01", tz="UTC"),
        pd.Timestamp("2018-09-01", tz="UTC"),
    ]
    pred = run_backtest(Spy, X, y, months)
    for m, last in zip(months, seen, strict=True):
        assert last < m - TRAIN_GAP + pd.Timedelta(hours=1)
    assert pred.index.min() == months[0] and len(pred) == (31 + 30) * 24


@pytest.mark.parametrize(
    "make",
    [lambda: RidgeForecaster(1.0), lambda: LGBMForecaster(True, {"n_estimators": 50})],
)
def test_models_beat_seasonal_naive_on_synthetic(synthetic, make):
    X, y = build_features(synthetic, "actual")
    pred = run_backtest(
        make, X, y, [pd.Timestamp("2018-11-01", tz="UTC")], train_start="2017-02-01"
    )
    mae = (y.loc[pred.index] - pred).abs().mean()
    snaive = (y - y.shift(168)).loc[pred.index].abs().mean()
    assert mae < snaive


def test_tso_bias_correction_removes_constant_offset():
    idx = pd.date_range("2019-01-01", periods=24 * 60, freq="h", tz="UTC")
    y = pd.Series(50_000 + 1000 * np.sin(np.arange(len(idx)) / 5), index=idx)
    tso = y - 1500  # TSO always 1.5 GW too low
    bc = tso_bias_corrected(y, tso)
    assert (y - bc).iloc[-24:].abs().max() < 1e-6


def test_dm_and_interval_metrics():
    idx = pd.date_range("2019-01-01", periods=24 * 100, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    y = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    good, bad = y + rng.normal(0, 0.1, len(idx)), y + rng.normal(0, 1, len(idx))
    r = diebold_mariano(y, good, bad)
    assert r["DM_stat"] < 0 and r["p_value"] < 0.01
    iv = interval_metrics(y, y - 1, y + 1)
    assert iv["coverage_%"] == 100 and iv["mean_width_MW"] == 2
    assert pinball_loss(y, y, 0.9) == 0
