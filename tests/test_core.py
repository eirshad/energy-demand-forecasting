import numpy as np
import pandas as pd
import pytest

from energy_demand_forecasting.baselines import (
    naive_day,
    seasonal_naive_week,
    weekday_mean,
)
from energy_demand_forecasting.evaluation import evaluate, metrics
from energy_demand_forecasting.features import calendar_features, holiday_share


@pytest.fixture
def hourly():
    idx = pd.date_range("2019-01-01", periods=24 * 7 * 6, freq="h", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)


def test_baselines_only_use_lags_of_at_least_24h(hourly):
    assert naive_day(hourly).iloc[24] == hourly.iloc[0]
    assert seasonal_naive_week(hourly).iloc[168] == hourly.iloc[0]
    wm = weekday_mean(hourly, weeks=4)
    assert wm.iloc[: 168 * 4].isna().all()
    t = 168 * 4
    assert wm.iloc[t] == np.mean([hourly.iloc[t - 168 * k] for k in range(1, 5)])


def test_metrics_known_values():
    y = pd.Series([100.0, 200.0])
    m = metrics(y, pd.Series([110.0, 180.0]))
    assert m["MAE"] == pytest.approx(15.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["MAPE_%"] == pytest.approx(10.0)


def test_evaluate_ignores_missing_predictions(hourly):
    preds = pd.DataFrame({"perfect": hourly, "lagged": naive_day(hourly)})
    res = evaluate(hourly, preds, {"all": ("2019-01-01", "2019-02-28")})
    assert res.loc[("all", "perfect"), "MAE"] == 0
    assert res.loc[("all", "lagged"), "n"] == len(hourly) - 24


def test_holiday_share():
    s = holiday_share(pd.DatetimeIndex(["2019-12-25", "2019-06-20", "2019-07-10"]))
    assert s.iloc[0] == 1.0  # Christmas: national
    assert 0 < s.iloc[1] < 1  # Corpus Christi: regional
    assert s.iloc[2] == 0.0


def test_calendar_features_use_local_time_and_dst():
    idx = pd.date_range("2019-03-31 00:00", periods=3, freq="h", tz="UTC")
    f = calendar_features(idx)
    assert f["hour"].tolist() == [1, 3, 4]  # CET -> CEST jump at 02:00 local
