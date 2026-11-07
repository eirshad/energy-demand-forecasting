"""End-to-end pipeline: features -> (tuning) -> rolling backtests -> metrics.

    uv run run-backtest            # use cached tuned parameters if present
    uv run run-backtest --tune     # re-tune on 2018 first (~30-40 min)

Outputs:
    data/processed/predictions.parquet   all out-of-sample forecasts (2019-01 .. 2020-09)
    reports/results/*.csv|json           metrics, significance tests, intervals, parameters
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from energy_demand_forecasting.backtest import month_starts, run_backtest
from energy_demand_forecasting.baselines import (
    naive_2day,
    seasonal_naive_week,
    tso_bias_corrected,
    weekday_mean,
)
from energy_demand_forecasting.data import PROJECT_ROOT
from energy_demand_forecasting.evaluation import (
    PERIODS,
    conformal_from_point,
    conformalize_quantiles,
    daily_peak_mae,
    diebold_mariano,
    evaluate,
    interval_metrics,
    pinball_loss,
)
from energy_demand_forecasting.features import TSO_FORECAST, load_dataset
from energy_demand_forecasting.forecast_features import build_features
from energy_demand_forecasting.models import LGBMForecaster, RidgeForecaster

RESULTS = PROJECT_ROOT / "reports" / "results"
PRED_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.parquet"
PARAMS_PATH = RESULTS / "best_params.json"
TEST_MONTHS = month_starts("2019-01-01", "2020-09-01")
NOISE_SEED = 42
QUANTILES = (0.1, 0.9)


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def get_params(
    X: pd.DataFrame, y: pd.Series, tune: bool, n_trials: int, t0: float
) -> dict:
    if PARAMS_PATH.exists() and not tune:
        _log(f"using cached parameters from {PARAMS_PATH.name}", t0)
        return json.loads(PARAMS_PATH.read_text())
    from energy_demand_forecasting.tuning import tune_lgbm, tune_ridge

    params = {"ridge": tune_ridge(X, y)}
    _log(f"ridge tuned: {params['ridge']}", t0)
    for name, norm in [("lgbm_raw", False), ("lgbm_norm", True)]:
        params[name] = tune_lgbm(X, y, normalise=norm, n_trials=n_trials)
        _log(f"{name} tuned: {params[name]}", t0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    return params


def _lgbm_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in {"val_MAE", "n_trials"}}


def run(tune: bool = False, n_trials: int = 20) -> pd.DataFrame:
    t0 = time.time()
    df = load_dataset()
    X_act, y = build_features(df, "actual")
    X_none, _ = build_features(df, "none")
    X_noisy, _ = build_features(df, "noisy", seed=NOISE_SEED)
    _log(f"features built: {X_act.shape}", t0)

    params = get_params(X_act, y, tune, n_trials, t0)
    p_raw, p_norm = _lgbm_params(params["lgbm_raw"]), _lgbm_params(params["lgbm_norm"])

    P = {}
    P["ridge"] = run_backtest(
        lambda: RidgeForecaster(params["ridge"]["alpha"]), X_act, y, TEST_MONTHS
    )
    _log("ridge done", t0)
    # Weather variants: 'actual' (oracle weather), 'none' (no weather features),
    # 'noisy' (trained on actuals, tested with simulated forecast errors = realistic).
    variants = {
        "actual": (X_act, X_act),
        "none": (X_none, X_none),
        "noisy": (X_act, X_noisy),
    }
    for v, (Xtr, Xte) in variants.items():
        for name, norm, p in [("lgbm_raw", False, p_raw), ("lgbm_norm", True, p_norm)]:
            P[f"{name}__{v}"] = run_backtest(
                lambda norm=norm, p=p: LGBMForecaster(norm, p),
                Xtr,
                y,
                TEST_MONTHS,
                X_test=Xte,
            )
            _log(f"{name} [{v}] done", t0)
        P[f"ensemble__{v}"] = (P[f"lgbm_raw__{v}"] + P[f"lgbm_norm__{v}"]) / 2
    for q in QUANTILES:
        for v, (Xtr, Xte) in {
            "actual": variants["actual"],
            "noisy": variants["noisy"],
        }.items():
            P[f"q{int(q * 100)}__{v}"] = run_backtest(
                lambda q=q: LGBMForecaster(True, p_norm, quantile=q),
                Xtr,
                y,
                TEST_MONTHS,
                X_test=Xte,
            )
            _log(f"quantile {q} [{v}] done", t0)

    preds = pd.DataFrame(P)
    tso = df[TSO_FORECAST]
    bench = pd.DataFrame(
        {
            "naive_2day": naive_2day(y),
            "snaive_week": seasonal_naive_week(y),
            "weekday_mean_4w": weekday_mean(y),
            "tso": tso,
            "tso_bias_corrected": tso_bias_corrected(y, tso),
        }
    ).loc[preds.index]
    preds = bench.join(preds)
    preds.insert(0, "actual", y.loc[preds.index])
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds = add_conformal_intervals(preds)
    preds.to_parquet(PRED_PATH)
    _log(f"predictions saved: {preds.shape}", t0)

    save_metrics(preds)
    _log("metrics saved - done", t0)
    return preds


def save_metrics(preds: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    y = preds["actual"]
    point = [c for c in preds.columns if c != "actual" and not c.startswith("q")]
    m = evaluate(y, preds[point])
    m["peak_MAE"] = [
        round(
            daily_peak_mae(
                y.loc[PERIODS[p][0] : PERIODS[p][1]],
                preds[c].loc[PERIODS[p][0] : PERIODS[p][1]],
            ),
            2,
        )
        for p, c in m.index
    ]
    m.to_csv(RESULTS / "metrics.csv")

    dm_rows = []
    for period, (s, e) in PERIODS.items():
        for model in ["ensemble__noisy", "ensemble__actual", "ensemble__none"]:
            for ref in ["tso", "tso_bias_corrected", "snaive_week", "ridge"]:
                r = diebold_mariano(
                    y.loc[s:e], preds[model].loc[s:e], preds[ref].loc[s:e]
                )
                dm_rows.append({"period": period, "model": model, "vs": ref, **r})
    pd.DataFrame(dm_rows).round(4).to_csv(RESULTS / "dm_tests.csv", index=False)

    iv_rows = []
    methods = {
        "quantile_lgbm": ("q10__{v}", "q90__{v}"),
        "cqr": ("q_cqr_lo__{v}", "q_cqr_hi__{v}"),
        "conformal_point": ("q_cp_lo__{v}", "q_cp_hi__{v}"),
    }
    for period, (s, e) in PERIODS.items():
        for v in ["actual", "noisy"]:
            for method, (lo_c, hi_c) in methods.items():
                lo = preds[lo_c.format(v=v)].loc[s:e]
                hi = preds[hi_c.format(v=v)].loc[s:e]
                ys = y.loc[s:e]
                iv_rows.append(
                    {
                        "period": period,
                        "weather": v,
                        "method": method,
                        **interval_metrics(ys, lo, hi),
                        "pinball_q10": pinball_loss(ys, lo, 0.1),
                        "pinball_q90": pinball_loss(ys, hi, 0.9),
                    }
                )
    pd.DataFrame(iv_rows).round(2).to_csv(RESULTS / "intervals.csv", index=False)


def add_conformal_intervals(preds: pd.DataFrame) -> pd.DataFrame:
    """Calibrated 80 % intervals: CQR on the quantile models, and conformal around the ensemble.

    Calibration uses only out-of-sample errors from earlier forecasts, so the first week of the
    test period has no interval (NaN) and is excluded from interval metrics.
    """
    y = preds["actual"]
    out = preds.copy()
    for v in ["actual", "noisy"]:
        lo, hi = conformalize_quantiles(y, preds[f"q10__{v}"], preds[f"q90__{v}"])
        out[f"q_cqr_lo__{v}"], out[f"q_cqr_hi__{v}"] = lo, hi
        lo, hi = conformal_from_point(y, preds[f"ensemble__{v}"])
        out[f"q_cp_lo__{v}"], out[f"q_cp_hi__{v}"] = lo, hi
    return out


def recompute_metrics() -> None:
    """Re-derive calibrated intervals and all metrics from saved predictions (seconds)."""
    preds = pd.read_parquet(PRED_PATH)
    preds = add_conformal_intervals(preds[[c for c in preds if not c.startswith("q_")]])
    preds.to_parquet(PRED_PATH)
    save_metrics(preds)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--metrics-only",
        action="store_true",
        help="only recompute intervals/metrics from saved predictions",
    )
    ap.add_argument(
        "--tune", action="store_true", help="re-run hyperparameter tuning on 2018"
    )
    ap.add_argument(
        "--trials", type=int, default=20, help="Optuna trials per LightGBM model"
    )
    args = ap.parse_args()
    if args.metrics_only:
        recompute_metrics()
        return
    run(tune=args.tune, n_trials=args.trials)


if __name__ == "__main__":
    main()
