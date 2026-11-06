"""Hyperparameter tuning on 2018 only (the test years 2019-2020 are never touched).

Validation folds: one month per quarter of 2018 (Jan, Apr, Jul, Oct), each with an expanding
training window, exactly like the backtest. Objective: MAE.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from energy_demand_forecasting.backtest import run_backtest
from energy_demand_forecasting.models import LGBMForecaster, RidgeForecaster

TUNE_MONTHS = [
    pd.Timestamp(m, tz="UTC")
    for m in ["2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01"]
]


def _mae(pred: pd.Series, y: pd.Series) -> float:
    return float(np.mean(np.abs(y.loc[pred.index] - pred)))


def tune_ridge(X: pd.DataFrame, y: pd.Series) -> dict:
    scores = {
        a: _mae(run_backtest(lambda a=a: RidgeForecaster(a), X, y, TUNE_MONTHS), y)
        for a in [0.1, 1, 10, 100]
    }
    best = min(scores, key=scores.get)
    return {"alpha": best, "val_MAE": scores[best]}


def tune_lgbm(
    X: pd.DataFrame, y: pd.Series, normalise: bool, n_trials: int = 20, seed: int = 0
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 200, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        pred = run_backtest(
            lambda: LGBMForecaster(normalise, params), X, y, TUNE_MONTHS
        )
        return _mae(pred, y)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    # Start from the defaults so tuning can only improve on them.
    from energy_demand_forecasting.models import DEFAULT_LGBM_PARAMS

    study.enqueue_trial(
        {k: v for k, v in DEFAULT_LGBM_PARAMS.items() if k != "subsample_freq"}
    )
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "val_MAE": study.best_value, "n_trials": n_trials}
