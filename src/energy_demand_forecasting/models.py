"""Forecasting models with a common fit/predict interface.

All models receive the full feature matrix (including LEVEL_COL). "Normalised" models predict
load / level28 (the 28-day mean load known at issue time) and see load-valued features divided
by the same level, so they generalise across slow level changes (trend, data-coverage drift,
COVID) that tree models cannot extrapolate.
"""

from __future__ import annotations

from typing import ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from energy_demand_forecasting.forecast_features import LEVEL_COL, LOAD_COLS

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def _normalise(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    cols = [c for c in LOAD_COLS if c in Xn]
    Xn[cols] = Xn[cols].div(Xn[LEVEL_COL], axis=0)
    return Xn


class LGBMForecaster:
    def __init__(
        self,
        normalise: bool = True,
        params: dict | None = None,
        quantile: float | None = None,
        seed: int = 0,
    ):
        self.normalise, self.quantile, self.seed = normalise, quantile, seed
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    def _model(self) -> lgb.LGBMRegressor:
        kw = (
            {"objective": "quantile", "alpha": self.quantile}
            if self.quantile
            else {"objective": "l2"}
        )
        return lgb.LGBMRegressor(
            **self.params, **kw, random_state=self.seed, verbose=-1
        )

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        return _normalise(X) if self.normalise else X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LGBMForecaster:
        target = y / X[LEVEL_COL] if self.normalise else y
        self.features_ = list(X.columns)
        self.model_ = self._model().fit(self._prep(X), target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model_.predict(self._prep(X[self.features_]))
        return p * X[LEVEL_COL].to_numpy() if self.normalise else p


class RidgeForecaster:
    """Interpretable linear benchmark on the normalised target.

    Hour × day-type is one-hot encoded (24 × 3 profile shapes); everything else is linear.
    """

    CATEGORICAL: ClassVar[list[str]] = ["hour_daytype", "dow", "month"]

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def _design(X: pd.DataFrame) -> pd.DataFrame:
        Z = _normalise(X)
        daytype = np.where(
            Z.holiday_share >= 0.5,
            2,
            np.where(Z.dow == 6, 2, np.where(Z.dow == 5, 1, 0)),
        )
        Z = Z.assign(hour_daytype=Z.hour.astype(int) * 3 + daytype)
        return Z.drop(columns=["hour", LEVEL_COL])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecaster:
        Z = self._design(X)
        num = [c for c in Z.columns if c not in self.CATEGORICAL]
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL),
                ("num", StandardScaler(), num),
            ]
        )
        self.features_ = list(X.columns)
        self.model_ = make_pipeline(pre, Ridge(alpha=self.alpha)).fit(
            Z, y / X[LEVEL_COL]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features_]
        return self.model_.predict(self._design(X)) * X[LEVEL_COL].to_numpy()
