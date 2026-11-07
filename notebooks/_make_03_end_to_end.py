"""Generates notebooks/03_end_to_end.ipynb. Run from the project root:

uv run python notebooks/_make_03_end_to_end.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_end_to_end.ipynb
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 03 — End-to-end day-ahead load forecasting for Germany

Data → data check → leakage-safe features → models → rolling backtest → evaluation →
significance → error analysis → uncertainty → explainability.

**Forecasting task.** At **09:00 UTC on day D-1** (≈ when the German TSOs publish theirs),
forecast all 24 hourly loads of UTC day D. The newest load value available is the hour starting
07:00 UTC on D-1, so target hours are 17–40 h after the last observation.

**Evaluation.** Monthly refits with an expanding window; hyperparameters tuned on 2018 only;
test on **2019** (normal year) and **Jan–Sep 2020** (COVID stress test). Benchmarks: naive
baselines, the TSOs' official day-ahead forecast and a bias-corrected version of it.

All logic lives in `src/energy_demand_forecasting/`; this notebook orchestrates and analyses.
The heavy backtest (~35 min) is produced by `uv run run-backtest` and loaded here; set
`RUN_FULL_PIPELINE = True` to recompute it inside the notebook."""),
    code("""RUN_FULL_PIPELINE = False

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from energy_demand_forecasting import pipeline
from energy_demand_forecasting.data import PROJECT_ROOT
from energy_demand_forecasting.evaluation import PERIODS, diebold_mariano, metrics
from energy_demand_forecasting.features import TARGET, TSO_FORECAST, calendar_features, load_dataset
from energy_demand_forecasting.forecast_features import LEVEL_COL, ORIGIN_OFFSET, build_features
from energy_demand_forecasting.models import LGBMForecaster, _normalise

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
FIG, RES = PROJECT_ROOT / "reports" / "figures", PROJECT_ROOT / "reports" / "results"
save = lambda fig, name: fig.savefig(FIG / f"{name}.png")

df = load_dataset()
print(df.shape, df.index.min(), "->", df.index.max())"""),
    md("""## 1. Data and data-quality check

Target: `DE_load_actual_entsoe_transparency` (ENTSO-E, via OPSD). Comparing its annual totals
with official German gross electricity consumption shows its **coverage drifts upward**
(≈81 % → 86 %) — the series is not a fixed share of national demand. That matters for how we
read the TSO forecast's bias."""),
    code("""check = pd.read_csv(RES / "data_check_annual.csv", index_col=0)
display(check)
err = (df[TARGET] - df[TSO_FORECAST]) / 1000
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
check["coverage_ratio"].dropna().plot(ax=axes[0], marker="o")
axes[0].set(title="ENTSO-E actual load / official gross consumption", ylabel="ratio", xlabel="")
err.resample("ME").mean().plot(ax=axes[1], color="tab:red")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(title="TSO forecast bias by month (actual − forecast)", ylabel="GW", xlabel="")
save(fig, "10_data_check")"""),
    md("""## 2. Leakage-safe features

Calendar (local time, DST-aware), population-weighted holiday share, bridge days, distance
to holidays, holiday-cleaned weekly lag, load lags ≥ 48 h, load state at the issue-time origin
(last value, 24 h / 7 d / 28 d means), and weather (population-weighted means, per-city
temperature, heating/cooling degree-hours, smoothed temperature for thermal inertia).

**Leakage test:** multiply every load value after the issue-time cut-off by 3 — features for the
target day must not change."""),
    code("""X, y = build_features(df, "actual")
print(f"{X.shape[1]} features:", ", ".join(X.columns))

D = pd.Timestamp("2019-05-15", tz="UTC")
tampered = df.copy()
tampered.loc[tampered.index > D - ORIGIN_OFFSET, TARGET] *= 3
X_t, _ = build_features(tampered, "actual")
day = slice(D, D + pd.Timedelta(hours=23))
print("Features for", D.date(), "unchanged after tampering with the future:",
      X.loc[day].fillna(-1).equals(X_t.loc[day].fillna(-1)))"""),
    md("""## 3. Models and rolling backtest

| Model | Idea |
|---|---|
| `naive_2day`, `snaive_week`, `weekday_mean_4w` | naive references |
| `tso` / `tso_bias_corrected` | official TSO forecast; + its trailing 28-day mean error (known at issue time) |
| `ridge` | linear model, 24×3 hour × day-type profile, normalised target |
| `lgbm_raw` | LightGBM on load in MW |
| `lgbm_norm` | LightGBM on load / 28-day level (handles trend, coverage drift, COVID) |
| `ensemble` | mean of the two LightGBMs |
| `q10`, `q90` | quantile LightGBM → 80 % prediction interval |

Weather variants: **`actual`** (reanalysis = perfect weather forecast, optimistic),
**`noisy`** (trained on actuals, tested with simulated day-ahead forecast errors — *the
realistic setting and our headline*), **`none`** (no weather at all)."""),
    code("""params = json.loads((RES / "best_params.json").read_text())
display(pd.DataFrame(params).T)
if RUN_FULL_PIPELINE:
    preds = pipeline.run(tune=False)
else:
    preds = pd.read_parquet(pipeline.PRED_PATH)
print(preds.shape, preds.index.min(), "->", preds.index.max())"""),
    md("## 4. Headline results"),
    code("""m = pd.read_csv(RES / "metrics.csv", index_col=[0, 1])
headline = ["snaive_week", "tso", "tso_bias_corrected", "ridge",
            "lgbm_raw__noisy", "lgbm_norm__noisy", "ensemble__noisy", "ensemble__actual"]
tbl = m.loc[(slice(None), headline), ["MAE", "RMSE", "MAPE_%", "peak_MAE"]].copy()
tbl[["MAE", "RMSE", "peak_MAE"]] = (tbl[["MAE", "RMSE", "peak_MAE"]] / 1000).round(3)
tbl = tbl.rename(columns={"MAE": "MAE_GW", "RMSE": "RMSE_GW", "peak_MAE": "peak_MAE_GW"})
display(tbl)

plot = m["MAE"].unstack(0).loc[headline] / 1000
fig, ax = plt.subplots(figsize=(12, 4.5))
plot.plot.barh(ax=ax)
ax.invert_yaxis(); ax.set(title="MAE by model (lower is better)", xlabel="GW", ylabel="")
save(fig, "11_headline_mae")"""),
    code("""full = m[["MAE", "RMSE", "MAPE_%", "peak_MAE"]].unstack(0)
full.columns = [f"{a}|{b}" for a, b in full.columns]
full.sort_values("MAE|test_2019").round(1)"""),
    md("## 5. How much does weather (and weather-forecast error) matter?"),
    code("""rows = []
for v in ["none", "noisy", "actual"]:
    for p in PERIODS:
        rows.append({"weather": v, "period": p, "ensemble_MAE_GW": m.loc[(p, f"ensemble__{v}"), "MAE"] / 1000})
wv = pd.DataFrame(rows).pivot(index="weather", columns="period", values="ensemble_MAE_GW").loc[["none", "noisy", "actual"]]
display(wv.round(3))
fig, ax = plt.subplots(figsize=(7, 3.5))
wv.plot.bar(ax=ax, rot=0)
ax.set(title="Ensemble MAE by weather information", ylabel="GW", xlabel="")
ax.legend(title="period", loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "12_weather_variants")"""),
    md("""## 6. Are the differences statistically significant?

Diebold-Mariano test on daily mean absolute errors (Newey-West variance, HLN correction).
Negative statistic ⇒ our model is more accurate than the reference."""),
    code("""dm = pd.read_csv(RES / "dm_tests.csv")
dm["mean_diff_GW"] = (dm["mean_diff_MW"] / 1000).round(3)
dm.pivot_table(index=["period", "model"], columns="vs", values=["mean_diff_GW", "p_value"]).round(4)"""),
    md("## 7. Error analysis (2019, realistic weather)"),
    code("""s, e = PERIODS["test_2019"]
P = preds.loc[s:e]
cal = calendar_features(P.index)
cols = {"ensemble__noisy": "ensemble (noisy wx)", "tso": "TSO", "tso_bias_corrected": "TSO bias-corr."}
abs_err = P[list(cols)].sub(P.actual, axis=0).abs().rename(columns=cols) / 1000
day_type = pd.Series("workday", index=P.index)
day_type[cal.is_weekend == 1] = "weekend"
day_type[cal.holiday_share >= 0.5] = "holiday"
day_type[cal.is_xmas_period == 1] = "christmas"

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
abs_err.groupby(cal.hour).mean().plot(ax=axes[0], marker="o", ms=3)
axes[0].set(title="MAE by local hour", xlabel="hour", ylabel="GW")
abs_err.groupby(day_type).mean().loc[["workday", "weekend", "holiday", "christmas"]].plot.bar(ax=axes[1], rot=0)
axes[1].set(title="MAE by day type", ylabel="GW", xlabel="")
abs_err.groupby(cal.month).mean().plot(ax=axes[2], marker="o")
axes[2].set(title="MAE by month", xlabel="month", ylabel="GW")
save(fig, "13_error_breakdown")
abs_err.groupby(day_type).mean().round(3)"""),
    code("""bias = P[list(cols)].rsub(P.actual, axis=0).rename(columns=cols).mean() / 1000
print("Mean bias 2019 (actual − forecast, GW):"); print(bias.round(3).to_string())
daily_err = abs_err.groupby(abs_err.index.floor("D")).mean()
print("\\nWorst 8 days for the ensemble:")
daily_err.sort_values("ensemble (noisy wx)", ascending=False).head(8).round(2)"""),
    md("## 8. Example forecasts"),
    code("""windows = {"A normal winter week (Jan 2019)": ("2019-01-14", "2019-01-20"),
           "Ascension & bridge day (May/Jun 2019)": ("2019-05-27", "2019-06-02"),
           "Christmas 2019": ("2019-12-20", "2019-12-31"),
           "COVID lockdown onset (Mar 2020)": ("2020-03-09", "2020-03-29")}
fig, axes = plt.subplots(len(windows), 1, figsize=(14, 13))
for ax, (title, (a, b)) in zip(axes, windows.items()):
    w = preds.loc[a:f"{b} 23:00"] / 1000
    ax.fill_between(w.index, w.q_cp_lo__noisy, w.q_cp_hi__noisy, color="tab:blue", alpha=0.15, label="80% interval (conformal)")
    ax.plot(w.actual, color="k", lw=1.6, label="actual")
    ax.plot(w.ensemble__noisy, color="tab:blue", lw=1.2, label="ensemble (noisy wx)")
    ax.plot(w.tso, color="tab:orange", lw=1, alpha=0.9, label="TSO")
    ax.set(title=title, ylabel="GW")
axes[0].legend(ncols=4, loc="upper right")
fig.tight_layout()
save(fig, "14_example_forecasts")"""),
    md("## 9. Adapting to the COVID shock"),
    code("""s20, e20 = PERIODS["stress_2020"]
c = {"ensemble__noisy": "ensemble", "lgbm_raw__noisy": "LGBM raw", "lgbm_norm__noisy": "LGBM normalised",
     "tso": "TSO", "snaive_week": "seasonal naive"}
P20 = preds.loc[s20:e20]
roll = P20[list(c)].sub(P20.actual, axis=0).abs().rename(columns=c).resample("D").mean().rolling(7).mean() / 1000
fig, ax = plt.subplots(figsize=(13, 4))
roll.plot(ax=ax)
ax.axvspan(pd.Timestamp("2020-03-16", tz="UTC"), pd.Timestamp("2020-05-31", tz="UTC"), color="tab:red", alpha=0.08)
ax.set(title="7-day rolling MAE in 2020 (shaded: first lockdown)", ylabel="GW", xlabel="")
save(fig, "15_covid_adaptation")
lock = slice("2020-03-16", "2020-05-31")
(P20.loc[lock, list(c)].sub(P20.loc[lock, "actual"], axis=0).abs().mean() / 1000).rename(index=c).round(3).to_frame("MAE lockdown (GW)")"""),
    md("""## 10. Uncertainty: 80 % prediction intervals

Three methods: raw quantile LightGBM (q10/q90), **CQR** (quantiles widened by recent conformity scores) and
**conformal around the ensemble** (rolling empirical error quantiles). Calibration uses only errors known
at issue time (lagged 48 h, 56-day window), so the first week of 2019 has no interval."""),
    code("""iv = pd.read_csv(RES / "intervals.csv")
iv["mean_width_GW"] = (iv["mean_width_MW"] / 1000).round(2)
display(iv.drop(columns="mean_width_MW").pivot_table(index=["period", "weather"], columns="method", values=["coverage_%", "mean_width_GW"]))
yy = P.actual
lo, hi = P.q_cp_lo__noisy, P.q_cp_hi__noisy
ok = lo.notna()
cov_hour = ((yy >= lo) & (yy <= hi))[ok].groupby(cal.hour[ok]).mean() * 100
raw_cov = ((yy >= P.q10__noisy) & (yy <= P.q90__noisy)).groupby(cal.hour).mean() * 100
fig, ax = plt.subplots(figsize=(10, 3.5))
pd.DataFrame({"raw quantile LightGBM": raw_cov, "conformal (ensemble)": cov_hour}).plot.bar(ax=ax, color=["tab:gray", "tab:green"])
ax.axhline(80, color="k", ls="--", lw=1)
ax.set(title="Empirical coverage of the 80% interval by hour (2019, noisy weather)", ylabel="%", xlabel="hour")
ax.legend(loc="lower right")
save(fig, "16_interval_coverage")"""),
    md("""## 11. What drives the model? (SHAP)

Refit the normalised LightGBM on all data up to 2018-12-30 (as in the January 2019 fold) and
explain its 2019 predictions. Values are in units of the normalised target (load / 28-day level)."""),
    code("""p_norm = {k: v for k, v in params["lgbm_norm"].items() if k not in {"val_MAE", "n_trials"}}
tr = X.loc["2015-02-01":"2018-12-29 23:00"]
ok = tr.notna().all(axis=1)
fm = LGBMForecaster(True, p_norm).fit(tr[ok], y.loc[tr.index][ok])
sample = _normalise(X.loc["2019"]).sample(4000, random_state=0)
explainer = shap.TreeExplainer(fm.model_)
sv = explainer.shap_values(sample)
imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False)

fig = plt.figure(figsize=(9, 7))
shap.summary_plot(sv, sample, max_display=18, show=False)
plt.title("SHAP summary, normalised LightGBM (2019 sample)")
save(plt.gcf(), "17_shap_summary")
plt.show()
(imp.head(15) * 100).round(2).to_frame("mean |SHAP| (% of level)")"""),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, f in zip(axes, ["w_temperature_2m", "holiday_share", "w_shortwave_radiation"]):
    ax.scatter(sample[f], sv[:, sample.columns.get_loc(f)] * 100, s=3, alpha=0.4,
               c=sample["hour"], cmap="twilight")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title=f"SHAP dependence: {f}", xlabel=f, ylabel="effect (% of level)")
save(fig, "18_shap_dependence")"""),
    md("""## 12. Conclusions

* **Accuracy.** With realistic (noisy) weather, the LightGBM ensemble's MAE is **1.00 GW (1.84 %) in 2019**
  vs **1.99 GW** for the official TSO forecast and **1.55 GW** for the bias-corrected TSO forecast; in
  Jan–Sep 2020: **1.06 GW** vs 1.46 / 1.30 GW. All differences are significant (Diebold-Mariano p < 0.001).
* **Even without any weather data** the ensemble (1.13 GW) beats the bias-corrected TSO forecast — most of
  the edge comes from calendar/holiday modelling and level normalisation; weather adds ~11 % on top,
  and simulated weather-forecast error gives back about a quarter of that.
* **Peak-hour error** halves vs the TSO forecast (1.02 vs 2.14 GW in 2019).
* **Intervals.** Raw quantile LightGBM intervals were badly over-confident (44 % coverage for a nominal
  80 %); online conformal calibration fixes this (≈79–81 %).
* **Caveats.** Weather-forecast errors are simulated; the target series has documented coverage drift;
  OPSD ends in Sep 2020. See `reports/REPORT.md`."""),
]
nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "name": "energy-demand-forecasting",
            "display_name": "Python (energy-demand-forecasting)",
            "language": "python",
        }
    },
)
nbf.write(nb, "notebooks/03_end_to_end.ipynb")
print("wrote notebooks/03_end_to_end.ipynb")
