import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 02 — Baselines

**Setup:** day-ahead. Forecasts for day D are issued at 00:00 UTC on D, using only data up to D-1
(all baselines use lags ≥ 24 h). Scored on 2019 (normal year) and Jan–Sep 2020 (COVID stress test).
The TSOs' official day-ahead forecast is the bar to beat."""),
    code("""import matplotlib.pyplot as plt
import pandas as pd

from energy_demand_forecasting.baselines import naive_day, seasonal_naive_week, weekday_mean
from energy_demand_forecasting.data import PROJECT_ROOT
from energy_demand_forecasting.evaluation import PERIODS, evaluate
from energy_demand_forecasting.features import TARGET, TSO_FORECAST, calendar_features, load_dataset

df = load_dataset()
y = df[TARGET]
preds = pd.concat([naive_day(y), seasonal_naive_week(y), weekday_mean(y, 4),
                   df[TSO_FORECAST].rename("tso_dayahead")], axis=1)
res = evaluate(y, preds)
res.to_csv(PROJECT_ROOT / "reports" / "baselines_metrics.csv")
res"""),
    md("## Where do the baselines fail? Error by day type (2019)"),
    code("""cal = calendar_features(df.index)
s, e = PERIODS["test_2019"]
day_type = pd.Series("workday", index=df.index)
day_type[cal.is_weekend == 1] = "weekend"
day_type[(cal.holiday_share > 0.5) | (cal.is_xmas_period == 1)] = "holiday/xmas"
# A workday right after a weekend/holiday, or right before one, breaks the "same as yesterday" logic
abs_err = preds.sub(y, axis=0).abs().loc[s:e] / 1000
by_type = abs_err.groupby(day_type.loc[s:e]).mean().round(2)
by_type["share_of_hours"] = day_type.loc[s:e].value_counts(normalize=True).round(3)
by_type"""),
    code("""fig, ax = plt.subplots(figsize=(13, 4))
win = slice("2019-12-16", "2020-01-12")
(y.loc[win] / 1000).plot(ax=ax, color="k", lw=1.6, label="actual")
for c in ["snaive_week", "tso_dayahead"]:
    (preds[c].loc[win] / 1000).plot(ax=ax, lw=1, alpha=0.85, label=c)
ax.set(title="Christmas 2019: seasonal naive vs TSO forecast", ylabel="GW"); ax.legend()
fig.savefig(PROJECT_ROOT / "reports" / "figures" / "09_baselines_christmas.png", bbox_inches="tight")"""),
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
nbf.write(nb, "notebooks/02_baselines.ipynb")
