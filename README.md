# Energy Demand Forecasting

Forecasting electricity demand from real public data, using classical baselines,
statistical models and gradient boosting, evaluated with rolling-origin backtests.

## Setup
```bash
uv sync
uv run jupyter lab
```

## Project layout
See `CLAUDE.md` for conventions.

## Data
Re-fetch everything with `uv run fetch-data` (add `--force` to re-download). Retrieved 2026-11-07.

| Dataset | Source | Coverage | License / terms |
|---|---|---|---|
| OPSD Time series, 60-min single-index (release 2020-10-06, final release) | https://data.open-power-system-data.org/time_series/2020-10-06/ | 2015-01-01 – 2020-09-30, hourly, UTC | Compiled from ENTSO-E Transparency and TSO data; licensing follows the original sources (see `data/raw/opsd/datapackage.json`). Cite: *Open Power System Data. 2020. Data Package Time series. Version 2020-10-06.* |
| Open-Meteo Historical Weather API (ERA5-based reanalysis) | https://archive-api.open-meteo.com/v1/archive | Same period, hourly, 5 German cities | CC BY 4.0 — attribution to Open-Meteo.com |

**Target:** `DE_load_actual_entsoe_transparency` (Germany total load, MW).
**Benchmark:** `DE_load_forecast_entsoe_transparency`, the TSOs' official day-ahead forecast.
**Weather features:** temperature, humidity, wind, cloud cover and shortwave radiation, averaged over
Berlin, Hamburg, Munich, Cologne and Frankfurt, weighted by population (`w_*` columns), plus per-city temperature.

Interim outputs (`data/interim/`): `de_load.parquet`, `de_weather.parquet`, both 50,400 hourly rows on the same UTC index.
