"""Download and prepare raw data: OPSD hourly load + Open-Meteo historical weather.

Raw files land in data/raw/ (never edited). Germany-only, analysis-ready tables
are written to data/interim/.

Usage:
    uv run fetch-data            # download (skips files that already exist) + build interim
    uv run fetch-data --force    # re-download everything
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Pinned release (the final OPSD time-series release) for reproducibility.
OPSD_RELEASE = "2020-10-06"
OPSD_BASE = f"https://data.open-power-system-data.org/time_series/{OPSD_RELEASE}"
OPSD_FILES = ["time_series_60min_singleindex.csv", "datapackage.json"]

COUNTRY = "DE"
START_DATE = "2015-01-01"
END_DATE = "2020-09-30"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]
# Largest German metro areas; weights ~ city population (millions) as a proxy for demand.
CITIES = {
    "berlin": (52.52, 13.41, 3.7),
    "hamburg": (53.55, 9.99, 1.9),
    "munich": (48.14, 11.58, 1.5),
    "cologne": (50.94, 6.96, 1.1),
    "frankfurt": (50.11, 8.68, 0.77),
}


def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"  exists, skipping: {dest.relative_to(PROJECT_ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(chunk_size=1 << 20))
    tmp.rename(dest)
    print(
        f"  downloaded {dest.stat().st_size / 1e6:.1f} MB -> {dest.relative_to(PROJECT_ROOT)}"
    )
    return dest


def download_opsd(force: bool = False) -> None:
    print(f"OPSD time series (release {OPSD_RELEASE})")
    for name in OPSD_FILES:
        _download(f"{OPSD_BASE}/{name}", RAW_DIR / "opsd" / name, force)


def download_weather(force: bool = False) -> None:
    print(f"Open-Meteo hourly weather {START_DATE}..{END_DATE}")
    out_dir = RAW_DIR / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    for city, (lat, lon, _) in CITIES.items():
        dest = out_dir / f"{city}.json"
        if dest.exists() and not force:
            print(f"  exists, skipping: {dest.relative_to(PROJECT_ROOT)}")
            continue
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "GMT",
        }
        for attempt in range(1, 5):
            try:
                r = requests.get(OPEN_METEO_URL, params=params, timeout=180)
                r.raise_for_status()
                break
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
            ) as e:
                if attempt == 4:
                    raise
                wait = 10 * attempt
                print(
                    f"  {city}: attempt {attempt} failed ({type(e).__name__}), retrying in {wait}s"
                )
                time.sleep(wait)
        dest.write_text(r.text)
        print(f"  {city}: {len(r.content) / 1e6:.1f} MB")
        time.sleep(1)  # be polite to the free API


def build_load_table() -> pd.DataFrame:
    """Germany-level columns from the OPSD file, indexed by UTC timestamp."""
    path = RAW_DIR / "opsd" / OPSD_FILES[0]
    header = pd.read_csv(path, nrows=0).columns
    # Country-level columns only (e.g. DE_load_actual_entsoe_transparency), not
    # TSO zones (DE_50hertz_...) or the DE_LU bidding zone.
    keep = [
        c
        for c in header
        if c.startswith(f"{COUNTRY}_") and c.split("_")[1] in {"load", "solar", "wind"}
    ]
    df = pd.read_csv(
        path, usecols=["utc_timestamp", *keep], parse_dates=["utc_timestamp"]
    )
    df = df.set_index("utc_timestamp").sort_index()
    df.index = df.index.tz_convert("UTC")
    return df.loc[START_DATE:END_DATE]


def build_weather_table() -> pd.DataFrame:
    """Population-weighted national average of hourly weather, plus per-city temperature."""
    frames, weights = {}, {}
    for city, (_, _, w) in CITIES.items():
        payload = json.loads((RAW_DIR / "weather" / f"{city}.json").read_text())
        h = pd.DataFrame(payload["hourly"])
        h["time"] = pd.to_datetime(h["time"], utc=True)
        frames[city] = h.set_index("time")
        weights[city] = w
    total = sum(weights.values())
    weighted = sum(frames[c][WEATHER_VARS] * (weights[c] / total) for c in frames)
    weighted.columns = [f"w_{c}" for c in weighted.columns]
    per_city_temp = pd.DataFrame(
        {f"temp_{c}": f["temperature_2m"] for c, f in frames.items()}
    )
    out = weighted.join(per_city_temp)
    out.index.name = "utc_timestamp"
    return out


def build_interim() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    load = build_load_table()
    weather = build_weather_table()
    load.to_parquet(INTERIM_DIR / "de_load.parquet")
    weather.to_parquet(INTERIM_DIR / "de_weather.parquet")
    print(f"Interim: de_load {load.shape}, de_weather {weather.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download existing files"
    )
    args = parser.parse_args()
    download_opsd(args.force)
    download_weather(args.force)
    build_interim()


if __name__ == "__main__":
    main()
