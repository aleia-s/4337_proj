import json
import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import BLS_PARAMS

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# map JSON keys → final column names
COLUMN_RENAMES = {
    "emp_all":   "all_employees_thousands",
    "hrs_all":   "avg_weekly_hours_all",
    "earn_all":  "avg_hourly_earnings_all",
    "emp_prod":  "production_employees_thousands",
    "hrs_prod":  "avg_weekly_hours_production",
    "earn_prod": "avg_hourly_earnings_production",
    "unrate":    "unemployment_rate",
    "open":      "openings_thousands",
    "hires":     "hires_thousands",
    "sep":       "separations_thousands",
}

def build_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def fetch_bls_series(series_ids, start_year, end_year, api_key, session):
    payload = {
        "seriesid":        series_ids,
        "startyear":       str(start_year),
        "endyear":         str(end_year),
        "registrationkey": api_key
    }
    resp = session.post(BLS_API_URL, json=payload, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    series_list = resp.json().get("Results", {}).get("series", [])
    if not series_list:
        raise RuntimeError(f"No data for {series_ids}: {resp.text}")

    dfs = []
    for s in series_list:
        sid = s["seriesID"]
        rows = [
            {
                "date": pd.to_datetime(f"{item['year']}-{item['period'][1:]}-01"),
                sid: float(item["value"])
            }
            for item in s["data"] if item["period"].startswith("M")
        ]
        if rows:
            df = pd.DataFrame(rows).set_index("date")
            dfs.append(df)
    # align on months where all series have values
    return pd.concat(dfs, axis=1).dropna()


def main():
    # -- load everything from your central config.py --
    start_year = BLS_PARAMS["start_year"]
    end_year   = BLS_PARAMS["end_year"]
    api_key    = BLS_PARAMS["api_key"]
    codes_file = Path(BLS_PARAMS["series_codes_file"])
    output_csv = Path(BLS_PARAMS["output_csv"])

    # read your industries JSON
    codes = json.loads(codes_file.read_text())["industries"]

    session = build_session()
    parts = []

    for industry, metrics in codes.items():
        try:
            df = fetch_bls_series(
                series_ids=list(metrics.values()),
                start_year=start_year,
                end_year=end_year,
                api_key=api_key,
                session=session
            )
        except Exception as e:
            logger.error("Error fetching %s: %s", industry, e)
            continue

        # 1) series‐ID → JSON key (emp_all, hrs_all, …)
        df = df.rename(columns={v: k for k, v in metrics.items()})
        # 2) JSON key → final human name
        df = df.rename(columns=COLUMN_RENAMES)
        # 3) make y = all_employees_thousands
        df = df.rename(columns={"all_employees_thousands": "y"})

        df["industry"] = industry
        df = df.reset_index()

        cols = [
            "date", "industry", "y",
            "avg_weekly_hours_all", "avg_hourly_earnings_all",
            "production_employees_thousands",
            "avg_weekly_hours_production", "avg_hourly_earnings_production",
            "unemployment_rate",
            "openings_thousands", "hires_thousands", "separations_thousands"
        ]
        parts.append(df[cols])

    # stitch all industries together
    master = pd.concat(parts, ignore_index=True)
    master.sort_values(["date", "industry"], inplace=True)

    # simply write to your configured path
    master.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows × %d cols to %s", *master.shape, output_csv)


if __name__ == "__main__":
    main()
