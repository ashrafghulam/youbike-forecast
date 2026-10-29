"""
Turn the clean tables into a modelling table.

    python pipeline/features.py        # data/processed/*.parquet -> data/processed/features.parquet

One row = one station at one 10-minute slot t (the moment we make a forecast).
Target = bikes at the same station at t + 60 minutes.

Every feature only uses information that was available at time t:
current and past bike counts, the calendar, the weather observed at t,
and the weather *forecast* for t+60 as published at t.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent.parent
TZ = "Asia/Taipei"
SLOT_MIN = 10
HORIZON = 60 // SLOT_MIN          # 6 slots = 1 hour ahead
DAY = 24 * 60 // SLOT_MIN         # 144 slots
WEEK = 7 * DAY
N_NEIGHBOURS = 5
MAX_NEIGHBOUR_KM = 1.0
FILL_GAP_SLOTS = 2                # features may carry a value forward over a gap of up to 20 min

# Taiwan national holidays and make-up days inside or near the collection window.
# Add more if the collection is extended.
HOLIDAYS = pd.to_datetime([]).date


def pivot(snap: pd.DataFrame, col: str, slots: pd.DatetimeIndex, stations: pd.Index) -> pd.DataFrame:
    """Slot x station matrix, with NaN where there is no valid reading."""
    m = snap.pivot_table(index="slot", columns="station_id", values=col, aggfunc="last")
    return m.reindex(index=slots, columns=stations)


def neighbour_matrix(stations: pd.DataFrame) -> sparse.csr_matrix:
    """Row-normalised matrix: each station averages its nearest stations (within 1 km)."""
    coords = np.radians(stations[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    dist, idx = tree.query(coords, k=N_NEIGHBOURS + 1)      # first hit is the station itself
    dist_km = dist[:, 1:] * 6371.0
    idx = idx[:, 1:]
    keep = dist_km <= MAX_NEIGHBOUR_KM
    rows = np.repeat(np.arange(len(stations)), N_NEIGHBOURS)[keep.ravel()]
    cols = idx.ravel()[keep.ravel()]
    w = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(stations),) * 2)
    return w


def neighbour_mean(values: pd.DataFrame, w: sparse.csr_matrix) -> pd.DataFrame:
    """Average of neighbours' values, ignoring neighbours with no reading."""
    v = values.values
    have = ~np.isnan(v)
    total = sparse.csr_matrix.dot(w, np.where(have, v, 0).T).T
    count = sparse.csr_matrix.dot(w, have.T.astype(float)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def stack(m: pd.DataFrame, name: str) -> pd.Series:
    s = m.stack(future_stack=True)
    s.index.names = ["slot", "station_id"]
    return s.rename(name).astype("float32")


def weather_features(slots: pd.DatetimeIndex, points, obs: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Weather per (weather_point, slot) using only what was known at the slot:
    - the latest observation at or before t (within 1 hour)
    - the forecast for the hour containing t+60 min, from the latest fetch at or before t
    """
    grid = pd.MultiIndex.from_product([points, slots], names=["weather_point", "slot"]).to_frame(index=False)
    grid["target_hour"] = (grid["slot"] + pd.Timedelta(minutes=60)).dt.floor("h")
    grid = grid.sort_values("slot")

    o = obs[["weather_point", "time", "temperature_2m", "precipitation", "relative_humidity_2m",
             "wind_speed_10m", "weather_code"]].rename(columns={
        "time": "obs_time", "temperature_2m": "temp_now", "precipitation": "precip_now",
        "relative_humidity_2m": "humidity_now", "wind_speed_10m": "wind_now", "weather_code": "weather_code_now"})
    grid = pd.merge_asof(grid, o.sort_values("obs_time"), left_on="slot", right_on="obs_time",
                         by="weather_point", direction="backward", tolerance=pd.Timedelta("60min"))

    f = fc[["weather_point", "issued_at", "valid_time", "precipitation", "precipitation_probability",
            "temperature_2m"]].rename(columns={
        "valid_time": "target_hour", "precipitation": "fc_precip_1h",
        "precipitation_probability": "fc_precip_prob_1h", "temperature_2m": "fc_temp_1h"})
    f["issued_slot"] = f["issued_at"].dt.floor(f"{SLOT_MIN}min")
    grid = pd.merge_asof(grid, f.drop(columns="issued_at").sort_values("issued_slot"), left_on="slot",
                         right_on="issued_slot", by=["weather_point", "target_hour"], direction="backward",
                         tolerance=pd.Timedelta("60min"))

    keep = ["slot", "weather_point", "temp_now", "precip_now", "humidity_now", "wind_now", "weather_code_now",
            "fc_precip_1h", "fc_precip_prob_1h", "fc_temp_1h"]
    return grid[keep]


def build_features(processed: Path) -> pd.DataFrame:
    snap = pd.read_parquet(processed / "snapshots.parquet")
    stations = pd.read_parquet(processed / "stations.parquet").set_index("station_id")
    obs = pd.read_parquet(processed / "weather.parquet")
    fc = pd.read_parquet(processed / "weather_forecast.parquet")

    snap = snap[snap["is_valid"]]
    stations = stations.loc[stations.index.isin(snap["station_id"].unique())]
    slots = pd.date_range(snap["slot"].min(), snap["slot"].max(), freq=f"{SLOT_MIN}min")
    sids = stations.index

    raw = pivot(snap, "bikes", slots, sids)                       # exactly as observed
    bikes = raw.ffill(limit=FILL_GAP_SLOTS)                       # for features only
    docks = pivot(snap, "docks", slots, sids).ffill(limit=FILL_GAP_SLOTS)
    cap = stations["capacity"].reindex(sids).values
    fill = bikes / cap

    w = neighbour_matrix(stations.reset_index())
    nbr_fill = neighbour_mean(fill, w)

    cols = {
        # target: what we want to predict (never forward-filled)
        "y": raw.shift(-HORIZON),
        # current state
        "bikes_now": bikes,
        "docks_now": docks,
        "fill_now": fill,
        # recent trend
        "chg_10m": bikes - bikes.shift(1),
        "chg_30m": bikes - bikes.shift(3),
        "chg_60m": bikes - bikes.shift(6),
        # the target time, one day and one week earlier (seasonal memory)
        "bikes_yday_target": bikes.shift(DAY - HORIZON),
        "bikes_week_target": bikes.shift(WEEK - HORIZON),
        "chg_yday_next_hour": bikes.shift(DAY - HORIZON) - bikes.shift(DAY),
        # neighbouring stations
        "nbr_fill_now": nbr_fill,
        "nbr_chg_30m": nbr_fill - nbr_fill.shift(3),
    }
    df = pd.concat([stack(m, k) for k, m in cols.items()], axis=1).reset_index()

    # calendar, described at the target time (t + 60 min)
    target_time = df["slot"] + pd.Timedelta(minutes=60)
    df["target_hour"] = (target_time.dt.hour + target_time.dt.minute / 60).astype("float32")
    df["dow"] = target_time.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_holiday"] = target_time.dt.date.isin(HOLIDAYS).astype("int8")

    # station properties
    st = stations[["city", "district_en", "lat", "lon", "capacity", "weather_point"]].reset_index()
    st["city"] = st["city"].astype("category")
    st["district_en"] = st["district_en"].astype("category")
    st["nbr_count"] = np.asarray(w.sum(axis=1)).ravel().astype("int8")
    df = df.merge(st, on="station_id", how="left")

    # weather
    wf = weather_features(slots, stations["weather_point"].unique(), obs, fc)
    df = df.merge(wf, on=["slot", "weather_point"], how="left")

    df["date"] = df["slot"].dt.date
    return df.drop(columns=["weather_point"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    df = build_features(args.processed)
    df.to_parquet(args.processed / "features.parquet", index=False)
    print(f"features: {len(df):,} rows, {df['station_id'].nunique()} stations, "
          f"{df['slot'].nunique()} slots, rows with target: {df['y'].notna().sum():,}")


if __name__ == "__main__":
    main()
