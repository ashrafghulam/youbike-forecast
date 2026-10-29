"""
Build clean tables from the raw snapshot history.

    python pipeline/build_dataset.py            # reads data/raw, writes data/processed

Nothing in data/raw is ever changed. Delete data/processed and run this again
to get exactly the same tables back.

Outputs (data/processed/):
    snapshots.parquet         one row per station per 10-minute slot
    stations.parquet          one row per station (location, capacity, nearest weather point)
    weather.parquet           observed weather per point, every 15 minutes
    weather_forecast.parquet  hourly forecast as it was known at each fetch
    quality_report.md         row counts, gaps and flags
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TZ = "Asia/Taipei"
SLOT = "10min"          # the collector runs every 10 minutes
STALE_MINUTES = 30      # station reading older than this at fetch time is flagged

# Weather grid used by the collector (one point per ~10 km cell with stations).
WEATHER_GRID = json.loads((ROOT / "collector" / "weather_points.json").read_text())

# The first file (30 Oct 2026, 02:49 Taipei) used 9 fixed points and stored
# only the raw response; these are their names, in request order.
LEGACY_POINTS = [
    ("xinyi", 25.033, 121.565),
    ("main_station", 25.047, 121.517),
    ("shilin", 25.080, 121.525),
    ("gongguan", 25.013, 121.535),
    ("neihu", 25.062, 121.600),
    ("banqiao", 25.013, 121.465),
    ("sanchong", 25.060, 121.460),
    ("zhonghe", 24.993, 121.510),
    ("xindian", 24.975, 121.540),
]


def read_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def fetch_time(path: Path) -> pd.Timestamp:
    # file name is the UTC fetch time, e.g. 20261029T184948Z.json.gz
    return pd.to_datetime(path.name.split(".")[0], format="%Y%m%dT%H%M%SZ", utc=True).tz_convert(TZ)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- stations

def parse_taipei(rows, fetched_at):
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "fetched_at": fetched_at,
        "city": "taipei",
        "station_id": df["sno"].astype(str),
        "name_zh": df["sna"].str.replace("YouBike2.0_", "", regex=False),
        "name_en": df["snaen"].str.replace("YouBike2.0_", "", regex=False),
        "district_en": df["sareaen"].str.replace(" Dist.", "", regex=False).str.strip(),
        "lat": to_num(df["latitude"]),
        "lon": to_num(df["longitude"]),
        "capacity": to_num(df["Quantity"]),
        "bikes": to_num(df["available_rent_bikes"]),
        "docks": to_num(df["available_return_bikes"]),
        "ebikes": np.nan,  # Taipei feed does not split e-bikes
        "active": df["act"].astype(str) == "1",
        "station_time": pd.to_datetime(df["mday"], errors="coerce").dt.tz_localize(TZ),
    })


def parse_ntpc(rows, fetched_at):
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "fetched_at": fetched_at,
        "city": "new_taipei",
        "station_id": df["sno"].astype(str),
        "name_zh": df["sna"].str.replace("YouBike2.0_", "", regex=False),
        "name_en": df["snaen"].str.replace("YouBike2.0_", "", regex=False),
        "district_en": df["sareaen"].str.replace(" Dist", "", regex=False).str.strip(),
        "lat": to_num(df["lat"]),
        "lon": to_num(df["lng"]),
        "capacity": to_num(df["tot_quantity"]),
        "bikes": to_num(df["sbi_quantity"]),
        "docks": to_num(df["bemp"]),
        "ebikes": to_num(df["eyb_quantity"]),
        "active": df["act"].astype(str) == "1",
        "station_time": pd.to_datetime(df["mday"], format="%Y%m%dT%H%M%S", errors="coerce").dt.tz_localize(TZ),
    })


PARSERS = {"taipei": parse_taipei, "ntpc": parse_ntpc}


def load_snapshots(raw: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, bad = [], []
    for source, parser in PARSERS.items():
        for path in sorted((raw / source).glob("*/*.json.gz")):
            try:
                frames.append(parser(read_gz(path), fetch_time(path)))
            except Exception as e:  # a corrupt file is reported, not fatal
                bad.append(f"{path.relative_to(raw)}: {e!r}")
    if not frames:
        raise SystemExit("no readable station files found:\n" + "\n".join(bad[:10]))
    return pd.concat(frames, ignore_index=True), bad


def clean_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["station_id", "bikes", "capacity"]).copy()
    df["slot"] = df["fetched_at"].dt.floor(SLOT)

    # If two fetches land in the same slot (e.g. a manual run), keep the later one.
    df = df.sort_values("fetched_at").drop_duplicates(["station_id", "slot"], keep="last")

    df["staleness_min"] = (df["fetched_at"] - df["station_time"]).dt.total_seconds() / 60
    # bikes + docks is often less than capacity (docks out of service, bikes under repair)
    df["unavailable"] = df["capacity"] - df["bikes"] - df["docks"]
    df["fill_ratio"] = (df["bikes"] / df["capacity"]).where(df["capacity"] > 0)

    df["flag_inactive"] = ~df["active"]
    df["flag_stale"] = df["staleness_min"] > STALE_MINUTES
    df["flag_bad_count"] = (df["bikes"] < 0) | (df["docks"] < 0) | (df["capacity"] <= 0)
    df["is_valid"] = ~(df["flag_inactive"] | df["flag_stale"] | df["flag_bad_count"])

    cols = ["slot", "fetched_at", "station_time", "staleness_min", "city", "station_id",
            "capacity", "bikes", "docks", "ebikes", "unavailable", "fill_ratio",
            "active", "flag_inactive", "flag_stale", "flag_bad_count", "is_valid"]
    return df[cols].sort_values(["station_id", "slot"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_stations(raw_snap: pd.DataFrame) -> pd.DataFrame:
    """Static info per station, taken from its most recent reading."""
    last = raw_snap.sort_values("fetched_at").groupby("station_id").tail(1)
    seen = raw_snap.groupby("station_id")["fetched_at"].agg(first_seen="min", last_seen="max")
    st = last[["station_id", "city", "name_zh", "name_en", "district_en", "lat", "lon", "capacity"]]
    st = st.merge(seen, on="station_id").reset_index(drop=True)

    pts = np.array([(p["lat"], p["lon"]) for p in WEATHER_GRID])
    d = haversine_km(st["lat"].values[:, None], st["lon"].values[:, None], pts[:, 0], pts[:, 1])
    st["weather_point"] = [WEATHER_GRID[i]["id"] for i in d.argmin(axis=1)]
    st["weather_point_km"] = d.min(axis=1).round(2)
    return st.sort_values("station_id").reset_index(drop=True)


# ---------------------------------------------------------------- weather

def load_weather(raw: Path):
    obs, fc, bad = [], [], []
    for path in sorted((raw / "weather").glob("*/*.json.gz")):
        try:
            data = read_gz(path)
            issued = fetch_time(path)
            if isinstance(data, dict):            # current format
                names = [p["id"] for p in data["points"]]
                data = data["response"]
            else:                                 # legacy 9-point format
                names = [p[0] for p in LEGACY_POINTS]
            for name, loc in zip(names, data):
                cur = dict(loc["current"])
                cur.pop("interval", None)
                cur["time"] = pd.Timestamp(cur["time"]).tz_localize(TZ)
                obs.append({"weather_point": name, **cur})
                h = loc["hourly"]
                for i, t in enumerate(h["time"]):
                    row = {k: v[i] for k, v in h.items() if k != "time"}
                    fc.append({"weather_point": name, "issued_at": issued,
                               "valid_time": pd.Timestamp(t).tz_localize(TZ), **row})
        except Exception as e:
            bad.append(f"{path.relative_to(raw)}: {e!r}")

    obs = pd.DataFrame(obs).drop_duplicates(["weather_point", "time"], keep="last")
    fc = pd.DataFrame(fc)
    return obs.sort_values(["weather_point", "time"]).reset_index(drop=True), fc, bad


# ---------------------------------------------------------------- report

def gap_summary(snap: pd.DataFrame) -> tuple[int, int, list[str]]:
    slots = snap["slot"].drop_duplicates().sort_values()
    if len(slots) < 2:
        return len(slots), 0, []
    expected = pd.date_range(slots.iloc[0], slots.iloc[-1], freq=SLOT)
    missing = expected.difference(slots)
    # group missing slots into runs for readability
    runs, start, prev = [], None, None
    for t in missing:
        if start is None:
            start = prev = t
        elif t - prev == pd.Timedelta(SLOT):
            prev = t
        else:
            runs.append((start, prev)); start = prev = t
    if start is not None:
        runs.append((start, prev))
    text = [f"{a:%m-%d %H:%M} to {b:%m-%d %H:%M} ({int((b - a) / pd.Timedelta(SLOT)) + 1} slots)" for a, b in runs]
    return len(expected), len(missing), text


def write_report(path, snap, stations, obs, bad):
    n_exp, n_miss, runs = gap_summary(snap)
    by_city = snap.groupby("city").agg(stations=("station_id", "nunique"), rows=("station_id", "size"),
                                       valid_pct=("is_valid", "mean"))
    lines = [
        "# Data quality report",
        "",
        f"Built: {pd.Timestamp.now(tz=TZ):%Y-%m-%d %H:%M} (Taipei time)",
        f"Period: {snap['slot'].min():%Y-%m-%d %H:%M} to {snap['slot'].max():%Y-%m-%d %H:%M} (Taipei time)",
        "",
        "## Coverage",
        "",
        f"- 10-minute slots expected: {n_exp}, missing: {n_miss}"
        + (f" ({n_miss / n_exp:.1%})" if n_exp else ""),
        f"- Stations: {stations.shape[0]}",
        f"- Station-slot rows: {len(snap):,}",
        f"- Weather observations: {len(obs):,}",
        "",
        "| city | stations | rows | valid |",
        "|---|---|---|---|",
    ]
    for city, r in by_city.iterrows():
        lines.append(f"| {city} | {int(r.stations)} | {int(r.rows):,} | {r.valid_pct:.1%} |")
    lines += [
        "",
        "## Flags (rows are kept, only marked)",
        "",
        f"- inactive station: {int(snap['flag_inactive'].sum()):,}",
        f"- stale reading (> {STALE_MINUTES} min old at fetch): {int(snap['flag_stale'].sum()):,}",
        f"- impossible counts: {int(snap['flag_bad_count'].sum()):,}",
        f"- bikes + docks < capacity (normal, broken or reserved docks): "
        f"{(snap['unavailable'] > 0).mean():.1%} of rows",
        "",
        "## Missing time slots",
        "",
    ]
    lines += [f"- {r}" for r in runs[:50]] or ["- none"]
    if len(runs) > 50:
        lines.append(f"- ... and {len(runs) - 50} more")
    lines += ["", "## Unreadable raw files", ""] + ([f"- {b}" for b in bad] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_snap, bad1 = load_snapshots(args.raw)
    stations = build_stations(raw_snap)
    snap = clean_snapshots(raw_snap)
    obs, fc, bad2 = load_weather(args.raw)

    snap.to_parquet(args.out / "snapshots.parquet", index=False)
    stations.to_parquet(args.out / "stations.parquet", index=False)
    obs.to_parquet(args.out / "weather.parquet", index=False)
    fc.to_parquet(args.out / "weather_forecast.parquet", index=False)
    write_report(args.out / "quality_report.md", snap, stations, obs, bad1 + bad2)

    print(f"snapshots: {len(snap):,} rows | stations: {len(stations)} | "
          f"weather obs: {len(obs)} | forecasts: {len(fc)} | unreadable files: {len(bad1) + len(bad2)}")


if __name__ == "__main__":
    main()
