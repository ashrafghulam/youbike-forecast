"""
Write the forecast files used by the 3D map website (docs/index.html).

    python web/make_forecast.py            # uses the local data/ folder

Runs inside the collector job after every snapshot. It uses the same code as
training and the app (app/live.py -> build_dataset -> features) and the saved
model, so the website shows exactly what the model predicts.

Outputs (docs/data/):
    stations.json   static station info (only changes when stations change)
    forecast.json   per station: [bikes now, free docks, forecast +1 h, top 3 reasons, last 2 h of counts]
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import live  # noqa: E402

OUT = ROOT / "docs" / "data"
LOW = 2

# Inputs grouped into reasons a rider understands (same groups as pipeline/drivers.py)
REASONS = [
    ("Bikes at the station now", ["bikes_now", "docks_now", "fill_now", "capacity"]),
    ("Trend in the last hour", ["chg_10m", "chg_30m", "chg_60m"]),
    ("Same time yesterday / last week", ["bikes_yday_target", "bikes_week_target", "chg_yday_next_hour"]),
    ("Nearby stations", ["nbr_fill_now", "nbr_chg_30m", "nbr_count"]),
    ("Time of day and weekday", ["target_hour", "dow", "is_weekend", "is_holiday"]),
    ("Location", ["city", "district_en", "lat", "lon"]),
    ("Weather", ["temp_now", "precip_now", "humidity_now", "wind_now",
                 "fc_precip_1h", "fc_precip_prob_1h", "fc_temp_1h"]),
]


def load_model():
    meta = json.loads((ROOT / "models" / "feature_list.json").read_text())
    text = (ROOT / "models" / "lgbm.txt").read_bytes().decode("utf-8").replace("\r\n", "\n")
    return lgb.Booster(model_str=text), meta


def latest_fetch_time(data_dir: Path) -> datetime:
    manifests = sorted((data_dir / "manifest").glob("*.jsonl"))
    for m in reversed(manifests):
        lines = [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
        ok = [x for x in lines if x.get("status") == "ok"]
        if ok:
            return datetime.fromisoformat(ok[-1]["fetched_at_utc"])
    return datetime.now(timezone.utc)


def main():
    data_dir = ROOT / "data"
    now = latest_fetch_time(data_dir)
    feats, recent, info = live.build_live(history=live.LocalHistory(data_dir), fetch_now=False, now=now)
    feats = feats[feats["bikes_now"].notna()].reset_index(drop=True)

    model, meta = load_model()
    X = feats[meta["features"]].copy()
    for c in meta["categorical"]:
        cats = meta["categories"][c]
        v = X[c].astype(str)
        X[c] = pd.Categorical(v.where(v.isin(cats)), categories=cats)

    change = model.predict(X)
    contrib = model.predict(X, pred_contrib=True)[:, :-1]          # per-feature effect on the change, in bikes
    cap = feats["capacity"].to_numpy()
    fc = np.clip(feats["bikes_now"].to_numpy() + change, 0, cap)

    idx = {f: i for i, f in enumerate(meta["features"])}
    group_vals = np.column_stack([contrib[:, [idx[f] for f in cols if f in idx]].sum(axis=1)
                                  for _, cols in REASONS])

    # last 2 hours of counts per station, on a fixed 10-minute grid (None = no reading)
    last = recent["slot"].max()
    hist_slots = pd.date_range(last - pd.Timedelta(hours=2), last, freq="10min")
    hist = (recent.pivot_table(index="station_id", columns="slot", values="bikes", aggfunc="last")
            .reindex(columns=hist_slots))

    stations, forecast = [], {}
    for i, r in feats.iterrows():
        sid = r["station_id"]
        stations.append({
            "id": sid, "en": r.get("name_en") or sid, "zh": r.get("name_zh") or "",
            "d": str(r["district_en"]), "city": str(r["city"]),
            "lat": round(float(r["lat"]), 5), "lon": round(float(r["lon"]), 5), "cap": int(cap[i]),
        })
        top = np.argsort(-np.abs(group_vals[i]))[:3]
        docks = r["docks_now"]
        forecast[sid] = [
            int(r["bikes_now"]),
            None if pd.isna(docks) else int(docks),
            round(float(fc[i]), 1),
            [[int(g), round(float(group_vals[i, g]), 1)] for g in top],
            [None if pd.isna(v) else int(v) for v in
             (hist.loc[sid].to_numpy() if sid in hist.index else [np.nan] * len(hist_slots))],
        ]

    OUT.mkdir(parents=True, exist_ok=True)
    stations.sort(key=lambda s: s["id"])
    st_text = json.dumps({"stations": stations}, ensure_ascii=False, separators=(",", ":"))
    st_path = OUT / "stations.json"
    if not st_path.exists() or st_path.read_text(encoding="utf-8") != st_text:
        st_path.write_text(st_text, encoding="utf-8")

    data_time = pd.Timestamp(info["fetched_at"]).tz_convert("Asia/Taipei")
    out = {
        "data_time": data_time.isoformat(timespec="minutes"),
        "forecast_time": (data_time + pd.Timedelta(hours=1)).isoformat(timespec="minutes"),
        "reasons": [name for name, _ in REASONS],
        "hist_slots": [t.isoformat(timespec="minutes") for t in hist_slots],
        "low": LOW,
        "s": forecast,
    }
    (OUT / "forecast.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"forecast for {len(forecast)} stations, data {data_time:%Y-%m-%d %H:%M} (Taipei)")


if __name__ == "__main__":
    main()
