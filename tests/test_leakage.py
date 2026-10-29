"""
Leakage check: changing anything after time T must not change any feature at or before T.

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import features  # noqa: E402

TZ = "Asia/Taipei"


def make_tables(folder: Path, seed: int = 0, future_noise_after=None):
    """Small fake history: 20 stations, 3 days, every 10 minutes."""
    rng = np.random.default_rng(seed)
    slots = pd.date_range("2026-11-02 00:00", periods=3 * 144, freq="10min", tz=TZ)
    n = 20
    stations = pd.DataFrame({
        "station_id": [f"5001{i:05d}" for i in range(n)],
        "city": "taipei", "name_zh": "x", "name_en": "x", "district_en": "Daan",
        "lat": 25.03 + rng.uniform(0, 0.01, n), "lon": 121.54 + rng.uniform(0, 0.01, n),
        "capacity": 30, "first_seen": slots[0], "last_seen": slots[-1], "weather_point": "p1",
        "weather_point_km": 1.0,
    })
    bikes = rng.integers(0, 30, size=(len(slots), n)).astype(float)
    rain = rng.uniform(0, 3, len(slots))
    if future_noise_after is not None:  # rewrite the future only
        later = slots > future_noise_after
        bikes[later] = rng.integers(0, 30, size=(later.sum(), n))
        rain[later] = rng.uniform(0, 30, later.sum())

    snap = pd.concat([pd.DataFrame({
        "slot": t, "station_id": stations.station_id, "bikes": bikes[i], "docks": 30 - bikes[i],
        "capacity": 30, "is_valid": True}) for i, t in enumerate(slots)])
    obs = pd.DataFrame({"weather_point": "p1", "time": slots, "temperature_2m": 20.0, "precipitation": rain,
                        "relative_humidity_2m": 80, "wind_speed_10m": 3.0, "weather_code": 1})
    fc = pd.DataFrame([{"weather_point": "p1", "issued_at": t, "valid_time": t.floor("h") + pd.Timedelta(hours=h),
                        "precipitation": rain[i], "precipitation_probability": 10, "temperature_2m": 20.0}
                       for i, t in enumerate(slots) for h in range(3)])
    folder.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(folder / "snapshots.parquet")
    stations.to_parquet(folder / "stations.parquet")
    obs.to_parquet(folder / "weather.parquet")
    fc.to_parquet(folder / "weather_forecast.parquet")


def test_future_does_not_change_past_features(tmp_path):
    cutoff = pd.Timestamp("2026-11-03 12:00", tz=TZ)
    make_tables(tmp_path / "a")
    make_tables(tmp_path / "b", future_noise_after=cutoff)

    a = features.build_features(tmp_path / "a")
    b = features.build_features(tmp_path / "b")
    feature_cols = [c for c in a.columns if c not in ("y",)]
    past_a = a[a["slot"] <= cutoff].reset_index(drop=True)
    past_b = b[b["slot"] <= cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(past_a[feature_cols], past_b[feature_cols])
    # the target itself SHOULD differ near the cutoff, which shows the test can detect a change
    assert not past_a["y"].equals(past_b["y"])
