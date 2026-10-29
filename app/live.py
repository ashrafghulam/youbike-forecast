"""
Live forecast: build the features for "right now" with the same code used in training.

1. Fetch the current YouBike + weather data (same functions as the collector).
2. Download the few past snapshots the features need (last 2 hours, ~1 day ago,
   ~1 week ago) from the collector's history on GitHub.
3. Run build_dataset + features on that small set, keep only the latest slot.
"""

import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "collector"))
sys.path.insert(0, str(ROOT / "pipeline"))
import build_dataset  # noqa: E402
import collect  # noqa: E402
import features  # noqa: E402

REPO_RAW = "https://raw.githubusercontent.com/ashrafghulam/youbike-forecast/main/data"
HISTORY_WINDOWS = [            # (start, end) relative to now: which past snapshots to download
    (timedelta(minutes=-130), timedelta(0)),                       # recent trend + chart
    (timedelta(hours=-24, minutes=-20), timedelta(hours=-23)),      # same time yesterday
    (timedelta(days=-7, minutes=-20), timedelta(days=-7, hours=1)),  # same time last week
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": collect.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


class GitHubHistory:
    """Reads the collector history straight from the GitHub repo."""

    def manifest(self, day: str) -> list[dict]:
        try:
            text = _get(f"{REPO_RAW}/manifest/{day}.jsonl").decode("utf-8")
        except Exception:
            return []
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def file(self, rel: str) -> bytes:
        return _get(f"{REPO_RAW}/{rel}")


class LocalHistory:
    """Same interface, reading a local data folder (for testing or running offline)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def manifest(self, day: str) -> list[dict]:
        p = self.data_dir / "manifest" / f"{day}.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def file(self, rel: str) -> bytes:
        return (self.data_dir / rel).read_bytes()


def needed_history(history, now: datetime) -> list[dict]:
    """Manifest entries (successful station + weather fetches) inside the history windows."""
    days = set()
    for a, b in HISTORY_WINDOWS:
        for t in (now + a, now + b):
            days.add(t.strftime("%Y-%m-%d"))
    entries = [e for d in sorted(days) for e in history.manifest(d) if e.get("status") == "ok"]
    keep = []
    for e in entries:
        t = datetime.fromisoformat(e["fetched_at_utc"])
        if any(now + a <= t <= now + b for a, b in HISTORY_WINDOWS):
            keep.append(e)
    return keep


def build_live(history=None, fetch_now=True, now=None):
    """Return (features of the latest slot, recent snapshots for charts, info dict)."""
    history = history or GitHubHistory()
    now = now or datetime.now(timezone.utc)
    info = {"history_files": 0, "live_fetch": {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"
        collect.ROOT = tmp                      # collector writes into the temp folder

        for e in needed_history(history, now):
            dest = tmp / e["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(history.file(e["file"]))
                info["history_files"] += 1
            except Exception:
                pass                            # a missing old file only means a missing lag

        if fetch_now:
            for name, fn in collect.SOURCES.items():
                try:
                    payload, n = fn()
                    collect.save_raw(name, datetime.now(timezone.utc), payload)
                    info["live_fetch"][name] = f"ok ({n})"
                except Exception as err:
                    info["live_fetch"][name] = f"failed: {err!r}"[:200]

        processed = tmp / "processed"
        processed.mkdir()
        raw_snap, _ = build_dataset.load_snapshots(raw)
        stations = build_dataset.build_stations(raw_snap)
        snap = build_dataset.clean_snapshots(raw_snap)
        obs, fc, _ = build_dataset.load_weather(raw)
        snap.to_parquet(processed / "snapshots.parquet", index=False)
        stations.to_parquet(processed / "stations.parquet", index=False)
        obs.to_parquet(processed / "weather.parquet", index=False)
        fc.to_parquet(processed / "weather_forecast.parquet", index=False)

        latest = snap["slot"].max()
        feats = features.build_features(processed, output_slots=[latest])

    recent = snap[snap["slot"] >= latest - pd.Timedelta(hours=2)][["slot", "station_id", "bikes", "docks", "capacity"]]
    feats = feats.merge(stations[["station_id", "name_en", "name_zh"]], on="station_id", how="left")
    info["latest_slot"] = latest
    info["fetched_at"] = snap.loc[snap["slot"] == latest, "fetched_at"].max()
    return feats, recent, info
