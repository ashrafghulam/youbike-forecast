"""
YouBike + weather snapshot collector.

Run once per schedule tick (GitHub Actions cron, or system cron).
Each run:
  1. fetches every source (Taipei YouBike, New Taipei YouBike, Open-Meteo weather)
  2. retries each source with exponential backoff
  3. saves the raw response *unchanged* as gzip, named by UTC fetch time
  4. appends one line per source to a daily manifest (status, counts, errors)

Design choices:
  - Raw responses are stored as-is. Parsing happens later, so a parsing bug
    never loses data and the dataset can always be rebuilt from raw history.
  - Sources are independent: one failing does not stop the others.
  - Only the Python standard library is used, so the job has nothing to install.
  - Exit code is 1 if any source failed, so the failure is visible in the
    scheduler, but successful sources are still saved.
"""

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
USER_AGENT = "youbike-forecast-collector/1.0 (research take-home)"
TIMEOUT = 30           # seconds per HTTP request
MAX_ATTEMPTS = 4       # per request
BACKOFF_BASE = 3       # seconds: 3, 6, 12 ...

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NTPC_URL = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json"
NTPC_PAGE_SIZE = 1000
NTPC_MAX_PAGES = 20    # safety stop

# Weather points: a coarse grid over Taipei + New Taipei urban area.
# Open-Meteo accepts several coordinates in one call.
WEATHER_POINTS = [
    (25.033, 121.565),  # Xinyi / Taipei 101
    (25.047, 121.517),  # Taipei Main Station
    (25.080, 121.525),  # Shilin / Zhongshan north
    (25.013, 121.535),  # Gongguan / NTU
    (25.062, 121.600),  # Neihu / Songshan
    (25.013, 121.465),  # Banqiao (New Taipei)
    (25.060, 121.460),  # Sanchong / Xinzhuang (New Taipei)
    (24.993, 121.510),  # Zhonghe / Yonghe (New Taipei)
    (24.975, 121.540),  # Xindian (New Taipei)
]
WEATHER_VARS = "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,cloud_cover"


def http_get(url: str, params: dict | None = None) -> bytes:
    """GET with retries and exponential backoff. Raises the last error."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read()
            json.loads(body)  # make sure it is valid JSON before we accept it
            return body
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as e:
            last_err = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
    raise RuntimeError(f"failed after {MAX_ATTEMPTS} attempts: {last_err!r}")


def save_raw(source: str, ts: datetime, payload: bytes) -> Path:
    """Write gzip atomically: temp file, then rename, so a crash never leaves half a file."""
    day = ts.strftime("%Y-%m-%d")
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "raw" / source / day / f"{stamp}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(payload)
    tmp.replace(out)
    return out


def log_manifest(ts: datetime, record: dict) -> None:
    path = ROOT / "manifest" / f"{ts.strftime('%Y-%m-%d')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- sources ----------

def fetch_taipei() -> tuple[bytes, int]:
    body = http_get(TAIPEI_URL)
    data = json.loads(body)
    n = len(data) if isinstance(data, list) else len(data.get("retVal", []))
    return body, n


def fetch_ntpc() -> tuple[bytes, int]:
    """New Taipei API is paged; collect all pages into one JSON list."""
    rows = []
    for page in range(NTPC_MAX_PAGES):
        body = http_get(NTPC_URL, {"page": page, "size": NTPC_PAGE_SIZE})
        chunk = json.loads(body)
        if not isinstance(chunk, list):
            raise RuntimeError(f"unexpected NTPC response type: {type(chunk).__name__}")
        rows.extend(chunk)
        if len(chunk) < NTPC_PAGE_SIZE:
            break
    return json.dumps(rows, ensure_ascii=False).encode("utf-8"), len(rows)


def fetch_weather() -> tuple[bytes, int]:
    """Current conditions + next 3 hours of forecast for each grid point.
    The forecast is what a live system would actually know at prediction time."""
    params = {
        "latitude": ",".join(str(p[0]) for p in WEATHER_POINTS),
        "longitude": ",".join(str(p[1]) for p in WEATHER_POINTS),
        "current": WEATHER_VARS,
        "hourly": WEATHER_VARS + ",precipitation_probability",
        "forecast_hours": 3,
        "timezone": "Asia/Taipei",
    }
    body = http_get("https://api.open-meteo.com/v1/forecast", params)
    data = json.loads(body)
    return body, len(data) if isinstance(data, list) else 1


SOURCES = {
    "taipei": fetch_taipei,
    "ntpc": fetch_ntpc,
    "weather": fetch_weather,
}


def main() -> int:
    only = set(sys.argv[1:])  # optional: run a subset, e.g. `python collect.py taipei`
    failed = []
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        ts = datetime.now(timezone.utc)
        t0 = time.monotonic()
        rec = {"fetched_at_utc": ts.isoformat(timespec="seconds"), "source": name}
        try:
            payload, n = fn()
            path = save_raw(name, ts, payload)
            rec.update(status="ok", records=n, bytes=len(payload), file=str(path.relative_to(ROOT)))
            print(f"[ok]   {name:8s} {n:5d} records  {len(payload)/1024:8.1f} KB  -> {path.relative_to(ROOT)}")
        except Exception as e:  # never let one source kill the run
            rec.update(status="error", error=repr(e)[:500])
            failed.append(name)
            print(f"[FAIL] {name:8s} {e!r}", file=sys.stderr)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        log_manifest(ts, rec)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
