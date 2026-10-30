# Dataset

This describes how the raw snapshots become the tables used for modelling, what each column means, and the known problems in the data.

## Rebuilding from scratch

```bash
pip install -r requirements.txt
python pipeline/build_dataset.py
```

The script reads `data/raw/` (never modifies it) and writes `data/processed/`. `data/processed/` is not committed to git, because anyone can rebuild it from the raw files with the command above, and running it twice on the same raw files gives the same tables.

## Sources

| Source | What it gives | Update rate at source |
|---|---|---|
| YouBike Taipei live JSON | bikes and free docks for ~1,820 Taipei stations | about every minute |
| New Taipei open data API | same for ~1,610 New Taipei stations, plus e-bike count | about every minute |
| Open-Meteo forecast API | weather now + next 3 hours, 24 points | every 15 minutes |

The collector takes a snapshot of all three every 10 minutes (see the README for the pipeline).

## Time conventions

- File names in `data/raw/` use **UTC** fetch time, e.g. `20261029T184948Z.json.gz`.
- Every timestamp in the processed tables is **Asia/Taipei** time and timezone-aware.
- `slot` is the fetch time rounded **down** to 10 minutes. It is the time index used for modelling. A row in slot 08:10 means "what the station looked like at some point between 08:10 and 08:20".

## Tables

### `snapshots.parquet`: one row per station per 10-minute slot

| column | meaning |
|---|---|
| `slot` | 10-minute time slot (Taipei time) |
| `fetched_at` | exact time the collector downloaded the data |
| `station_time` | time the station itself last reported (`mday` in the feeds) |
| `staleness_min` | `fetched_at - station_time`, in minutes |
| `city` | `taipei` or `new_taipei` |
| `station_id` | YouBike station number (`sno`). Taipei starts with `5001`, New Taipei with `5002` |
| `capacity` | total docks at the station |
| `bikes` | bikes available to rent. **This is the prediction target.** |
| `docks` | empty docks available to return a bike |
| `ebikes` | of `bikes`, how many are e-bikes (New Taipei only, empty for Taipei) |
| `unavailable` | `capacity - bikes - docks` (see known issues) |
| `fill_ratio` | `bikes / capacity` |
| `active` | station is in service (`act == 1`) |
| `flag_inactive`, `flag_stale`, `flag_bad_count` | quality flags, explained below |
| `is_valid` | no flag raised. Use this to filter for training. |

### `stations.parquet`: one row per station

`station_id`, `city`, `name_zh`, `name_en`, `district_en`, `lat`, `lon`, `capacity`, `first_seen`, `last_seen`, `weather_point` (nearest weather grid point) and `weather_point_km` (distance to it).

Names and capacity come from the station's most recent reading. The "YouBike2.0_" prefix is removed from names.

### `weather.parquet`: observed weather

One row per weather point per 15-minute observation. Open-Meteo sometimes revises the value for a period on a later fetch; the **first** version is kept, because that is what was known at the time. Columns: `temperature_2m` (°C), `relative_humidity_2m` (%), `apparent_temperature` (°C), `precipitation` and `rain` (mm), `weather_code` (WMO code), `wind_speed_10m` (km/h), `cloud_cover` (%).

### `weather_forecast.parquet`: forecast as known at the time

One row per weather point × fetch (`issued_at`) × forecast hour (`valid_time`), with the same variables plus `precipitation_probability`.

This table exists to avoid leakage. A model running live at 08:00 can't know the weather actually observed at 09:00, only what was forecast for 09:00 at 08:00. Using observed future weather in training would make the results look better than they could be in real use.

## Weather points

The collector asks Open-Meteo for 24 points: one per 0.1° (~10 km) grid cell that contains YouBike stations, placed at the average position of those stations (`collector/weather_points.json`). Every station is within 6.6 km of a point (2.9 km on average). Rain in Taipei is often very local, so a single city-wide value would hide a lot.

The very first snapshot (30 Oct 2026, 02:49 Taipei time) used 9 hand-picked points that covered central Taipei only. The builder reads both formats. Those weather rows keep their old point names and don't match any station's `weather_point`, so they are effectively unused.

## Cleaning rules

Rows are **flagged, not deleted**, so the decision of what to exclude is visible and can be changed later.

| flag | rule | why |
|---|---|---|
| `flag_inactive` | `act != 1` | station out of service; its counts mean nothing |
| `flag_stale` | `staleness_min > 30` | the station stopped reporting, so the "current" count is old |
| `flag_bad_count` | negative bikes or docks, or capacity ≤ 0 | impossible values |

Other steps:
- Rows with no station id, bike count or capacity are dropped (none so far).
- If two fetches fall in the same slot (for example a manual run next to a scheduled one), the later fetch is kept.
- A raw file that can't be read is listed in `quality_report.md` instead of stopping the build.

## Known issues

- **bikes + docks is often less than capacity.** In the first snapshot this was true for ~47% of stations. It is normal for YouBike: docks can be broken or locked, and bikes can be under repair. It is kept as `unavailable` and not treated as an error.
- **Scheduled runs can start late.** GitHub Actions doesn't guarantee exact timing, so some slots will be empty and the gap between two snapshots is not always exactly 10 minutes. `quality_report.md` lists every missing slot.
- **Station list changes.** Stations can be added, removed or renamed during collection. `first_seen` and `last_seen` show this.
- **New Taipei numbers arrive as text** in the API and are converted to numbers. A value that fails to convert becomes empty.
- **No history before 30 Oct 2026.** The dataset only covers the collection period, so it contains no holidays and only a small range of weather.

## Quality report

Each build writes `data/processed/quality_report.md` with the period covered, the number of stations and rows per city, the share of valid rows, flag counts, every missing time slot and any unreadable raw file.
