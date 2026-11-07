# YouBike 1-hour-ahead availability forecast

Predict how many bikes a Taipei / New Taipei YouBike station will have one hour from now, and explain what drives it.

## Results in short

- **Data:** 9 days (30 Oct – 7 Nov 2026), every 10 minutes, 3,446 stations in Taipei and New Taipei plus weather at 24 points. 4.4 million station readings, 0.4% of time slots missing, no failed fetches. Collected by GitHub Actions; collection stopped on 8 Nov 2026 01:20.
- **Model:** LightGBM, one model for all stations, predicting the change in bikes over the next hour.
- **Validation:** one fold per day, always trained on earlier days only, 7 test days (1–7 Nov), 3.4 million test forecasts.

| | MAE (bikes) | vs "same as now" |
|---|---|---|
| same as now (naive baseline) | 2.81 | – |
| random forest | 2.63 | +6% |
| **LightGBM + night rule** | **2.48** | **+12%** |

Better than the baseline on all 7 test days (+5% to +18%, growing with more training days) and at every hour from 06:00 to 01:00, most at rush hours (+28% at 07:00). Two warning models catch 78% of stations that run empty within the hour (baseline 64%).

- **Drivers:** the current count, then the daily routine (how the station changed at the same hour yesterday), time of day, and neighbouring stations. Weather barely matters in this data because it hardly rained.
- **What-if:** if nearby stations empty fast, about 90 more stations (of 3,399) are expected to be nearly empty at the evening rush.

Details: [docs/MODEL.md](docs/MODEL.md). Data: [docs/DATASET.md](docs/DATASET.md). Website and app: [docs/APP.md](docs/APP.md).

## Project layout

| folder | contents |
|---|---|
| `collector/`, `.github/workflows/` | the collector and its schedule |
| `data/raw/`, `data/manifest/` | every raw snapshot, unchanged, and a log of every fetch |
| `pipeline/` | dataset build, features, training and validation, drivers |
| `models/` | final models (trained on all 9 days) |
| `reports/` | validation results, driver importance, what-if scenarios, charts (`reports/figures/`) |
| `tests/` | leakage test |
| `docs/` | documentation and the map website (`docs/index.html`) |
| `web/` | writes the website's forecast file |
| `app/` | Streamlit analysis app |

## Collection pipeline

GitHub Actions took a snapshot every 10 minutes with `collector/collect.py`, from 30 Oct to 8 Nov 2026. The workflow is now disabled; to collect again: Actions tab -> collect -> Enable workflow, then Run workflow.

GitHub's scheduler is best effort (runs start late or get skipped, and a new schedule can take hours to begin), so the job doesn't depend on it for timing. Each run is a long job that loops for ~5.5 hours, taking a snapshot at every 10-minute mark and pushing it right away. Before it ends, it starts the next run itself; the hourly schedule is a second backup. The `concurrency` group keeps exactly one collector running, with the next one waiting in the queue.

Each snapshot fetches:

| Source | Endpoint | Notes |
|---|---|---|
| `taipei` | YouBike Taipei live JSON | one list, all stations |
| `ntpc` | New Taipei open data API | paged (`page`, `size=1000`), merged into one list |
| `weather` | Open-Meteo forecast API | current conditions + next 3 h forecast at 24 points covering every station |

Raw responses are saved unchanged, gzip-compressed:

```
data/raw/<source>/<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>.json.gz   # UTC fetch time
data/manifest/<YYYY-MM-DD>.jsonl                            # one line per source per run
```

How it handles failures:

- each HTTP request is retried 4 times with backoff (3 s, 6 s, 12 s)
- sources are independent, so one failing does not stop the others
- files are written to a temp name and then renamed, so there are no half-written files
- every attempt, failed or not, is logged in the manifest, so gaps are measurable later
- a run where any source failed shows red in the Actions tab, but whatever succeeded is still committed
- only one collector runs at a time (`concurrency` group); if a push clashes with another change to the same file, the collector keeps its own newer version (`git pull --rebase -X theirs`), and a failed sync is undone before the next snapshot, so one clash can't block a whole job (this happened once, on 31 Oct 13:00–13:20, and cost 3 snapshots)
- each snapshot is pushed as soon as it is taken, so a crashed job loses at most one snapshot
- a run stopped by GitHub still starts the next one (its last step always runs); this happened once, on 4 Nov 02:17, and no snapshot was lost

Keeping the raw data untouched means the clean dataset can always be rebuilt from scratch, and a parsing bug never loses data.

## Building the dataset

```bash
pip install -r requirements.txt
python pipeline/build_dataset.py     # data/raw -> data/processed
```

See [docs/DATASET.md](docs/DATASET.md) for the tables, columns, cleaning rules and known issues.

## Model

```bash
python pipeline/build_dataset.py # raw -> clean tables (9 full days)
python pipeline/features.py      # features.parquet
python pipeline/train.py         # validation reports + final model
python pipeline/drivers.py       # importance, partial dependence, what-if scenarios
python -m pytest tests/          # checks that no future data leaks into features
```

`train.py` takes about 25 minutes and up to ~5.5 GB of memory on the 9 days (`--no-cv` trains only the final models).

See [docs/MODEL.md](docs/MODEL.md).

## Website and app

- **Map website:** a static page (`docs/index.html`, deck.gl + MapLibre). After every snapshot the collector runs `web/make_forecast.py`, which writes `docs/data/forecast.json` with the forecast and the top three reasons for each station. The page loads instantly and refreshes every 5 minutes; since collection stopped it shows the last forecast and says so.
- **Analysis app:** `streamlit run app/app.py`. Station history, "why" for each forecast, what-if scenarios and validation results.

See [docs/APP.md](docs/APP.md).

## Run the collector locally

```bash
python collector/collect.py            # all sources
python collector/collect.py taipei     # one source
```

Only the Python standard library is needed (3.10+).
