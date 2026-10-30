# YouBike 1-hour-ahead availability forecast

Predict how many bikes a Taipei / New Taipei YouBike station will have one hour from now, and explain what drives it.

## Status

- [x] Collector running (from 30 Oct 2026)
- [x] Dataset build (`pipeline/build_dataset.py`, documented in [docs/DATASET.md](docs/DATASET.md))
- [x] Baselines, model and time-based validation (`pipeline/features.py`, `pipeline/train.py`, [docs/MODEL.md](docs/MODEL.md)), results after the final run
- [x] Drivers and what-if scenarios (`pipeline/drivers.py`), results after the final run
- [x] 3D live map website (`docs/index.html`, refreshed after every snapshot) and analysis app (`app/app.py`, [docs/APP.md](docs/APP.md))
- [ ] Presentation

## Collection pipeline

GitHub Actions takes a snapshot every 10 minutes with `collector/collect.py`.

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
- only one collector runs at a time (`concurrency` group), and pushes retry on conflict
- each snapshot is pushed as soon as it is taken, so a crashed job loses at most one snapshot

Keeping the raw data untouched means the clean dataset can always be rebuilt from scratch, and a parsing bug never loses data.

## Building the dataset

```bash
pip install -r requirements.txt
python pipeline/build_dataset.py     # data/raw -> data/processed
```

See [docs/DATASET.md](docs/DATASET.md) for the tables, columns, cleaning rules and known issues.

## Model

```bash
python pipeline/features.py      # features.parquet
python pipeline/train.py         # validation reports + final model
python pipeline/drivers.py       # importance, partial dependence, what-if scenarios
python -m pytest tests/          # checks that no future data leaks into features
```

See [docs/MODEL.md](docs/MODEL.md).

## Website and app

- **3D live map:** https://ashrafghulam.github.io/youbike-forecast/ . A static page (`docs/index.html`, deck.gl + MapLibre). After every snapshot the collector runs `web/make_forecast.py`, which writes `docs/data/forecast.json` with the forecast and the top three reasons for each station. The page loads instantly and refreshes every 5 minutes.
- **Analysis app:** https://youbike-forecast.streamlit.app (`streamlit run app/app.py`). Station history, "why" for each forecast, what-if scenarios and validation results.

See [docs/APP.md](docs/APP.md).

## Run the collector locally

```bash
python collector/collect.py            # all sources
python collector/collect.py taipei     # one source
```

Only the Python standard library is needed (3.10+).
