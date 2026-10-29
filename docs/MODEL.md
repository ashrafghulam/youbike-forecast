# Model

## The task

At any 10-minute slot `t`, for every station, predict the number of bikes available at `t + 60 min`.

Run everything from the raw data:

```bash
python pipeline/build_dataset.py   # raw snapshots -> clean tables
python pipeline/features.py        # clean tables -> features.parquet
python pipeline/train.py           # validation, reports/, final model in models/
python -m pytest tests/            # leakage test
```

## Features

Each feature uses only what was known at time `t`.

| group | features | idea |
|---|---|---|
| current state | `bikes_now`, `docks_now`, `fill_now` | the best single clue is where the station is now |
| recent trend | `chg_10m`, `chg_30m`, `chg_60m` | a station that's emptying tends to keep emptying for a while |
| daily / weekly memory | `bikes_yday_target`, `bikes_week_target`, `chg_yday_next_hour` | bikes at the target time 1 day and 7 days ago, and how yesterday's same hour changed |
| neighbours | `nbr_fill_now`, `nbr_chg_30m`, `nbr_count` | average of the 5 nearest stations within 1 km. Riders move to the next station when one is empty, and nearby stations often share a pattern (office area, MRT exit) |
| calendar | `target_hour`, `dow`, `is_weekend`, `is_holiday` | commuting peaks, weekend leisure trips. Described at the target time |
| location | `city`, `district_en`, `lat`, `lon`, `capacity` | where the station is and how big it is |
| weather now | `temp_now`, `precip_now`, `humidity_now`, `wind_now` | latest observation at or before `t` |
| weather forecast | `fc_precip_1h`, `fc_precip_prob_1h`, `fc_temp_1h` | forecast for the target hour, **as published at `t`** |

Two details that matter for honesty:

- The target is never filled in. If the station has no reading at `t + 60`, the row is not used for training or testing. Features may carry the last value forward over a gap of up to 20 minutes (a late GitHub run), the target may not.
- Future weather comes from the forecast that existed at time `t`, not from what actually happened. In real use the actual weather of the next hour isn't known yet.

## Baselines

| baseline | prediction |
|---|---|
| persistence | same as now |
| same time yesterday | bikes at the target time, 24 h earlier |
| same time last week | bikes at the target time, 7 days earlier |
| historical average | average for this station, this hour, weekday or weekend, over the training days |

When a baseline has no value (for example "last week" during the first 7 days), it falls back to "same as now".

## Model

LightGBM gradient boosting, one model for all stations.

- It predicts the **change** from now (`y - bikes_now`), and the forecast is `bikes_now + change`, clipped to `[0, capacity]`. Most of the information is in the current count, so learning the change is easier and keeps forecasts sensible.
- Squared-error loss, fixed settings and a fixed 500 trees. Nothing is tuned on the test days.

## Validation

Day-by-day expanding window:

```
fold 1: train days 1-2        test day 3
fold 2: train days 1-3        test day 4
...
```

- A training row is dropped if its **target** time (`t + 60 min`) falls inside the test day. Without this, the last hour before midnight would train on bikes counted in the test day.
- Test days with less than 12 hours of data are skipped.
- Metrics are averaged across folds, weighted by the number of test rows.

`tests/test_leakage.py` checks this automatically. It builds features twice, once with the real future and once with a randomly rewritten future after a cutoff, and checks that every feature up to the cutoff is identical. I also checked that the test fails when a deliberately leaky feature is added.

## Metrics

- **MAE**: average error in bikes. The main number.
- **RMSE**: punishes big misses more.
- **within 2**: share of forecasts within 2 bikes of the truth.
- **near-empty alert** (≤ 2 bikes in one hour) and **near-full alert** (≤ 2 free docks): precision and recall. These are the cases a rider actually suffers from.
- **improvement vs persistence**: `1 - MAE_model / MAE_persistence`.

`reports/cv_by_hour.csv` breaks the error down by hour of day, because "same as now" is almost perfect at 3 AM and the real test is the rush hours.

## Results

To be filled in after the final run on the complete data (planned 8 Nov 2026).
