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

## Methods compared

Baselines: same as now, same time yesterday, same time last week, historical average (below).
Models: linear regression, random forest (sampled to 200k rows), LightGBM, and LightGBM with a night rule.
Warning models: two LightGBM classifiers, "empty in one hour" (≤ 2 bikes) and "full in one hour" (≤ 2 free docks). Each threshold is picked on the last 20% of the *training* period (best F1), never on the test day.

Night rule: for target hours 00:00–05:59 the forecast is "same as now". In validation the model was slightly worse than "same as now" at night (almost nothing moves), so this is a measured choice, not a guess.

Extra score: the NTU score (from the NTU Machine Learning 2023 YouBike task) weights each error by how close the station is to empty or full, so mistakes at empty/full stations count up to 3× more.

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
- Absolute-error (L1) loss, because the main metric is MAE and most stations barely change in an hour, so the median change is a better target than the mean. On the first day (trained on 02:00–10:00, tested on 11:00–13:50) L1 gave MAE 2.77, squared error 2.88 (persistence 2.88). This is a single small check and will be confirmed by the full day-by-day validation.
- Fixed settings and a fixed 500 trees. Nothing is tuned on the test days.

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

## Drivers and what-if scenarios

`python pipeline/drivers.py` explains the final model in three ways. All numbers are in bikes, because the model predicts the change in bikes over the next hour.

1. **Importance (SHAP).** For 30,000 random rows, SHAP splits each forecast into a contribution per input. The average size of these contributions shows what the model relies on. Inputs are also grouped (current state, recent trend, yesterday / last week, neighbours, time, location, weather). The group value is computed on the summed contributions per row, so related inputs that share credit aren't counted twice.
   Output: `reports/driver_importance.csv`, `figures/importance_groups.png`, `figures/importance_features.png`.

2. **How the forecast moves with one input (partial dependence).** Take real rows, set one input to a fixed value for all of them, and average the forecast. Repeat for each value. Done for the hour of day (weekday vs weekend), the rain forecast, and the neighbours' change over the last 30 minutes.
   Output: `reports/partial_dependence.csv`, `figures/pd_hour.png`, `figures/pd_rain.png`, `figures/pd_neighbours.png`.

3. **What-if scenarios.** Take every station at a real weekday evening moment (17:30, forecasting 18:30) and change one thing: heavy rain, no rain, Saturday instead of a weekday, neighbours emptying, the station itself draining, a 6 °C warmer evening. For each, count how many stations the model expects to be nearly empty and how much the forecasts move.
   Output: `reports/whatif_scenarios.csv`.

Limits of this analysis:
- It explains what **the model** learned from ~9 days of autumn data, not proven cause and effect.
- Changing one input alone can create combinations that never happen (heavy rain with no clouds). Scenarios stay close to real conditions to limit this.
- With only a few rainy hours in the data, the rain effect is uncertain.

## Results

### Day-by-day validation (2 Nov 2026): trained on Fri 30 + Sat 31 Oct, tested on all of Sun 1 Nov

487,000 test forecasts on a day the models never saw.

| method | MAE (bikes) | within 2 bikes | NTU score | vs "same as now" |
|---|---|---|---|---|
| same as now (baseline) | 2.69 | 65% | 0.164 | – |
| same time yesterday | 5.66 | 36% | 0.343 | −111% |
| historical average (station × hour × weekend) | 5.44 | 34% | 0.330 | −102% |
| linear regression | 2.77 | 53% | 0.185 | −3% |
| random forest | 2.75 | 53% | 0.186 | −2% |
| LightGBM | 2.57 | 58% | 0.164 | +4.5% |
| **LightGBM + night rule** | **2.56** | 59% | **0.163** | **+4.9%** |

("Same time last week" still falls back to "same as now": there is no week-old data yet.)

By hour of day, LightGBM beats "same as now" in every hour from 06:00 to 23:00 (largest gain at 18:00: 3.77 vs 4.13 bikes) and is slightly worse at 00:00 and 02:00–05:00, which is what the night rule covers.

Warnings, one hour ahead:

| | empty: caught | empty: alarm right | full: caught | full: alarm right |
|---|---|---|---|---|
| same as now | 63% | 63% | 59% | 59% |
| LightGBM count forecast | 50% | 70% | 40% | 77% |
| **warning models** (AUC 0.91 empty, 0.95 full) | **75%** | 58% | **64%** | 55% |

What this says:
- Only LightGBM beats the naive baseline; linear regression and random forest do not. The forecast-of-change approach plus neighbours and recent trend is what helps.
- Methods that ignore the current count (yesterday, historical average) are far worse one hour ahead. The current count carries most of the information.
- The gain over the baseline is modest (about 5%) but consistent across the day, and largest at rush hours, which is when riders need it.

### First fair test (1 Nov 2026): trained on Friday 30 Oct, tested on all of Saturday 31 Oct

474,000 test forecasts. The model never saw a weekend before this test.

| method | MAE (bikes) | within 2 bikes | NTU score | vs "same as now" |
|---|---|---|---|---|
| same as now (baseline) | 2.74 | 64% | 0.169 | – |
| same time yesterday | 5.65 | 39% | 0.338 | −106% |
| linear regression | 2.96 | 49% | 0.198 | −8% |
| random forest | 2.79 | 53% | 0.189 | −2% |
| LightGBM | 2.64 | 58% | 0.170 | +3.7% |
| **LightGBM + night rule** | **2.62** | 59% | **0.168** | **+4.5%** |

("Same time last week" and "historical average" fall back to "same as now" here: there is no week-old data yet, and the training day had no weekend.)

Warnings (did we warn about stations that really ran empty / full one hour later?):

| | empty: caught | empty: alarm right | full: caught | full: alarm right |
|---|---|---|---|---|
| same as now | 63% | 62% | 55% | 56% |
| LightGBM count forecast | 50% | 68% | 32% | 78% |
| **warning models** (AUC 0.90 empty, 0.94 full) | **80%** | 53% | **57%** | 51% |

The warning models catch far more stations that run empty, at the cost of more false alarms. For a rider, a missed warning (walking to an empty station) costs more than a false one, so the website uses them.

### Live check of the published forecasts (31 Oct 07:50 – 1 Nov 03:30)

Every forecast the website showed, compared with the real count one hour later (389,000 forecasts, model trained on Friday only):
- the website's "now" and "1 hour ago" matched the saved official data in 100% of cases;
- MAE 2.78 vs 2.90 for "same as now" (4% better); for stations that really changed by 3+ bikes, 5.41 vs 6.21 (13% better);
- better than "same as now" in every hour from 07:00 to 23:00, slightly worse from 00:00 to 04:00, which led to the night rule.

The final validation, with one fold per day over the full collection period, is planned for 8 Nov 2026.
