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
- Absolute-error (L1) loss, because the main metric is MAE and most stations barely change in an hour, so the median change is a better target than the mean. On the first day (trained on 02:00–10:00, tested on 11:00–13:50) L1 gave MAE 2.77, squared error 2.88 (persistence 2.88). This was a single small check; squared error was not re-tested in the final validation.
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

### Final validation: 7 test days, 1–7 Nov 2026

Data: 30 Oct 02:50 to 7 Nov 23:50 (9 days, see [DATASET.md](DATASET.md)). Each test day is predicted by models trained only on the days before it (2 days for the first fold, 8 for the last), with the purge gap described above. 3.4 million test forecasts in total. Reproduce with `python pipeline/train.py`; the numbers are in `reports/cv_folds.csv`, `reports/cv_summary.json`, `reports/cv_by_hour.csv` and `reports/cv_alert_auc.csv`.

**All methods, averaged over the 7 test days**

| method | MAE (bikes) | RMSE | within 2 bikes | NTU score | vs "same as now" |
|---|---|---|---|---|---|
| same as now (baseline) | 2.81 | 4.94 | 65% | 0.174 | – |
| same time yesterday | 5.18 | 7.62 | 39% | 0.317 | −84% |
| same time last week ¹ | 3.40 | 5.61 | 59% | 0.210 | −21% |
| historical average (station × hour × weekend) | 4.47 | 6.52 | 37% | 0.280 | −59% |
| linear regression | 2.82 | 4.40 | 54% | 0.189 | −0.4% |
| random forest | 2.63 | 4.19 | 57% | 0.179 | +6.3% |
| LightGBM | 2.48 | 4.16 | 61% | 0.160 | +11.8% |
| **LightGBM + night rule** | **2.48** | 4.18 | **62%** | **0.159** | **+11.9%** |

¹ A week of history exists only from Fri 6 Nov. On the 5 earlier test days this baseline falls back to "same as now". On the two days where it is real it is far worse than "same as now": 4.47 and 5.43 against 2.91 and 2.76.

**By test day** (MAE in bikes)

| test day | trained on | same as now | LightGBM + night rule | gain |
|---|---|---|---|---|
| Sun 1 Nov | 2 days | 2.69 | 2.56 | +4.8% |
| Mon 2 Nov | 3 days | 2.68 | 2.52 | +5.8% |
| Tue 3 Nov | 4 days | 2.87 | 2.55 | +11.2% |
| Wed 4 Nov | 5 days | 2.88 | 2.40 | +16.7% |
| Thu 5 Nov | 6 days | 2.89 | 2.38 | +17.9% |
| Fri 6 Nov | 7 days | 2.91 | 2.42 | +16.9% |
| Sat 7 Nov | 8 days | 2.76 | 2.53 | +8.5% |

**By hour of the forecast target**, all test days together (MAE in bikes, LightGBM without the night rule)

| hours | same as now | LightGBM | gain |
|---|---|---|---|
| 02–05 (night) | 0.61–0.98 | 0.67–0.98 | −1% to −8% (night rule uses "same as now") |
| 07 | 4.35 | 3.13 | +28% |
| 08 | 4.95 | 3.78 | +24% |
| 12 | 2.87 | 2.62 | +9% |
| 17 | 4.44 | 3.76 | +15% |
| 18 | 4.60 | 3.87 | +16% |

Every hour from 06:00 to 01:00 is better than "same as now"; only 02:00–05:00 is worse, which is what the night rule covers.

**Warnings one hour ahead**, averaged over the 7 test days

| | empty: caught | empty: alarm right | full: caught | full: alarm right |
|---|---|---|---|---|
| same as now | 64% | 64% | 53% | 54% |
| LightGBM count forecast | 59% | 69% | 37% | 76% |
| **warning models** (AUC 0.91 empty, 0.95 full) | **78%** | 60% | **61%** | 52% |

The thresholds chosen on the training period were stable across folds (0.25–0.35 for empty, 0.20–0.30 for full), and the AUC stayed between 0.905 and 0.914 (empty) and 0.942 and 0.955 (full).

**What this says**

- LightGBM beats the naive baseline on every one of the 7 test days, by 12% on average.
- The gain grows with more training data: about 5% with 2–3 days, about 17% with 5–7 days. Saturday 7 Nov (+8.5%) is lower; at that point the model had seen only one earlier Saturday.
- The gain is largest at rush hours (+28% at 07:00, +16% at 18:00), which is when riders most need a forecast.
- Random forest also beats the baseline (+6%), linear regression does not. The current count plus non-linear effects of trend, neighbours and time of day is what helps.
- Methods that ignore the current count (yesterday, last week, historical average) are far worse one hour ahead.
- The warning models catch more stations that really run empty (78% vs 64%) and full (61% vs 53%), with slightly more false alarms. For a rider, a missed warning costs more than a false one, so the website uses them.

### Drivers (final model, trained on all 9 days)

`python pipeline/drivers.py`, on 30,000 random rows. Numbers are the average size of each group's contribution to the forecast, in bikes (mean |SHAP|).

| what the model uses | effect (bikes) | strongest single inputs |
|---|---|---|
| current state | 0.85 | bikes now, how full the station is |
| yesterday / last week | 0.70 | how this station changed in the same hour yesterday; bikes at the target time yesterday |
| time of day / weekday | 0.26 | target hour, day of week |
| neighbouring stations | 0.25 | how full nearby stations are, and how they changed in the last 30 min |
| recent trend | 0.18 | change over the last 10 / 30 / 60 min |
| location | 0.10 | district |
| weather | 0.06 | mostly temperature (which also tracks the time of day) |

- The **current count** matters most, and the **daily routine** is next: the most useful single input after the count is how the same station changed in the same hour yesterday. "Same time yesterday" alone is a poor forecast (MAE 5.18), but as a correction to the current count it carries a lot of information.
- **Time of day**: on weekdays the model expects stations to lose the most bikes for targets around 07:00–09:00 (on average −0.67 bikes per station at 08:00, against −0.40 at weekends), with a second, smaller dip in the evening (`figures/pd_hour.png`).
- **Neighbours**: if nearby stations lost 30% of their fill in the last 30 minutes, the expected change is −0.76 bikes; if they stayed level, −0.21 (`figures/pd_neighbours.png`). People go to the next station when theirs runs out, and stations around one MRT exit or office block empty together.
- **Weather** has almost no effect, because there was almost no rain during collection (at most 0.7 mm in 15 minutes). The rain curve (`figures/pd_rain.png`) is flat above 0.5 mm, because the model never saw heavier rain. This is a limit of the data, not evidence that rain doesn't matter.

### What-if scenarios

Every station at a real weekday evening moment, Fri 6 Nov 17:30, forecasting 18:30 (3,399 stations). One input is changed, everything else stays real. Base: 579 stations expected to be nearly empty (≤ 2 bikes).

| scenario | what changes | stations nearly empty | avg change per station | stations moving > 2 bikes |
|---|---|---|---|---|
| base | real conditions | 579 | – | – |
| neighbours emptying fast | nearby stations lost 15% more of their bikes in the last 30 min | **668 (+89)** | −0.44 | 37 |
| same moment on a Saturday | weekday → Saturday | **504 (−75)** | +0.13 (average size 0.70) | 243 |
| heavy rain next hour | 5 mm/h forecast (90%), raining 2 mm now | 533 (−46) | +0.13 | 84 |
| station draining | this station lost 4 more bikes in 30 min | 582 (+3) | −0.19 | 9 |
| 6 °C warmer | temperature now and next hour +6 °C | 567 (−12) | 0.00 | 1 |
| dry next hour | no rain now or forecast | 581 (+2) | 0.00 | 0 |

- **Neighbours emptying** is the strongest scenario: about 90 more stations would be nearly empty. A practical use: when one area starts to drain, warn about the stations around it, not only the one already empty.
- **Saturday instead of Friday** changes the most stations (243 move by more than 2 bikes), but in both directions (average size of the change 0.70 bikes, average change only +0.13), so some stations gain and others lose.
- **Heavy rain** shows fewer empty stations (fewer people ride), but the model only saw light rain, so this number should not be trusted. More data in the rainy season is needed.
- A station's **own recent drain** moves the forecast less than expected (−0.19 bikes for 4 bikes lost). The model gives more weight to the daily routine and the neighbours than to the last 30 minutes at one station.

Limits: this explains what the model learned from 9 days of autumn data, not proven cause and effect, and changing one input alone can create combinations that rarely happen.

### Earlier checks

Before the final run, the same validation was run on shorter periods (train Fri → test Sat 31 Oct: +4.5%; train Fri–Sat → test Sun 1 Nov: +4.9%), and the forecasts published on the website were checked against what really happened:

**Live check of the published forecasts (31 Oct 07:50 – 1 Nov 03:30)**

Every forecast the website showed, compared with the real count one hour later (389,000 forecasts, model trained on Friday only):
- the website's "now" and "1 hour ago" matched the saved official data in 100% of cases;
- MAE 2.78 vs 2.90 for "same as now" (4% better); for stations that really changed by 3+ bikes, 5.41 vs 6.21 (13% better);
- better than "same as now" in every hour from 07:00 to 23:00, slightly worse from 00:00 to 04:00, which led to the night rule.
