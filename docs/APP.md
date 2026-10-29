# App

A Streamlit dashboard for the live one-hour forecast.

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Pages

- **Map**: every station, coloured by the forecast for one hour from now (near empty, ok, near full), with how many stations will be near empty or near full compared with now, and the 15 stations expected to lose the most bikes.
- **Station**: pick a station to see the last two hours as collected and the forecast one hour ahead.
- **What if**: change the rain forecast, the temperature, the day type or the time of day, and see how many stations would be near empty or near full and which forecasts move most.
- **Model results**: validation table, error by hour of day, driver charts and the what-if scenarios from `reports/`.
- **How it works**: a short description of the pipeline.

## How the live forecast is made

1. The app fetches the current Taipei, New Taipei and weather data, using the collector's own functions.
2. It reads the collector manifests on GitHub and downloads only the past snapshots the features need: the last ~2 hours, around the same time yesterday, and around the same time last week.
3. It runs `build_dataset.py` and `features.py` on this small set and keeps the latest 10-minute slot.
4. The saved model (`models/lgbm.txt`) predicts the change for every station.

Because live forecasting uses the same code as training, a feature can't be computed one way in training and another way in the app. Results are cached for 5 minutes; the sidebar has a refresh button.

## Settings (environment variables, all optional)

| variable | effect |
|---|---|
| `YOUBIKE_LOCAL_DATA` | read history from a local `data/` folder instead of GitHub |
| `YOUBIKE_NO_FETCH` | don't fetch live data; use only the history |
| `YOUBIKE_MODELS` | folder with `lgbm.txt` and `feature_list.json` (default `models/`) |
| `YOUBIKE_REPORTS` | folder with validation reports (default `reports/`) |

## Limits

- Live data depends on three public APIs. If one fails, the app says which one in the header, and shows what it could get.
- In the first week, "same time last week" has no data, so that feature is empty.
- The forecast is only as good as the latest collected snapshot: if the collector has a gap, the trend features are weaker.
