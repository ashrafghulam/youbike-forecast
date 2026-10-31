# Website and app

## Live map (website)

Two static pages published with GitHub Pages, no server: `docs/index.html` (map) and `docs/station.html` (one station). Shared code in `docs/common.js` and `docs/style.css`.

**Map page**
- Each station is a bike pin when zoomed in, a dot when zoomed out. Colour = what to expect in one hour: green (bikes available, 6+), orange (only a few, 3–5), red (no bikes, ≤ 2), blue (no parking, ≤ 2 free docks).
- Find a station three ways: an area dropdown (all 42 districts), a search box with suggestions while typing (English or Chinese, e.g. "MRT exit 5", "台大"), or "Near me" (browser location).
- A summary for the chosen area ("In 1 hour: 516 stations with no bikes…"); clicking a line highlights only those stations.
- Click a pin: pop-up with 1 hour ago / now / in 1 hour, a plain-language verdict and a link to the station page. Optional 3D view and dark mode.

**Station page** (`station.html?id=<station number>`, shareable)
- Verdict, 1 hour ago / now / in 1 hour, a chart of the last 2 hours with the forecast, "Why this forecast?" in plain sentences, and the 6 nearest stations with a small map.

**Data:** `docs/data/stations.json` and `docs/data/forecast.json`, written by `web/make_forecast.py` inside the collector job after every snapshot, with the same feature code and model as training. Per station: bikes now, free docks, forecast, top 3 reasons (LightGBM's built-in SHAP values, `pred_contrib`, summed into seven groups) and the last 2 hours of counts. If this step fails, collection carries on.

## Analysis app (Streamlit)

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
