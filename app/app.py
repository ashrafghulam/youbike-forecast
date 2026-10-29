"""
YouBike 1-hour forecast dashboard.

    streamlit run app/app.py
"""

import json
import os
import sys
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import live  # noqa: E402

MODELS = Path(os.environ.get("YOUBIKE_MODELS", ROOT / "models"))
REPORTS = Path(os.environ.get("YOUBIKE_REPORTS", ROOT / "reports"))
LOW = 2
STATUS = {  # label -> colour (RGB)
    "near empty (≤2 bikes)": (227, 73, 72),
    "ok": (42, 120, 214),
    "near full (≤2 docks)": (98, 80, 214),
}

st.set_page_config(page_title="YouBike 1-hour forecast", page_icon="🚲", layout="wide")


# ------------------------------------------------------------------ loading

@st.cache_resource
def load_model():
    path = MODELS / "lgbm.txt"
    if not path.exists():
        return None, None
    meta = json.loads((MODELS / "feature_list.json").read_text())
    return lgb.Booster(model_file=str(path)), meta


@st.cache_data(ttl=300, show_spinner="Fetching live data and recent history…")
def load_live():
    local = os.environ.get("YOUBIKE_LOCAL_DATA")      # optional: read history from a local folder
    history = live.LocalHistory(Path(local)) if local else live.GitHubHistory()
    return live.build_live(history=history, fetch_now=not os.environ.get("YOUBIKE_NO_FETCH"))


def predict(model, meta, X: pd.DataFrame) -> np.ndarray:
    X = X.copy()
    for c in meta["categorical"]:
        cats = meta["categories"][c]
        vals = X[c].astype(str)
        X[c] = pd.Categorical(vals.where(vals.isin(cats)), categories=cats)   # unseen -> missing
    change = model.predict(X[meta["features"]])
    return np.clip(X["bikes_now"].values + change, 0, X["capacity"].values)


def status(bikes, usable):
    return np.where(np.round(bikes) <= LOW, "near empty (≤2 bikes)",
                    np.where(usable - np.round(bikes) <= LOW, "near full (≤2 docks)", "ok"))


def station_map(df: pd.DataFrame, value_col: str, tooltip: str):
    d = df.copy()
    d["color"] = d["status"].map(STATUS).apply(list)
    d["radius"] = 25 + 2 * np.sqrt(d["capacity"])
    layer = pdk.Layer("ScatterplotLayer", d, get_position="[lon, lat]", get_fill_color="color",
                      get_radius="radius", radius_min_pixels=2, radius_max_pixels=9, pickable=True,
                      opacity=0.85, stroked=False)
    view = pdk.ViewState(latitude=d["lat"].mean(), longitude=d["lon"].mean(), zoom=10.5)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="light",
                             tooltip={"html": tooltip}), width="stretch")
    st.caption("  ·  ".join(f"<span style='color:rgb{c}'>●</span> {k}" for k, c in STATUS.items()),
               unsafe_allow_html=True)


# ------------------------------------------------------------------ page

model, meta = load_model()
st.title("YouBike: bikes one hour from now")

try:
    feats, recent, info = load_live()
except Exception as e:
    st.error(f"Could not build the live data: {e!r}")
    st.stop()

feats = feats[feats["bikes_now"].notna()].copy()
feats["usable"] = (feats["bikes_now"] + feats["docks_now"]).fillna(feats["capacity"])
if model is not None:
    feats["forecast"] = predict(model, meta, feats)
else:
    feats["forecast"] = feats["bikes_now"]
feats["forecast_r"] = feats["forecast"].round(1)
feats["status"] = status(feats["forecast"], feats["usable"])
feats["status_now"] = status(feats["bikes_now"], feats["usable"])
feats["name"] = feats["name_en"].fillna(feats["station_id"])

slot = info["latest_slot"]
st.caption(f"Data as of {info['fetched_at']:%a %d %b %Y %H:%M} (Taipei) · forecast for "
           f"{info['fetched_at'] + pd.Timedelta(hours=1):%H:%M} · {len(feats):,} stations · "
           f"{info['history_files']} history files used · live fetch: "
           + ", ".join(f"{k} {v}" for k, v in info["live_fetch"].items()))
if model is None:
    st.warning("No trained model found in models/. The map shows current counts only.")

with st.sidebar:
    st.header("Filter")
    cities = st.multiselect("City", ["taipei", "new_taipei"], default=["taipei", "new_taipei"],
                            format_func=lambda c: "Taipei" if c == "taipei" else "New Taipei")
    districts = sorted(feats.loc[feats["city"].astype(str).isin(cities), "district_en"].astype(str).unique())
    pick_d = st.multiselect("District", districts)
    if st.button("Refresh live data"):
        load_live.clear()
        st.rerun()
view = feats[feats["city"].astype(str).isin(cities)]
if pick_d:
    view = view[view["district_en"].astype(str).isin(pick_d)]

tab_map, tab_station, tab_whatif, tab_results, tab_about = st.tabs(
    ["Map", "Station", "What if", "Model results", "How it works"])

# ---------------- map
with tab_map:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", f"{len(view):,}")
    ne_now, ne_fc = (view["status_now"] == "near empty (≤2 bikes)").sum(), (view["status"] == "near empty (≤2 bikes)").sum()
    nf_now, nf_fc = (view["status_now"] == "near full (≤2 docks)").sum(), (view["status"] == "near full (≤2 docks)").sum()
    c2.metric("Near empty in 1 h", f"{ne_fc:,}", f"{ne_fc - ne_now:+,} vs now", delta_color="inverse")
    c3.metric("Near full in 1 h", f"{nf_fc:,}", f"{nf_fc - nf_now:+,} vs now", delta_color="inverse")
    c4.metric("Bikes at stations, in 1 h", f"{view['forecast'].sum():,.0f}",
              f"{view['forecast'].sum() - view['bikes_now'].sum():+,.0f} vs now")
    station_map(view, "forecast",
                "<b>{name}</b><br/>{district_en}<br/>now: {bikes_now} bikes<br/>in 1 h: {forecast_r} bikes")
    st.subheader("Biggest expected drops in the next hour")
    drops = view.assign(change=view["forecast"] - view["bikes_now"]).nsmallest(15, "change")
    st.dataframe(drops[["name", "district_en", "bikes_now", "forecast", "change", "capacity"]]
                 .rename(columns={"name": "station", "district_en": "district", "bikes_now": "now",
                                  "forecast": "in 1 h"}).round(1), hide_index=True, width="stretch")

# ---------------- station
with tab_station:
    options = view.sort_values("name")
    label = options["name"] + " · " + options["district_en"].astype(str) + " · " + options["station_id"]
    choice = st.selectbox("Station", label.tolist())
    row = options.loc[label == choice].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Bikes now", int(row["bikes_now"]))
    b.metric("Forecast in 1 h", f"{row['forecast']:.1f}", f"{row['forecast'] - row['bikes_now']:+.1f}")
    c.metric("Capacity (usable now)", f"{int(row['capacity'])} ({int(row['usable'])})")
    hist = recent[recent["station_id"] == row["station_id"]][["slot", "bikes"]]
    fc_point = pd.DataFrame({"slot": [slot, slot + pd.Timedelta(hours=1)],
                             "bikes": [row["bikes_now"], row["forecast"]]})
    line = alt.Chart(hist).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("slot:T", title="time (Taipei)"), y=alt.Y("bikes:Q", title="bikes", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("slot:T", format="%H:%M"), "bikes:Q"])
    pts = alt.Chart(hist).mark_circle(size=40, color="#2a78d6").encode(x="slot:T", y="bikes:Q")
    fc_line = alt.Chart(fc_point).mark_line(strokeDash=[5, 4], strokeWidth=2, color="#eb6834").encode(
        x="slot:T", y="bikes:Q")
    fc_dot = alt.Chart(fc_point.tail(1)).mark_circle(size=90, color="#eb6834").encode(
        x="slot:T", y="bikes:Q", tooltip=[alt.Tooltip("bikes:Q", title="forecast", format=".1f")])
    st.altair_chart(line + pts + fc_line + fc_dot, width="stretch")
    st.caption("Blue: last 2 hours as collected. Orange dashed: forecast for one hour ahead.")

# ---------------- what if
with tab_whatif:
    if model is None:
        st.info("What-if needs a trained model.")
    else:
        st.write("Change the conditions and see how the one-hour forecast reacts for the stations in view.")
        w1, w2, w3 = st.columns(3)
        rain = w1.slider("Rain forecast for the next hour (mm)", 0.0, 10.0,
                         float(np.nan_to_num(view["fc_precip_1h"].mean())), 0.5)
        temp_shift = w2.slider("Temperature change (°C)", -8, 8, 0)
        day = w3.radio("Day type", ["as now", "weekday", "Saturday", "Sunday"], horizontal=True)
        hour_shift = st.slider("Move the clock (hours), keeping the stations as they are now", -6, 6, 0)

        X = view.copy()
        X["fc_precip_1h"] = rain
        X["fc_precip_prob_1h"] = 90.0 if rain > 0 else 0.0
        X["precip_now"] = min(rain, X["precip_now"].fillna(0).max()) if rain > 0 else 0.0
        X["temp_now"] += temp_shift
        X["fc_temp_1h"] += temp_shift
        if day == "weekday":
            X["dow"], X["is_weekend"] = 2, 0
        elif day == "Saturday":
            X["dow"], X["is_weekend"] = 5, 1
        elif day == "Sunday":
            X["dow"], X["is_weekend"] = 6, 1
        X["target_hour"] = (X["target_hour"] + hour_shift) % 24
        X["forecast"] = predict(model, meta, X)
        X["status"] = status(X["forecast"], X["usable"])

        base_ne = (view["status"] == "near empty (≤2 bikes)").sum()
        new_ne = (X["status"] == "near empty (≤2 bikes)").sum()
        base_nf = (view["status"] == "near full (≤2 docks)").sum()
        new_nf = (X["status"] == "near full (≤2 docks)").sum()
        diff = X["forecast"] - view["forecast"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Near empty in 1 h", f"{new_ne:,}", f"{new_ne - base_ne:+,} vs real conditions", delta_color="inverse")
        m2.metric("Near full in 1 h", f"{new_nf:,}", f"{new_nf - base_nf:+,} vs real conditions", delta_color="inverse")
        m3.metric("Stations whose forecast moves > 1 bike", f"{(diff.abs() > 1).sum():,}")
        X["forecast_r"] = X["forecast"].round(1)
        station_map(X, "forecast",
                    "<b>{name}</b><br/>now: {bikes_now} bikes<br/>in 1 h (scenario): {forecast_r} bikes")
        moved = X.assign(change_vs_real=diff).reindex(diff.abs().sort_values(ascending=False).index).head(15)
        st.dataframe(moved[["name", "district_en", "bikes_now", "forecast", "change_vs_real"]]
                     .rename(columns={"name": "station", "district_en": "district", "bikes_now": "now",
                                      "forecast": "in 1 h (scenario)"}).round(1),
                     hide_index=True, width="stretch")
        st.caption("The model only knows conditions seen in the collected data (about 9 days of autumn). "
                   "Scenarios far from that, such as a typhoon, are guesses.")

# ---------------- results
with tab_results:
    summ = REPORTS / "cv_summary.json"
    if not summ.exists():
        st.info("Validation results appear here after running pipeline/train.py.")
    else:
        s = pd.DataFrame(json.loads(summ.read_text())).T
        s = s.rename(columns={"mae": "MAE (bikes)", "rmse": "RMSE", "within_2": "within 2 bikes",
                              "mae_improvement_vs_persistence": "MAE gain vs 'same as now'"})
        st.subheader("Day-by-day validation (train on earlier days, test on the next day)")
        st.dataframe(s.sort_values("MAE (bikes)").round(3), width="stretch")
        hourly = REPORTS / "cv_by_hour.csv"
        if hourly.exists():
            h = pd.read_csv(hourly).melt("hour", var_name="method", value_name="MAE")
            st.altair_chart(alt.Chart(h).mark_line(strokeWidth=2).encode(
                x=alt.X("hour:Q", title="hour of day (target)"), y=alt.Y("MAE:Q", title="MAE (bikes)"),
                color=alt.Color("method:N", scale=alt.Scale(range=["#2a78d6", "#eb6834"])),
                tooltip=["hour", "method", alt.Tooltip("MAE:Q", format=".2f")]), width="stretch")
        figs = REPORTS / "figures"
        for name in ["importance_groups", "pd_hour", "pd_rain", "pd_neighbours"]:
            if (figs / f"{name}.png").exists():
                st.image(str(figs / f"{name}.png"))
        sc = REPORTS / "whatif_scenarios.csv"
        if sc.exists():
            sc = pd.read_csv(sc)
            st.subheader("What-if scenarios on a real evening rush hour")
            st.caption(sc["moment"].iloc[0])
            st.dataframe(sc.drop(columns="moment").round(2), hide_index=True, width="stretch")

# ---------------- about
with tab_about:
    st.markdown("""
**Pipeline**

1. A scheduled job (GitHub Actions, every 10 minutes) saves the raw YouBike data for Taipei and
   New Taipei and the Open-Meteo weather at 24 points into the repository.
2. `build_dataset.py` turns the raw history into clean tables. Nothing raw is changed, so it can always be rebuilt.
3. `features.py` builds one row per station per 10 minutes, with only information known at that moment.
4. `train.py` compares the model with simple baselines, testing day by day on days the model has not seen.
5. This app fetches the live data, downloads the few past snapshots it needs from the repository,
   and runs **the same code** as training to make the forecast.

Code and documentation: [github.com/ashrafghulam/youbike-forecast](https://github.com/ashrafghulam/youbike-forecast)
""")
