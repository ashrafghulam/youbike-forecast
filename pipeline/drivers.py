"""
What drives the forecast, and what-if scenarios.

    python pipeline/drivers.py      # needs models/lgbm.txt and data/processed/features.parquet

Outputs (reports/):
    driver_importance.csv           mean |SHAP| per feature and per feature group
    whatif_scenarios.csv            scenario results on a real evening rush hour
    partial_dependence.csv          how the forecast moves as one input changes
    figures/*.png                   charts for the presentation

The model predicts the *change* in bikes over the next hour, so every SHAP value
and every scenario below is in bikes: "+1.5" means 1.5 more bikes than otherwise.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = 30_000
LOW = 2

GROUPS = {
    "current state": ["bikes_now", "docks_now", "fill_now"],
    "recent trend": ["chg_10m", "chg_30m", "chg_60m"],
    "yesterday / last week": ["bikes_yday_target", "bikes_week_target", "chg_yday_next_hour"],
    "neighbouring stations": ["nbr_fill_now", "nbr_chg_30m", "nbr_count"],
    "time of day / weekday": ["target_hour", "dow", "is_weekend", "is_holiday"],
    "location": ["city", "district_en", "lat", "lon", "capacity"],
    "weather": ["temp_now", "precip_now", "humidity_now", "wind_now",
                "fc_precip_1h", "fc_precip_prob_1h", "fc_temp_1h"],
}

# chart styling (reference palette, light mode)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(models: Path, features: Path):
    text = (models / "lgbm.txt").read_bytes().decode("utf-8").replace("\r\n", "\n")  # Windows line endings
    model = lgb.Booster(model_str=text)
    meta = json.loads((models / "feature_list.json").read_text())
    df = pd.read_parquet(features)
    df = df[df["y"].notna() & df["bikes_now"].notna()].reset_index(drop=True)
    for c in meta["categorical"]:
        df[c] = pd.Categorical(df[c].astype(str), categories=meta["categories"][c])
    return model, meta["features"], df


def predict_change(model, X):
    return model.predict(X)


def forecast(model, X):
    return np.clip(X["bikes_now"].values + model.predict(X), 0, X["capacity"].values)


# ------------------------------------------------------------------ SHAP

def shap_importance(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    per_feature = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    per_group = {}
    for g, cols in GROUPS.items():
        idx = [X.columns.get_loc(c) for c in cols if c in X.columns]
        # sum SHAP inside a group per row first, so the group is treated as one input
        per_group[g] = np.abs(sv[:, idx].sum(axis=1)).mean()
    per_group = pd.Series(per_group).sort_values(ascending=False)
    return sv, per_feature, per_group


def plot_bars(series: pd.Series, title: str, xlabel: str, path: Path):
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1))
    ax.barh(s.index, s.values, color=BLUE, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v, y, f"  {v:.2f}", va="center", color=INK2, fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.max() * 1.18)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ partial dependence

def partial_dependence(model, X, feature, grid):
    """Average forecast change when `feature` is set to each value in `grid` for every row."""
    out = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        out.append(predict_change(model, Xv).mean())
    return np.array(out)


def plot_line(x, ys: dict, title, xlabel, ylabel, path, xticks=None):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (name, y), color in zip(ys.items(), [BLUE, ORANGE]):
        ax.plot(x, y, color=color, linewidth=2, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if len(ys) > 1:
        ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ what-if

def pick_rush_hour(df):
    """The latest weekday slot at 17:30 (target 18:30), or the busiest slot if none exists."""
    s = df["slot"]
    cand = df[(s.dt.hour == 17) & (s.dt.minute == 30) & (s.dt.dayofweek < 5)]
    if cand.empty:
        busiest = df.groupby("slot")["chg_60m"].apply(lambda v: v.abs().mean()).idxmax()
        return df[df["slot"] == busiest]
    return cand[cand["slot"] == cand["slot"].max()]


def scenarios(model, base: pd.DataFrame, features):
    X0 = base[features]
    p0 = forecast(model, X0)

    def run(name, change: dict, note):
        X = X0.copy()
        for k, v in change.items():
            X[k] = v(X) if callable(v) else v
        p = forecast(model, X)
        return {
            "scenario": name, "what changes": note,
            "avg change vs base (bikes)": float((p - p0).mean()),
            "avg size of change (bikes)": float(np.abs(p - p0).mean()),
            "stations near empty": int((np.round(p) <= LOW).sum()),
            "stations moving > 2 bikes": int((np.abs(p - p0) > 2).sum()),
        }

    rows = [{
        "scenario": "base", "what changes": "real conditions at this time",
        "avg change vs base (bikes)": 0.0, "avg size of change (bikes)": 0.0, "stations near empty": int((np.round(p0) <= LOW).sum()),
        "stations moving > 2 bikes": 0,
    }]
    rows.append(run("heavy rain next hour",
                    {"fc_precip_1h": 5.0, "fc_precip_prob_1h": 90.0, "precip_now": 2.0},
                    "forecast 5 mm/h (90%), already raining 2 mm"))
    rows.append(run("dry next hour",
                    {"fc_precip_1h": 0.0, "fc_precip_prob_1h": 0.0, "precip_now": 0.0},
                    "no rain now or forecast"))
    rows.append(run("same moment on a Saturday",
                    {"dow": 5, "is_weekend": 1}, "weekday -> Saturday, everything else equal"))
    rows.append(run("neighbours emptying fast",
                    {"nbr_chg_30m": lambda X: X["nbr_chg_30m"] - 0.15},
                    "nearby stations lost 15% more of their bikes in the last 30 min"))
    rows.append(run("station draining",
                    {"chg_30m": lambda X: X["chg_30m"] - 4, "chg_60m": lambda X: X["chg_60m"] - 6},
                    "this station lost 4 more bikes in 30 min (6 in 60 min)"))
    rows.append(run("hot evening",
                    {"temp_now": lambda X: X["temp_now"] + 6, "fc_temp_1h": lambda X: X["fc_temp_1h"] + 6},
                    "6 °C warmer now and next hour"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = ap.parse_args()
    figs = args.reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    model, features, df = load(args.models, args.features)
    X = df[features].sample(min(SAMPLE, len(df)), random_state=0)

    # 1. importance
    sv, per_feature, per_group = shap_importance(model, X)
    imp = pd.concat([per_group.rename("mean_abs_shap").to_frame().assign(level="group"),
                     per_feature.rename("mean_abs_shap").to_frame().assign(level="feature")])
    imp.index.name = "name"
    imp.to_csv(args.reports / "driver_importance.csv")
    plot_bars(per_group, "What drives the 1-hour forecast",
              "average effect on the forecast (bikes, mean |SHAP|)", figs / "importance_groups.png")
    plot_bars(per_feature.head(15), "Top 15 single inputs",
              "average effect on the forecast (bikes, mean |SHAP|)", figs / "importance_features.png")

    # 2. how the forecast moves when one input changes
    pd_rows = []
    hours = np.arange(0, 24, 0.5)
    wd = X[X["is_weekend"] == 0].head(5000)
    we = wd.assign(is_weekend=1, dow=5)
    pd_wd, pd_we = partial_dependence(model, wd, "target_hour", hours), partial_dependence(model, we, "target_hour", hours)
    plot_line(hours, {"weekday": pd_wd, "weekend": pd_we}, "Expected change in bikes by time of day",
              "hour (at the forecast target)", "change in next hour (bikes)", figs / "pd_hour.png",
              xticks=range(0, 25, 3))
    pd_rows += [{"input": "target_hour", "value": h, "group": "weekday", "avg_change": v} for h, v in zip(hours, pd_wd)]
    pd_rows += [{"input": "target_hour", "value": h, "group": "weekend", "avg_change": v} for h, v in zip(hours, pd_we)]

    rain = np.array([0, 0.5, 1, 2, 3, 5, 8])
    pd_rain = partial_dependence(model, X.head(5000), "fc_precip_1h", rain)
    plot_line(rain, {"all stations": pd_rain}, "Expected change in bikes vs rain forecast",
              "forecast rain next hour (mm)", "change in next hour (bikes)", figs / "pd_rain.png")
    pd_rows += [{"input": "fc_precip_1h", "value": r, "group": "all", "avg_change": v} for r, v in zip(rain, pd_rain)]

    nbr = np.linspace(-0.3, 0.3, 13)
    pd_nbr = partial_dependence(model, X.head(5000), "nbr_chg_30m", nbr)
    plot_line(nbr * 100, {"all stations": pd_nbr}, "Expected change in bikes vs neighbours' last 30 min",
              "change in neighbours' fill over 30 min (percentage points)", "change in next hour (bikes)",
              figs / "pd_neighbours.png")
    pd_rows += [{"input": "nbr_chg_30m", "value": n, "group": "all", "avg_change": v} for n, v in zip(nbr, pd_nbr)]
    pd.DataFrame(pd_rows).to_csv(args.reports / "partial_dependence.csv", index=False)

    # 3. what-if on a real moment
    base = pick_rush_hour(df)
    sc = scenarios(model, base, features)
    sc.insert(0, "moment", f"{base['slot'].iloc[0]:%a %d %b %H:%M} -> forecast for +1 h ({len(base)} stations)")
    sc.to_csv(args.reports / "whatif_scenarios.csv", index=False)

    print("importance by group (bikes):")
    print(per_group.round(3).to_string())
    print("\nwhat-if:", sc["moment"].iloc[0])
    print(sc.drop(columns="moment").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
