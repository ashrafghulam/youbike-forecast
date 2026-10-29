"""
Baselines, model, and time-respecting validation.

    python pipeline/train.py                 # uses data/processed/features.parquet

Validation is day by day (expanding window):
    fold 1: train on days 1..k        test on day k+1
    fold 2: train on days 1..k+1      test on day k+2
    ...
Training rows whose *target* time (t + 60 min) falls inside the test day are
removed, so nothing from the test period reaches the model ("purge gap").

Outputs:
    reports/cv_folds.csv       metrics per fold and method
    reports/cv_summary.json    averaged metrics
    reports/cv_by_hour.csv     MAE by hour of day, model vs persistence
    models/lgbm.txt            final model trained on all data
    models/feature_list.json
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MIN_TRAIN_DAYS = 2
MIN_TEST_SLOTS = 72               # a test day needs at least 12 hours of data to count as a fold
LOW = 2                           # "near empty" = at most 2 bikes, "near full" = at most 2 free docks

FEATURES = [
    "bikes_now", "docks_now", "fill_now", "chg_10m", "chg_30m", "chg_60m",
    "bikes_yday_target", "bikes_week_target", "chg_yday_next_hour",
    "nbr_fill_now", "nbr_chg_30m", "nbr_count",
    "target_hour", "dow", "is_weekend", "is_holiday",
    "city", "district_en", "lat", "lon", "capacity",
    "temp_now", "precip_now", "humidity_now", "wind_now",
    "fc_precip_1h", "fc_precip_prob_1h", "fc_temp_1h",
]
CATEGORICAL = ["city", "district_en"]

PARAMS = dict(
    objective="regression",       # squared error on the *change* in bikes
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
NUM_ROUNDS = 500                  # fixed, so no test data is used to pick it


def train_model(train: pd.DataFrame) -> lgb.Booster:
    # The model predicts the change from now; the forecast is bikes_now + change.
    ds = lgb.Dataset(train[FEATURES], label=train["y"] - train["bikes_now"],
                     categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)


def predict(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    pred = df["bikes_now"].values + model.predict(df[FEATURES])
    return np.clip(pred, 0, df["capacity"].values)


def historical_average(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Average bikes for this station, at this hour, weekday vs weekend, in the training period."""
    key = ["station_id", "hour", "is_weekend"]
    tr = train.assign(hour=train["target_hour"].astype(int))
    te = test.assign(hour=test["target_hour"].astype(int))
    prof = tr.groupby(key, observed=True)["y"].mean().rename("hist")
    out = te.join(prof, on=key)["hist"]
    return out.fillna(te["bikes_now"]).values      # unseen combination: fall back to "now"


def alert_scores(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tp = int((true & pred).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int(true.sum()), 1)


def metrics(y: np.ndarray, pred: np.ndarray, usable_docks: np.ndarray) -> dict:
    """usable_docks = bikes + free docks right now (capacity minus broken docks)."""
    err = pred - y
    rounded = np.round(pred)
    e_prec, e_rec = alert_scores(y <= LOW, rounded <= LOW)
    f_prec, f_rec = alert_scores(usable_docks - y <= LOW, usable_docks - rounded <= LOW)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "within_2": float(np.mean(np.abs(err) <= 2)),     # share of forecasts within 2 bikes
        # alerts a rider would care about, one hour ahead
        "near_empty_precision": e_prec, "near_empty_recall": e_rec,
        "near_full_precision": f_prec, "near_full_recall": f_rec,
        "n": int(len(y)),
    }


def cross_validate(df: pd.DataFrame):
    df = df[df["y"].notna() & df["bikes_now"].notna()].copy()
    df["target_time"] = df["slot"] + pd.Timedelta(minutes=60)
    days = sorted(df["date"].unique())
    if len(days) <= MIN_TRAIN_DAYS:
        raise SystemExit(f"need more than {MIN_TRAIN_DAYS} days of data, have {len(days)}")

    rows, hourly = [], []
    for test_day in days[MIN_TRAIN_DAYS:]:
        test_start = pd.Timestamp(test_day).tz_localize(df["slot"].dt.tz)
        train = df[df["target_time"] < test_start]                 # purge: target must be before test day
        test = df[df["date"] == test_day]
        if train.empty or test["slot"].nunique() < MIN_TEST_SLOTS:
            print(f"  {test_day}: skipped (not enough data for a fold)")
            continue

        model = train_model(train)
        y = test["y"].values
        usable = (test["bikes_now"] + test["docks_now"]).fillna(test["capacity"]).values
        preds = {
            "persistence (same as now)": test["bikes_now"].values,
            "same time yesterday": test["bikes_yday_target"].fillna(test["bikes_now"]).values,
            "same time last week": test["bikes_week_target"].fillna(test["bikes_now"]).values,
            "historical average": historical_average(train, test),
            "lightgbm": predict(model, test),
        }
        for name, p in preds.items():
            rows.append({"test_day": str(test_day), "train_days": len(train["date"].unique()),
                         "method": name, **metrics(y, p, usable)})

        h = test.assign(hour=test["target_hour"].astype(int),
                        err_model=np.abs(preds["lightgbm"] - y),
                        err_persist=np.abs(preds["persistence (same as now)"] - y))
        hourly.append(h[["hour", "err_model", "err_persist"]])
        print(f"  {test_day}: train rows {len(train):,}  test rows {len(test):,}  "
              f"MAE model {rows[-1]['mae']:.3f}  persistence {rows[-5]['mae']:.3f}")

    folds = pd.DataFrame(rows)
    by_hour = (pd.concat(hourly).groupby("hour")[["err_model", "err_persist"]].mean()
               .rename(columns={"err_model": "mae_lightgbm", "err_persist": "mae_persistence"}))
    return folds, by_hour


def summarise(folds: pd.DataFrame) -> dict:
    # weight each fold by its number of rows
    out = {}
    for name, g in folds.groupby("method"):
        w = g["n"] / g["n"].sum()
        out[name] = {m: float((g[m] * w).sum()) for m in g.columns
                     if m not in ("test_day", "train_days", "method", "n")}
    base = out["persistence (same as now)"]["mae"]
    for name in out:
        out[name]["mae_improvement_vs_persistence"] = 1 - out[name]["mae"] / base
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--no-cv", action="store_true",
                    help="skip validation and only train the final model (e.g. before 3 days of data exist)")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    if not args.no_cv:
        print("cross-validation (one fold per test day):")
        folds, by_hour = cross_validate(df)
        summary = summarise(folds)

        folds.to_csv(args.reports / "cv_folds.csv", index=False)
        by_hour.to_csv(args.reports / "cv_by_hour.csv")
        (args.reports / "cv_summary.json").write_text(json.dumps(summary, indent=2))

        print("\naverage over folds:")
        print(pd.DataFrame(summary).T.round(3).to_string())

    # final model on everything, for the app and the driver analysis
    full = df[df["y"].notna() & df["bikes_now"].notna()]
    model = train_model(full)
    model.save_model(str(args.models / "lgbm.txt"))
    (args.models / "feature_list.json").write_text(json.dumps(
        {"features": FEATURES, "categorical": CATEGORICAL,
         "categories": {c: list(full[c].astype("category").cat.categories) for c in CATEGORICAL}}, indent=2))
    print(f"\nfinal model trained on {len(full):,} rows -> {args.models / 'lgbm.txt'}")


if __name__ == "__main__":
    main()
