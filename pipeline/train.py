"""
Baselines, models, and time-respecting validation.

    python pipeline/train.py                       # validation + final models
    python pipeline/train.py --min-train-days 1    # early on, when only a few days exist
    python pipeline/train.py --no-cv               # final models only

Validation is day by day (expanding window):
    fold 1: train on days 1..k        test on day k+1
    fold 2: train on days 1..k+1      test on day k+2
    ...
Training rows whose *target* time (t + 60 min) falls inside the test day are
removed, so nothing from the test period reaches the model ("purge gap").

Compared on every fold:
    baselines      same as now, same time yesterday, same time last week, historical average
    models         linear regression, random forest, LightGBM, LightGBM + night rule
    alert models   two LightGBM classifiers: "will be empty" and "will be full" in one hour

Outputs:
    reports/cv_folds.csv, reports/cv_summary.json, reports/cv_by_hour.csv
    models/lgbm.txt, models/alert_empty.txt, models/alert_full.txt, models/feature_list.json
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
MIN_TEST_SLOTS = 72               # a test day needs at least 12 hours of data to count as a fold
LOW = 2                           # "empty" = at most 2 bikes, "full" = at most 2 free docks
NIGHT_HOURS = set(range(0, 6))    # target hours where "same as now" is used (see docs/MODEL.md)
RF_SAMPLE = 200_000               # random forest is trained on a sample to keep it fast

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
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

PARAMS = dict(
    # Absolute error (L1) on the *change* in bikes. MAE is the main metric, and the
    # typical (median) change at a station is near zero, so L1 fits better than
    # squared error. A first check on day 1 agreed (MAE 2.77 vs 2.88 for squared error).
    objective="regression_l1",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42,
)
ALERT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
NUM_ROUNDS = 500                  # fixed, so no test data is used to pick it
ALERT_ROUNDS = 300


# ------------------------------------------------------------------ helpers

def usable_docks(df: pd.DataFrame) -> np.ndarray:
    """Bikes + free docks right now = docks that actually work (capacity minus broken ones)."""
    return (df["bikes_now"] + df["docks_now"]).fillna(df["capacity"]).to_numpy()


def alert_labels(df: pd.DataFrame) -> dict:
    y, usable = df["y"].to_numpy(), usable_docks(df)
    return {"empty": y <= LOW, "full": usable - y <= LOW}


def is_night(df: pd.DataFrame) -> np.ndarray:
    return df["target_hour"].astype(int).isin(NIGHT_HOURS).to_numpy()


# ------------------------------------------------------------------ models

def train_model(train: pd.DataFrame) -> lgb.Booster:
    # The model predicts the change from now; the forecast is bikes_now + change.
    ds = lgb.Dataset(train[FEATURES], label=train["y"] - train["bikes_now"],
                     categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)


def predict(model: lgb.Booster, df: pd.DataFrame, night_rule: bool = False) -> np.ndarray:
    pred = np.clip(df["bikes_now"].values + model.predict(df[FEATURES]), 0, df["capacity"].values)
    if night_rule:
        pred = np.where(is_night(df), df["bikes_now"].values, pred)
    return pred


def linear_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med = train[NUMERIC].median().fillna(0)          # columns with no data yet (e.g. last week) -> 0
    lr = LinearRegression().fit(train[NUMERIC].fillna(med), train["y"] - train["bikes_now"])
    return np.clip(test["bikes_now"].values + lr.predict(test[NUMERIC].fillna(med)), 0, test["capacity"].values)


def forest_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    s = train.sample(min(RF_SAMPLE, len(train)), random_state=0)
    med = s[NUMERIC].median().fillna(0)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0)
    rf.fit(s[NUMERIC].fillna(med), s["y"] - s["bikes_now"])
    return np.clip(test["bikes_now"].values + rf.predict(test[NUMERIC].fillna(med)), 0, test["capacity"].values)


def historical_average(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Average bikes for this station, at this hour, weekday vs weekend, in the training period."""
    key = ["station_id", "hour", "is_weekend"]
    tr = train.assign(hour=train["target_hour"].astype(int))
    te = test.assign(hour=test["target_hour"].astype(int))
    prof = tr.groupby(key, observed=True)["y"].mean().rename("hist")
    return te.join(prof, on=key)["hist"].fillna(te["bikes_now"]).values


def _fit_alert(train: pd.DataFrame, label: np.ndarray) -> lgb.Booster:
    ds = lgb.Dataset(train[FEATURES], label=label.astype(int), categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(ALERT_PARAMS, ds, num_boost_round=ALERT_ROUNDS)


def _best_threshold(prob: np.ndarray, truth: np.ndarray) -> float:
    """Threshold with the best F1 (balance of 'caught' and 'right when given')."""
    best, best_f1 = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.05):
        p = prob >= t
        tp = (p & truth).sum()
        f1 = 2 * tp / max(p.sum() + truth.sum(), 1)
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return round(best, 2)


def train_alerts(train: pd.DataFrame) -> dict:
    """Two classifiers. Each threshold is chosen on the last 20% of the *training* period,
    then the classifier is refit on all training data. The test period is never used."""
    out = {}
    cut = train["slot"].quantile(0.8)
    inner_tr, inner_va = train[train["slot"] < cut - pd.Timedelta(hours=1)], train[train["slot"] >= cut]
    lab_tr, lab_va, lab_all = alert_labels(inner_tr), alert_labels(inner_va), alert_labels(train)
    for kind in ("empty", "full"):
        m = _fit_alert(inner_tr, lab_tr[kind])
        thr = _best_threshold(m.predict(inner_va[FEATURES]), lab_va[kind])
        out[kind] = (_fit_alert(train, lab_all[kind]), thr)
    return out


# ------------------------------------------------------------------ metrics

def alert_scores(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tp = int((true & pred).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int(true.sum()), 1)


def ntu_score(y, pred, cap):
    """Error weighted by how close the station is to empty or full (NTU ML 2023 YouBike task).
    A mistake at an empty or full station counts up to 3x more than at a half-full one."""
    cap = np.maximum(cap, 1)
    f = y / cap
    return float(np.mean(3 * np.abs(y - pred) / cap * (np.abs(f - 1 / 3) + np.abs(f - 2 / 3))))


def metrics(test: pd.DataFrame, pred: np.ndarray, alerts=None) -> dict:
    """alerts: optional {"empty": bool array, "full": bool array}; otherwise derived from the count forecast."""
    y, usable, cap = test["y"].to_numpy(), usable_docks(test), test["capacity"].to_numpy()
    err = pred - y
    truth = alert_labels(test)
    if alerts is None:
        r = np.round(pred)
        alerts = {"empty": r <= LOW, "full": usable - r <= LOW}
    e_prec, e_rec = alert_scores(truth["empty"], alerts["empty"])
    f_prec, f_rec = alert_scores(truth["full"], alerts["full"])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "within_2": float(np.mean(np.abs(err) <= 2)),
        "ntu_score": ntu_score(y, pred, cap),
        "empty_caught": e_rec, "empty_alarm_right": e_prec,
        "full_caught": f_rec, "full_alarm_right": f_prec,
        "n": int(len(y)),
    }


# ------------------------------------------------------------------ validation

def cross_validate(df: pd.DataFrame, min_train_days: int):
    df = df[df["y"].notna() & df["bikes_now"].notna()].copy()
    df["target_time"] = df["slot"] + pd.Timedelta(minutes=60)
    days = sorted(df["date"].unique())
    if len(days) <= min_train_days:
        raise SystemExit(f"need more than {min_train_days} days of data, have {len(days)}")

    rows, hourly, aucs = [], [], []
    for test_day in days[min_train_days:]:
        test_start = pd.Timestamp(test_day).tz_localize(df["slot"].dt.tz)
        train = df[df["target_time"] < test_start]                 # purge: target must be before test day
        test = df[df["date"] == test_day]
        if train.empty or test["slot"].nunique() < MIN_TEST_SLOTS:
            print(f"  {test_day}: skipped (only {test['slot'].nunique()} time slots, need {MIN_TEST_SLOTS})")
            continue

        model = train_model(train)
        lgbm = predict(model, test)
        lgbm_night = predict(model, test, night_rule=True)
        alerts = train_alerts(train)
        probs = {k: m.predict(test[FEATURES]) for k, (m, _) in alerts.items()}
        flags = {k: probs[k] >= alerts[k][1] for k in probs}
        truth = alert_labels(test)
        aucs.append({"test_day": str(test_day),
                     **{f"auc_{k}": float(roc_auc_score(truth[k], probs[k])) for k in probs},
                     **{f"threshold_{k}": alerts[k][1] for k in alerts}})

        preds = {
            "same as now (baseline)": (test["bikes_now"].to_numpy(), None),
            "same time yesterday": (test["bikes_yday_target"].fillna(test["bikes_now"]).to_numpy(), None),
            "same time last week": (test["bikes_week_target"].fillna(test["bikes_now"]).to_numpy(), None),
            "historical average": (historical_average(train, test), None),
            "linear regression": (linear_forecast(train, test), None),
            "random forest": (forest_forecast(train, test), None),
            "lightgbm": (lgbm, None),
            "lightgbm + night rule": (lgbm_night, None),
            "lightgbm + night rule + alert models": (lgbm_night, flags),
        }
        for name, (p, fl) in preds.items():
            rows.append({"test_day": str(test_day), "train_days": len(train["date"].unique()),
                         "method": name, **metrics(test, p, fl)})

        h = test.assign(hour=test["target_hour"].astype(int), err_model=np.abs(lgbm - test["y"]),
                        err_persist=np.abs(test["bikes_now"] - test["y"]))
        hourly.append(h[["hour", "err_model", "err_persist"]])
        print(f"  {test_day}: train {len(train):,} rows ({len(train['date'].unique())} days), test {len(test):,} rows")

    folds = pd.DataFrame(rows)
    by_hour = (pd.concat(hourly).groupby("hour")[["err_model", "err_persist"]].mean()
               .rename(columns={"err_model": "mae_lightgbm", "err_persist": "mae_same_as_now"}))
    return folds, by_hour, pd.DataFrame(aucs)


def summarise(folds: pd.DataFrame) -> dict:
    out = {}
    for name, g in folds.groupby("method", sort=False):
        w = g["n"] / g["n"].sum()
        out[name] = {m: float((g[m] * w).sum()) for m in g.columns if m not in ("test_day", "train_days", "method", "n")}
    base = out["same as now (baseline)"]["mae"]
    for name in out:
        out[name]["mae_gain_vs_baseline"] = 1 - out[name]["mae"] / base
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet")
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--min-train-days", type=int, default=2)
    ap.add_argument("--no-cv", action="store_true", help="skip validation, only train the final models")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    if not args.no_cv:
        print("validation (one fold per test day):")
        folds, by_hour, aucs = cross_validate(df, args.min_train_days)
        summary = summarise(folds)
        folds.to_csv(args.reports / "cv_folds.csv", index=False)
        by_hour.to_csv(args.reports / "cv_by_hour.csv")
        aucs.to_csv(args.reports / "cv_alert_auc.csv", index=False)
        (args.reports / "cv_summary.json").write_text(json.dumps(summary, indent=2))
        print("\naverage over folds:")
        print(pd.DataFrame(summary).T.round(3).to_string())
        print("\nalert models:")
        print(aucs.round(3).to_string(index=False))

    # final models on everything, for the website, the app and the driver analysis
    full = df[df["y"].notna() & df["bikes_now"].notna()]
    model = train_model(full)
    model.save_model(str(args.models / "lgbm.txt"))
    alerts = train_alerts(full)
    for kind, (m, _) in alerts.items():
        m.save_model(str(args.models / f"alert_{kind}.txt"))
    (args.models / "feature_list.json").write_text(json.dumps(
        {"features": FEATURES, "categorical": CATEGORICAL,
         "categories": {c: list(full[c].astype("category").cat.categories) for c in CATEGORICAL},
         "night_hours": sorted(NIGHT_HOURS), "low": LOW,
         "alert_thresholds": {k: thr for k, (_, thr) in alerts.items()}}, indent=2))
    print(f"\nfinal models trained on {len(full):,} rows -> {args.models}")


if __name__ == "__main__":
    main()
