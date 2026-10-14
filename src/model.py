"""Models, baselines and two honest evaluation schemes.

1. Temporal split (same athletes, future days): for every athlete, the first 70%
   of days train the model and the last 30% test it. This mirrors a real
   monitoring system that learns from the season so far.
2. Leave-one-athlete-out (new athletes): the model never sees the test athlete.
   This shows how well it generalises to someone new joining the squad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "target_readiness_next_day"
RANDOM_STATE = 42


def make_models() -> dict:
    return {
        "Ridge regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 20)),
        ),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def modelling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where tomorrow's readiness is known and today's report exists."""
    return df[df[TARGET].notna() & df["readiness"].notna()].copy()


def temporal_split(df: pd.DataFrame, train_frac: float = 0.7):
    parts_train, parts_test = [], []
    for _, g in df.groupby("participant"):
        g = g.sort_values("date")
        cut = int(len(g) * train_frac)
        parts_train.append(g.iloc[:cut])
        parts_test.append(g.iloc[cut:])
    return pd.concat(parts_train), pd.concat(parts_test)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Within ±1 point": float(np.mean(np.abs(y_true - y_pred) <= 1.0)),
    }


def low_day_thresholds(train: pd.DataFrame) -> pd.Series:
    """A low-readiness day is clearly worse than usual *for that athlete*:
    at or below their own mean minus one standard deviation, estimated on the
    training period only. Personal thresholds matter because people use the
    0-10 scale differently."""
    stats = train.groupby("participant")[TARGET].agg(["mean", "std"])
    return stats["mean"] - stats["std"].fillna(0)


def low_readiness_auc(test: pd.DataFrame, preds: np.ndarray, thresholds: pd.Series) -> float:
    """How well a readiness prediction ranks tomorrow's low-readiness days."""
    is_low = (test[TARGET] <= test["participant"].map(thresholds)).astype(int)
    if is_low.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(is_low, -np.asarray(preds)))


def evaluate_temporal(df: pd.DataFrame, features: list[str]):
    rows = modelling_rows(df)
    train, test = temporal_split(rows)
    y_train, y_test = train[TARGET], test[TARGET]

    low_thresholds = low_day_thresholds(train)
    personal_mean = train.groupby("participant")[TARGET].mean()

    preds = {
        "Baseline: same as today": test["readiness"].to_numpy(),
        "Baseline: personal average": test["participant"].map(personal_mean)
        .fillna(y_train.mean())
        .to_numpy(),
    }
    fitted = {}
    for name, model in make_models().items():
        model.fit(train[features], y_train)
        preds[name] = model.predict(test[features])
        fitted[name] = model

    results = []
    for name, p in preds.items():
        m = regression_metrics(y_test, p)
        m["Low-readiness AUC"] = low_readiness_auc(test, p, low_thresholds)
        results.append({"Model": name, **m})

    test = test.assign(**{f"pred: {k}": v for k, v in preds.items()})
    return pd.DataFrame(results), fitted, train, test


def evaluate_new_athletes(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = modelling_rows(df)
    groups = rows["participant"]
    all_true, all_pred = [], {k: [] for k in ["Baseline: same as today", *make_models()]}

    for train_idx, test_idx in LeaveOneGroupOut().split(rows, groups=groups):
        train, test = rows.iloc[train_idx], rows.iloc[test_idx]
        all_true.append(test[TARGET].to_numpy())
        all_pred["Baseline: same as today"].append(test["readiness"].to_numpy())
        for name, model in make_models().items():
            model.fit(train[features], train[TARGET])
            all_pred[name].append(model.predict(test[features]))

    y_true = np.concatenate(all_true)
    return pd.DataFrame(
        [{"Model": k, **regression_metrics(y_true, np.concatenate(v))} for k, v in all_pred.items()]
    )


# ---------------------------------------------------------------------------
# Classification: "will tomorrow be a low-readiness day for this athlete?"
# ---------------------------------------------------------------------------

def make_classifiers() -> dict:
    return {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def alert_metrics(y_true: np.ndarray, score: np.ndarray, flag_share: float = 0.2) -> dict:
    """Coach-facing view: if staff check the 20% of athlete-days with the
    highest risk score, how many real low days do they catch (recall), and
    how many flags are correct (precision)?"""
    # Flag exactly the top k days; ties (common with whole-number scores) are
    # broken with a fixed random order so no method is favoured.
    k = int(round(flag_share * len(score)))
    tiebreak = np.random.default_rng(RANDOM_STATE).random(len(score))
    order = np.lexsort((tiebreak, -np.asarray(score)))
    flagged = np.zeros(len(score), dtype=bool)
    flagged[order[:k]] = True
    tp = int((flagged & (y_true == 1)).sum())
    return {
        "Recall @20% flagged": tp / max(int(y_true.sum()), 1),
        "Precision @20% flagged": tp / max(int(flagged.sum()), 1),
    }


def evaluate_low_readiness(df: pd.DataFrame, features: list[str]):
    rows = modelling_rows(df)
    train, test = temporal_split(rows)
    thr = low_day_thresholds(train)
    y_train = (train[TARGET] <= train["participant"].map(thr)).astype(int).to_numpy()
    y_test = (test[TARGET] <= test["participant"].map(thr)).astype(int).to_numpy()

    scores = {
        # Higher score = higher risk of a low day tomorrow
        "Baseline: low readiness today": -test["readiness"].to_numpy(),
        "Baseline: below personal norm today": -test["readiness_vs_baseline"]
        .fillna(0)
        .to_numpy(),
    }
    fitted = {}
    for name, clf in make_classifiers().items():
        clf.fit(train[features], y_train)
        scores[name] = clf.predict_proba(test[features])[:, 1]
        fitted[name] = clf

    results = []
    for name, sc in scores.items():
        row = {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, sc),
            "PR-AUC": average_precision_score(y_test, sc),
            **alert_metrics(y_test, sc),
        }
        if not name.startswith("Baseline"):
            row["Brier score"] = brier_score_loss(y_test, sc)
        results.append(row)

    summary = {
        "low_day_rate_train": float(y_train.mean()),
        "low_day_rate_test": float(y_test.mean()),
        "n_test": int(len(y_test)),
    }
    test = test.assign(low_day_next=y_test,
                       **{f"risk: {k}": v for k, v in scores.items()})
    return pd.DataFrame(results), fitted, summary, test
