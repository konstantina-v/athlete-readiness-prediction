"""Train the low-readiness classifier and save it as a new model version.

    python train.py --data data/raw            # train + register a new version
    python train.py --data data/raw --promote  # ...and put it in production

Two stages, as in a real deployment:
1. Evaluate: fit on the first 70% of each athlete's days, score on the last 30%.
   These held-out scores go into the model card, so we know how good it is.
2. Refit on all labelled data and save. The production model should learn from
   everything we have; the held-out scores from stage 1 describe its quality.
"""

from __future__ import annotations

import argparse
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn

from src import registry
from src.features import MIN_LOGGED_SESSIONS, available_features, build_features
from src.load_data import load_pmdata
from src.model import (
    TARGET,
    evaluate_low_readiness,
    low_day_thresholds,
    make_classifiers,
    modelling_rows,
)

MODEL_NAME = "Gradient boosting"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def feature_profile(df: pd.DataFrame, features: list[str]) -> dict:
    """Summary of each feature in the training data. Stored with the model so
    that monitoring can later check whether new data looks different."""
    profile = {}
    for f in features:
        s = df[f]
        q = s.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        profile[f] = {
            "missing_rate": float(s.isna().mean()),
            "mean": None if s.notna().sum() == 0 else float(s.mean()),
            "quantiles": {str(k): (None if pd.isna(v) else float(v)) for k, v in q.items()},
        }
    return profile


def train(data_dir: str, promote: bool = False) -> str:
    df = build_features(load_pmdata(data_dir))
    features = available_features(df)
    rows = modelling_rows(df)

    # Stage 1: honest evaluation on unseen future days
    results, _, info, _ = evaluate_low_readiness(df, features)
    holdout = results.set_index("Model").loc[MODEL_NAME].to_dict()
    best_baseline = results[results["Model"].str.startswith("Baseline")]["ROC-AUC"].max()
    print(f"Held-out ROC-AUC {holdout['ROC-AUC']:.3f} "
          f"(best simple rule {best_baseline:.3f})")

    # Stage 2: refit on all labelled data
    thresholds = low_day_thresholds(rows)
    y = (rows[TARGET] <= rows["participant"].map(thresholds)).astype(int)
    model = make_classifiers()[MODEL_NAME]
    model.fit(rows[features], y)

    metadata = {
        "task": "Predict whether tomorrow will be a low-readiness day for the athlete",
        "model_type": f"{MODEL_NAME} ({type(model).__name__})",
        "positive_class": "low-readiness day: next-day readiness <= athlete's mean - 1 SD",
        "features": features,
        "training_data": {
            "source": "PMData (Thambawita et al., 2020)",
            "athletes": int(rows["participant"].nunique()),
            "rows": int(len(rows)),
            "date_range": [str(rows["date"].min().date()), str(rows["date"].max().date())],
            "low_day_rate": float(y.mean()),
            "load_features_masked_for": sorted(
                df.loc[~df["logs_training"], "participant"].unique().tolist()),
            "min_logged_sessions_rule": MIN_LOGGED_SESSIONS,
        },
        "personal_low_day_thresholds": {k: round(float(v), 3) for k, v in thresholds.items()},
        "evaluation": {
            "method": "temporal split: first 70% of each athlete's days train, last 30% test",
            "test_rows": info["n_test"],
            "test_low_day_rate": round(info["low_day_rate_test"], 3),
            "metrics": {k: round(float(v), 4) for k, v in holdout.items()
                        if not pd.isna(v)},
            "best_baseline_roc_auc": round(float(best_baseline), 4),
        },
        "feature_profile": feature_profile(rows, features),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "git_commit": _git_commit(),
        },
    }
    version = registry.save_model(model, metadata)
    print(f"Saved model version {version}")

    if promote or registry.production_version() is None:
        reason = "manual promotion" if promote else "first model, no production model yet"
        registry.promote(version, reason)
        print(f"Promoted {version} to production ({reason})")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--promote", action="store_true",
                        help="Put this version in production")
    args = parser.parse_args()
    train(args.data, args.promote)
