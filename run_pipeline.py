"""Run the full athlete-readiness pipeline.

    python run_pipeline.py --data data/raw

Outputs:
    reports/figures/*.png           figures used in the README
    reports/results.md              evaluation tables
    reports/athlete_daily_features.csv   tidy table, ready for Power BI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src import plots
from src.features import FEATURE_LABELS, available_features, build_features
from src.load_data import load_pmdata
from src.model import TARGET, evaluate_low_readiness, evaluate_new_athletes, evaluate_temporal

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def main(data_dir: str):
    print(f"Loading PMData from {data_dir} ...")
    daily = load_pmdata(data_dir)
    df = build_features(daily)
    features = available_features(df)

    n_athletes = df["participant"].nunique()
    n_rows = int(df[TARGET].notna().sum())
    print(f"{n_athletes} athletes, {len(df)} athlete-days, {n_rows} with a next-day target")
    print(f"Using {len(features)} features")

    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "athlete_daily_features.csv", index=False)

    # 1. Same athletes, future days
    temporal, fitted, train, test = evaluate_temporal(df, features)
    # 2. Completely new athletes
    new_athletes = evaluate_new_athletes(df, features)

    print("\nTemporal split (last 30% of each athlete's days):")
    print(temporal.round(3).to_string(index=False))
    print("\nLeave-one-athlete-out:")
    print(new_athletes.round(3).to_string(index=False))

    # 3. Classification: will tomorrow be a low-readiness day for this athlete?
    low, low_fitted, low_info, low_test = evaluate_low_readiness(df, features)
    print(f"\nLow-readiness day classifier (low-day rate in test: "
          f"{low_info['low_day_rate_test']:.1%} of {low_info['n_test']} days):")
    print(low.round(3).to_string(index=False))
    non_loggers = sorted(df.loc[~df["logs_training"], "participant"].unique())

    # Explain the gradient boosting model with SHAP on the test period
    gb = fitted["Gradient boosting"]
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(test[features])
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features)
    importance.index = [FEATURE_LABELS.get(f, f) for f in importance.index]

    # Figures: pick the athlete with the most reported days as the example
    example = df.groupby("participant")["readiness"].count().idxmax()
    plots.athlete_monitoring(df, example, FIGURES / "athlete_monitoring.png")
    plots.model_comparison(temporal, FIGURES / "model_comparison.png",
                           "Next-day readiness: error on unseen future days")
    plots.prediction_example(test, example, "pred: Gradient boosting",
                             FIGURES / "prediction_example.png")
    plots.feature_importance(importance, FIGURES / "feature_importance.png")
    best_clf = low[~low["Model"].str.startswith("Baseline")].sort_values("ROC-AUC").iloc[-1]["Model"]
    plots.low_day_alerts(low_test, example, f"risk: {best_clf}",
                         FIGURES / "low_day_alerts.png")

    with open(REPORTS / "results.md", "w") as f:
        f.write(f"# Results\n\n{n_athletes} athletes, {n_rows} athlete-days with a next-day target.\n\n")
        f.write("## Same athletes, unseen future days (temporal 70/30 split)\n\n")
        f.write(temporal.round(3).to_markdown(index=False))
        f.write("\n\n## New athletes (leave-one-athlete-out)\n\n")
        f.write(new_athletes.round(3).to_markdown(index=False))
        f.write("\n\n## Low-readiness day classifier (temporal 70/30 split)\n\n")
        f.write(f"A low day is at or below the athlete's own mean minus 1 SD (training period). "
                f"Low-day rate: {low_info['low_day_rate_train']:.1%} (train), "
                f"{low_info['low_day_rate_test']:.1%} (test, {low_info['n_test']} days).\n\n")
        f.write(low.round(3).to_markdown(index=False))
        f.write(f"\n\nAthletes with load features masked (fewer than 5 logged sessions): "
                f"{', '.join(non_loggers) or 'none'}.")
        f.write("\n\n## Top drivers (mean |SHAP|, gradient boosting)\n\n")
        f.write(importance.sort_values(ascending=False).head(12).round(3)
                .rename("Mean |SHAP|").to_markdown())
        f.write("\n")
    print(f"\nSaved figures to {FIGURES}/ and tables to {REPORTS}/results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="Folder with p01, p02, ...")
    main(parser.parse_args().data)
