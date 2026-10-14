"""End-to-end smoke test on synthetic data with the PMData file format."""

from pathlib import Path

from src.features import available_features, build_features
from src.load_data import load_pmdata
from src.model import TARGET, evaluate_temporal
from tests.make_synthetic_pmdata import main as make_synthetic


def test_pipeline_runs(tmp_path: Path):
    make_synthetic(str(tmp_path), n=4)
    df = build_features(load_pmdata(tmp_path))
    assert df["participant"].nunique() == 4
    # Next-day target must equal the following calendar day's readiness
    g = df[df["participant"] == "p01"].reset_index(drop=True)
    assert g[TARGET].iloc[:-1].equals(g["readiness"].iloc[1:].reset_index(drop=True))
    results, *_ = evaluate_temporal(df, available_features(df))
    assert {"Gradient boosting", "Ridge regression"} <= set(results["Model"])
    assert (results["MAE"] > 0).all()
