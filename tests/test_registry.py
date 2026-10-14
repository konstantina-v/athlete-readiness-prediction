"""Train on synthetic data, save a version, promote it, and load it back."""

import os

import numpy as np

from tests.make_synthetic_pmdata import main as make_synthetic


def test_train_save_promote_load(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_synthetic(str(data), n=4)
    monkeypatch.chdir(tmp_path)  # registry writes to ./models

    from src import registry
    from train import train

    version = train(str(data))
    assert registry.production_version() == version  # first model is promoted

    model, meta = registry.load_production_model()
    assert meta["version"] == version
    assert set(meta["features"]) and "evaluation" in meta
    X = np.zeros((1, len(meta["features"])))
    p = model.predict_proba(X)[0, 1]
    assert 0.0 <= p <= 1.0
