"""A minimal model registry: every trained model gets its own versioned folder,
and a small pointer file says which version is in production.

    models/
        registry/
            v20261014-163000/
                model.joblib      the fitted model
                metadata.json     the "model card": data, features, scores
        production.json           {"version": "v20261014-163000", ...}

Why: in production you must always be able to answer "which model made this
prediction, what data was it trained on, and how good was it?", and roll back
to an older version if a new one misbehaves.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

MODELS_DIR = Path("models")
REGISTRY_DIR = MODELS_DIR / "registry"
PRODUCTION_POINTER = MODELS_DIR / "production.json"


def new_version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_model(model, metadata: dict, version: str | None = None) -> str:
    """Save a model and its model card under a new version folder."""
    version = version or new_version()
    folder = REGISTRY_DIR / version
    folder.mkdir(parents=True, exist_ok=False)  # never overwrite a version
    joblib.dump(model, folder / "model.joblib")
    metadata = {"version": version, **metadata}
    with open(folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return version


def load_model(version: str):
    """Return (model, metadata) for a given version."""
    folder = REGISTRY_DIR / version
    if not folder.exists():
        raise FileNotFoundError(f"No model version {version} in {REGISTRY_DIR}")
    model = joblib.load(folder / "model.joblib")
    with open(folder / "metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


def production_version() -> str | None:
    if not PRODUCTION_POINTER.exists():
        return None
    with open(PRODUCTION_POINTER) as f:
        return json.load(f)["version"]


def load_production_model():
    version = production_version()
    if version is None:
        raise FileNotFoundError("No production model yet. Run: python train.py --promote")
    return load_model(version)


def promote(version: str, reason: str) -> None:
    """Point production at a version. The previous version stays in the
    registry, so rolling back is just promoting it again."""
    load_model(version)  # fail early if the version does not exist
    previous = production_version()
    with open(PRODUCTION_POINTER, "w") as f:
        json.dump(
            {
                "version": version,
                "previous_version": previous,
                "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": reason,
            },
            f,
            indent=2,
        )


def list_versions() -> list[str]:
    if not REGISTRY_DIR.exists():
        return []
    return sorted(p.name for p in REGISTRY_DIR.iterdir() if p.is_dir())
