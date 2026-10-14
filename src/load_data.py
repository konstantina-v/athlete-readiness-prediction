"""Load the PMData sports logging dataset into one tidy daily table.

PMData (Thambawita et al., 2020) contains five months of data from 16 participants:
self-reported wellness and training sessions (PMSys app) plus Fitbit data.

Expected layout (only these four files per participant are needed):

    data/raw/
        p01/pmsys/wellness.csv
        p01/pmsys/srpe.csv
        p01/fitbit/sleep_score.csv
        p01/fitbit/resting_heart_rate.json
        p02/...

Output: one row per participant per calendar day.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

WELLNESS_COLS = [
    "readiness",
    "fatigue",
    "mood",
    "soreness",
    "stress",
    "sleep_quality",
    "sleep_duration_h",
]


def _to_date(series: pd.Series) -> pd.Series:
    """Parse timestamps of mixed formats and keep only the calendar date."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None).dt.normalize()


def load_wellness(path: Path) -> pd.DataFrame:
    """Morning wellness questionnaire (readiness, fatigue, mood, ...)."""
    df = pd.read_csv(path)
    df["date"] = _to_date(df["effective_time_frame"])
    cols = [c for c in WELLNESS_COLS if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # If someone reported twice on the same day, keep the last report.
    return (
        df.sort_values("effective_time_frame")
        .groupby("date")[cols]
        .last()
        .reset_index()
    )


def load_srpe(path: Path) -> pd.DataFrame:
    """Training sessions -> daily session-RPE load (RPE x minutes, Foster 2001)."""
    df = pd.read_csv(path)
    df["date"] = _to_date(df["end_date_time"])
    df["perceived_exertion"] = pd.to_numeric(df["perceived_exertion"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["session_load"] = df["perceived_exertion"] * df["duration_min"]

    def _first_activity(value) -> str:
        try:
            parsed = ast.literal_eval(str(value))
            return parsed[0] if isinstance(parsed, list) and parsed else str(value)
        except (ValueError, SyntaxError):
            return str(value)

    df["activity"] = df.get("activity_names", "").map(_first_activity)
    daily = (
        df.groupby("date")
        .agg(
            srpe_load=("session_load", "sum"),
            sessions=("session_load", "size"),
            training_minutes=("duration_min", "sum"),
        )
        .reset_index()
    )
    return daily


def load_sleep_score(path: Path) -> pd.DataFrame:
    """Fitbit sleep score. The timestamp is the wake-up time, so the date is
    the morning after the night of sleep."""
    df = pd.read_csv(path)
    df["date"] = _to_date(df["timestamp"])
    keep = {
        "overall_score": "sleep_score",
        "deep_sleep_in_minutes": "deep_sleep_min",
        "restlessness": "sleep_restlessness",
    }
    cols = [c for c in keep if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.groupby("date")[cols].mean().rename(columns=keep).reset_index()


def load_resting_hr(path: Path) -> pd.DataFrame:
    """Fitbit daily resting heart rate. Fitbit writes 0 when it has no estimate."""
    with open(path) as f:
        records = json.load(f)
    rows = []
    for rec in records:
        value = rec.get("value", {})
        hr = value.get("value") if isinstance(value, dict) else value
        rows.append({"dateTime": rec.get("dateTime"), "resting_hr": hr})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "resting_hr"])
    df["date"] = _to_date(df["dateTime"])
    df["resting_hr"] = pd.to_numeric(df["resting_hr"], errors="coerce").replace(0, np.nan)
    return df.groupby("date")["resting_hr"].mean().reset_index()


def load_participant(pdir: Path) -> pd.DataFrame | None:
    """Merge all sources for one participant onto a continuous daily index."""
    pid = pdir.name
    sources = {
        "wellness": (pdir / "pmsys" / "wellness.csv", load_wellness),
        "srpe": (pdir / "pmsys" / "srpe.csv", load_srpe),
        "sleep": (pdir / "fitbit" / "sleep_score.csv", load_sleep_score),
        "rhr": (pdir / "fitbit" / "resting_heart_rate.json", load_resting_hr),
    }
    frames = {}
    for name, (path, loader) in sources.items():
        if path.exists():
            try:
                frames[name] = loader(path)
            except Exception as exc:  # keep going if one file is malformed
                print(f"  [warn] {pid}/{name}: could not read ({exc})")

    if "wellness" not in frames or frames["wellness"].empty:
        print(f"  [skip] {pid}: no wellness data")
        return None

    # Use the period in which the participant actually reported wellness.
    w = frames["wellness"]
    days = pd.date_range(w["date"].min(), w["date"].max(), freq="D")
    daily = pd.DataFrame({"date": days})
    for df in frames.values():
        daily = daily.merge(df, on="date", how="left")

    # No logged session on a day means zero training load, not missing data.
    for col in ["srpe_load", "sessions", "training_minutes"]:
        if col not in daily:
            daily[col] = 0.0
        daily[col] = daily[col].fillna(0.0)

    daily.insert(0, "participant", pid)
    return daily


def load_pmdata(raw_dir: str | Path) -> pd.DataFrame:
    """Load every participant folder (p01, p02, ...) under raw_dir."""
    raw_dir = Path(raw_dir)
    pdirs = sorted(p for p in raw_dir.glob("p*") if p.is_dir())
    if not pdirs:
        raise FileNotFoundError(
            f"No participant folders found in {raw_dir}. "
            "See README.md for how to download PMData."
        )
    frames = [load_participant(p) for p in pdirs]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True)
