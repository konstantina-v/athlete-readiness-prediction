"""Training-load and readiness features, computed per athlete.

Framing: at the end of day t, predict the readiness the athlete will report on
the morning of day t+1. Every feature therefore uses only information that is
available by the evening of day t (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACUTE_DAYS = 7
CHRONIC_DAYS = 28

# Athletes who logged fewer sessions than this over the whole period almost
# certainly trained without logging. For them a missing log does not mean a
# rest day, so their load features are set to missing (not zero) and the
# models rely on their wellness, sleep and heart-rate data instead.
MIN_LOGGED_SESSIONS = 5

LOAD_FEATURES = [
    "srpe_load", "acute_load", "chronic_load", "acwr", "acwr_ewma",
    "monotony", "strain", "rest_days_7d",
]


def _ewma(series: pd.Series, days: int) -> pd.Series:
    """Exponentially weighted load, lambda = 2 / (N + 1) (Williams et al., 2017)."""
    return series.ewm(alpha=2 / (days + 1), adjust=False).mean()


def add_load_features(g: pd.DataFrame) -> pd.DataFrame:
    load = g["srpe_load"]
    g["acute_load"] = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).mean()
    g["chronic_load"] = load.rolling(CHRONIC_DAYS, min_periods=CHRONIC_DAYS // 2).mean()
    g["acwr"] = g["acute_load"] / g["chronic_load"].replace(0, np.nan)

    g["acwr_ewma"] = _ewma(load, ACUTE_DAYS) / _ewma(load, CHRONIC_DAYS).replace(0, np.nan)
    # The EWMA ratio is unstable before roughly one chronic window of data.
    g.loc[g.index[: CHRONIC_DAYS // 2], "acwr_ewma"] = np.nan

    weekly_mean = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).mean()
    weekly_std = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).std()
    g["monotony"] = weekly_mean / weekly_std.replace(0, np.nan)  # Foster (1998)
    g["strain"] = load.rolling(ACUTE_DAYS, min_periods=ACUTE_DAYS).sum() * g["monotony"]
    g["rest_days_7d"] = (load == 0).rolling(ACUTE_DAYS, min_periods=1).sum()
    return g


def add_physiology_features(g: pd.DataFrame) -> pd.DataFrame:
    """Deviations from the athlete's own recent baseline (past days only)."""
    if "resting_hr" in g:
        baseline = g["resting_hr"].shift(1).rolling(CHRONIC_DAYS, min_periods=7).median()
        g["rhr_vs_baseline"] = g["resting_hr"] - baseline
    if "sleep_score" in g:
        baseline = g["sleep_score"].shift(1).rolling(CHRONIC_DAYS, min_periods=7).median()
        g["sleep_score_vs_baseline"] = g["sleep_score"] - baseline
    return g


def add_wellness_features(g: pd.DataFrame) -> pd.DataFrame:
    r = g["readiness"]
    g["readiness_7d_mean"] = r.rolling(ACUTE_DAYS, min_periods=3).mean()
    personal = r.shift(1).rolling(CHRONIC_DAYS, min_periods=7)
    g["readiness_vs_baseline"] = r - personal.mean()
    g["readiness_change_1d"] = r - r.shift(1)
    return g


def add_target(g: pd.DataFrame) -> pd.DataFrame:
    """Tomorrow's self-reported readiness (0-10). The daily index is continuous,
    so shift(-1) is exactly the next calendar day."""
    g["target_readiness_next_day"] = g["readiness"].shift(-1)
    return g


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in daily.sort_values(["participant", "date"]).groupby("participant"):
        g = g.copy()
        g = add_load_features(g)
        g["logs_training"] = (g["srpe_load"] > 0).sum() >= MIN_LOGGED_SESSIONS
        if not g["logs_training"].iloc[0]:
            g[LOAD_FEATURES] = np.nan
        g = add_physiology_features(g)
        g = add_wellness_features(g)
        g = add_target(g)
        g["day_index"] = np.arange(len(g))
        out.append(g)
    return pd.concat(out, ignore_index=True)


# Features available by the evening of day t.
FEATURES = [
    # Today's wellness report
    "readiness", "fatigue", "mood", "soreness", "stress",
    "sleep_quality", "sleep_duration_h",
    # Wellness trends relative to the athlete's own baseline
    "readiness_7d_mean", "readiness_vs_baseline", "readiness_change_1d",
    # Training load
    "srpe_load", "acute_load", "chronic_load", "acwr", "acwr_ewma",
    "monotony", "strain", "rest_days_7d",
    # Fitbit physiology (last night's sleep, today's resting HR)
    "sleep_score", "deep_sleep_min", "sleep_restlessness",
    "sleep_score_vs_baseline", "resting_hr", "rhr_vs_baseline",
]

FEATURE_LABELS = {
    "readiness": "Readiness today",
    "fatigue": "Fatigue today",
    "mood": "Mood today",
    "soreness": "Soreness today",
    "stress": "Stress today",
    "sleep_quality": "Sleep quality (self-report)",
    "sleep_duration_h": "Sleep duration (self-report)",
    "readiness_7d_mean": "Readiness, 7-day mean",
    "readiness_vs_baseline": "Readiness vs personal baseline",
    "readiness_change_1d": "Readiness change since yesterday",
    "srpe_load": "Training load today (sRPE)",
    "acute_load": "Acute load (7 d)",
    "chronic_load": "Chronic load (28 d)",
    "acwr": "ACWR (rolling)",
    "acwr_ewma": "ACWR (EWMA)",
    "monotony": "Training monotony",
    "strain": "Training strain",
    "rest_days_7d": "Rest days, last 7",
    "sleep_score": "Fitbit sleep score",
    "deep_sleep_min": "Deep sleep (min)",
    "sleep_restlessness": "Sleep restlessness",
    "sleep_score_vs_baseline": "Sleep score vs baseline",
    "resting_hr": "Resting heart rate",
    "rhr_vs_baseline": "Resting HR vs baseline",
}


def available_features(df: pd.DataFrame) -> list[str]:
    """Drop features that are entirely missing (e.g. if a Fitbit file is absent)."""
    return [f for f in FEATURES if f in df and df[f].notna().any()]
