"""Static figures for the README (matplotlib).

Palette: categorical slots 1-2 (blue, orange) on a light surface, recessive
grey axes and grid. One y-axis per panel; different measures get their own panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY_MARK = "#a3a29c"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
BAND = "#eeede8"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
    }
)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def athlete_monitoring(df: pd.DataFrame, participant: str, path: Path):
    """Three aligned panels: daily load, ACWR with the 0.8-1.3 reference band,
    and morning readiness."""
    g = df[df["participant"] == participant].sort_values("date")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    ax.bar(g["date"], g["srpe_load"], width=0.8, color=BLUE, linewidth=0)
    ax.set_title("Daily training load (sRPE, AU)")

    ax = axes[1]
    ax.axhspan(0.8, 1.3, color=BAND, zorder=0)
    ax.plot(g["date"], g["acwr_ewma"], color=BLUE, linewidth=2)
    ax.set_title("Acute:chronic workload ratio (EWMA) – shaded band: 0.8–1.3")

    ax = axes[2]
    ax.plot(g["date"], g["readiness"], color=BLUE, linewidth=2, marker="o",
            markersize=3.5)
    ax.set_ylim(0, 10.5)
    ax.set_title("Morning readiness (0–10, self-reported)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.suptitle(f"Athlete monitoring view – participant {participant}",
                 x=0.125, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    _save(fig, path)


def model_comparison(results: pd.DataFrame, path: Path, title: str):
    r = results.sort_values("MAE", ascending=False)
    colors = [GREY_MARK if n.startswith("Baseline") else BLUE for n in r["Model"]]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(r) + 1.2))
    ax.barh(r["Model"], r["MAE"], color=colors, height=0.55, linewidth=0)
    for y, v in enumerate(r["MAE"]):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=TEXT)
    ax.set_xlabel("Mean absolute error, readiness points (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title(title)
    ax.set_xlim(0, r["MAE"].max() * 1.15)
    _save(fig, path)


def prediction_example(test: pd.DataFrame, participant: str, model_col: str, path: Path):
    g = test[test["participant"] == participant].sort_values("date")
    target_date = g["date"] + pd.Timedelta(days=1)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(target_date, g["target_readiness_next_day"], color=BLUE, linewidth=2,
            marker="o", markersize=4, label="Reported readiness")
    ax.plot(target_date, g[model_col], color=ORANGE, linewidth=2,
            label="Predicted the evening before")
    ax.set_ylim(0, 10.5)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Next-day readiness on unseen days – participant {participant}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _save(fig, path)


def feature_importance(importance: pd.Series, path: Path, top: int = 12):
    imp = importance.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6, linewidth=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Mean |SHAP value| (readiness points)")
    ax.set_title("What drives tomorrow's readiness prediction")
    _save(fig, path)


def low_day_alerts(test: pd.DataFrame, participant: str, risk_col: str, path: Path):
    """Risk score for each evening, with the days that turned out low marked."""
    g = test[test["participant"] == participant].sort_values("date")
    target_date = g["date"] + pd.Timedelta(days=1)
    low = g["low_day_next"] == 1
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(target_date, g[risk_col], color=BLUE, linewidth=2,
            label="Predicted risk of a low day (evening before)")
    ax.scatter(target_date[low], g.loc[low, risk_col], s=60, color=ORANGE,
               edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Day that turned out low")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk score")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Low-readiness alerts on unseen days – participant {participant}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _save(fig, path)
