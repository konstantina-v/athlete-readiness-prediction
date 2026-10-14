"""Create a small synthetic dataset with the same file formats as PMData.

Used only to test that the pipeline runs end to end without the real data.
The numbers are invented; never report results from this data.

    python tests/make_synthetic_pmdata.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_participant(pid: str, out: Path, rng: np.random.Generator, days: int = 150):
    dates = pd.date_range("2019-11-01", periods=days, freq="D")
    base_ready = rng.uniform(5, 8)
    base_rhr = rng.uniform(48, 65)

    sessions, wellness, sleep, rhr = [], [], [], []
    fatigue_state = 0.0
    for i, d in enumerate(dates):
        trains = rng.random() < 0.6
        load = 0.0
        if trains:
            rpe = int(rng.integers(3, 9))
            dur = int(rng.integers(30, 100))
            load = rpe * dur
            sessions.append({
                "end_date_time": f"{d.date()}T18:{rng.integers(0, 59):02d}:00.000Z",
                "activity_names": "['running']",
                "perceived_exertion": rpe,
                "duration_min": dur,
            })
        fatigue_state = 0.8 * fatigue_state + load / 400
        sleep_score = float(np.clip(rng.normal(78, 6) - 2 * fatigue_state, 40, 98))
        rhr_val = base_rhr + 1.5 * fatigue_state + rng.normal(0, 1.5)
        readiness = np.clip(round(base_ready - 0.6 * fatigue_state
                                  + (sleep_score - 78) / 10 + rng.normal(0, 0.8)), 0, 10)
        if rng.random() < 0.9:  # some missed reports
            wellness.append({
                "effective_time_frame": f"{d.date()}T08:15:00.000Z",
                "fatigue": int(np.clip(round(2 + fatigue_state), 1, 5)),
                "mood": int(rng.integers(2, 6)),
                "readiness": int(readiness),
                "sleep_duration_h": round(float(rng.normal(7.2, 0.7)), 1),
                "sleep_quality": int(rng.integers(2, 6)),
                "soreness": int(np.clip(round(2 + fatigue_state / 1.5), 1, 5)),
                "soreness_area": "[]",
                "stress": int(rng.integers(1, 5)),
            })
        sleep.append({
            "sleep_log_entry_id": i,
            "timestamp": f"{d.date()}T07:05:30Z",
            "overall_score": round(sleep_score),
            "composition_score": 20,
            "revitalization_score": 20,
            "duration_score": 40,
            "deep_sleep_in_minutes": int(rng.integers(40, 110)),
            "resting_heart_rate": round(rhr_val),
            "restlessness": round(float(rng.uniform(0.05, 0.12)), 3),
        })
        rhr.append({
            "dateTime": f"{d.date()} 00:00:00",
            "value": {"date": d.strftime("%m/%d/%y"),
                      "value": 0.0 if rng.random() < 0.05 else round(rhr_val, 2),
                      "error": 6.5},
        })

    (out / pid / "pmsys").mkdir(parents=True, exist_ok=True)
    (out / pid / "fitbit").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wellness).to_csv(out / pid / "pmsys" / "wellness.csv", index=False)
    pd.DataFrame(sessions).to_csv(out / pid / "pmsys" / "srpe.csv", index=False)
    pd.DataFrame(sleep).to_csv(out / pid / "fitbit" / "sleep_score.csv", index=False)
    with open(out / pid / "fitbit" / "resting_heart_rate.json", "w") as f:
        json.dump(rhr, f)


def main(out: str, n: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    out_path = Path(out)
    for i in range(1, n + 1):
        make_participant(f"p{i:02d}", out_path, rng)
    print(f"Wrote {n} synthetic participants to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    main(parser.parse_args().out)
