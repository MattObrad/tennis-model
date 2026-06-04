"""
calibrate_tennis.py -- Platt-scale calibration analysis for ITF Women Elo.

FINDING (2026-06-04):
  After the symmetric-K rebuild, the 2025 holdout is already well-calibrated:
  every bin is within ±4pp of the actual win rate.  Fitting Platt scaling on
  2023-2024 data (which showed +5-7pp underconfidence) and applying it to 2025
  made the Brier slightly WORSE (0.1973 → 0.1980) and pushed two high-prob bins
  into overconfidence territory.

  The underconfidence in 2021-2024 was a side-effect of the asymmetric-K ratings
  that have since been rebuilt.  predict_tennis.py uses raw elo_prob for now.

  Re-run this script after ~6 months of new data to decide whether calibration
  is warranted again.

Usage:
  python calibrate_tennis.py           # fit, report, save pkl
  python calibrate_tennis.py --no-save # fit and report only (no file written)
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# ── mirror backtest imports ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
import bisect
from dataclasses import dataclass
from typing import Optional

BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "data" / "tennis.db"
CAL_PATH  = BASE_DIR / "data" / "calibration_tennis.pkl"

SCALE = 400.0
VAL_START  = "2023-01-01";  VAL_END  = "2024-12-31"
HOLD_START = "2025-01-01";  HOLD_END = "2025-12-31"
MIN_MATCHES = 10   # must match predict_tennis.py gate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── tiny elo index (mirrors backtest_tennis.py) ────────────────────────────────
@dataclass
class Snap:
    date: str
    rid:  int
    ov:   float
    n:    int

def load_elo_index(conn: sqlite3.Connection) -> dict[str, list[Snap]]:
    raw = conn.execute(
        "SELECT player_name, match_date, id, overall_elo, matches_played "
        "FROM player_elo ORDER BY player_name, match_date ASC, id ASC"
    ).fetchall()
    idx: dict[str, list[Snap]] = {}
    for nm, dt, rid, ov, n in raw:
        idx.setdefault(nm, []).append(Snap(dt, rid, ov, n))
    return idx

def lookup(idx: dict, name: str, before: str) -> Optional[Snap]:
    snaps = idx.get(name)
    if not snaps:
        return None
    dates = [s.date for s in snaps]
    pos = bisect.bisect_left(dates, before)
    return snaps[pos - 1] if pos > 0 else None

def elo_prob(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / SCALE))


# ── load match data for a period ──────────────────────────────────────────────
def load_period(conn: sqlite3.Connection, idx: dict,
                start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where:
      X[i] = logit of elo_prob for one side of match i
      y[i] = 1 if that side won, 0 if lost

    Both sides of every match are included so the model trains on the
    full [0,1] probability range and is symmetric around 0.5.
    """
    rows = conn.execute(
        "SELECT tourney_date, winner_name, loser_name, score "
        "FROM matches WHERE tourney_date BETWEEN ? AND ? "
        "ORDER BY tourney_date ASC, id ASC",
        (start, end),
    ).fetchall()

    Xs, ys = [], []
    skipped = 0
    for date, wname, lname, score in rows:
        su = (score or "").upper()
        if "W/O" in su or "DEF" in su or "RET" in su:
            skipped += 1
            continue
        ws = lookup(idx, wname, date)
        ls = lookup(idx, lname, date)
        if ws is None or ls is None or ws.n < MIN_MATCHES or ls.n < MIN_MATCHES:
            continue
        p_w = elo_prob(ws.ov, ls.ov)
        # Clip to avoid log(0)
        p_w = max(1e-6, min(1 - 1e-6, p_w))
        p_l = 1.0 - p_w
        # Winner side (y=1)
        Xs.append(math.log(p_w / (1 - p_w)))
        ys.append(1)
        # Loser side (y=0)
        Xs.append(math.log(p_l / (1 - p_l)))
        ys.append(0)

    log.info("  Period %s–%s: %d matches → %d training pairs  (%d non-result skipped)",
             start[:4], end[:4], len(Xs) // 2, len(Xs), skipped)
    return np.array(Xs).reshape(-1, 1), np.array(ys)


# ── calibration helpers ────────────────────────────────────────────────────────
def apply_calibration(model: LogisticRegression, raw_probs: np.ndarray) -> np.ndarray:
    """Vectorised: raw_probs shape (N,), returns calibrated (N,)."""
    clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logits  = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def calibration_table(raw_probs: np.ndarray, outcomes: np.ndarray,
                      cal_probs: np.ndarray, label: str) -> None:
    """Print binned calibration curve for raw vs calibrated probabilities."""
    BIN_LO  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    BIN_HI  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
    BIN_LBL = ["50-55%", "55-60%", "60-65%", "65-70%",
               "70-75%", "75-80%", "80-85%", "85-90%", "90%+"]

    # We evaluate from the favourite's perspective so bins cover [0.5, 1.0].
    # Since both sides are in the array, select only p_fav >= 0.5 rows.
    fav_mask = raw_probs >= 0.50
    rp  = raw_probs[fav_mask]
    cp  = cal_probs[fav_mask]
    # outcome for p>=0.5 side: if raw >= 0.5 that side WAS the winner → y=1
    # (by construction: we appended winner first with y=1, then loser with y=0)
    out = outcomes[fav_mask]

    print(f"\n  CALIBRATION CURVE — {label}")
    print(f"  {'Bin':8} {'N':>6}  {'Raw%':>7}  {'Actual%':>8}  {'Raw gap':>8}  "
          f"{'Cal%':>7}  {'Cal gap':>8}")
    print("  " + "-" * 62)

    for lo, hi, lbl in zip(BIN_LO, BIN_HI, BIN_LBL):
        mask = (rp >= lo) & (rp < hi)
        n = mask.sum()
        if n < 10:
            continue
        actual  = out[mask].mean()
        raw_mid = rp[mask].mean()
        cal_mid = cp[mask].mean()
        raw_gap = actual - raw_mid
        cal_gap = actual - cal_mid
        raw_flag = "OK" if abs(raw_gap) <= 0.04 else ("underconf" if raw_gap > 0 else "OVERconf")
        cal_flag = "OK" if abs(cal_gap) <= 0.04 else ("underconf" if cal_gap > 0 else "OVERconf")
        print(f"  {lbl:8} {n:>6,}  {raw_mid*100:>6.1f}%  {actual*100:>7.1f}%  "
              f"{raw_gap*100:>+7.1f}%  {cal_mid*100:>6.1f}%  {cal_gap*100:>+7.1f}%  "
              f"({cal_flag})")

    # Overall Brier
    brier_raw = float(np.mean((raw_probs - outcomes) ** 2))
    brier_cal = float(np.mean((cal_probs - outcomes) ** 2))
    print(f"\n  Brier(raw)        = {brier_raw:.4f}")
    print(f"  Brier(calibrated) = {brier_cal:.4f}  "
          f"({'better' if brier_cal < brier_raw else 'WORSE'})")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true",
                    help="Fit and report but do not save calibration_tennis.pkl")
    args = ap.parse_args()

    conn    = sqlite3.connect(DB_PATH)
    log.info("Loading Elo index from player_elo...")
    idx     = load_elo_index(conn)

    # ── fit on validation 2023-2024 ──────────────────────────────────────────
    log.info("Building training data (2023-2024 validation)...")
    X_val, y_val = load_period(conn, idx, VAL_START, VAL_END)

    # Platt scaling: logistic regression on logit(raw_elo_prob)
    # C=1e6 → minimal regularisation (almost pure Platt)
    clf = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    clf.fit(X_val, y_val)
    a, b = float(clf.coef_[0][0]), float(clf.intercept_[0])
    log.info("Platt fit: a=%.4f  b=%.4f  (a>1 = stretches toward extremes)", a, b)

    # ── in-sample report (validation) ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CALIBRATION REPORT — ITF Women Elo  (Platt scaling)")
    print("  Fit on: 2023-2024 validation     Tested on: 2025 holdout")
    print(f"  Platt parameters: a={a:.4f}  b={b:.4f}")
    print("=" * 70)

    raw_val = 1.0 / (1.0 + np.exp(-X_val.ravel()))   # sigmoid(logit(p)) = p
    cal_val = apply_calibration(clf, raw_val)
    calibration_table(raw_val, y_val, cal_val, "2023-2024 VALIDATION (in-sample)")

    # ── holdout report (2025) ─────────────────────────────────────────────────
    log.info("Building holdout data (2025)...")
    X_hold, y_hold = load_period(conn, idx, HOLD_START, HOLD_END)
    raw_hold = 1.0 / (1.0 + np.exp(-X_hold.ravel()))
    cal_hold = apply_calibration(clf, raw_hold)
    calibration_table(raw_hold, y_hold, cal_hold, "2025 HOLDOUT (out-of-sample)")

    conn.close()

    # ── save ─────────────────────────────────────────────────────────────────
    if args.no_save:
        print("\n  --no-save: calibration NOT written.")
    else:
        CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CAL_PATH, "wb") as f:
            pickle.dump(clf, f)
        print(f"\n  Saved: {CAL_PATH}")
        print("  Load in predict_tennis.py: pickle.load(open(CAL_PATH, 'rb'))")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
