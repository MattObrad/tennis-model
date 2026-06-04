"""
backtest_tennis.py -- Walk-forward validation of the ITF Women Elo model.

Periods
-------
  Training   2015-2022  (context only -- Elo already built by elo_engine.py)
  Validation 2023-2024  (calibration check, threshold exploration)
  Holdout    2025       (first and final honest look)

Market proxy
------------
  We have no historical Kambi odds for 2023-2024 ITF matches, so we use
  WTA ranking-implied probability as a naive stand-in:

      P(player_A wins) = rank_B / (rank_A + rank_B)

  Note: the spec wrote `1/(1+rank_loser/rank_winner)` which inverts the
  direction (gives high probability to HIGHER rank numbers = worse players).
  The formula above is corrected: lower rank number = better player.

  IMPORTANT CAVEAT: the rank proxy is weaker than live Kambi odds.  An edge
  against WTA ranking does NOT guarantee an edge against Kambi.  Treat any
  positive ROI here as directional evidence only.  The real test is CLV on
  live Kambi odds over 30+ predictions.

  ~53% of val matches and ~76% of holdout matches have both players ranked.
  Unranked matches are included in Elo accuracy stats but excluded from
  edge / ROI analysis.

Usage
-----
  python backtest_tennis.py               # val + holdout
  python backtest_tennis.py --val-only    # 2023-2024 only
  python backtest_tennis.py --holdout-only  # 2025 only
  python backtest_tennis.py --min-matches 10  # require 10+ prior matches
"""

from __future__ import annotations

import argparse
import bisect
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data" / "tennis.db"

SCALE  = 400.0   # must match elo_engine.py

VAL_START  = "2023-01-01";  VAL_END  = "2024-12-31"
HOLD_START = "2025-01-01";  HOLD_END = "2025-12-31"

EDGE_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
DEFAULT_MIN_MATCHES = 5    # require each player to have >= N prior matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Elo math ──────────────────────────────────────────────────────────────────
def elo_prob(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / SCALE))


# ── In-memory Elo index ───────────────────────────────────────────────────────
@dataclass
class _Snap:
    """One row from player_elo, used for point-in-time lookup."""
    date: str
    rid:  int
    ov:   float   # overall_elo
    cl:   float   # clay_elo
    ha:   float   # hard_elo
    gr:   float   # grass_elo
    n:    int     # matches_played


def load_elo_index(conn: sqlite3.Connection) -> dict[str, list[_Snap]]:
    """
    Pull the entire player_elo table into a dict keyed by player_name.
    Each list is sorted (match_date ASC, id ASC) for bisect lookup.
    ~547K rows -- loads in ~2 seconds.
    """
    log.info("Loading player_elo index (~547K rows)...")
    raw = conn.execute(
        """
        SELECT player_name, match_date, id,
               overall_elo, clay_elo, hard_elo, grass_elo, matches_played
        FROM   player_elo
        ORDER  BY player_name, match_date ASC, id ASC
        """
    ).fetchall()
    idx: dict[str, list[_Snap]] = {}
    for nm, dt, rid, ov, cl, ha, gr, n in raw:
        if nm not in idx:
            idx[nm] = []
        idx[nm].append(_Snap(dt, rid, ov, cl, ha, gr, n))
    log.info("Index built: %d players, %d snapshots.", len(idx), len(raw))
    return idx


def lookup_elo(idx: dict, name: str, before_date: str) -> Optional[_Snap]:
    """
    Return the most recent _Snap for `name` with match_date < before_date.
    O(log n) via bisect.  Returns None if player has no prior history.

    Since entries are (date ASC, id ASC), bisect_left on dates gives the
    first index where date >= before_date; the entry just before it is the
    last with date < before_date (and the highest id on that date, which is
    what we want for within-tournament ordering).
    """
    entries = idx.get(name)
    if not entries:
        return None
    dates = [e.date for e in entries]
    pos   = bisect.bisect_left(dates, before_date)
    return entries[pos - 1] if pos > 0 else None


# ── Match record ──────────────────────────────────────────────────────────────
@dataclass
class Rec:
    date:          str
    year:          int
    surface:       str
    level:         str
    winner_name:   str
    loser_name:    str
    # Elo outputs
    p_winner:      float        # P(actual winner wins) from Elo
    p_fav:         float        # max(p_winner, 1-p_winner) >= 0.5
    fav_is_winner: bool         # True  → Elo-favourite = actual winner
    w_n:           int          # winner's prior match count
    l_n:           int          # loser's prior match count
    # Rank proxy (None if either player unranked)
    rank_p_winner: Optional[float] = None   # P(winner wins) from WTA rank
    rank_p_fav:    Optional[float] = None   # P(Elo-fav wins) from WTA rank
    edge:          Optional[float] = None   # p_fav - rank_p_fav


# ── Data loading ──────────────────────────────────────────────────────────────
def load_period(
    conn:      sqlite3.Connection,
    idx:       dict,
    start:     str,
    end:       str,
    min_n:     int,
) -> list[Rec]:
    raw = conn.execute(
        """
        SELECT tourney_date, data_year, surface, tourney_level,
               winner_name, loser_name, winner_rank, loser_rank
        FROM   matches
        WHERE  tourney_date BETWEEN ? AND ?
        ORDER  BY tourney_date ASC, id ASC
        """,
        (start, end),
    ).fetchall()

    records: list[Rec] = []
    skipped_no_elo = 0
    skipped_min_n  = 0

    for date, yr, surf, lvl, wname, lname, w_rank, l_rank in raw:
        if surf not in ("Clay", "Hard", "Grass"):
            surf = "Hard"

        ws = lookup_elo(idx, wname, date)
        ls = lookup_elo(idx, lname, date)

        if ws is None or ls is None:
            skipped_no_elo += 1
            continue

        if ws.n < min_n or ls.n < min_n:
            skipped_min_n += 1
            continue

        # Elo probability
        p_w  = elo_prob(ws.ov, ls.ov)
        p_fav = p_w if p_w >= 0.5 else 1.0 - p_w
        fav_is_winner = p_w >= 0.5

        # Rank proxy: P(winner wins) = loser_rank / (winner_rank + loser_rank)
        # Lower WTA rank = better player, so loser's (higher) rank goes in numerator.
        rank_p_w = rank_p_fav = edge = None
        if w_rank and l_rank and w_rank > 0 and l_rank > 0:
            rank_p_w   = l_rank / (w_rank + l_rank)
            rank_p_fav = rank_p_w if fav_is_winner else (1.0 - rank_p_w)
            edge       = p_fav - rank_p_fav

        records.append(Rec(
            date=date, year=yr, surface=surf, level=lvl,
            winner_name=wname, loser_name=lname,
            p_winner=p_w, p_fav=p_fav, fav_is_winner=fav_is_winner,
            w_n=ws.n, l_n=ls.n,
            rank_p_winner=rank_p_w, rank_p_fav=rank_p_fav, edge=edge,
        ))

    log.info(
        "  Loaded %d records  (skipped: %d no-Elo, %d under min_n=%d)",
        len(records), skipped_no_elo, skipped_min_n, min_n,
    )
    return records


# ── Metric helpers ────────────────────────────────────────────────────────────
def _acc(recs: list[Rec]) -> float:
    return sum(r.fav_is_winner for r in recs) / len(recs) if recs else 0.0

def _brier(recs: list[Rec]) -> float:
    return sum((r.p_winner - 1.0) ** 2 for r in recs) / len(recs) if recs else 0.5

def _logloss(recs: list[Rec]) -> float:
    return (-sum(math.log(max(r.p_winner, 1e-7)) for r in recs) / len(recs)
            if recs else math.log(2))

def _rank_acc(recs: list[Rec]) -> Optional[float]:
    sub = [r for r in recs if r.rank_p_winner is not None]
    if not sub:
        return None
    return sum(1 for r in sub if r.rank_p_winner > 0.5) / len(sub)

# Calibration bins: 50-55%, 55-60%, ... 95%+
_BIN_LO  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
_BIN_HI  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
_BIN_LBL = ["50-55%","55-60%","60-65%","65-70%","70-75%",
            "75-80%","80-85%","85-90%","90-95%","95%+"]

def _bin_idx(p_fav: float) -> int:
    for i, hi in enumerate(_BIN_HI):
        if p_fav < hi:
            return i
    return len(_BIN_LBL) - 1


# ── Output helpers ────────────────────────────────────────────────────────────
W = 64
BAR = "-" * W

def _hdr(title: str) -> None:
    print();  print("=" * W);  print(f"  {title}");  print("=" * W)


def print_overview(recs: list[Rec], label: str) -> None:
    n     = len(recs)
    n_rk  = sum(1 for r in recs if r.rank_p_fav is not None)
    ra    = _rank_acc(recs)

    print(f"\n  {label}")
    print(f"  {'Metric':<28} {'Elo':>9}  {'Rank proxy':>11}  {'Baseline':>9}")
    print(f"  {BAR}")
    print(f"  {'Accuracy (fav wins)':<28} {_acc(recs)*100:>8.1f}%  "
          f"{(ra*100 if ra else 0):>10.1f}%  {'50.0%':>9}")
    print(f"  {'Brier Score':<28} {_brier(recs):>9.4f}  "
          f"{'—':>11}  {'0.2500':>9}")
    print(f"  {'Log-Loss':<28} {_logloss(recs):>9.4f}  "
          f"{'—':>11}  {'0.6931':>9}")
    print(f"  {'Matches evaluated':<28} {n:>9,}  {n_rk:>10,}  {'':>9}")
    print(f"  {'% with WTA rankings':<28} {'':>9}  {n_rk/n*100:>10.1f}%  {'':>9}")


def print_calibration(recs: list[Rec], label: str = "") -> None:
    if label:
        print(f"\n  CALIBRATION — {label}")
    else:
        print(f"\n  CALIBRATION CURVE")
    print(f"  {'Pred%':<10} {'Actual%':>8} {'N':>8}   {'Gap':>6}  verdict")
    print(f"  {BAR}")

    bins = defaultdict(list)
    for r in recs:
        bins[_bin_idx(r.p_fav)].append(int(r.fav_is_winner))

    any_bad = False
    for i, lbl in enumerate(_BIN_LBL):
        vals = bins[i]
        if not vals:
            continue
        actual = sum(vals) / len(vals)
        mid    = (_BIN_LO[i] + _BIN_HI[i]) / 2
        gap    = actual - mid
        flag   = ("ok      " if abs(gap) <= 0.04
                  else ("OVERCONF" if gap < -0.04 else "underconf"))
        if "OVER" in flag:
            any_bad = True
        bar = ("+" if gap >= 0 else "-") * min(8, int(abs(gap) * 100))
        print(f"  {lbl:<10} {actual*100:>7.1f}% {len(vals):>8,}  {gap*100:>+5.1f}%  "
              f"{flag}  {bar}")

    if any_bad:
        print(f"\n  *** OVERCONFIDENT bins detected: model overstates win probability.")
        print(f"      Consider adding Platt calibration before live deployment.")


def print_edge_analysis(recs: list[Rec], label: str = "") -> None:
    """
    For each edge threshold, show bets count, fav win%, implied odds, and ROI.
    ROI = flat $1 bets at fair (no-vig) rank-implied decimal odds.
    Apply ~7% mental discount for Kambi vig before interpreting as real edge.
    """
    ranked = [r for r in recs if r.edge is not None]
    if not ranked:
        print("\n  No ranked matches available for edge analysis.")
        return

    if label:
        print(f"\n  EDGE ANALYSIS — {label}")
    else:
        print(f"\n  EDGE ANALYSIS")
    print(f"  Min    {'N Bets':>8} {'Win%':>7} {'Avg Odds':>10} {'ROI (fair)':>11}  signal")
    print(f"  edge   {BAR[6:]}")

    results = {}
    for thresh in EDGE_THRESHOLDS:
        bets = [r for r in ranked if r.edge >= thresh]
        if len(bets) < 10:
            continue
        wins = sum(r.fav_is_winner for r in bets)
        w_pct = wins / len(bets)
        avg_odds = sum(1.0 / r.rank_p_fav for r in bets) / len(bets)
        profits  = [(1.0 / r.rank_p_fav - 1.0) if r.fav_is_winner else -1.0
                    for r in bets]
        roi = sum(profits) / len(bets)
        signal = ("+viable" if roi > 0.07
                  else ("~breakeven" if roi > -0.02 else "-negative"))
        print(f"  >={thresh*100:.0f}%  {len(bets):>8,} {w_pct*100:>6.1f}%  "
              f"{avg_odds:>9.2f}x  {roi*100:>+9.1f}%  {signal}")
        results[thresh] = roi

    print(f"\n  * ROI uses fair 1/rank_implied odds.  Real Kambi adds ~5-7% vig.")
    print(f"    Subtract 7pp to estimate real-world ROI.")

    return results


def print_surface_breakdown(recs: list[Rec]) -> None:
    print(f"\n  SURFACE BREAKDOWN")
    print(f"  {'Surface':<8} {'N':>7} {'Elo Acc':>9} {'Rank Acc':>10} "
          f"{'Brier':>8} {'N-edge bets':>13}")
    print(f"  {BAR}")

    for surf in ("Clay", "Hard", "Grass"):
        sr = [r for r in recs if r.surface == surf]
        if not sr:
            continue
        ra    = _rank_acc(sr)
        edge8 = sum(1 for r in sr if r.edge is not None and r.edge >= 0.08)
        print(f"  {surf:<8} {len(sr):>7,} {_acc(sr)*100:>8.1f}%  "
              f"{(ra*100 if ra else 0):>9.1f}%  {_brier(sr):>8.4f}  "
              f"{edge8:>13,}")


def print_level_breakdown(recs: list[Rec]) -> None:
    print(f"\n  TOURNEY LEVEL BREAKDOWN (top levels by count)")
    print(f"  {'Level':<8} {'N':>7} {'Elo Acc':>9} {'Brier':>8} {'N edge>=8%':>12}")
    print(f"  {BAR}")

    by_level: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        by_level[r.level].append(r)

    for lvl, lr in sorted(by_level.items(), key=lambda kv: -len(kv[1]))[:8]:
        edge8 = sum(1 for r in lr if r.edge is not None and r.edge >= 0.08)
        print(f"  {lvl:<8} {len(lr):>7,} {_acc(lr)*100:>8.1f}%  "
              f"{_brier(lr):>8.4f}  {edge8:>12,}")


def print_large_edge_test(recs: list[Rec]) -> None:
    """
    Optimizer's curse check: do very large Elo advantages win proportionally more?
    If yes → model is well-calibrated at extremes.
    If actual < predicted → OVERCONFIDENT at extremes (optimizer's curse).
    """
    print(f"\n  OPTIMIZER'S CURSE CHECK")
    print(f"  (When Elo says 70%, does the favourite win 70% of the time?)")
    print(f"  {'Elo fav%':<12} {'N':>7} {'Predicted%':>12} {'Actual%':>10} {'Gap':>8}  verdict")
    print(f"  {BAR}")

    buckets = [
        (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.00),
    ]
    for lo, hi in buckets:
        sub = [r for r in recs if lo <= r.p_fav < hi]
        if len(sub) < 20:
            continue
        avg_pred = sum(r.p_fav for r in sub) / len(sub)
        actual   = _acc(sub)
        gap      = actual - avg_pred
        verdict  = ("ok       " if abs(gap) < 0.05
                    else ("underconf" if gap > 0.05 else "OVERCONF "))
        print(f"  {lo*100:.0f}-{hi*100:.0f}%      {len(sub):>7,} "
              f"{avg_pred*100:>11.1f}%  {actual*100:>9.1f}%  "
              f"{gap*100:>+6.1f}%  {verdict}")


def print_year_consistency(val: list[Rec], hold: list[Rec]) -> None:
    """
    Shows per-year accuracy and edge ROI to detect decay.
    Key question: does signal degrade from 2023 → 2024 → 2025?
    """
    all_recs = val + hold
    by_year  = defaultdict(list)
    for r in all_recs:
        by_year[r.year].append(r)

    print(f"\n  YEAR-BY-YEAR CONSISTENCY (>=8% edge threshold)")
    print(f"  {'Year':<6} {'N':>7} {'Elo Acc':>9} {'Edge N':>8} "
          f"{'Edge Win%':>11} {'ROI (fair)':>12}  trend")
    print(f"  {BAR}")

    prev_acc = None
    for yr in sorted(by_year.keys()):
        yr_recs = by_year[yr]
        acc     = _acc(yr_recs)
        bets    = [r for r in yr_recs if r.edge is not None and r.edge >= 0.08]
        if bets:
            w_pct   = sum(r.fav_is_winner for r in bets) / len(bets)
            profits = [(1.0 / r.rank_p_fav - 1.0) if r.fav_is_winner else -1.0
                       for r in bets]
            roi      = sum(profits) / len(bets)
            roi_str  = f"{roi*100:>+10.1f}%"
            wpct_str = f"{w_pct*100:>10.1f}%"
        else:
            wpct_str = "      N/A"
            roi_str  = "       N/A"
        trend = ""
        if prev_acc is not None:
            d = acc - prev_acc
            trend = ("stable" if abs(d) < 0.02 else ("decay" if d < -0.02 else "improv"))
        prev_acc = acc
        print(f"  {yr:<6} {len(yr_recs):>7,} {acc*100:>8.1f}%  "
              f"{len(bets):>8,}{wpct_str}{roi_str}  {trend}")


def print_honest_verdict(val: list[Rec], hold: list[Rec]) -> None:
    val_acc  = _acc(val)
    hold_acc = _acc(hold)
    acc_delta = hold_acc - val_acc

    val_bets  = [r for r in val  if r.edge is not None and r.edge >= 0.08]
    hold_bets = [r for r in hold if r.edge is not None and r.edge >= 0.08]

    def roi(bets):
        if not bets:
            return None
        profits = [(1.0 / r.rank_p_fav - 1.0) if r.fav_is_winner else -1.0
                   for r in bets]
        return sum(profits) / len(profits)

    val_roi  = roi(val_bets)
    hold_roi = roi(hold_bets)

    print()
    print("=" * W)
    print("  HONEST ASSESSMENT")
    print("=" * W)
    print()

    # Accuracy stability
    if abs(acc_delta) <= 0.02:
        print(f"  [+] Elo accuracy stable: {val_acc*100:.1f}% (val) -> {hold_acc*100:.1f}% (hold). No decay.")
    elif acc_delta < -0.02:
        print(f"  [!] Elo accuracy DEGRADED: {val_acc*100:.1f}% (val) -> {hold_acc*100:.1f}% (hold). Investigate.")
    else:
        print(f"  [+] Elo accuracy improved: {val_acc*100:.1f}% (val) -> {hold_acc*100:.1f}% (hold).")

    # ROI signal
    for period, r, bets, lbl in [
        (val_roi,  val_roi,  val_bets,  "Validation"),
        (hold_roi, hold_roi, hold_bets, "Holdout   "),
    ]:
        if r is None:
            print(f"  [?] {lbl}: no ranked bets at >=8% edge.")
            continue
        real_roi = r - 0.07  # rough vig discount
        if r > 0.07:
            tag = "[+]"
            msg = f"After ~7% vig: {real_roi*100:+.1f}%. Promising."
        elif r > -0.02:
            tag = "[~]"
            msg = f"After ~7% vig: {real_roi*100:+.1f}%. Likely breakeven or negative live."
        else:
            tag = "[-]"
            msg = f"After ~7% vig: {real_roi*100:+.1f}%. Negative even with no vig."
        print(f"  {tag} {lbl}: ROI(fair)={r*100:+.1f}% on {len(bets):,} bets. {msg}")

    print()
    print("  WHAT THIS BACKTEST CANNOT TELL YOU:")
    print("  1. Whether Kambi's ITF odds are priced sharper than WTA rankings.")
    print("     (They almost certainly are -- rankings are public, Kambi pays quants.)")
    print("  2. Whether 'edges' in the rank proxy survive Kambi's vig.")
    print("  3. CLV: does our model price match before or after Kambi moves?")
    print("  4. Real sample sizes: 2025 has 21K matches but Kambi posts only ~327.")
    print("     We have 327 live samples, not 21K -- vol is much lower.")
    print()

    # Final recommendation
    if hold_roi is not None and hold_roi > -0.02 and abs(acc_delta) <= 0.03:
        print("  RECOMMENDATION: Paper mode. Run predict_tennis.py for 3-4 weeks.")
        print("  Track CLV on every alert. If CLV > 0% on 50+ bets: go signal.")
        print("  If CLV <= 0% on 50+ bets: Elo has no live edge against Kambi.")
    elif hold_roi is not None and hold_roi < -0.05:
        print("  RECOMMENDATION: Do NOT deploy. Elo cannot beat WTA rank proxy.")
        print("  If Elo can't beat rankings, it cannot beat Kambi's professional")
        print("  odds. Investigate K-factor, scale, or surface weighting.")
    else:
        print("  RECOMMENDATION: Marginal signal. Paper mode with tight thresholds (>=10%).")
        print("  Use surface Elo only where player has >= 20 surface matches.")

    print("=" * W)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ITF Women Elo model")
    parser.add_argument("--val-only",      action="store_true")
    parser.add_argument("--holdout-only",  action="store_true")
    parser.add_argument("--min-matches",   type=int, default=DEFAULT_MIN_MATCHES,
                        help=f"Min prior matches per player (default {DEFAULT_MIN_MATCHES})")
    args = parser.parse_args()

    run_val  = not args.holdout_only
    run_hold = not args.val_only
    min_n    = args.min_matches

    conn = sqlite3.connect(DB_PATH)
    elo_idx = load_elo_index(conn)

    val_recs: list[Rec]  = []
    hold_recs: list[Rec] = []

    # ── Validation 2023-2024 ─────────────────────────────────────────────────
    if run_val:
        log.info("Processing validation period 2023-2024 (min_matches=%d)...", min_n)
        val_recs = load_period(conn, elo_idx, VAL_START, VAL_END, min_n)
        n_rk = sum(1 for r in val_recs if r.edge is not None)

        _hdr(f"VALIDATION: 2023-2024   N={len(val_recs):,}   "
             f"Ranked={n_rk:,} ({n_rk/len(val_recs)*100:.0f}%)")
        print_overview(val_recs, "OVERALL METRICS")
        print_calibration(val_recs, "2023-2024")
        print_edge_analysis(val_recs, "2023-2024")
        print_surface_breakdown(val_recs)
        print_level_breakdown(val_recs)

    # ── Holdout 2025 ─────────────────────────────────────────────────────────
    if run_hold:
        log.info("Processing holdout 2025 (min_matches=%d)...", min_n)
        hold_recs = load_period(conn, elo_idx, HOLD_START, HOLD_END, min_n)
        n_rk = sum(1 for r in hold_recs if r.edge is not None)

        print()
        print("*" * W)
        print("  *** HOLDOUT 2025 — First and final look ***")
        print("*" * W)
        _hdr(f"HOLDOUT: 2025   N={len(hold_recs):,}   "
             f"Ranked={n_rk:,} ({n_rk/len(hold_recs)*100:.0f}%)")
        print_overview(hold_recs, "OVERALL METRICS")
        print_calibration(hold_recs, "2025 holdout")
        print_edge_analysis(hold_recs, "2025 holdout")
        print_surface_breakdown(hold_recs)
        print_level_breakdown(hold_recs)
        print_large_edge_test(hold_recs)

    if run_val and run_hold:
        print_year_consistency(val_recs, hold_recs)
        print_honest_verdict(val_recs, hold_recs)

    conn.close()


if __name__ == "__main__":
    main()
