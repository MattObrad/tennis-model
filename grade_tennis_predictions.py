"""
grade_tennis_predictions.py -- Brier / calibration validation on the FULL
prediction set, not just qualifying alerts.

predict_tennis.py writes EVERY scanned ITF Women match to tennis.db
`predictions` (player1_model_prob, player1_fair_prob, player1_kambi_odds, ...)
regardless of whether it cleared the live alert gate (edge >= 8% AND prob >= 70%).
Only a handful qualify as alerts, but dozens of predictions are made daily.

This script validates Elo calibration against the de-vigged Kambi line on the
ENTIRE gradeable sample (any prediction whose result has since published into
the `matches` table), while the live alert gate stays untouched at 8%/70%.

For each gradeable prediction:
    actual     = 1 if player1 won, 0 if player2 won
    model_prob = player1_model_prob   (Elo)
    kambi_prob = player1_fair_prob     (de-vigged Kambi opening line)

Overall Brier is orientation-invariant -- (p - y)^2 == ((1-p) - (1-y))^2 -- so
player1-orientation is fine for the headline numbers. The calibration TABLE is
reported from the favoured side's perspective (p >= 0.5) so bins cover
[0.5, 1.0], matching the convention in calibrate_tennis.py.

Read-only: opens tennis.db in query-only mode and never touches alerts.db or
any live config.

Usage:
    python3 grade_tennis_predictions.py
    python3 grade_tennis_predictions.py --tennis-db /home/picks/tennis.db
    python3 grade_tennis_predictions.py --start 2026-06-06 --end 2026-06-12
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import unicodedata
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Name normalisation -- VERBATIM from grade_tennis_results.py
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c)).strip()


def _names_match(stored: str, lookup: str) -> bool:
    """
    Flexible name comparison that handles two real-world mismatches between
    tennisexplorer-scraped rows and Kambi/Sackmann full names:

    1. Hyphens:  "Diana Ioana Simionescu"  <-> "Diana-Ioana Simionescu"
    2. Abbreviated "Last I." vs full "First Last":
                 "Scott K."  <-> "Katrina Scott"
    """
    def clean(s: str) -> str:
        return _norm(s).replace("-", " ").replace(".", "").strip()

    sc = clean(stored)
    lc = clean(lookup)

    if sc == lc:
        return True

    sp = sc.split()
    lp = lc.split()
    if len(sp) >= 2 and len(sp[-1]) == 1 and len(lp) >= 2:
        abbr_last    = " ".join(sp[:-1])
        abbr_initial = sp[-1]
        full_last    = " ".join(lp[1:])
        full_initial = lp[0][0]
        if abbr_last == full_last and abbr_initial == full_initial:
            return True

    return False


# ---------------------------------------------------------------------------
# Result lookup -- mirrors grade_tennis_results._get_match_result
# ---------------------------------------------------------------------------

def _player1_result(
    conn: sqlite3.Connection,
    p1_name: str,
    p2_name: str,
    pred_date: str,
) -> int | None:
    """
    Return 1 if player1 won, 0 if player1 lost, None if the match isn't yet in
    `matches`. Searches a -1/+2 day window around the prediction date (same as
    the production grader) and resolves the EXACT pairing via _names_match so a
    player who appears in several rounds that week is graded on the right match.
    """
    rows = conn.execute(
        """SELECT winner_name, loser_name FROM matches
           WHERE tourney_date BETWEEN date(?, '-1 day') AND date(?, '+2 days')""",
        (pred_date, pred_date),
    ).fetchall()

    for winner, loser in rows:
        if _names_match(winner, p1_name) and _names_match(loser, p2_name):
            return 1
        if _names_match(loser, p1_name) and _names_match(winner, p2_name):
            return 0
    return None


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# Calibration bins (favoured-side perspective: p >= 0.5). Lower bound inclusive,
# upper exclusive; last bin catches 1.0.
_BINS = [
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70-80%", 0.70, 0.80),
    ("80-90%", 0.80, 0.90),
    ("90%+",   0.90, 1.01),
]


class Graded:
    """One gradeable prediction, oriented to the model's favoured side."""
    __slots__ = ("p_fav", "k_fav", "won", "p1_model", "k1", "y1")

    def __init__(self, p1_model: float, k1: float, y1: int):
        # player1-oriented raw values (for orientation-invariant overall Brier)
        self.p1_model = p1_model
        self.k1 = k1
        self.y1 = y1
        # favoured-side orientation (model picks the side with prob >= 0.5)
        if p1_model >= 0.5:
            self.p_fav, self.k_fav, self.won = p1_model, k1, y1
        else:
            self.p_fav, self.k_fav, self.won = 1.0 - p1_model, 1.0 - k1, 1 - y1


def _overall(graded: list[Graded]) -> tuple[int, float, float, float]:
    """Return (n, elo_brier, kambi_brier, baseline_brier). Orientation-invariant."""
    n = len(graded)
    elo   = _mean([(g.p1_model - g.y1) ** 2 for g in graded])
    kambi = _mean([(g.k1       - g.y1) ** 2 for g in graded])
    base  = _mean([(0.5        - g.y1) ** 2 for g in graded])
    return n, elo, kambi, base


def _calibration_table(graded: list[Graded]) -> list[tuple]:
    """Per-bucket (label, n, avg_pred, actual_winrate, elo_brier, kambi_brier)."""
    out = []
    for label, lo, hi in _BINS:
        sub = [g for g in graded if lo <= g.p_fav < hi]
        if not sub:
            out.append((label, 0, None, None, None, None))
            continue
        avg_pred = _mean([g.p_fav for g in sub])
        actual   = _mean([g.won for g in sub])
        elo_b    = _mean([(g.p_fav - g.won) ** 2 for g in sub])
        kam_b    = _mean([(g.k_fav - g.won) ** 2 for g in sub])
        out.append((label, len(sub), avg_pred, actual, elo_b, kam_b))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_block(title: str, graded: list[Graded]) -> None:
    W = 78
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)

    if not graded:
        print("  (no gradeable predictions in this slice)")
        return

    n, elo, kambi, base = _overall(graded)
    print(f"  n = {n}")
    print(f"  Elo   Brier = {elo:.4f}")
    print(f"  Kambi Brier = {kambi:.4f}   (de-vigged opening line)")
    print(f"  0.50  Brier = {base:.4f}   (coin-flip baseline)")
    diff = kambi - elo
    if elo < kambi:
        print(f"  -> Elo beats Kambi by {diff:.4f}  (lower is better) -- edge signal")
    elif elo > kambi:
        print(f"  -> Kambi beats Elo by {-diff:.4f}  (lower is better) -- no edge")
    else:
        print(f"  -> Elo == Kambi")

    print()
    print(f"  Calibration (favoured side, p >= 0.5):")
    hdr = f"  {'bucket':<8} {'n':>4} {'avg_pred':>9} {'actual':>8} {'elo_Brier':>10} {'kambi_Brier':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, bn, avg_pred, actual, elo_b, kam_b in _calibration_table(graded):
        if bn == 0:
            print(f"  {label:<8} {bn:>4} {'--':>9} {'--':>8} {'--':>10} {'--':>12}")
        else:
            print(f"  {label:<8} {bn:>4} {avg_pred*100:>8.1f}% {actual*100:>7.1f}% "
                  f"{elo_b:>10.4f} {kam_b:>12.4f}")
    # TOTAL row
    tot_avg = _mean([g.p_fav for g in graded])
    tot_act = _mean([g.won for g in graded])
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<8} {len(graded):>4} {tot_avg*100:>8.1f}% {tot_act*100:>7.1f}% "
          f"{elo:>10.4f} {kambi:>12.4f}")


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

_COLOR_GREEN  = 3066993   # go live
_COLOR_YELLOW = 15844367  # close
_COLOR_RED    = 15158332  # stay paper


def _discord_post(webhook_url: str, payload: dict) -> bool:
    if not webhook_url:
        return False
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url,
            data    = data,
            headers = {"Content-Type": "application/json; charset=utf-8",
                       "User-Agent": "ObServatory/1.0"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        print(f"Discord post failed: {exc}", file=sys.stderr)
        return False


def _post_brier_summary(
    n_all: int, elo_all: float, kambi_all: float,
    n_bet: int, elo_bet: float, kambi_bet: float,
    alert_prob: float,
) -> bool:
    """Post weekly Brier validation summary to DISCORD_WEBHOOK_TENNIS."""
    webhook = os.environ.get("DISCORD_WEBHOOK_TENNIS", "").strip()
    if not webhook:
        return False

    gap = kambi_bet - elo_bet  # positive = Elo better than Kambi on bettable subset
    if elo_bet < kambi_bet:
        status = "GO LIVE — Elo beats market on bettable subset"
        color  = _COLOR_GREEN
    elif gap > -0.01:  # Kambi ahead but by < 0.01
        status = "CLOSE — monitor closely"
        color  = _COLOR_YELLOW
    else:
        status = "STAY PAPER"
        color  = _COLOR_RED

    pct = int(alert_prob * 100)
    desc = (
        f"Overall (n={n_all}): Elo {elo_all:.3f} vs Kambi {kambi_all:.3f}\n"
        f"Bettable {pct}%+ (n={n_bet}): Elo {elo_bet:.3f} vs Kambi {kambi_bet:.3f}\n\n"
        f"**Status: {status}**"
    )

    payload = {"embeds": [{
        "title":       "🎾 Weekly Brier Validation",
        "description": desc,
        "color":       color,
        "footer":      {"text": "ObServatory Tennis Model"},
    }]}
    return _discord_post(webhook, payload)


# ---------------------------------------------------------------------------
# DB location
# ---------------------------------------------------------------------------

def _resolve_db(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for cand in (
        Path("/home/picks/tennis.db"),     # VPS
        here / "data" / "tennis.db",        # local dev layout
        here / "tennis.db",
    ):
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return here / "tennis.db"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Brier/calibration on ALL tennis predictions.")
    ap.add_argument("--tennis-db", default=None, metavar="PATH")
    ap.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                    help="Only predictions with prediction_date >= START")
    ap.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                    help="Only predictions with prediction_date <= END")
    ap.add_argument("--alert-prob", type=float, default=0.70,
                    help="Live alert probability threshold for the filtered view (default 0.70)")
    args = ap.parse_args(argv)

    db_path = _resolve_db(args.tennis_db)
    if not db_path.exists():
        print(f"ERROR: tennis.db not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")

    max_match = conn.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
    n_pred    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # Pull predictions, deduped to the latest write per event_id (predict_tennis
    # can re-scan an event; mirror grade_tennis_results' "latest row" rule).
    where = []
    params: list = []
    if args.start:
        where.append("prediction_date >= ?"); params.append(args.start)
    if args.end:
        where.append("prediction_date <= ?"); params.append(args.end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"""SELECT prediction_date, event_id, player1_name, player2_name,
                   player1_model_prob, player1_fair_prob
            FROM predictions
            {clause}
            ORDER BY created_at ASC, id ASC""",
        params,
    ).fetchall()

    latest: dict[str, tuple] = {}
    for r in rows:
        latest[r[1]] = r  # later write wins (ordered ASC)

    pred_dates = [r[0] for r in latest.values()]
    date_lo = min(pred_dates) if pred_dates else None
    date_hi = max(pred_dates) if pred_dates else None

    graded: list[Graded] = []
    ungradeable = 0
    skipped_null = 0
    for pred_date, event_id, p1, p2, p1_model, p1_fair in latest.values():
        if p1_model is None or p1_fair is None:
            skipped_null += 1
            continue
        y1 = _player1_result(conn, p1, p2, pred_date)
        if y1 is None:
            ungradeable += 1
            continue
        graded.append(Graded(float(p1_model), float(p1_fair), int(y1)))

    conn.close()

    W = 78
    print("=" * W)
    print("  TENNIS PREDICTION VALIDATION -- Elo vs Kambi (full prediction set)")
    print("=" * W)
    print(f"  tennis.db                  : {db_path}")
    print(f"  matches latest tourney_date: {max_match}")
    print(f"  predictions in table       : {n_pred}  ({len(latest)} unique events"
          + (f", {args.start}..{args.end} filter" if (args.start or args.end) else "")
          + ")")
    if date_lo:
        print(f"  prediction_date range      : {date_lo} .. {date_hi}")
    print(f"  gradeable (result found)   : {len(graded)}")
    print(f"  not yet gradeable          : {ungradeable}  (result not in matches yet)")
    if skipped_null:
        print(f"  skipped (null prob)        : {skipped_null}")

    if not graded:
        print("\n  No gradeable predictions -- results have not published into "
              "`matches` yet.\n  Re-run after the scraper backfills.")
        return 0

    _print_block("ALL PREDICTIONS", graded)

    fav_alert = [g for g in graded if g.p_fav >= args.alert_prob]
    _print_block(
        f"FILTERED TO MODEL PROB >= {args.alert_prob*100:.0f}%  (live alert threshold -- 'what we'd bet')",
        fav_alert,
    )

    # Discord weekly summary
    n_all, elo_all, kambi_all, _ = _overall(graded)
    if fav_alert:
        n_bet, elo_bet, kambi_bet, _ = _overall(fav_alert)
        sent = _post_brier_summary(
            n_all, elo_all, kambi_all,
            n_bet, elo_bet, kambi_bet,
            args.alert_prob,
        )
        if sent:
            print("  Discord Brier summary posted.")
        else:
            print("  Discord Brier summary skipped (DISCORD_WEBHOOK_TENNIS not set or post failed).")
    else:
        print("  Discord Brier summary skipped (no bettable-subset predictions to report).")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
