"""
grade_tennis_alerts.py -- Auto-grade pending TENNIS bets in alerts.db.

Pipeline:
    1. Query bet_alerts WHERE sport='TENNIS' AND graded=0 AND alert_date < today
    2. Look up the match result in tennis.db (winner_name / loser_name)
    3. Grade WIN (player won) or LOSS (player lost)
    4. Compute profit_units from the stored American odds
    5. Update bet_alerts: result, profit_units, graded=1, graded_at

Cron (daily after Sackmann backfill, 11am UTC works since we already run
collect_tennis_vps.py at 5am):
    0 11 * * *  cd /home/picks && ALERTS_DB_PATH=/home/picks/alerts.db \\
        python3 grade_tennis_alerts.py >> /home/picks/logs/grade_tennis_alerts.log 2>&1

Usage:
    python grade_tennis_alerts.py
    python grade_tennis_alerts.py --date 2026-05-28   # re-grade a specific date
    python grade_tennis_alerts.py --dry-run           # show grades without writing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import unicodedata
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

_DIR      = Path(__file__).resolve().parent
_LOG_DIR  = _DIR / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_DIR / "grade_tennis_alerts.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

_ALERTS_DB = os.environ.get("ALERTS_DB_PATH") or str(_DIR / "alerts.db")
_TENNIS_DB = os.environ.get("TENNIS_DB_PATH")  or str(_DIR / "tennis.db")

# Grade window: search tennis.db for a match within ±N days of the alert date.
# ITF tournaments span one week; the tourney_date is the Monday start date,
# so a match played on Friday is stored with the Monday tourney_date. Use 10
# days to safely bracket any scheduling slippage.
MATCH_WINDOW_DAYS = 10

# ── odds helpers ───────────────────────────────────────────────────────────────

def decimal_from_american(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (-100 / odds + 1)


def profit_for_result(result: str, odds: int) -> float:
    """Flat 1-unit stake profit/loss."""
    if result == "WIN":
        return round(decimal_from_american(odds) - 1.0, 4)
    if result == "LOSS":
        return -1.0
    return 0.0   # PUSH


# ── name normalisation ─────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c)).strip()


# ── result lookup ──────────────────────────────────────────────────────────────

def _get_opponent(tennis_conn: sqlite3.Connection, game_id: str, player_name: str) -> str:
    """Look up the opponent for this alert from tennis.db predictions via event_id."""
    row = tennis_conn.execute(
        """SELECT player1_name, player2_name FROM predictions
           WHERE event_id = ? ORDER BY created_at DESC LIMIT 1""",
        (game_id,),
    ).fetchone()
    if not row:
        return ""
    p1, p2 = row
    norm_p = _norm(player_name)
    if _norm(p1) == norm_p:
        return p2
    if _norm(p2) == norm_p:
        return p1
    return ""


def find_result(
    tennis_conn: sqlite3.Connection,
    player_name: str,
    alert_date: str,
    game_id: str = "",
) -> str | None:
    """
    Return 'WIN', 'LOSS', or None (not yet published in tennis.db).

    Looks for the player in the matches table within MATCH_WINDOW_DAYS of
    the alert date.  When game_id is provided, resolves the specific opponent
    from tennis.db predictions so we grade the correct match (not just any
    match the player appeared in that week — a player can play multiple rounds).
    """
    lo_date = (datetime.fromisoformat(alert_date)
               - timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()
    hi_date = (datetime.fromisoformat(alert_date)
               + timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()

    rows = tennis_conn.execute(
        """
        SELECT winner_name, loser_name
        FROM   matches
        WHERE  tourney_date BETWEEN ? AND ?
        """,
        (lo_date, hi_date),
    ).fetchall()

    norm_player = _norm(player_name)

    # Try to resolve the specific opponent to avoid grading the wrong match
    # when a player won earlier rounds but lost the one we bet.
    opponent = _get_opponent(tennis_conn, game_id, player_name) if game_id else ""
    norm_opp = _norm(opponent) if opponent else ""

    for winner, loser in rows:
        w_norm = _norm(winner)
        l_norm = _norm(loser)
        if w_norm == norm_player:
            if norm_opp and l_norm != norm_opp:
                continue   # right player, wrong match
            return "WIN"
        if l_norm == norm_player:
            if norm_opp and w_norm != norm_opp:
                continue   # right player, wrong match
            return "LOSS"
    return None


# ── Discord helper ─────────────────────────────────────────────────────────────

def _discord_post(webhook_url: str, payload: dict) -> bool:
    if not webhook_url:
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ObServatory/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 204)
    except Exception as exc:
        log.warning("Discord post failed: %s", exc)
        return False


# ── main grading loop ──────────────────────────────────────────────────────────

def grade_date(
    alerts_conn: sqlite3.Connection,
    tennis_conn: sqlite3.Connection,
    date_et: str,
    dry_run: bool = False,
) -> dict:
    """Grade all ungraded TENNIS bets for a given alert_date. Returns summary."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    bets = alerts_conn.execute(
        "SELECT id, player_name, direction, line, odds, game_id "
        "FROM bet_alerts "
        "WHERE sport = 'TENNIS' AND graded = 0 AND alert_date = ?",
        (date_et,),
    ).fetchall()

    if not bets:
        return {"date": date_et, "total": 0, "graded": 0, "pending": 0}

    log.info("  %s: %d ungraded TENNIS bets to process", date_et, len(bets))

    graded_count = 0
    pending_count = 0

    for row in bets:
        bid, player_name, direction, line, odds, game_id = row
        result = find_result(tennis_conn, player_name, date_et, game_id=game_id or "")

        if result is None:
            log.debug("  No result yet for %s on %s", player_name, date_et)
            pending_count += 1
            continue

        profit = profit_for_result(result, int(odds))
        log.info("  %-28s  %s  odds=%+d  profit=%+.2f",
                 player_name, result, int(odds), profit)

        if not dry_run:
            alerts_conn.execute(
                """
                UPDATE bet_alerts SET
                    actual_result = ?,
                    result        = ?,
                    profit_units  = ?,
                    graded        = 1,
                    graded_at     = ?
                WHERE id = ?
                """,
                (None, result, profit, now, bid),
            )
            graded_count += 1

    if not dry_run:
        alerts_conn.commit()

    return {
        "date":    date_et,
        "total":   len(bets),
        "graded":  graded_count if not dry_run else 0,
        "pending": pending_count,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Auto-grade pending TENNIS bets in alerts.db.")
    p.add_argument("--date",      default=None, metavar="YYYY-MM-DD",
                   help="Grade only this specific date (default: all ungraded before today)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Show what would be graded without writing anything")
    p.add_argument("--alerts-db", default=None)
    p.add_argument("--tennis-db", default=None)
    args = p.parse_args(argv)

    alerts_db = args.alerts_db or _ALERTS_DB
    tennis_db = args.tennis_db or _TENNIS_DB
    today_utc = datetime.now(timezone.utc).date().isoformat()

    log.info("=== grade_tennis_alerts starting ===")
    log.info("  alerts.db : %s", alerts_db)
    log.info("  tennis.db : %s", tennis_db)
    if args.dry_run:
        log.info("  DRY RUN — no writes")

    for path, label in [(alerts_db, "alerts.db"), (tennis_db, "tennis.db")]:
        if not Path(path).exists():
            log.error("%s not found: %s", label, path)
            return 1

    alerts_conn = sqlite3.connect(alerts_db)
    tennis_conn = sqlite3.connect(tennis_db)
    tennis_conn.execute("PRAGMA query_only = ON")

    # Determine dates to grade
    if args.date:
        dates = [args.date]
    else:
        rows = alerts_conn.execute(
            "SELECT DISTINCT alert_date FROM bet_alerts "
            "WHERE sport = 'TENNIS' AND graded = 0 AND alert_date < ? "
            "ORDER BY alert_date",
            (today_utc,),
        ).fetchall()
        dates = [r[0] for r in rows]

    if not dates:
        log.info("No ungraded TENNIS bets before %s.", today_utc)
        alerts_conn.close()
        tennis_conn.close()
        return 0

    log.info("Dates to grade: %s", dates)

    total_graded = 0
    for d in dates:
        summary = grade_date(alerts_conn, tennis_conn, d, dry_run=args.dry_run)
        total_graded += summary["graded"]
        log.info(
            "  %s: %d/%d graded, %d still pending (Sackmann not yet published)",
            d, summary["graded"], summary["total"], summary["pending"],
        )

    tennis_conn.close()
    alerts_conn.close()

    log.info("=== Done. %d TENNIS bet(s) graded. ===", total_graded)

    # Post grading summary to DISCORD_WEBHOOK_RESULTS if anything was graded
    if total_graded > 0 and not args.dry_run:
        webhook = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
        if webhook:
            _discord_post(webhook, {"embeds": [{
                "title":       "🎾 Tennis Bets Graded",
                "description": (f"**{total_graded}** bet{'s' if total_graded != 1 else ''} "
                                f"graded across {len(dates)} date{'s' if len(dates) != 1 else ''}."),
                "color":       3066993,   # green
                "footer":      {"text": "ObServatory Tennis Model"},
            }]})

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
