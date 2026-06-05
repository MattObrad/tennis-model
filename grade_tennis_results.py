"""
grade_tennis_results.py -- Grade pending tennis bets and post per-bet Discord results.

Pipeline:
    1. Load ungraded TENNIS bets from alerts.db (result='PENDING', graded=0)
    2. Resolve opponent from tennis.db predictions (via event_id / game_id)
    3. Find the specific match in tennis.db matches by player+opponent+date window
    4. Grade WIN or LOSS; profit_units from stored American odds
    5. Compute CLV from VPS Postgres closing snapshot (optional; skipped if PG unavailable)
    6. Post per-bet Discord embed

Usage:
    python grade_tennis_results.py
    python grade_tennis_results.py --date 2026-06-04
    python grade_tennis_results.py --dry-run
    python grade_tennis_results.py --db /home/picks/alerts.db \\
                                   --tennis-db /home/picks/tennis.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
for _env in (_DIR / ".env", _DIR / "mlb" / ".env"):
    if _env.exists():
        load_dotenv(dotenv_path=_env, override=False, encoding="utf-8-sig")
        break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ET           = timedelta(hours=-4)
_ALERTS_DB   = str(_DIR / "alerts.db")
_TENNIS_DB   = str(_DIR / "tennis.db")

# Grade window: ITF tourney_date is the Monday start; a Friday match has the
# same tourney_date as Monday. Use 10 days to bracket any week-long tournament.
MATCH_WINDOW_DAYS = 10

# Discord colors
_COLOR_WIN  = 3066993   # green
_COLOR_LOSS = 15158332  # red
_COLOR_EVEN = 9807270   # grey


# ---------------------------------------------------------------------------
# Name normalisation (mirrors grade_tennis_alerts.py and predict_tennis.py)
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    nfd = unicodedata.normalize("NFD", name.lower())
    return "".join(c for c in nfd if not unicodedata.combining(c)).strip()


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def _implied_prob(odds: int) -> float:
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def _american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def _devig(odds_p1: int, odds_p2: int) -> tuple[float, float]:
    """Proportional de-vig for a two-way market. Returns (fair_p1, fair_p2)."""
    imp1 = _implied_prob(odds_p1)
    imp2 = _implied_prob(odds_p2)
    total = imp1 + imp2
    return imp1 / total, imp2 / total


# ---------------------------------------------------------------------------
# Discord helper
# ---------------------------------------------------------------------------

def _discord_post(webhook_url: str, payload: dict) -> bool:
    if not webhook_url:
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url,
            data    = data,
            headers = {"Content-Type": "application/json", "User-Agent": "ObServatory/1.0"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Postgres (optional — for CLV only)
# ---------------------------------------------------------------------------

def _open_pg(config_path: str | None = None):
    """Open VPS Postgres connection for CLV lookup. Returns conn or None."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        log.warning("psycopg2 not installed — CLV computation skipped.")
        return None

    # Prefer config file; fall back to env vars
    cfg: dict = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f).get("vps", {})
        except Exception:
            pass

    host     = cfg.get("host")     or os.environ.get("VPS_DB_HOST",     "localhost")
    port     = cfg.get("port")     or int(os.environ.get("VPS_DB_PORT",  "5432"))
    dbname   = cfg.get("database") or os.environ.get("VPS_DB_NAME",     "picksdb")
    user     = cfg.get("user")     or os.environ.get("VPS_DB_USER",     "picksuser")
    password = os.environ.get("VPS_DB_PASSWORD", cfg.get("password", ""))

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user,
            password=password, connect_timeout=10,
        )
        conn.autocommit = True
        log.info("Postgres connected for CLV lookup.")
        return conn
    except Exception as exc:
        log.warning("Postgres unavailable — CLV skipped: %s", exc)
        return None


def _fetch_clv(
    pg_conn,
    game_id: str,
    alert_player: str,
    alert_fair: float,
) -> float | None:
    """
    Compute closing-line value (in probability points) for one alert.

    CLV = close_fair_prob_for_alert_player - alert_fair_prob
    Positive CLV: market moved toward our player by close (we got good early price).

    Requires VPS Postgres to read props_snapshots around game_time.
    Returns None if data is unavailable.
    """
    if pg_conn is None:
        return None

    try:
        import psycopg2.extras
        cur = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Step 1: get game_time for this event
        cur.execute("SELECT game_time FROM games WHERE event_id = %s", (game_id,))
        row = cur.fetchone()
        if not row:
            return None
        game_time = row["game_time"]

        # Step 2: get the last snapshot for EACH player in this event before game_time
        cur.execute(
            """
            SELECT DISTINCT ON (player_name)
                player_name, over_odds
            FROM   props_snapshots
            WHERE  event_id    = %s
              AND  line        IS NULL
              AND  over_odds   IS NOT NULL
              AND  snapshot_time < %s
            ORDER  BY player_name, snapshot_time DESC
            """,
            (game_id, game_time),
        )
        rows = cur.fetchall()
        if len(rows) < 2:
            return None

        # Build odds dict keyed by player_name
        odds_map: dict[str, int] = {r["player_name"]: r["over_odds"] for r in rows}

        # Identify our player vs opponent using normalised name match
        norm_alert = _norm(alert_player)
        our_name = next(
            (n for n in odds_map if _norm(n) == norm_alert), None
        )
        if our_name is None:
            return None
        opp_names = [n for n in odds_map if n != our_name]
        if not opp_names:
            return None
        opp_name = opp_names[0]

        # Step 3: de-vig the closing odds
        close_fair_us, _ = _devig(odds_map[our_name], odds_map[opp_name])

        # CLV = closing market fair prob - alert fair prob
        # Positive = market agrees with us more at close = we got good early price
        return round(close_fair_us - alert_fair, 4)

    except Exception as exc:
        log.warning("CLV lookup failed for game %s: %s", game_id, exc)
        return None


# ---------------------------------------------------------------------------
# Elo/opponent lookup from tennis.db predictions
# ---------------------------------------------------------------------------

def _get_prediction_info(tennis_conn: sqlite3.Connection, event_id: str, player_name: str):
    """
    Return (player_elo, opponent_elo, opponent_name, model_prob, fair_prob) for
    the alerted player from tennis.db predictions, or None if not found.
    Uses normalised name matching to handle minor Kambi display-name variations.
    """
    rows = tennis_conn.execute(
        """SELECT player1_name, player2_name,
                  player1_overall_elo, player2_overall_elo,
                  player1_model_prob,  player2_model_prob,
                  player1_fair_prob,   player2_fair_prob
           FROM predictions
           WHERE event_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (event_id,),
    ).fetchone()
    if not rows:
        return None
    p1, p2, elo1, elo2, mp1, mp2, fp1, fp2 = rows
    norm_p = _norm(player_name)
    if _norm(p1) == norm_p:
        return (elo1, elo2, p2, mp1, fp1)
    if _norm(p2) == norm_p:
        return (elo2, elo1, p1, mp2, fp2)
    return None


# ---------------------------------------------------------------------------
# Match result lookup
# ---------------------------------------------------------------------------

def _get_match_result(
    tennis_conn: sqlite3.Connection,
    event_id: str,
    player_name: str,
    alert_date: str,
) -> str | None:
    """
    Return 'WIN', 'LOSS', or None (match not yet in tennis.db).

    Resolves the specific opponent from the predictions table (stored at alert
    time) so we grade the EXACT match that was bet — not just any match the
    player appeared in during the tournament week (they could win 3 rounds
    before losing the one we bet, or vice versa).

    Uses accent-normalised comparison throughout; both predictions and matches
    may use slightly different encodings for accented characters.
    """
    # Step 1: resolve opponent from predictions
    pred = _get_prediction_info(tennis_conn, event_id, player_name)
    if pred is None:
        log.debug("No prediction row found for event=%s player=%s", event_id, player_name)
        return None
    _, _, opponent_name, _, _ = pred

    # Step 2: search matches within ±MATCH_WINDOW_DAYS filtered by both players
    lo = (datetime.fromisoformat(alert_date) - timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()
    hi = (datetime.fromisoformat(alert_date) + timedelta(days=MATCH_WINDOW_DAYS)).date().isoformat()

    rows = tennis_conn.execute(
        "SELECT winner_name, loser_name FROM matches WHERE tourney_date BETWEEN ? AND ?",
        (lo, hi),
    ).fetchall()

    norm_p   = _norm(player_name)
    norm_opp = _norm(opponent_name)

    for winner, loser in rows:
        w_norm = _norm(winner)
        l_norm = _norm(loser)
        if w_norm == norm_p and l_norm == norm_opp:
            return "WIN"
        if l_norm == norm_p and w_norm == norm_opp:
            return "LOSS"

    return None


# ---------------------------------------------------------------------------
# Per-date grading
# ---------------------------------------------------------------------------

def grade_date(
    alerts_conn: sqlite3.Connection,
    tennis_conn: sqlite3.Connection | None,
    date_et: str,
    pg_conn=None,
    dry_run: bool = False,
) -> dict:
    """Grade all ungraded TENNIS bets for date_et. Returns summary dict."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bets = alerts_conn.execute(
        """SELECT id, player_name, direction, line, odds, implied_prob, game_id
           FROM bet_alerts
           WHERE sport = 'TENNIS' AND graded = 0 AND alert_date = ?""",
        (date_et,),
    ).fetchall()

    if not bets:
        return {"date": date_et, "total": 0, "graded": 0, "pending": 0}

    graded = 0
    pending = 0

    for b in bets:
        bid         = b["id"]
        player_name = b["player_name"]
        open_odds   = int(b["odds"])
        game_id     = b["game_id"] or ""
        alert_fair  = float(b["implied_prob"]) if b["implied_prob"] else None

        # Check for result in tennis.db
        if tennis_conn is None:
            pending += 1
            continue

        result = _get_match_result(tennis_conn, game_id, player_name, date_et)
        if result is None:
            pending += 1
            continue

        profit = (_american_to_decimal(open_odds) - 1.0) if result == "WIN" else -1.0

        # Compute CLV from closing Postgres snapshot
        clv = None
        clv_beat = None
        if alert_fair is not None and game_id:
            clv = _fetch_clv(pg_conn, game_id, player_name, alert_fair)
            if clv is not None:
                clv_beat = 1 if clv > 0 else 0

        log.info("  %-28s  %s  profit=%+.2f  CLV=%s",
                 player_name, result, profit,
                 f"{clv*100:+.2f}pp" if clv is not None else "n/a")

        if not dry_run:
            alerts_conn.execute(
                """UPDATE bet_alerts SET
                     actual_result = ?, result = ?, profit_units = ?,
                     clv = ?, clv_beat = ?,
                     graded = 1, graded_at = ?
                   WHERE id = ?""",
                (1.0 if result == "WIN" else 0.0, result, profit,
                 clv, clv_beat, now, bid),
            )
            graded += 1

    if not dry_run:
        alerts_conn.commit()

    return {"date": date_et, "total": len(bets), "graded": graded, "pending": pending}


# ---------------------------------------------------------------------------
# Discord embed
# ---------------------------------------------------------------------------

def _post_tennis_results_embed(
    alerts_conn: sqlite3.Connection,
    tennis_conn: sqlite3.Connection | None,
    webhook: str,
    date_et: str,
) -> None:
    rows = alerts_conn.execute(
        """SELECT player_name, direction, line, odds, model_prob, implied_prob,
                  edge_prob, actual_result, result, profit_units, clv, clv_beat,
                  game_id
           FROM bet_alerts
           WHERE sport = 'TENNIS' AND alert_date = ?
             AND result IN ('WIN', 'LOSS', 'PUSH', 'PENDING')
           ORDER BY result != 'PENDING', player_name""",
        (date_et,),
    ).fetchall()

    if not rows:
        return

    resolved = [r for r in rows if r["result"] != "PENDING"]
    pending  = [r for r in rows if r["result"] == "PENDING"]

    net   = sum((r["profit_units"] or 0.0) for r in resolved)
    color = _COLOR_WIN if net >= 0 else _COLOR_LOSS

    lines = []
    for r in resolved + pending:
        emoji   = {"WIN": "[WIN]", "LOSS": "[LOSS]", "PUSH": "[PUSH]"}.get(r["result"], "[PEND]")
        outcome = {"WIN": "WON",   "LOSS": "LOST",   "PUSH": "PUSH"}.get(r["result"], "PENDING")
        pname   = r["player_name"]
        pnl     = f"{r['profit_units']:+.2f}u" if r["profit_units"] is not None else ""

        pred_info = None
        if tennis_conn:
            pred_info = _get_prediction_info(tennis_conn, r["game_id"], pname)

        if pred_info:
            p_elo, opp_elo, opp_name, mp, fp = pred_info
        else:
            p_elo = opp_elo = None
            opp_name = ""
            mp  = float(r["model_prob"])   if r["model_prob"]   else None
            fp  = float(r["implied_prob"]) if r["implied_prob"] else None

        if r["result"] == "PENDING":
            lines.append(
                f"[PEND] {pname} vs {opp_name} -- PENDING" if opp_name
                else f"[PEND] {pname} -- PENDING"
            )
            continue

        line1 = f"{emoji} {pname} -- {outcome} {pnl}"
        opp_s = f"vs {opp_name}  |  ITF Women" if opp_name else "ITF Women"
        elo_s = (f"Elo: {p_elo:.0f} vs {opp_elo:.0f}  |  "
                 if p_elo and opp_elo else "")
        prob_s = (f"Our P: {mp*100:.0f}%  |  Market: {fp*100:.0f}%"
                  if mp and fp else "")
        clv_s = (f"  |  CLV: {float(r['clv'])*100:+.2f}pp"
                 if r["clv"] is not None else "")
        line3 = f"{elo_s}{prob_s}{clv_s}"
        lines.append(f"{line1}\n{opp_s}" + (f"\n{line3}" if line3.strip() else ""))

    wins   = sum(1 for r in resolved if r["result"] == "WIN")
    losses = sum(1 for r in resolved if r["result"] == "LOSS")
    clvs   = [float(r["clv"]) * 100 for r in resolved if r["clv"] is not None]
    avg_clv_s = f"CLV: {sum(clvs)/len(clvs):+.2f}pp  |  " if clvs else ""
    stat = f"[CHART] {wins}-{losses} | {net:+.2f}u  |  {avg_clv_s}ObServatory Tennis Model"

    CHUNK = 4
    chunks = [lines[i:i+CHUNK] for i in range(0, max(len(lines), 1), CHUNK)]
    n = len(chunks)
    for idx, chunk in enumerate(chunks):
        suffix = f" ({idx+1}/{n})" if n > 1 else ""
        desc   = "\n\n".join(chunk) + f"\n\n{stat}"
        _discord_post(webhook, {"embeds": [{
            "title":       f"🎾 Tennis Results -- {date_et}{suffix}",
            "color":       color,
            "description": desc,
            "footer":      {"text": "ObServatory Tennis Model"},
        }]})
        log.info("Discord tennis results embed sent for %s%s.", date_et, suffix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Grade pending tennis bets and post results.")
    p.add_argument("--date",      default=None, metavar="YYYY-MM-DD")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--db",        default=None, metavar="PATH")
    p.add_argument("--tennis-db", default=None, metavar="PATH")
    p.add_argument("--config",    default=None, metavar="PATH",
                   help="Config JSON for VPS Postgres creds (for CLV)")
    args = p.parse_args(argv)

    alerts_db = args.db        or os.environ.get("ALERTS_DB_PATH")  or _ALERTS_DB
    tennis_db = args.tennis_db or os.environ.get("TENNIS_DB_PATH")  or _TENNIS_DB

    # Default config search: tennis_config.json in same dir
    config_path = args.config or str(_DIR / "tennis_config.json")

    today_et = (datetime.now(timezone.utc) + ET).date().isoformat()
    log.info("=== grade_tennis_results starting ===")

    if not Path(alerts_db).exists():
        log.error("alerts.db not found: %s", alerts_db)
        return 1

    alerts_conn = sqlite3.connect(alerts_db)
    alerts_conn.row_factory = sqlite3.Row

    tennis_conn = None
    if Path(tennis_db).exists():
        tennis_conn = sqlite3.connect(tennis_db)
        tennis_conn.row_factory = sqlite3.Row
        tennis_conn.execute("PRAGMA query_only = ON")
    else:
        log.warning("tennis.db not found at %s — grading and Elo data unavailable.", tennis_db)

    # Optional Postgres for CLV
    pg_conn = _open_pg(config_path)

    if args.date:
        dates = [args.date]
    else:
        rows = alerts_conn.execute(
            "SELECT DISTINCT alert_date FROM bet_alerts "
            "WHERE sport='TENNIS' AND graded=0 AND alert_date < ? ORDER BY alert_date",
            (today_et,),
        ).fetchall()
        dates = [r["alert_date"] for r in rows]

    if not dates:
        log.info("No ungraded TENNIS bets before %s.", today_et)
        alerts_conn.close()
        if tennis_conn: tennis_conn.close()
        if pg_conn:     pg_conn.close()
        return 0

    log.info("Dates to grade: %s", dates)

    total_graded = 0
    for d in dates:
        summary = grade_date(alerts_conn, tennis_conn, d, pg_conn=pg_conn, dry_run=args.dry_run)
        total_graded += summary["graded"]
        log.info("  %s: %d/%d graded, %d pending",
                 d, summary["graded"], summary["total"], summary["pending"])

    log.info("=== Done. %d tennis bets graded. ===", total_graded)

    if total_graded > 0 and not args.dry_run:
        webhook = os.environ.get("DISCORD_WEBHOOK_RESULTS", "").strip()
        if webhook:
            for d in dates:
                _post_tennis_results_embed(alerts_conn, tennis_conn, webhook, d)

    alerts_conn.close()
    if tennis_conn: tennis_conn.close()
    if pg_conn:     pg_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
