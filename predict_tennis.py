"""
predict_tennis.py -- Daily ITF Women tennis edge finder.

Pipeline:
  1. VPS Postgres  → today's ITF Women match-winner props (latest snapshot)
  2. Pair sides    → group both players per event, compute two-way de-vig
  3. Name matching → fuzzy-match Kambi names to Sackmann names (for Elo lookup)
  4. Elo lookup    → point-in-time overall Elo for each player
  5. Model prob    → elo_win_prob(elo_p1, elo_p2) for each side
  6. Edge filter   → edge >= 8%  AND  model_prob >= 55%
  7. Extreme flag  → |model_prob - fair_prob| > 20% triggers !EXTREME warning
  8. Write         → tennis.db predictions + alerts tables
  9. Write         → D:/models/alerts.db  bet_alerts  (sport='TENNIS')
  10. Notify       → Discord via notify_tennis.py  (skipped in paper mode)

Usage:
  python predict_tennis.py                    # today's date
  python predict_tennis.py --date 2026-06-03  # specific date
  python predict_tennis.py --dry-run          # show everything, write nothing

Paper mode (config.json betting.paper_mode = true):
  Predictions and alerts ARE written to DB for tracking.
  Discord is NOT sent.
  All output is prefixed [PAPER MODE].
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR        = Path(__file__).resolve().parent
CONFIG_PATH = _DIR / "config.json"

# DB_PATH and ALERTS_DB are resolved from config after load_config().
# Defaults are local Windows paths; VPS tennis_config.json overrides them.
_DB_PATH_DEFAULT    = _DIR / "data" / "tennis.db"
_ALERTS_DB_DEFAULT  = Path("D:/models/alerts.db")

load_dotenv(_DIR / ".env", override=False, encoding="utf-8-sig")

# ── logging ────────────────────────────────────────────────────────────────────
_log_dir = _DIR / "logs"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_dir / "predict_tennis.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── config ─────────────────────────────────────────────────────────────────────
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# ── Elo constants ──────────────────────────────────────────────────────────────
SCALE     = 400.0
START_ELO = 1500.0


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / SCALE))


# ── American odds helpers ──────────────────────────────────────────────────────
def american_to_decimal(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def devig(odds_p1: int, odds_p2: int) -> tuple[float, float, float]:
    """
    Two-way de-vig: both sides of the match winner market are posted.
    Returns (fair_p1, fair_p2, overround).
    fair_p1 + fair_p2 == 1.0 by construction.
    """
    dec1 = american_to_decimal(odds_p1)
    dec2 = american_to_decimal(odds_p2)
    imp1 = 1.0 / dec1
    imp2 = 1.0 / dec2
    overround = imp1 + imp2
    return imp1 / overround, imp2 / overround, overround


# ── fuzzy name matcher ─────────────────────────────────────────────────────────
def _normalize(name: str) -> str:
    """Lowercase + strip accents for fuzzy comparison."""
    nfkd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


@dataclass
class MatchResult:
    sackmann_name: str
    score: float          # 0-1, 1=perfect
    method: str           # 'exact' | 'alias' | 'fuzzy' | 'none'


class FuzzyMatcher:
    """
    Matches Kambi player names to Sackmann names for Elo lookup.

    Priority:
      1. Exact match (case-insensitive, accent-stripped)
      2. player_aliases table in tennis.db
      3. Fuzzy match (SequenceMatcher ratio) against all known Sackmann names
         — accepted at >= FUZZY_THRESHOLD; suggested-only below that

    Kambi and Sackmann both use "First Last" format, so most names
    match directly.  Aliases table handles the exceptions.
    """
    FUZZY_THRESHOLD = 0.82

    def __init__(self, elo_conn: sqlite3.Connection):
        # Load all known Sackmann player names (distinct from matches)
        rows = elo_conn.execute(
            "SELECT DISTINCT winner_name FROM matches UNION SELECT DISTINCT loser_name FROM matches"
        ).fetchall()
        self._sackmann = [r[0] for r in rows]
        self._norm_map  = {_normalize(n): n for n in self._sackmann}

        # Load aliases
        rows2 = elo_conn.execute(
            "SELECT kambi_name, sackmann_name FROM player_aliases"
        ).fetchall()
        self._aliases = {_normalize(k): s for k, s in rows2}

        log.info("FuzzyMatcher: %d Sackmann names, %d aliases loaded.",
                 len(self._sackmann), len(self._aliases))

    def match(self, kambi_name: str) -> MatchResult:
        nk = _normalize(kambi_name)

        # 1. Alias table
        if nk in self._aliases:
            return MatchResult(self._aliases[nk], 1.0, "alias")

        # 2. Exact (accent-stripped)
        if nk in self._norm_map:
            return MatchResult(self._norm_map[nk], 1.0, "exact")

        # 3. Fuzzy
        best_score = 0.0
        best_name  = ""
        for sn, sack_name in self._norm_map.items():
            s = SequenceMatcher(None, nk, sn).ratio()
            if s > best_score:
                best_score, best_name = s, sack_name

        if best_score >= self.FUZZY_THRESHOLD:
            return MatchResult(best_name, best_score, "fuzzy")

        # 4. No match
        return MatchResult("", best_score, "none")


# ── Elo lookup ─────────────────────────────────────────────────────────────────
def get_player_elo(
    conn: sqlite3.Connection, player_name: str, before_date: str
) -> dict:
    row = conn.execute(
        """
        SELECT overall_elo, clay_elo, hard_elo, grass_elo,
               matches_played, clay_matches, hard_matches, grass_matches
        FROM   player_elo
        WHERE  player_name = ?
          AND  match_date  < ?
        ORDER  BY match_date DESC, id DESC
        LIMIT  1
        """,
        (player_name, before_date),
    ).fetchone()
    if row:
        keys = ("overall_elo", "clay_elo", "hard_elo", "grass_elo",
                "matches_played", "clay_matches", "hard_matches", "grass_matches")
        return dict(zip(keys, row))
    return {
        "overall_elo": START_ELO, "clay_elo": START_ELO,
        "hard_elo": START_ELO,    "grass_elo": START_ELO,
        "matches_played": 0,      "clay_matches": 0,
        "hard_matches": 0,        "grass_matches": 0,
    }


# ── data structures ────────────────────────────────────────────────────────────
@dataclass
class RawSide:
    """One player/side from VPS props_snapshots."""
    event_id:  str
    game_time: datetime
    player_name: str    # Kambi display name
    over_odds:   int


@dataclass
class Prediction:
    """Full prediction for one match (both sides evaluated)."""
    event_id:      str
    game_time:     datetime
    prediction_date: str
    # Player 1 = the one with the modelled edge (or just P1 by order)
    p1_kambi:      str
    p1_sackmann:   str
    p1_match:      str           # 'exact' | 'alias' | 'fuzzy' | 'none'
    p1_match_score: float
    p1_elo:        float
    p1_n:          int
    p1_odds:       int
    p1_model_prob: float
    p1_fair_prob:  float
    p1_edge:       float
    # Player 2
    p2_kambi:      str
    p2_sackmann:   str
    p2_match:      str
    p2_match_score: float
    p2_elo:        float
    p2_n:          int
    p2_odds:       int
    p2_model_prob: float
    p2_fair_prob:  float
    p2_edge:       float
    # Market
    overround:     float
    # Alert candidate (the side with the edge)
    alert_player:  str           # kambi name of edge side
    alert_edge:    float
    alert_prob:    float
    alert_fair:    float
    alert_odds:    int
    extreme_flag:  bool
    qualifies:     bool          # edge >= threshold AND prob >= threshold


# ── VPS query ──────────────────────────────────────────────────────────────────
def fetch_itf_props(pg_conn, target_date: str) -> list[RawSide]:
    """
    Pull the latest snapshot per player per event for ITF Women match-winner
    props on target_date.  Uses DISTINCT ON to get one row per (event, player).
    """
    cur = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT DISTINCT ON (ps.event_id, ps.player_name)
            g.event_id,
            g.game_time,
            ps.player_name,
            ps.over_odds,
            ps.snapshot_time
        FROM   props_snapshots ps
        JOIN   games            g  ON g.event_id = ps.event_id
        WHERE  g.league        = 'ITF Women'
          AND  ps.line         IS NULL
          AND  ps.over_odds    IS NOT NULL
          AND  g.game_time::date = %(dt)s::date
          AND  g.game_time      > NOW() - INTERVAL '30 minutes'
        ORDER  BY ps.event_id, ps.player_name, ps.snapshot_time DESC
        """,
        {"dt": target_date},
    )
    rows = cur.fetchall()
    log.info("VPS: %d ITF Women match-winner rows for %s", len(rows), target_date)
    return [
        RawSide(
            event_id   = r["event_id"],
            game_time  = r["game_time"],
            player_name= r["player_name"],
            over_odds  = r["over_odds"],
        )
        for r in rows
    ]


def pair_sides(raw: list[RawSide]) -> list[tuple[RawSide, RawSide]]:
    """
    Group RawSide rows by event_id.  Only include events where exactly
    2 sides are present (both players' odds available for proper de-vig).
    Logs events with 1 or 3+ sides for diagnostics.
    """
    by_event: dict[str, list[RawSide]] = {}
    for r in raw:
        by_event.setdefault(r.event_id, []).append(r)

    pairs = []
    skipped_one_sided = 0
    for eid, sides in by_event.items():
        if len(sides) == 2:
            pairs.append((sides[0], sides[1]))
        else:
            log.warning("Event %s has %d sides (expected 2) -- skipped.", eid, len(sides))
            skipped_one_sided += 1

    log.info("Paired: %d complete matches, %d skipped (one-sided).",
             len(pairs), skipped_one_sided)
    return pairs


# ── prediction engine ──────────────────────────────────────────────────────────
def predict_match(
    p1: RawSide,
    p2: RawSide,
    matcher: FuzzyMatcher,
    elo_conn: sqlite3.Connection,
    prediction_date: str,
    config: dict,
) -> Prediction:
    min_edge   = config["betting"]["min_edge"]
    min_prob   = config["betting"]["min_model_prob"]
    min_career = config["betting"].get("min_career_matches", 10)

    # Name matching
    m1 = matcher.match(p1.player_name)
    m2 = matcher.match(p2.player_name)

    # Elo lookup -- use sackmann name if matched, else try kambi name directly
    elo1 = get_player_elo(elo_conn, m1.sackmann_name or p1.player_name, prediction_date)
    elo2 = get_player_elo(elo_conn, m2.sackmann_name or p2.player_name, prediction_date)

    # Model probabilities (from overall Elo)
    p1_prob = elo_win_prob(elo1["overall_elo"], elo2["overall_elo"])
    p2_prob = 1.0 - p1_prob

    # Two-way de-vig
    fair_p1, fair_p2, overround = devig(p1.over_odds, p2.over_odds)

    # Guard: overround < 1.0 means the two-way market sums to < 100% implied —
    # de-vig would inflate both probabilities above the raw price, creating fake edges.
    # This usually indicates a data error or an in-play market that's partially settled.
    if overround < 1.0:
        log.warning(
            "Sub-100%% overround (%.3f) for %s vs %s — skipping (data error or settled market).",
            overround, p1.player_name, p2.player_name,
        )

    # Edges
    edge1 = p1_prob - fair_p1
    edge2 = p2_prob - fair_p2

    # Determine alert side (higher positive edge)
    if edge1 >= edge2:
        alert_player = p1.player_name
        alert_edge   = edge1
        alert_prob   = p1_prob
        alert_fair   = fair_p1
        alert_odds   = p1.over_odds
    else:
        alert_player = p2.player_name
        alert_edge   = edge2
        alert_prob   = p2_prob
        alert_fair   = fair_p2
        alert_odds   = p2.over_odds

    # Reliability gate: both players must have enough career history for the
    # Elo rating to mean anything. Mirrors backtest_tennis.py's min_matches
    # filter (the backtest excluded these; live used to bet them -- the bug).
    enough_history = (elo1["matches_played"] >= min_career
                      and elo2["matches_played"] >= min_career)

    extreme_flag = abs(alert_edge) > 0.20

    # EV gate: a negative-EV bet loses money in expectation regardless of edge.
    # ev = model_prob * decimal_odds - 1 (already computed correctly in write paths).
    alert_dec = american_to_decimal(alert_odds)
    alert_ev  = alert_prob * alert_dec - 1.0

    max_odds     = config["betting"].get("max_odds_american", 300)
    qualifies    = (alert_edge >= min_edge
                    and alert_prob >= min_prob
                    and enough_history
                    and alert_ev > 0.0
                    and abs(alert_odds) <= max_odds
                    and not extreme_flag
                    and overround >= 1.0)

    return Prediction(
        event_id       = p1.event_id,
        game_time      = p1.game_time,
        prediction_date= prediction_date,
        p1_kambi       = p1.player_name,
        p1_sackmann    = m1.sackmann_name,
        p1_match       = m1.method,
        p1_match_score = m1.score,
        p1_elo         = elo1["overall_elo"],
        p1_n           = elo1["matches_played"],
        p1_odds        = p1.over_odds,
        p1_model_prob  = p1_prob,
        p1_fair_prob   = fair_p1,
        p1_edge        = edge1,
        p2_kambi       = p2.player_name,
        p2_sackmann    = m2.sackmann_name,
        p2_match       = m2.method,
        p2_match_score = m2.score,
        p2_elo         = elo2["overall_elo"],
        p2_n           = elo2["matches_played"],
        p2_odds        = p2.over_odds,
        p2_model_prob  = p2_prob,
        p2_fair_prob   = fair_p2,
        p2_edge        = edge2,
        overround      = overround,
        alert_player   = alert_player,
        alert_edge     = alert_edge,
        alert_prob     = alert_prob,
        alert_fair     = alert_fair,
        alert_odds     = alert_odds,
        extreme_flag   = extreme_flag,
        qualifies      = qualifies,
    )


# ── output helpers ─────────────────────────────────────────────────────────────
def print_match_report(preds: list[Prediction], matcher_stats: dict) -> None:
    """Print full match scan table with match quality and edges."""
    n = len(preds)
    n_matched  = sum(1 for p in preds
                     if p.p1_match != "none" and p.p2_match != "none")
    n_partial  = sum(1 for p in preds
                     if (p.p1_match == "none") != (p.p2_match == "none"))
    n_unmatched = n - n_matched - n_partial
    n_edge      = sum(1 for p in preds if p.qualifies)

    print()
    print(f"  ITF Women matches found today: {n}")
    print(f"  Both players matched          : {n_matched}")
    print(f"  Partial (one side unmatched)  : {n_partial}")
    print(f"  Neither matched               : {n_unmatched}")
    print(f"  Qualifying edges (>= threshold): {n_edge}")
    print()

    # Full match table
    print(f"  {'Event':>12}  {'P1 Kambi':<28} {'M':4} {'Elo1':>6}  "
          f"{'P2 Kambi':<28} {'M':4} {'Elo2':>6}  {'Ovr':>5}  {'Edge':>6}  {'Qual'}")
    print(f"  {'-'*130}")

    for p in sorted(preds, key=lambda x: -abs(x.alert_edge)):
        m1 = p.p1_match[0].upper()  # E/A/F/N
        m2 = p.p2_match[0].upper()
        edge_str = f"{p.alert_edge*100:+.1f}%"
        qual_str = "YES" if p.qualifies else ("EXT" if p.extreme_flag else "---")
        print(
            f"  {p.event_id:>12}  {p.p1_kambi:<28} {m1:4} {p.p1_elo:>6.0f}  "
            f"{p.p2_kambi:<28} {m2:4} {p.p2_elo:>6.0f}  "
            f"{p.overround:>5.3f}  {edge_str:>6}  {qual_str}"
        )

    print()
    print("  Match key: E=exact  A=alias  F=fuzzy  N=not matched")
    print("  Elo=1500 means no prior history found for that player")


def print_unmatched(preds: list[Prediction]) -> None:
    """Log unmatched players with best fuzzy suggestion."""
    unmatched = []
    for p in preds:
        if p.p1_match == "none":
            unmatched.append((p.p1_kambi, p.p1_match_score))
        if p.p2_match == "none":
            unmatched.append((p.p2_kambi, p.p2_match_score))
    if not unmatched:
        return
    print()
    print(f"  UNMATCHED PLAYERS ({len(unmatched)}) -- consider adding to player_aliases:")
    for name, score in sorted(unmatched, key=lambda x: x[0]):
        print(f"    {name!r:<35}  best fuzzy score: {score:.2f}")
    print("  To add alias: INSERT INTO player_aliases (kambi_name, sackmann_name) VALUES (...)")


def print_alert_preview(p: Prediction, paper_mode: bool) -> None:
    """Pretty-print what the Discord embed would contain."""
    paper = "[PAPER MODE] " if paper_mode else ""
    extreme = "  [!EXTREME -- gap > 20%, verify before betting]" if p.extreme_flag else ""
    dec = american_to_decimal(p.alert_odds)
    ev  = p.alert_prob * dec - 1.0

    print()
    print(f"  {'-'*58}")
    print(f"  {paper}[Tennis Edge Alert]{extreme}")
    print(f"  {'-'*58}")
    print(f"  Player          : {p.alert_player}")
    opp = p.p2_kambi if p.alert_player == p.p1_kambi else p.p1_kambi
    print(f"  Opponent        : {opp}")
    print(f"  Surface         : Unknown (not in VPS feed)")
    print(f"  Predicted Win%  : {p.alert_prob*100:.1f}%")
    print(f"  Market Implied% : {p.alert_fair*100:.1f}%")
    print(f"  Edge            : +{p.alert_edge*100:.1f}%")
    print(f"  Odds            : {p.alert_odds:+d}  ({dec:.2f}x)")
    print(f"  EV              : {ev*100:+.1f}%")
    print(f"  Game time       : {p.game_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Event ID        : {p.event_id}")
    print(f"  {'-'*58}")


# ── DB writes ──────────────────────────────────────────────────────────────────
def write_predictions(preds: list[Prediction], elo_conn: sqlite3.Connection) -> None:
    """Upsert all predictions to tennis.db predictions table."""
    sql = """
        INSERT OR IGNORE INTO predictions (
            prediction_date, event_id,
            player1_name, player2_name,
            player1_overall_elo, player2_overall_elo,
            player1_rank, player2_rank,
            player1_model_prob, player2_model_prob,
            player1_kambi_odds, player2_kambi_odds,
            player1_fair_prob, player2_fair_prob,
            player1_edge, player2_edge,
            notified, created_at
        ) VALUES (?,?,?,?,?,?,NULL,NULL,?,?,?,?,?,?,?,?,0,CURRENT_TIMESTAMP)
    """
    with elo_conn:
        for p in preds:
            elo_conn.execute(sql, (
                p.prediction_date, p.event_id,
                p.p1_kambi, p.p2_kambi,
                p.p1_elo, p.p2_elo,
                p.p1_model_prob, p.p2_model_prob,
                p.p1_odds, p.p2_odds,
                p.p1_fair_prob, p.p2_fair_prob,
                p.p1_edge, p.p2_edge,
            ))
    log.info("Wrote %d predictions to tennis.db.", len(preds))


def write_tennis_alerts(
    alerts: list[Prediction], elo_conn: sqlite3.Connection
) -> list[tuple[Prediction, int]]:
    """
    Write qualifying edges to tennis.db alerts table.
    Returns list of (pred, alert_id) for DB cross-reference.
    """
    results = []
    with elo_conn:
        for p in alerts:
            pred_id = elo_conn.execute(
                "SELECT id FROM predictions WHERE prediction_date=? AND event_id=?",
                (p.prediction_date, p.event_id),
            ).fetchone()
            pid = pred_id[0] if pred_id else None

            opp = p.p2_kambi if p.alert_player == p.p1_kambi else p.p1_kambi
            cur = elo_conn.execute(
                """
                INSERT OR IGNORE INTO alerts
                    (prediction_id, player_name, opponent_name, surface, tourney_level,
                     model_prob, fair_prob, edge, odds, discord_sent, result, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,0,'pending',CURRENT_TIMESTAMP)
                """,
                (pid, p.alert_player, opp, "Unknown", "ITF",
                 p.alert_prob, p.alert_fair, p.alert_edge, p.alert_odds),
            )
            results.append((p, cur.lastrowid))
    log.info("Wrote %d alerts to tennis.db.", len(results))
    return results


def write_unified_alerts(
    alerts: list[Prediction],
    alerts_conn: sqlite3.Connection,
    model_version: str,
) -> None:
    """
    Write qualifying edges to D:/models/alerts.db bet_alerts (sport='TENNIS').
    This keeps the unified weekly summary working across all three sports.
    """
    now = datetime.now()
    with alerts_conn:
        for p in alerts:
            opp = p.p2_kambi if p.alert_player == p.p1_kambi else p.p1_kambi
            dec = american_to_decimal(p.alert_odds)
            ev  = p.alert_prob * dec - 1.0
            # half-Kelly: f* = (p*dec - 1) / (dec - 1) = ev / (dec - 1), then ×0.5.
            # NOTE: numerator is the RETURN edge (ev), NOT the probability edge
            # (alert_prob - fair_prob). Using the probability edge here was the bug.
            kelly_h = 0.5 * (ev / max(dec - 1.0, 0.01))
            alerts_conn.execute(
                """
                INSERT OR IGNORE INTO bet_alerts (
                    sport, model_version, alert_date, alert_time,
                    game_id, player_name, market_type, direction, line,
                    odds, predicted_value, model_prob, implied_prob,
                    edge_prob, ev, kelly_half, result, notified, graded
                ) VALUES (?,?,?,?,?,?,?,?,NULL,?,NULL,?,?,?,?,?,?,0,0)
                """,
                (
                    "TENNIS", model_version,
                    now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                    p.event_id, p.alert_player, "Match Winner", "WIN",
                    p.alert_odds,
                    p.alert_prob, p.alert_fair, p.alert_edge,
                    ev, kelly_h,
                    "PENDING",
                ),
            )
    log.info("Wrote %d rows to alerts.db (sport=TENNIS).", len(alerts))


def mark_notified(
    elo_conn: sqlite3.Connection,
    event_id: str,
    prediction_date: str,
    alerts_conn: Optional[sqlite3.Connection] = None,
    player_name: str = "",
) -> None:
    ts = datetime.utcnow().isoformat()
    elo_conn.execute(
        "UPDATE predictions SET notified=1, notified_at=? "
        "WHERE event_id=? AND prediction_date=?",
        (ts, event_id, prediction_date),
    )
    elo_conn.commit()
    if alerts_conn and player_name:
        alerts_conn.execute(
            "UPDATE bet_alerts SET notified=1, notified_at=? "
            "WHERE sport='TENNIS' AND game_id=? AND player_name=?",
            (ts, event_id, player_name),
        )
        alerts_conn.commit()


def was_notified(
    elo_conn: sqlite3.Connection, event_id: str, prediction_date: str
) -> bool:
    row = elo_conn.execute(
        "SELECT notified FROM predictions WHERE event_id=? AND prediction_date=?",
        (event_id, prediction_date),
    ).fetchone()
    return bool(row and row[0])


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Daily ITF Women tennis prediction")
    parser.add_argument("--date",    default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all DB writes and Discord sends")
    parser.add_argument("--config",  default=None, metavar="PATH",
                        help="Config file path (default: config.json next to script)")
    args = parser.parse_args()

    config       = load_config(args.config)
    paper_mode   = config["betting"]["paper_mode"]
    min_edge     = config["betting"]["min_edge"]
    min_prob     = config["betting"]["min_model_prob"]
    model_ver    = f"{config['model']['name']}_v{config['model']['version']}"
    vps_cfg      = config["vps"]
    target_date  = args.date or datetime.now().strftime("%Y-%m-%d")

    # Resolve DB paths from config (allows VPS tennis_config.json to override)
    db_path    = Path(config.get("data", {}).get("db_path",    str(_DB_PATH_DEFAULT)))
    alerts_db  = Path(config.get("data", {}).get("alerts_db_path", str(_ALERTS_DB_DEFAULT)))

    mode_tag = "[DRY RUN]" if args.dry_run else ("[PAPER MODE]" if paper_mode else "[LIVE]")
    log.info("predict_tennis.py %s  date=%s", mode_tag, target_date)
    log.info("  tennis.db : %s", db_path)
    log.info("  alerts.db : %s", alerts_db)

    # ── connections ────────────────────────────────────────────────────────────
    try:
        pg_conn = psycopg2.connect(
            host     = vps_cfg["host"],
            port     = vps_cfg.get("port", 5432),
            dbname   = vps_cfg["database"],
            user     = vps_cfg["user"],
            password = os.environ.get("VPS_DB_PASSWORD", vps_cfg.get("password", "")),
            connect_timeout = 15,
        )
    except psycopg2.OperationalError as exc:
        log.error("Cannot connect to VPS Postgres: %s", exc)
        return 1

    elo_conn    = sqlite3.connect(db_path)
    elo_conn.execute("PRAGMA journal_mode=WAL")
    alerts_conn = sqlite3.connect(alerts_db) if not args.dry_run else None

    # ── fetch & pair ───────────────────────────────────────────────────────────
    raw   = fetch_itf_props(pg_conn, target_date)
    if not raw:
        log.warning("No ITF Women match-winner props found for %s.", target_date)
        pg_conn.close()
        elo_conn.close()
        return 0

    pairs = pair_sides(raw)
    if not pairs:
        log.warning("No complete (both-sided) matches found for %s.", target_date)
        pg_conn.close()
        elo_conn.close()
        return 0

    pg_conn.close()

    # ── predictions ────────────────────────────────────────────────────────────
    matcher = FuzzyMatcher(elo_conn)
    preds   = [
        predict_match(p1, p2, matcher, elo_conn, target_date, config)
        for p1, p2 in pairs
    ]

    alerts = [p for p in preds if p.qualifies]

    # ── report ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  ITF Women Tennis - {target_date}  {mode_tag}")
    print("=" * 62)
    print_match_report(preds, {})
    print_unmatched(preds)

    if not alerts:
        print(f"\n  No qualifying edges today (min_edge={min_edge*100:.0f}%,"
              f" min_prob={min_prob*100:.0f}%).")
    else:
        print(f"\n  QUALIFYING EDGES ({len(alerts)}):")
        for a in sorted(alerts, key=lambda x: -x.alert_edge):
            print_alert_preview(a, paper_mode or args.dry_run)

    # ── writes ─────────────────────────────────────────────────────────────────
    if not args.dry_run:
        write_predictions(preds, elo_conn)
        if alerts:
            write_tennis_alerts(alerts, elo_conn)
            if alerts_conn:
                write_unified_alerts(alerts, alerts_conn, model_ver)

    # ── Discord ────────────────────────────────────────────────────────────────
    if alerts and not args.dry_run and not paper_mode:
        from notify_tennis import send_summary, send_tennis_alert
        if len(alerts) > 1:
            send_summary(len(alerts), config)
        for a in sorted(alerts, key=lambda x: -x.alert_edge):
            if not was_notified(elo_conn, a.event_id, target_date):
                opp = a.p2_kambi if a.alert_player == a.p1_kambi else a.p1_kambi
                # Convert UTC game_time to CT (CDT = UTC-5 in summer)
                try:
                    ct_str = (a.game_time - timedelta(hours=5)).strftime("%-I:%M %p CT")
                except Exception:
                    ct_str = ""
                sent = send_tennis_alert({
                    "player_name":   a.alert_player,
                    "opponent_name": opp,
                    "tourney_name":  "ITF Women",
                    "model_prob":    a.alert_prob,
                    "fair_prob":     a.alert_fair,
                    "edge":          a.alert_edge,
                    "odds":          a.alert_odds,
                    "event_id":      a.event_id,
                    "extreme_flag":  a.extreme_flag,
                    "game_time_ct":  ct_str,
                }, config)
                if sent:
                    mark_notified(elo_conn, a.event_id, target_date,
                                  alerts_conn=alerts_conn,
                                  player_name=a.alert_player)
    elif alerts and paper_mode and not args.dry_run:
        log.info("[PAPER MODE] %d alert(s) not sent to Discord.", len(alerts))

    elo_conn.close()
    if alerts_conn:
        alerts_conn.close()

    log.info("Done. %d matches, %d edges.", len(preds), len(alerts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
