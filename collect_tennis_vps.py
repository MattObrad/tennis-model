"""
collect_tennis_vps.py -- VPS-side daily ITF Women match data updater.

Designed to run at 5am UTC every day via cron.  Lightweight: only
downloads the current year's Sackmann CSV, inserts new matches, then
rebuilds Elo ratings from scratch (~3-5 seconds for the full history).

Full rebuild (not incremental) keeps the implementation simple and
eliminates any risk of stale/partial Elo states from mid-year inserts.

Usage:
    python collect_tennis_vps.py              # current calendar year
    python collect_tennis_vps.py --year 2026  # explicit year
    python collect_tennis_vps.py --skip-elo   # collect only, no Elo rebuild
    python collect_tennis_vps.py --dry-run    # show what would happen

Cron example (5am UTC daily):
    0 5 * * * cd /home/picks && python collect_tennis_vps.py >> /home/picks/logs/collect_tennis.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import requests

# ── paths (VPS-side defaults) ─────────────────────────────────────────────────
_DIR    = Path(__file__).resolve().parent
DB_PATH = _DIR / "tennis.db"     # /home/picks/tennis.db on VPS

SACKMANN_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
    "wta_matches_qual_itf_{year}.csv"
)

# ── logging ───────────────────────────────────────────────────────────────────
_log_dir = _DIR / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_dir / "collect_tennis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── normalization (mirrors collect_tennis.py) ─────────────────────────────────
_SURFACE_MAP = {"clay": "Clay", "hard": "Hard", "grass": "Grass", "carpet": "Hard"}

def _normalize_surface(raw) -> str:
    s = str(raw).strip().lower() if raw and str(raw).strip().lower() != "nan" else ""
    return _SURFACE_MAP.get(s, "Hard")

def _normalize_level(raw) -> str:
    s = str(raw).strip() if raw and str(raw).strip().lower() not in ("nan","none","") else ""
    if not s:       return "ITF"
    if s.upper().startswith("W") and s[1:].isdigit(): return s.upper()
    if s.isdigit(): return f"W{s}"
    return s.upper()

def _parse_date(raw) -> str | None:
    s = str(raw).strip().split(".")[0]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else None

def _safe_int(v):
    try:
        f = float(v)
        import math
        return int(f) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None

def _safe_str(v):
    s = str(v).strip()
    return None if s.lower() in ("", "nan", "none") else s


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_year(year: int, session: requests.Session) -> str | None:
    url = SACKMANN_URL.format(year=year)
    try:
        resp = session.get(url, timeout=30)
    except requests.RequestException as exc:
        log.error("Network error fetching %d: %s", year, exc)
        return None
    if resp.status_code == 404:
        log.warning("Year %d not yet published on GitHub.", year)
        return None
    if not resp.ok:
        log.error("HTTP %d fetching year %d.", resp.status_code, year)
        return None
    log.info("Downloaded %d bytes for year %d.", len(resp.text), year)
    return resp.text


def parse_csv(text: str, year: int) -> list[dict]:
    try:
        import pandas as pd
        df = pd.read_csv(StringIO(text), dtype=str, low_memory=False,
                         encoding_errors="replace")
    except Exception as exc:
        log.error("CSV parse error for year %d: %s", year, exc)
        return []

    rows = []
    for _, r in df.iterrows():
        d = _parse_date(r.get("tourney_date"))
        if not d:
            continue
        wn = _safe_str(r.get("winner_name"))
        ln = _safe_str(r.get("loser_name"))
        if not wn or not ln:
            continue
        # Drop walkovers / defaults (no match played). RET kept for grading;
        # elo_engine.py excludes RET from rating updates.
        score_u = (_safe_str(r.get("score")) or "").upper()
        if "W/O" in score_u or "DEF" in score_u or "WALKOVER" in score_u:
            continue
        rows.append({
            "tourney_id":    _safe_str(r.get("tourney_id")) or f"UNK-{year}",
            "tourney_name":  _safe_str(r.get("tourney_name")) or "Unknown",
            "tourney_date":  d,
            "tourney_level": _normalize_level(r.get("tourney_level")),
            "surface":       _normalize_surface(r.get("surface")),
            "round":         _safe_str(r.get("round")),
            "winner_name":   wn,
            "loser_name":    ln,
            "winner_id":     _safe_str(r.get("winner_id")),
            "loser_id":      _safe_str(r.get("loser_id")),
            "winner_rank":   _safe_int(r.get("winner_rank")),
            "loser_rank":    _safe_int(r.get("loser_rank")),
            "score":         _safe_str(r.get("score")),
            "minutes":       _safe_int(r.get("minutes")),
            "data_year":     year,
        })
    return rows


def upsert_matches(conn: sqlite3.Connection, rows: list[dict]) -> tuple[int, int]:
    sql = """
        INSERT OR IGNORE INTO matches
            (tourney_id, tourney_name, tourney_date, tourney_level, surface,
             round, winner_name, loser_name, winner_id, loser_id,
             winner_rank, loser_rank, score, minutes, data_year)
        VALUES
            (:tourney_id, :tourney_name, :tourney_date, :tourney_level, :surface,
             :round, :winner_name, :loser_name, :winner_id, :loser_id,
             :winner_rank, :loser_rank, :score, :minutes, :data_year)
    """
    inserted = skipped = 0
    with conn:
        for row in rows:
            cur = conn.execute(sql, row)
            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1
    return inserted, skipped


# ── Elo rebuild (import from co-deployed elo_engine.py) ──────────────────────
def rebuild_elo(conn: sqlite3.Connection) -> None:
    """
    Full Elo rebuild from all matches in DB.
    Imports build_ratings from elo_engine.py (must be in the same directory).
    Takes ~3-5 seconds for 273K+ matches.
    """
    try:
        from elo_engine import build_ratings  # co-deployed alongside this script
    except ImportError:
        log.error("Cannot import elo_engine — Elo rebuild skipped. "
                  "Ensure elo_engine.py is in the same directory.")
        return

    t0 = time.time()
    log.info("Rebuilding Elo ratings (full pass)...")
    _, state = build_ratings(conn)
    elapsed = time.time() - t0
    n_players = len(state)
    n_rows    = conn.execute("SELECT COUNT(*) FROM player_elo").fetchone()[0]
    log.info("Elo rebuild complete: %d players, %d rows in %.1fs.",
             n_players, n_rows, elapsed)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="VPS-side daily ITF Women match data updater"
    )
    parser.add_argument("--year",     type=int, default=datetime.now().year,
                        help="Year to fetch (default: current year)")
    parser.add_argument("--skip-elo", action="store_true",
                        help="Skip Elo rebuild after collecting")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would happen without writing to DB")
    args = parser.parse_args()

    if not DB_PATH.exists():
        log.error("tennis.db not found at %s. Deploy it first with deploy_tennis.py.", DB_PATH)
        return 1

    log.info("collect_tennis_vps.py -- year=%d  db=%s", args.year, DB_PATH)

    # Count existing rows for this year before update
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    before = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE data_year = ?", (args.year,)
    ).fetchone()[0]
    total_before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    log.info("DB before: %d total matches (%d for year %d).",
             total_before, before, args.year)

    session = requests.Session()
    session.headers["User-Agent"] = "ObServatory-Tennis-VPS/1.0"

    text = fetch_year(args.year, session)
    if not text:
        log.warning("No data fetched for year %d -- no changes made.", args.year)
        conn.close()
        return 0

    rows = parse_csv(text, args.year)
    log.info("Parsed %d match rows from CSV.", len(rows))

    if args.dry_run:
        log.info("[DRY RUN] Would insert up to %d rows -- no changes written.", len(rows))
        conn.close()
        return 0

    inserted, skipped = upsert_matches(conn, rows)
    after = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE data_year = ?", (args.year,)
    ).fetchone()[0]

    log.info(
        "Matches year=%d: before=%d  after=%d  new=%d  dupes=%d",
        args.year, before, after, inserted, skipped,
    )

    if inserted == 0:
        log.info("No new matches -- Elo ratings are already current.")
    elif not args.skip_elo:
        rebuild_elo(conn)
    else:
        log.info("--skip-elo set -- Elo rebuild deferred.")

    conn.close()
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
