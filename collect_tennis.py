"""
collect_tennis.py -- Pull Jeff Sackmann ITF Women's match data into tennis.db

Usage:
    python collect_tennis.py                  # fetch all years 2015-2026
    python collect_tennis.py --years 2024     # single year
    python collect_tennis.py --years 2020-2026
    python collect_tennis.py --rebuild        # drop & recreate matches + player_elo

What it does:
  1. Creates data/tennis.db with the full model schema
  2. Downloads wta_matches_qual_itf_YYYY.csv from GitHub (2015-2026)
  3. Caches raw CSVs in data/raw/ to avoid re-downloading
  4. Normalises surface, tourney_level, dates
  5. Inserts into matches table (UNIQUE constraint skips dupes on re-run)
  6. Prints per-year row counts + summary stats

GitHub source:
  https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_qual_itf_YYYY.csv

Note on filename: Sackmann uses 'wta_matches_qual_itf' (with 'qual') for
the full ITF Women's circuit -- the name is historical; it covers ALL rounds,
not just qualifying.  tourney_level in the file is numeric ('15','35','50')
not prefixed -- we normalise to 'W15','W35','W50', etc.
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

import pandas as pd
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
DB_PATH   = DATA_DIR / "tennis.db"

SACKMANN_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
    "wta_matches_qual_itf_{year}.csv"
)
DEFAULT_YEARS = range(2015, 2027)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── schema ────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

-- ── matches ──────────────────────────────────────────────────────────────────
-- One row per match outcome (winner/loser).  Source: Sackmann ITF CSVs.
CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY,
    tourney_id    TEXT    NOT NULL,
    tourney_name  TEXT    NOT NULL,
    tourney_date  DATE    NOT NULL,  -- ISO YYYY-MM-DD; tournament start date
    tourney_level TEXT    NOT NULL,  -- W15 W25 W35 W50 W60 W80 W100 W125
    surface       TEXT    NOT NULL,  -- Clay | Hard | Grass  (Carpet → Hard)
    round         TEXT,              -- R128 R64 R32 R16 QF SF F
    winner_name   TEXT    NOT NULL,
    loser_name    TEXT    NOT NULL,
    winner_id     TEXT,              -- Sackmann internal player ID
    loser_id      TEXT,
    winner_rank   INTEGER,           -- WTA ranking at match time (often NULL in ITF)
    loser_rank    INTEGER,
    score         TEXT,              -- e.g. "6-3 6-2"
    minutes       INTEGER,
    data_year     INTEGER NOT NULL,  -- source CSV year
    UNIQUE(tourney_id, round, winner_name, loser_name)
);
CREATE INDEX IF NOT EXISTS idx_matches_date    ON matches (tourney_date);
CREATE INDEX IF NOT EXISTS idx_matches_winner  ON matches (winner_name, tourney_date);
CREATE INDEX IF NOT EXISTS idx_matches_loser   ON matches (loser_name,  tourney_date);
CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches (surface, tourney_date);

-- ── player_elo ────────────────────────────────────────────────────────────────
-- Point-in-time Elo ratings.  One row per player per match played.
-- Lookup pattern: WHERE player_name=? AND match_date < ? ORDER BY match_date DESC LIMIT 1
CREATE TABLE IF NOT EXISTS player_elo (
    id             INTEGER PRIMARY KEY,
    player_name    TEXT    NOT NULL,
    match_date     DATE    NOT NULL,
    overall_elo    REAL    NOT NULL DEFAULT 1500.0,
    clay_elo       REAL    NOT NULL DEFAULT 1500.0,
    hard_elo       REAL    NOT NULL DEFAULT 1500.0,
    grass_elo      REAL    NOT NULL DEFAULT 1500.0,
    matches_played INTEGER NOT NULL DEFAULT 0,
    clay_matches   INTEGER NOT NULL DEFAULT 0,
    hard_matches   INTEGER NOT NULL DEFAULT 0,
    grass_matches  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_elo_lookup ON player_elo (player_name, match_date);

-- ── predictions ───────────────────────────────────────────────────────────────
-- One row per match per daily prediction run.
-- Stores full Elo inputs, market data, and computed edge.
-- notified guards against duplicate Discord alerts across two daily runs.
CREATE TABLE IF NOT EXISTS predictions (
    id                   INTEGER PRIMARY KEY,
    prediction_date      DATE      NOT NULL,
    event_id             TEXT,                   -- Kambi event_id from VPS
    player1_name         TEXT      NOT NULL,      -- canonical Kambi-side name
    player2_name         TEXT      NOT NULL,
    surface              TEXT,
    tourney_name         TEXT,
    tourney_level        TEXT,
    -- Elo at prediction time (fetched point-in-time from player_elo)
    player1_overall_elo  REAL,
    player2_overall_elo  REAL,
    player1_surface_elo  REAL,
    player2_surface_elo  REAL,
    -- Ranking & context features
    player1_rank         INTEGER,
    player2_rank         INTEGER,
    player1_h2h_wins     INTEGER   DEFAULT 0,
    player2_h2h_wins     INTEGER   DEFAULT 0,
    player1_surface_wr   REAL,                   -- win rate, last 20 on this surface
    player2_surface_wr   REAL,
    elo_diff             REAL,                   -- player1 − player2 overall
    surface_elo_diff     REAL,                   -- player1 − player2 surface
    -- Model output
    player1_model_prob   REAL,
    player2_model_prob   REAL,
    -- Market (from Kambi VPS)
    player1_kambi_odds   INTEGER,                -- American odds
    player2_kambi_odds   INTEGER,
    player1_fair_prob    REAL,                   -- after two-way de-vig
    player2_fair_prob    REAL,
    -- Edge
    player1_edge         REAL,                   -- model_prob − fair_prob
    player2_edge         REAL,
    -- Notification guard (prevents duplicate Discord sends across two daily runs)
    notified             INTEGER   DEFAULT 0,
    notified_at          TIMESTAMP,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prediction_date, event_id)
);

-- ── alerts ────────────────────────────────────────────────────────────────────
-- Predictions that crossed the edge threshold (local mirror).
-- predict_tennis.py also writes to alerts.db (unified tracker).
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY,
    prediction_id  INTEGER REFERENCES predictions(id),
    player_name    TEXT    NOT NULL,
    opponent_name  TEXT    NOT NULL,
    surface        TEXT,
    tourney_level  TEXT,
    tourney_name   TEXT,
    match_date     DATE,
    model_prob     REAL    NOT NULL,
    fair_prob      REAL    NOT NULL,
    edge           REAL    NOT NULL,
    odds           INTEGER NOT NULL,   -- American
    discord_sent   INTEGER DEFAULT 0,
    result         TEXT,               -- 'win' | 'loss' | 'pending'
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── player_aliases ────────────────────────────────────────────────────────────
-- Maps Kambi display names → Sackmann CSV names.
-- Kambi uses "Last, First"; Sackmann uses "First Last".
CREATE TABLE IF NOT EXISTS player_aliases (
    id            INTEGER PRIMARY KEY,
    kambi_name    TEXT NOT NULL UNIQUE,
    sackmann_name TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ── normalization helpers ──────────────────────────────────────────────────────
# Sackmann uses title-case surfaces in modern files
_SURFACE_MAP: dict[str, str] = {
    "clay":    "Clay",
    "hard":    "Hard",
    "grass":   "Grass",
    "carpet":  "Hard",   # Carpet → Hard; virtually nonexistent in ITF post-2015
    "indoor":  "Hard",   # belt-and-suspenders
}

def normalize_surface(raw) -> str:
    s = str(raw).strip().lower() if raw and str(raw).strip().lower() != "nan" else ""
    return _SURFACE_MAP.get(s, "Hard")


def normalize_level(raw) -> str:
    """
    Sackmann qual_itf files store tourney_level as a bare number:
      '15' -> 'W15', '25' -> 'W25', '35' -> 'W35', '50' -> 'W50', etc.
    Some older files may already have 'W' prefix or use 'ITF' -- handle all.
    """
    s = str(raw).strip() if raw and str(raw).strip().lower() not in ("nan", "none", "") else ""
    if not s:
        return "ITF"
    # Already prefixed (e.g. 'W35')
    if s.upper().startswith("W") and s[1:].isdigit():
        return s.upper()
    # Bare numeric (e.g. '35') -- the common case in modern qual_itf files
    if s.isdigit():
        return f"W{s}"
    # Anything else: pass through upper-cased
    return s.upper()


def parse_tourney_date(raw) -> str | None:
    """
    Sackmann stores tourney_date as YYYYMMDD (integer in CSV).
    Returns ISO 'YYYY-MM-DD' string or None if unparseable.
    """
    s = str(raw).strip().split(".")[0]  # strip any float decimal
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def safe_int(v) -> int | None:
    try:
        f = float(v)
        return int(f) if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def safe_str(v) -> str | None:
    s = str(v).strip()
    return None if s.lower() in ("", "nan", "none") else s


# ── CSV fetch / cache ──────────────────────────────────────────────────────────
def fetch_year_csv(year: int, session: requests.Session, use_cache: bool = True) -> str | None:
    """
    Return raw CSV text for `year`.  Downloads from GitHub; caches to data/raw/.
    Returns None if the file doesn't exist (e.g. 2026 not yet published).
    """
    cache_path = RAW_DIR / f"wta_matches_qual_itf_{year}.csv"

    # NEVER serve the current year from cache: Sackmann appends to it weekly, so
    # a cached current-year file silently freezes the model on stale data. Only
    # completed past years are safe to cache.
    if year >= datetime.now().year:
        use_cache = False

    if use_cache and cache_path.exists():
        log.info("Year %d: loading from cache (%s)", year, cache_path.name)
        return cache_path.read_text(encoding="utf-8-sig")

    url = SACKMANN_URL.format(year=year)
    try:
        resp = session.get(url, timeout=30)
    except requests.RequestException as exc:
        log.error("Year %d: network error — %s", year, exc)
        return None

    if resp.status_code == 404:
        log.warning("Year %d: 404 — not yet published on GitHub", year)
        return None
    if not resp.ok:
        log.error("Year %d: HTTP %d — %s", year, resp.status_code, resp.text[:120])
        return None

    text = resp.text
    if use_cache:
        cache_path.write_text(text, encoding="utf-8")
        log.info("Year %d: downloaded %d bytes → cached (data/raw/)", year, len(text))
    else:
        log.info("Year %d: downloaded %d bytes (no cache)", year, len(text))

    return text


def parse_csv(text: str, year: int) -> list[dict]:
    """Parse raw CSV text → list of match row dicts ready for INSERT."""
    try:
        df = pd.read_csv(StringIO(text), dtype=str, low_memory=False, encoding_errors="replace")
    except Exception as exc:
        log.error("Year %d: CSV parse error — %s", year, exc)
        return []

    rows: list[dict] = []
    skipped = 0

    for _, r in df.iterrows():
        tourney_date = parse_tourney_date(r.get("tourney_date"))
        if not tourney_date:
            skipped += 1
            continue

        winner = safe_str(r.get("winner_name"))
        loser  = safe_str(r.get("loser_name"))
        if not winner or not loser:
            skipped += 1
            continue

        # Drop walkovers / defaults: no match was played, so there is no
        # athletic result. (Retirements are kept here -- the result stands for
        # grading -- but elo_engine.py excludes them from rating updates.)
        score_u = (safe_str(r.get("score")) or "").upper()
        if "W/O" in score_u or "DEF" in score_u or "WALKOVER" in score_u:
            skipped += 1
            continue

        tourney_id   = safe_str(r.get("tourney_id"))   or f"UNK-{year}"
        tourney_name = safe_str(r.get("tourney_name")) or "Unknown"

        rows.append({
            "tourney_id":    tourney_id,
            "tourney_name":  tourney_name,
            "tourney_date":  tourney_date,
            "tourney_level": normalize_level(r.get("tourney_level")),
            "surface":       normalize_surface(r.get("surface")),
            "round":         safe_str(r.get("round")),
            "winner_name":   winner,
            "loser_name":    loser,
            "winner_id":     safe_str(r.get("winner_id")),
            "loser_id":      safe_str(r.get("loser_id")),
            "winner_rank":   safe_int(r.get("winner_rank")),
            "loser_rank":    safe_int(r.get("loser_rank")),
            "score":         safe_str(r.get("score")),
            "minutes":       safe_int(r.get("minutes")),
            "data_year":     year,
        })

    if skipped:
        log.warning("Year %d: skipped %d rows (bad date or missing names)", year, skipped)

    return rows


# ── DB helpers ────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    log.info("Schema OK at %s", DB_PATH)


def upsert_matches(conn: sqlite3.Connection, rows: list[dict]) -> tuple[int, int]:
    """
    Bulk-insert match rows.  UNIQUE constraint silently skips duplicates.
    Returns (inserted, skipped).
    """
    if not rows:
        return 0, 0

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


def print_summary(conn: sqlite3.Connection) -> None:
    """Print aggregate stats after the full load."""
    total    = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    dr       = conn.execute("SELECT MIN(tourney_date), MAX(tourney_date) FROM matches").fetchone()
    players  = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT winner_name AS n FROM matches
            UNION
            SELECT DISTINCT loser_name  AS n FROM matches
        )
    """).fetchone()[0]
    by_level = conn.execute(
        "SELECT tourney_level, COUNT(*) c FROM matches "
        "GROUP BY tourney_level ORDER BY c DESC"
    ).fetchall()
    by_surf  = conn.execute(
        "SELECT surface, COUNT(*) c FROM matches "
        "GROUP BY surface ORDER BY c DESC"
    ).fetchall()

    print()
    print("=" * 42)
    print(f"  Total matches   : {total:>10,}")
    print(f"  Date range      : {dr[0]} to {dr[1]}")
    print(f"  Unique players  : {players:>10,}")
    print()
    print("  By tourney level:")
    for level, cnt in by_level:
        bar = "#" * min(30, cnt // 500)
        print(f"    {level:<8} {cnt:>7,}  {bar}")
    print()
    print("  By surface:")
    for surf, cnt in by_surf:
        pct = cnt / total * 100
        print(f"    {surf:<8} {cnt:>7,}  ({pct:.1f}%)")
    print("=" * 42)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Sackmann ITF Women tennis data into tennis.db"
    )
    parser.add_argument(
        "--years", default="2015-2026",
        help="Year range 'START-END' or single year (default: 2015-2026)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop matches + player_elo tables and reload from scratch",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Always re-download CSVs even if cached in data/raw/",
    )
    args = parser.parse_args()

    # parse year range
    if "-" in args.years:
        lo, hi = args.years.split("-", 1)
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(args.years)]

    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    if args.rebuild:
        log.warning("--rebuild: dropping matches and player_elo")
        conn.execute("DROP TABLE IF EXISTS matches")
        conn.execute("DROP TABLE IF EXISTS player_elo")
        conn.commit()

    init_db(conn)

    session = requests.Session()
    session.headers["User-Agent"] = "ObServatory-Tennis/1.0 (tennis prediction model)"

    use_cache = not args.no_cache

    print()
    print(f"  {'Year':<6}  {'CSV rows':>9}  {'Inserted':>9}  {'Skipped':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}")

    total_inserted = total_skipped = 0

    for year in years:
        text = fetch_year_csv(year, session, use_cache=use_cache)
        if text is None:
            print(f"  {year:<6}  {'N/A':>9}  {'—':>9}  {'—':>9}")
            continue

        rows          = parse_csv(text, year)
        inserted, dup = upsert_matches(conn, rows)
        total_inserted += inserted
        total_skipped  += dup

        print(f"  {year:<6}  {len(rows):>9,}  {inserted:>9,}  {dup:>9,}")
        time.sleep(0.25)  # polite GitHub rate limit

    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}")
    print(f"  {'TOTAL':<6}  {'':>9}  {total_inserted:>9,}  {total_skipped:>9,}")

    print_summary(conn)
    conn.close()

    log.info("Done. DB: %s", DB_PATH)


if __name__ == "__main__":
    main()
