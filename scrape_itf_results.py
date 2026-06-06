"""
scrape_itf_results.py -- Tennisexplorer.com ITF Women same-day results scraper.

Eliminates the Sackmann 2-4 week lag by pulling ITF Women match results
from tennisexplorer.com and inserting them into tennis.db.

Source:
    https://www.tennisexplorer.com/results/?type=wta-single&year=Y&month=M&day=D
    Returns all WTA + ITF Women singles results for one calendar day.

Data per match:
    - Winner / loser (abbreviated "Last I." expanded to Sackmann "First Last")
    - Score reconstructed from set-score cells (tiebreak notation handled)
    - Surface + tournament level from tournament page (cached per tournament)
    - Round stored as 'UNK' (round detail requires per-match request; not needed
      for Elo computation or grading)

Overlap handling with Sackmann data:
    - tourney_id prefixed 'TE-{year}-{slug}' to distinguish from Sackmann IDs
    - Pre-check SELECT before every insert prevents duplicates regardless of
      tourney_id format differences
    - collect_tennis_vps.py has a _cleanup_te_rows() function that deletes TE-
      prefixed rows once Sackmann covers the same matches

Schema change:
    Adds a nullable 'notes' column to matches if not present (safe ALTER TABLE).

Usage:
    python3 scrape_itf_results.py                   # last 7 days, full run
    python3 scrape_itf_results.py --days 1          # today only
    python3 scrape_itf_results.py --date 2026-06-04 # specific date
    python3 scrape_itf_results.py --dry-run         # parse only, no DB writes
    python3 scrape_itf_results.py --verify-only     # scrape today, show stats, no writes or Elo
    python3 scrape_itf_results.py --skip-elo        # insert but skip Elo rebuild
    python3 scrape_itf_results.py --db /path/to/tennis.db

Cron (after collect_tennis_vps.py at 05:00 UTC):
    0 6 * * * cd /home/picks && python3 scrape_itf_results.py \\
              >> /home/picks/logs/scrape_itf.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR    = Path(__file__).resolve().parent
DB_PATH = _DIR / "tennis.db"          # /home/picks/tennis.db on VPS

# ── logging ───────────────────────────────────────────────────────────────────
_log_dir = _DIR / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_dir / "scrape_itf.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
TE_BASE   = "https://www.tennisexplorer.com"
TE_RESULT = TE_BASE + "/results/?type=wta-single&year={y}&month={m}&day={d}"

REQUEST_DELAY_MIN = 1.0   # seconds between requests
REQUEST_DELAY_MAX = 2.0
MAX_REQUESTS_PER_RUN = 50  # hard cap (daily = ~20; backfill 7-day = ~35)
MATCH_WINDOW_DAYS = 10     # ±days for pre-check duplicate detection

# Prize money → ITF level mapping ($ thresholds)
_PRIZE_LEVELS: list[tuple[int, str]] = [
    (150_000, "W125"),
    (125_000, "W125"),
    (100_000, "W100"),
    (80_000,  "W80"),
    (75_000,  "W75"),
    (60_000,  "W60"),
    (50_000,  "W50"),
    (35_000,  "W35"),
    (25_000,  "W25"),
    (15_000,  "W15"),
    (10_000,  "W15"),
    (0,       "ITF"),
]

_SURFACE_MAP = {
    "clay":    "Clay",
    "hard":    "Hard",
    "grass":   "Grass",
    "carpet":  "Hard",
    "indoor":  "Hard",
    "outdoor": "Hard",
}


# ── HTTP session ──────────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    })
    return s


# Counter for rate-limiting guard
_request_count = 0

def _fetch(url: str, session: requests.Session) -> requests.Response | None:
    """Rate-limited GET. Returns None on 403/429/network error; logs warning."""
    global _request_count
    _request_count += 1
    if _request_count > MAX_REQUESTS_PER_RUN:
        log.warning("MAX_REQUESTS_PER_RUN (%d) hit — skipping %s",
                    MAX_REQUESTS_PER_RUN, url)
        return None

    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
    try:
        r = session.get(url, timeout=20)
    except requests.RequestException as exc:
        log.warning("Request error for %s: %s", url, exc)
        return None

    if r.status_code in (403, 429):
        log.warning("HTTP %d blocked on %s — skipping gracefully", r.status_code, url)
        return None
    if not r.ok:
        log.warning("HTTP %d for %s", r.status_code, url)
        return None

    return r


# ── DB setup ──────────────────────────────────────────────────────────────────
def _setup_db(conn: sqlite3.Connection) -> None:
    """Ensure schema is up to date (adds 'notes' column if missing)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    if "notes" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN notes TEXT")
        conn.commit()
        log.info("Added 'notes' column to matches table.")


# ── tournament metadata ───────────────────────────────────────────────────────
def _prize_to_level(prize: int) -> str:
    for threshold, level in _PRIZE_LEVELS:
        if prize >= threshold:
            return level
    return "ITF"


def _fetch_tourn_meta(
    slug: str,
    year: int,
    session: requests.Session,
    cache: dict,
) -> dict:
    """
    Fetch surface and prize money for a tournament.
    Parses pattern '(60,000 $, clay, women)' from the tournament page.
    Returns {'surface': str, 'level': str, 'prize': int}.
    Cached by (slug, year).
    """
    key = f"{slug}/{year}"
    if key in cache:
        return cache[key]

    url = f"{TE_BASE}/{slug}/{year}/wta-women/"
    r = _fetch(url, session)

    meta = {"surface": "Hard", "level": "ITF", "prize": 0}  # safe defaults
    if r is None:
        cache[key] = meta
        return meta

    text = r.text
    # Pattern: (15,000 $, clay, women) or (60,000 $, hard, women)
    m = re.search(
        r'\(([0-9][0-9,]*)\s*\$\s*,\s*(clay|hard|grass|carpet|indoor|outdoor)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            prize = int(m.group(1).replace(",", ""))
        except ValueError:
            prize = 0
        surf_raw = m.group(2).lower()
        surface  = _SURFACE_MAP.get(surf_raw, "Hard")
        meta = {"surface": surface, "level": _prize_to_level(prize), "prize": prize}

    cache[key] = meta
    log.debug("Tournament %s: surface=%s level=%s prize=%d",
              slug, meta["surface"], meta["level"], meta["prize"])
    return meta


# ── score reconstruction ──────────────────────────────────────────────────────
def _extract_set_scores(cells: list, start_idx: int) -> list[str]:
    """
    Extract set score strings from cells starting at start_idx.
    Only includes cells containing pure integers in valid set-score range:
      0-7  (normal game count)
      60-79 (tiebreak-loss notation: e.g. 65 = won 6 games, 5 TB points → lost TB)
    Stops on 'info', decimal values (odds), or out-of-range integers.
    """
    scores: list[str] = []
    for cell in cells[start_idx:]:
        text = cell.get_text(strip=True)
        if not text or text.lower() == "info":
            break
        if "." in text:   # odds like "1.17" — stop
            break
        try:
            val = int(text)
        except ValueError:
            break
        if (0 <= val <= 7) or (60 <= val <= 79):
            scores.append(text)
        else:
            break   # unexpected value, stop
    return scores


def _reconstruct_score(winner_sets: list[str], loser_sets: list[str]) -> str | None:
    """
    Build Sackmann-format score string from paired set score lists.

    Tennisexplorer tiebreak-loss notation:
      Player who LOST a tiebreak shows "XY": X = games played (6), Y = their TB score
      e.g. "65" → player won 6 games + 5 tiebreak points → lost set 6-7(5)
      The other player (who won the TB) shows "7" normally.

    Examples:
      winner=[65,6,6], loser=[7,1,1]  →  "6-7(5) 6-1 6-1"
      winner=[6,6],    loser=[2,3]    →  "6-2 6-3"
      winner=[7,6],    loser=[63,4]   →  "7-6(3) 6-4"
    """
    if not winner_sets or not loser_sets:
        return None
    parts: list[str] = []
    for ws, ls in zip(winner_sets, loser_sets):
        try:
            wi = int(ws)
            li = int(ls)
        except ValueError:
            continue

        if wi >= 10:
            # Winner lost this set by tiebreak: "65" = 6-7(5)
            w_games = wi // 10
            w_tb    = wi % 10
            l_games = li          # should be 7
            parts.append(f"{w_games}-{l_games}({w_tb})")
        elif li >= 10:
            # Loser lost this set by tiebreak: "65" = loser's 6 games + 5 TB points
            l_games = li // 10
            l_tb    = li % 10
            w_games = wi          # should be 7
            parts.append(f"{w_games}-{l_games}({l_tb})")
        else:
            parts.append(f"{wi}-{li}")

    return " ".join(parts) if parts else None


# ── player name normalisation ─────────────────────────────────────────────────
def _strip_seed(raw: str) -> str:
    """
    Remove seed / status annotations from player name cell.
      "Brace C.(2)"  → "Brace C."
      "Smith J.(WC)" → "Smith J."
      "Doe A.(Q)"    → "Doe A."
    """
    return re.sub(r'\s*\([^\)]*\)\s*$', '', raw).strip()


def _parse_abbrev(name: str) -> tuple[str, str] | None:
    """
    Parse "Last I." or "Last Name I." (multi-word last names).
    Returns (last_name, initial) or None if format unrecognised.
    """
    m = re.match(r'^(.+)\s+([A-Z])\.?$', name)
    if m:
        return m.group(1).strip(), m.group(2)
    return None


def _resolve_name(
    abbrev: str,
    conn: sqlite3.Connection,
    cache: dict,
) -> tuple[str, bool]:
    """
    Expand abbreviated "Brace C." → "Cadence Brace" using Sackmann data in DB.

    Strategy:
      1. Strip seed (already done by caller)
      2. Parse Last + Initial
      3. Query matches table: WHERE winner_name LIKE 'C%' AND winner_name LIKE '% Brace'
      4. If exactly one candidate → resolved
      5. If multiple → try stricter filter (starts with 'Initial ', ends with ' Last')
      6. If still ambiguous or not found → return original, flag unresolved

    Returns (resolved_name, is_resolved).
    Do NOT fabricate names.
    """
    if abbrev in cache:
        cached = cache[abbrev]
        return (cached, True) if cached else (abbrev, False)

    parsed = _parse_abbrev(abbrev)
    if not parsed:
        cache[abbrev] = None
        return abbrev, False

    last, initial = parsed

    rows = conn.execute(
        """
        SELECT DISTINCT winner_name FROM matches
        WHERE  winner_name LIKE ? AND winner_name LIKE ?
        UNION
        SELECT DISTINCT loser_name FROM matches
        WHERE  loser_name LIKE ? AND loser_name LIKE ?
        LIMIT  20
        """,
        (f"{initial}%", f"% {last}",
         f"{initial}%", f"% {last}"),
    ).fetchall()

    candidates = [r[0] for r in rows]

    if len(candidates) == 1:
        cache[abbrev] = candidates[0]
        return candidates[0], True

    if len(candidates) > 1:
        # Stricter: must start with 'Initial ' and end with ' Last'
        strict = [
            c for c in candidates
            if c.upper().startswith(f"{initial.upper()} ") or
               (c[0].upper() == initial.upper() and
                c.upper().endswith(f" {last.upper()}"))
        ]
        if len(strict) == 1:
            cache[abbrev] = strict[0]
            return strict[0], True
        # Still ambiguous — don't guess
        log.debug("Ambiguous name '%s': candidates=%s", abbrev, candidates[:5])

    # Not found or ambiguous
    cache[abbrev] = None
    return abbrev, False


# ── duplicate detection ───────────────────────────────────────────────────────
def _match_exists(conn: sqlite3.Connection, winner: str, loser: str, date_str: str) -> bool:
    """
    Return True if (winner, loser) already exists in the matches table within
    ±MATCH_WINDOW_DAYS of date_str (covers full tournament week).
    """
    row = conn.execute(
        """
        SELECT 1 FROM matches
        WHERE  winner_name = ? AND loser_name = ?
          AND  tourney_date BETWEEN date(?, ?) AND date(?, ?)
        LIMIT  1
        """,
        (winner, loser,
         date_str, f"-{MATCH_WINDOW_DAYS} days",
         date_str, f"+{MATCH_WINDOW_DAYS} days"),
    ).fetchone()
    return row is not None


# ── match parsing ─────────────────────────────────────────────────────────────
def _parse_match_pair(
    winner_row,
    loser_row,
    tourn_name: str,
    slug: str,
) -> dict | None:
    """
    Parse a (winner_row, loser_row) HTML pair into a match dict.

    Winner row (has 'bott' class): time | name | sets_won | set1 | set2 | ... | odds | info
    Loser  row (no  'bott' class): name | sets_won | set1 | set2 | ...

    Returns None for walkovers (no set scores) or malformed rows.
    """
    w_cells = winner_row.find_all(["td", "th"])
    l_cells = loser_row.find_all(["td", "th"])

    if len(w_cells) < 3 or len(l_cells) < 2:
        return None

    # Winner row: cell[0]=time, cell[1]=name, cell[2]=sets_won, cell[3+]=set scores
    w_name_raw  = w_cells[1].get_text(strip=True)
    w_sets_won  = w_cells[2].get_text(strip=True)
    w_set_cells = _extract_set_scores(w_cells, 3)

    # Loser row: cell[0]=name, cell[1]=sets_won, cell[2+]=set scores
    l_name_raw  = l_cells[0].get_text(strip=True)
    l_sets_won  = l_cells[1].get_text(strip=True)
    l_set_cells = _extract_set_scores(l_cells, 2)

    # Skip walkovers: no set scores at all
    if not w_set_cells and not l_set_cells:
        return None

    # Parse sets won
    try:
        w_won = int(w_sets_won)
        l_won = int(l_sets_won)
    except ValueError:
        return None

    # Skip if loser "won" more sets than winner (malformed row or wrong pairing)
    if l_won > w_won:
        return None

    # Reconstruct score
    score = _reconstruct_score(w_set_cells, l_set_cells)
    # Append RET if this looks like a retirement (completed sets < 2 for normal match)
    if w_won < 2 and score:
        score = score + " RET"
    elif w_won < 2 and not score:
        score = "RET"

    # Strip seeds
    w_abbrev = _strip_seed(w_name_raw)
    l_abbrev = _strip_seed(l_name_raw)

    if not w_abbrev or not l_abbrev:
        return None

    return {
        "winner_abbrev": w_abbrev,
        "loser_abbrev":  l_abbrev,
        "score":         score,
        "tourn_name":    tourn_name,
        "slug":          slug,
    }


# ── daily scrape ──────────────────────────────────────────────────────────────
def scrape_day(
    date_str: str,
    session: requests.Session,
    conn: sqlite3.Connection,
    meta_cache: dict,
    name_cache: dict,
    dry_run: bool,
) -> dict:
    """
    Scrape ITF Women results for one date.
    Returns summary dict with stats and sample matches.
    """
    y, m, d = date_str.split("-")
    url = TE_RESULT.format(y=y, m=int(m), d=int(d))
    log.info("Scraping %s -> %s", date_str, url)

    r = _fetch(url, session)
    if r is None:
        return {"date": date_str, "found": 0, "inserted": 0,
                "skipped": 0, "unresolved": 0, "samples": [],
                "status": "fetch_failed"}

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table", class_="result")
    if not tables:
        log.warning("No result table on %s", date_str)
        return {"date": date_str, "found": 0, "inserted": 0,
                "skipped": 0, "unresolved": 0, "samples": [],
                "status": "no_table"}

    main_table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows_html   = main_table.find_all("tr")

    # ── parse rows: tournament headers + match pairs ──────────────────────────
    raw_matches: list[dict] = []
    cur_name: str | None = None
    cur_slug: str | None = None
    i = 0

    while i < len(rows_html):
        row  = rows_html[i]
        cls  = row.get("class", [])

        if "head" in cls:
            link = row.find("a")
            href = link.get("href", "") if link else ""
            # Only ITF Women events (wta-women in href)
            if "wta-women" in href:
                # Tournament name: strip trailing " S 1 2 ..." header columns
                cells = row.find_all(["td", "th"])
                raw_name = cells[0].get_text(strip=True) if cells else ""
                # Extract slug from href: "/sumter-itf/2026/wta-women/" → "sumter-itf"
                parts = href.strip("/").split("/")
                slug  = parts[0] if parts else ""
                # Filter to ITF-only (skip French Open, Wimbledon, etc.)
                is_itf = ("itf" in slug.lower() or "itf" in raw_name.lower())
                if is_itf:
                    cur_name = raw_name
                    cur_slug = slug
                else:
                    cur_name = None
                    cur_slug = None
            else:
                cur_name = None
                cur_slug = None
            i += 1
            continue

        # Match pairs: winner row has 'bott' class
        if cur_name and ("bott" in cls) and ("fRow" in cls or "one" in cls or "two" in cls):
            # Next row should be the loser (no 'bott')
            if i + 1 < len(rows_html):
                next_row  = rows_html[i + 1]
                next_cls  = next_row.get("class", [])
                if "bott" not in next_cls and "head" not in next_cls:
                    parsed = _parse_match_pair(row, next_row, cur_name, cur_slug)
                    if parsed:
                        raw_matches.append(parsed)
                    i += 2
                    continue
        i += 1

    # ── fetch tournament metadata for unique slugs ────────────────────────────
    unique_slugs = {m["slug"] for m in raw_matches if m.get("slug")}
    for slug in sorted(unique_slugs):
        if f"{slug}/{y}" not in meta_cache:
            _fetch_tourn_meta(slug, int(y), session, meta_cache)

    # ── resolve names, check duplicates, build rows ───────────────────────────
    inserted   = 0
    skipped    = 0
    unresolved = 0
    samples: list[dict] = []

    for match in raw_matches:
        slug = match["slug"]
        meta = meta_cache.get(f"{slug}/{y}", {"surface": "Hard", "level": "ITF", "prize": 0})

        w_full, w_ok = _resolve_name(match["winner_abbrev"], conn, name_cache)
        l_full, l_ok = _resolve_name(match["loser_abbrev"],  conn, name_cache)

        if not w_ok:
            unresolved += 1
            log.debug("Unresolved winner: '%s'", match["winner_abbrev"])
        if not l_ok:
            unresolved += 1
            log.debug("Unresolved loser:  '%s'", match["loser_abbrev"])

        # Pre-check: already in DB from Sackmann or previous scrape?
        if _match_exists(conn, w_full, l_full, date_str):
            skipped += 1
            continue

        notes = None if (w_ok and l_ok) else "unresolved_name"

        row_data = {
            "tourney_id":    f"TE-{y}-{slug}",
            "tourney_name":  match["tourn_name"],
            "tourney_date":  date_str,
            "tourney_level": meta["level"],
            "surface":       meta["surface"],
            "round":         "UNK",
            "winner_name":   w_full,
            "loser_name":    l_full,
            "winner_id":     None,
            "loser_id":      None,
            "winner_rank":   None,
            "loser_rank":    None,
            "score":         match["score"],
            "minutes":       None,
            "data_year":     int(y),
            "notes":         notes,
        }

        if not dry_run:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO matches
                        (tourney_id, tourney_name, tourney_date, tourney_level,
                         surface, round, winner_name, loser_name,
                         winner_id, loser_id, winner_rank, loser_rank,
                         score, minutes, data_year, notes)
                    VALUES
                        (:tourney_id, :tourney_name, :tourney_date, :tourney_level,
                         :surface, :round, :winner_name, :loser_name,
                         :winner_id, :loser_id, :winner_rank, :loser_rank,
                         :score, :minutes, :data_year, :notes)
                    """,
                    row_data,
                )
            except sqlite3.Error as exc:
                log.error("Insert error for %s vs %s: %s", w_full, l_full, exc)
                continue

        inserted += 1

        if len(samples) < 5:
            samples.append({
                "winner":        w_full,
                "loser":         l_full,
                "score":         match["score"],
                "surface":       meta["surface"],
                "level":         meta["level"],
                "tournament":    match["tourn_name"],
                "date":          date_str,
                "w_resolved":    w_ok,
                "l_resolved":    l_ok,
            })

    if not dry_run:
        conn.commit()

    log.info(
        "  %s: found=%d  inserted=%d  skipped=%d  unresolved_names=%d  requests_so_far=%d",
        date_str, len(raw_matches), inserted, skipped, unresolved, _request_count,
    )

    return {
        "date":       date_str,
        "found":      len(raw_matches),
        "inserted":   inserted,
        "skipped":    skipped,
        "unresolved": unresolved,
        "samples":    samples,
        "status":     "ok",
    }


# ── cleanup: remove TE- rows superseded by Sackmann ──────────────────────────
def cleanup_te_rows(conn: sqlite3.Connection) -> int:
    """
    Delete TE- prefixed rows where an equivalent Sackmann row now exists
    (same winner+loser within ±12 days, non-TE tourney_id).
    Called by collect_tennis_vps.py after Sackmann data is loaded.
    Returns count of rows deleted.
    """
    result = conn.execute(
        """
        DELETE FROM matches
        WHERE  tourney_id LIKE 'TE-%'
          AND  EXISTS (
                   SELECT 1 FROM matches m2
                   WHERE  m2.winner_name = matches.winner_name
                     AND  m2.loser_name  = matches.loser_name
                     AND  m2.tourney_id  NOT LIKE 'TE-%'
                     AND  ABS(julianday(m2.tourney_date) -
                              julianday(matches.tourney_date)) <= 12
               )
        """
    )
    conn.commit()
    return result.rowcount


# ── Elo rebuild ───────────────────────────────────────────────────────────────
def rebuild_elo(conn: sqlite3.Connection) -> None:
    """Full Elo rebuild. ~3-5s for 273K+ matches; simpler than incremental."""
    try:
        from elo_engine import build_ratings
    except ImportError:
        log.error("Cannot import elo_engine — Elo rebuild skipped. "
                  "Ensure elo_engine.py is in the same directory.")
        return

    t0 = time.time()
    log.info("Rebuilding Elo ratings (full pass)...")
    _, state = build_ratings(conn)
    elapsed = time.time() - t0
    n_rows  = conn.execute("SELECT COUNT(*) FROM player_elo").fetchone()[0]
    log.info("Elo rebuild complete: %d players, %d rows, %.1fs",
             len(state), n_rows, elapsed)


# ── verification display ──────────────────────────────────────────────────────
def _print_verification(summaries: list[dict]) -> None:
    """Print a structured verification report for one or more scraped days."""
    sep = "=" * 65

    total_found    = sum(s["found"]      for s in summaries)
    total_inserted = sum(s["inserted"]   for s in summaries)
    total_skipped  = sum(s["skipped"]    for s in summaries)
    total_unres    = sum(s["unresolved"] for s in summaries)

    print()
    print(sep)
    print("  SCRAPE VERIFICATION REPORT")
    print(sep)
    print(f"  {'Date':<12} {'Found':>6} {'Inserted':>9} {'Skipped':>8} "
          f"{'Unresolved':>11} {'Status'}")
    print(f"  {'-'*58}")
    for s in summaries:
        print(f"  {s['date']:<12} {s['found']:>6} {s['inserted']:>9} "
              f"{s['skipped']:>8} {s['unresolved']:>11} {s.get('status','')}")

    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<12} {total_found:>6} {total_inserted:>9} "
          f"{total_skipped:>8} {total_unres:>11}")

    # Sample matches
    samples = [samp for s in summaries for samp in s.get("samples", [])][:5]
    if samples:
        print()
        print("  SAMPLE MATCHES")
        print(f"  {'-'*58}")
        for samp in samples:
            w_flag = "" if samp["w_resolved"] else " [UNRESOLVED]"
            l_flag = "" if samp["l_resolved"] else " [UNRESOLVED]"
            print(f"  {samp['date']} | {samp['level']} | {samp['surface']}")
            print(f"    {samp['tournament']}")
            print(f"    W: {samp['winner']}{w_flag}")
            print(f"    L: {samp['loser']}{l_flag}")
            print(f"    Score: {samp['score'] or 'n/a'}")
            print()

    # Name resolution rate
    total_names   = 2 * total_inserted + 2 * total_skipped
    pct_resolved  = 100 * (1 - total_unres / max(total_names, 1))
    print(f"  Name resolution rate: {pct_resolved:.0f}% "
          f"({total_unres} unresolved -- will be stored as-is with notes='unresolved_name')")
    print(f"  HTTP requests used: {_request_count} / {MAX_REQUESTS_PER_RUN}")
    print(sep)
    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    global _request_count
    _request_count = 0

    parser = argparse.ArgumentParser(
        description="Scrape ITF Women match results from tennisexplorer.com"
    )
    parser.add_argument("--days",        type=int, default=7,
                        help="Number of past days to scrape (default: 7)")
    parser.add_argument("--date",        default=None, metavar="YYYY-MM-DD",
                        help="Scrape a single specific date")
    parser.add_argument("--db",          default=None, metavar="PATH",
                        help="Path to tennis.db (default: ./tennis.db)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Parse and show stats; no DB writes, no Elo rebuild")
    parser.add_argument("--verify-only", action="store_true",
                        help="Scrape today only, show verification, no writes")
    parser.add_argument("--skip-elo",    action="store_true",
                        help="Insert matches but skip Elo rebuild")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        log.error("tennis.db not found at %s", db_path)
        return 1

    # Build date list
    if args.verify_only:
        dates = [date.today().isoformat()]
        args.dry_run = True   # verify-only implies dry-run
    elif args.date:
        dates = [args.date]
    else:
        today = date.today()
        dates = [(today - timedelta(days=i)).isoformat()
                 for i in range(args.days - 1, -1, -1)]

    log.info("=== scrape_itf_results.py starting ===")
    log.info("Dates: %s  dry_run=%s  verify_only=%s",
             dates, args.dry_run, args.verify_only)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _setup_db(conn)

    before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    session    = _make_session()
    meta_cache: dict = {}
    name_cache: dict = {}
    summaries:  list[dict] = []

    for d_str in dates:
        if _request_count >= MAX_REQUESTS_PER_RUN:
            log.warning("Request cap reached — stopping early at date %s", d_str)
            break
        summary = scrape_day(d_str, session, conn, meta_cache, name_cache, args.dry_run)
        summaries.append(summary)

    # Verification display (always shown)
    _print_verification(summaries)

    if args.dry_run:
        log.info("DRY RUN -- no DB writes performed.")
        conn.close()
        return 0

    after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    new_total = after - before
    log.info("DB matches: before=%d  after=%d  net_new=%d", before, after, new_total)

    # Elo rebuild if any new matches were inserted
    if new_total > 0 and not args.skip_elo:
        rebuild_elo(conn)
    elif new_total == 0:
        log.info("No new matches inserted — Elo ratings already current.")
    else:
        log.info("--skip-elo set — Elo rebuild deferred.")

    conn.close()
    log.info("=== scrape_itf_results.py done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
