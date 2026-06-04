"""
elo_engine.py -- Point-in-time Elo ratings for ITF Women's tennis.

Usage:
    python elo_engine.py            # build ratings + full calibration report
    python elo_engine.py --rebuild  # drop player_elo, recompute, then report
    python elo_engine.py --report-only  # skip rebuild, show report from existing ratings

Importable API (used by predict_tennis.py and backtest_tennis.py):
    from elo_engine import get_player_elo, elo_win_prob, SURFACE_KEY

Processing order guarantee:
    Matches are processed tourney_date ASC, id ASC.
    Since collect_tennis.py loads each CSV in natural Sackmann order (which
    sequences rounds R128->R64->R32->R16->QF->SF->F within each tournament),
    the `id` column already encodes the correct intra-tournament ordering.
    No lookahead: a player's rating at row N is built only from rows 0..N-1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "data" / "tennis.db"

K_BASE          = 32      # K-factor for established players
K_EARLY         = 40      # K-factor for early career (< EARLY_THRESHOLD matches)
EARLY_THRESHOLD = 30      # career match count threshold for K_EARLY
SCALE           = 400.0   # Elo scale (same as chess; standard for tennis)
START_ELO       = 1500.0  # default rating for unseen players

# Surfaces we track separately; anything else folds into overall only
SURFACE_KEY = {"Clay": "clay", "Hard": "hard", "Grass": "grass"}
SURFACES    = list(SURFACE_KEY.keys())

CALIB_START = "2020-01-01"
CALIB_END   = "2025-12-31"

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Elo math ──────────────────────────────────────────────────────────────────
def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """P(A wins) given Elo ratings. Standard formula: 1/(1+10^(-(a-b)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / SCALE))


def _k(matches_played: int) -> float:
    return K_EARLY if matches_played < EARLY_THRESHOLD else K_BASE


# ── player state (in-memory during build) ─────────────────────────────────────
class _PlayerState:
    """Mutable Elo state for one player, updated match-by-match."""
    __slots__ = ("overall", "clay", "hard", "grass",
                 "n", "n_clay", "n_hard", "n_grass")

    def __init__(self):
        self.overall = START_ELO
        self.clay    = START_ELO
        self.hard    = START_ELO
        self.grass   = START_ELO
        self.n       = 0   # career match count
        self.n_clay  = 0
        self.n_hard  = 0
        self.n_grass = 0

    def surface_elo(self, surface: str) -> float:
        if surface == "Clay":  return self.clay
        if surface == "Hard":  return self.hard
        if surface == "Grass": return self.grass
        return self.overall

    def surface_n(self, surface: str) -> int:
        if surface == "Clay":  return self.n_clay
        if surface == "Hard":  return self.n_hard
        if surface == "Grass": return self.n_grass
        return self.n

    def apply_result(self, surface: str, won: bool) -> None:
        """Update overall + surface Elo after one match side."""
        # overall already updated by caller; increment counters here
        if surface == "Clay":
            self.n_clay += 1
        elif surface == "Hard":
            self.n_hard += 1
        elif surface == "Grass":
            self.n_grass += 1
        self.n += 1


# ── importable lookup (for predict_tennis.py) ─────────────────────────────────
def get_player_elo(
    conn: sqlite3.Connection,
    player_name: str,
    before_date: str,
) -> dict:
    """
    Return the most recent Elo ratings for `player_name` strictly before
    `before_date` (ISO string 'YYYY-MM-DD').

    Uses `id DESC` as a tiebreaker within the same date so that when multiple
    matches are stored with the same tourney_date, the most-recently-processed
    one (latest round) is returned.

    Returns a dict with keys:
        overall_elo, clay_elo, hard_elo, grass_elo,
        matches_played, clay_matches, hard_matches, grass_matches
    Falls back to starter defaults (1500) if no data found.
    """
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
        "overall_elo":    START_ELO,
        "clay_elo":       START_ELO,
        "hard_elo":       START_ELO,
        "grass_elo":      START_ELO,
        "matches_played": 0,
        "clay_matches":   0,
        "hard_matches":   0,
        "grass_matches":  0,
    }


# ── main build ────────────────────────────────────────────────────────────────
def build_ratings(conn: sqlite3.Connection) -> list[dict]:
    """
    Process every match in the DB chronologically; write player_elo table.

    Returns list of calibration dicts (one per match in CALIB_START..CALIB_END)
    containing pre-match Elos so the caller can run calibration analysis without
    hitting the DB again.
    """
    log.info("Loading matches from DB...")
    rows = conn.execute(
        """
        SELECT id, tourney_date, tourney_level, surface,
               round, winner_name, loser_name, score
        FROM   matches
        ORDER  BY tourney_date ASC, id ASC
        """
    ).fetchall()
    log.info("Loaded %d matches — building Elo ratings...", len(rows))

    state: dict[str, _PlayerState] = {}
    elo_rows: list[tuple]    = []   # to bulk-insert into player_elo
    calib_data: list[dict]   = []   # pre-match snapshots for calibration

    skipped_nonmatch = 0
    for i, (mid, date, level, surface, rnd, wname, lname, score) in enumerate(rows):

        # Skip non-results: walkovers / defaults / retirements do not reflect
        # on-court ability and pollute Elo. (W/O, DEF kept out of `matches` going
        # forward by collect_tennis.py; RET stays in `matches` for grading but is
        # excluded here.) These never increment ratings or match counts.
        su = (score or "").upper()
        if "W/O" in su or "DEF" in su or "RET" in su:
            skipped_nonmatch += 1
            continue

        # Ensure surface is one of our three canonical values
        if surface not in SURFACE_KEY:
            surface = "Hard"   # fallback for anything non-standard

        # Initialise player states on first encounter
        if wname not in state: state[wname] = _PlayerState()
        if lname not in state: state[lname] = _PlayerState()
        ws = state[wname]
        ls = state[lname]

        # ── capture PRE-MATCH ratings ─────────────────────────────────────────
        pre_w_ov = ws.overall
        pre_l_ov = ls.overall
        pre_w_sf = ws.surface_elo(surface)
        pre_l_sf = ls.surface_elo(surface)

        # Collect calibration data (2020-2025 window)
        in_calib = CALIB_START <= date <= CALIB_END
        if in_calib:
            calib_data.append({
                "date":         date,
                "surface":      surface,
                "elo_diff":     pre_w_ov - pre_l_ov,    # winner − loser
                "surface_diff": pre_w_sf - pre_l_sf,
            })

        # ── update OVERALL Elo ────────────────────────────────────────────────
        # SYMMETRIC K: one K per match (same for winner and loser) so the points
        # the winner gains exactly equal the points the loser sheds. With per-side
        # K (the old code) a K=40 newcomer beating a K=32 veteran minted +8*(1-E)
        # net points every upset, inflating the pool mean (~1543 in 2015 -> ~1767
        # in 2026). A single K conserves the pool sum, so the mean stays ~1500 and
        # the 1500 default for unseen players is once again a neutral prior.
        # K is set by the LESS-experienced player so provisional ratings still
        # adapt quickly, applied equally to both sides.
        K_ov  = _k(min(ws.n, ls.n))
        E_w   = elo_win_prob(pre_w_ov, pre_l_ov)
        delta = K_ov * (1.0 - E_w)
        ws.overall = pre_w_ov + delta
        ls.overall = pre_l_ov - delta

        # ── update SURFACE Elo (same symmetric treatment) ─────────────────────
        K_sf     = _k(min(ws.surface_n(surface), ls.surface_n(surface)))
        E_ws     = elo_win_prob(pre_w_sf, pre_l_sf)
        delta_sf = K_sf * (1.0 - E_ws)
        if surface == "Clay":
            ws.clay  = pre_w_sf + delta_sf
            ls.clay  = pre_l_sf - delta_sf
            ws.n_clay += 1;  ls.n_clay += 1
        elif surface == "Hard":
            ws.hard  = pre_w_sf + delta_sf
            ls.hard  = pre_l_sf - delta_sf
            ws.n_hard += 1;  ls.n_hard += 1
        elif surface == "Grass":
            ws.grass = pre_w_sf + delta_sf
            ls.grass = pre_l_sf - delta_sf
            ws.n_grass += 1; ls.n_grass += 1

        ws.n += 1
        ls.n += 1

        # ── accumulate post-match rows for DB write ───────────────────────────
        for st, nm in ((ws, wname), (ls, lname)):
            elo_rows.append((
                nm, date,
                st.overall, st.clay, st.hard, st.grass,
                st.n, st.n_clay, st.n_hard, st.n_grass,
            ))

        if (i + 1) % 50_000 == 0:
            log.info("  Processed %d / %d matches...", i + 1, len(rows))

    log.info("Skipped %d non-result rows (W/O, DEF, RET) for Elo.", skipped_nonmatch)

    # ── bulk-write player_elo ─────────────────────────────────────────────────
    log.info("Writing %d rows to player_elo...", len(elo_rows))
    conn.execute("DELETE FROM player_elo")
    conn.executemany(
        """
        INSERT INTO player_elo
            (player_name, match_date,
             overall_elo, clay_elo, hard_elo, grass_elo,
             matches_played, clay_matches, hard_matches, grass_matches)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        elo_rows,
    )
    conn.commit()
    log.info("player_elo written: %d rows for %d unique players.",
             len(elo_rows), len(state))

    return calib_data, state


# ── calibration helpers ───────────────────────────────────────────────────────
def _bin_label(p_fav: float) -> str:
    """Return bucket label for the favorite's predicted win probability."""
    lo = int(p_fav * 100)
    lo = max(50, min(lo, 94))   # clamp to [50, 94]
    lo = (lo // 5) * 5          # round down to nearest 5
    hi = lo + 5
    return f"{lo}-{hi}%"


_BIN_ORDER = [
    "50-55%", "55-60%", "60-65%", "65-70%",
    "70-75%", "75-80%", "80-85%", "85-90%",
    "90-95%", "95-100%",
]


def run_calibration(calib_data: list[dict]) -> None:
    """
    Full calibration report from pre-match Elo snapshots.

    For each match, we know:
      - elo_diff = winner_elo - loser_elo  (positive = winner was favourite)
      - outcome  = 1 always (winner won, by definition of the data label)

    Calibration framing: "Did the player Elo predicted as FAVOURITE actually win?"
      - P_fav = max(P(winner wins), P(loser wins)) >= 0.5
      - fav_won = (elo_diff > 0)   → 1 if favourite won, 0 if underdog won
    Bin by P_fav and compare to actual favourite win rate.
    This answers: "When Elo said 65%, did the favourite win 65% of the time?"

    Brier score & log-loss are computed from the raw P(winner wins),
    where the actual outcome is always 1.  Lower = better.
    """
    if not calib_data:
        print("No calibration data.")
        return

    n = len(calib_data)

    # ── per-record metrics ────────────────────────────────────────────────────
    p_winner_list   = []   # P(winner wins) using overall Elo
    p_winner_s_list = []   # P(winner wins) using surface Elo
    bins: dict[str, list] = {b: [] for b in _BIN_ORDER}
    surf_stats: dict[str, dict] = {s: {"correct": 0, "total": 0,
                                       "brier": 0.0} for s in SURFACES}

    for d in calib_data:
        diff    = d["elo_diff"]
        p_w     = elo_win_prob(0, -diff)   # equiv: 1/(1+10^(-diff/400))
        p_w_s   = elo_win_prob(0, -d["surface_diff"])
        p_fav   = p_w if diff >= 0 else 1.0 - p_w
        fav_won = 1 if diff > 0 else (0 if diff < 0 else None)

        p_winner_list.append(p_w)
        p_winner_s_list.append(p_w_s)

        # calibration bin
        label = _bin_label(p_fav)
        if fav_won is not None:
            bins[label].append(fav_won)

        # per-surface
        surf = d["surface"]
        if surf in surf_stats:
            surf_stats[surf]["total"]   += 1
            surf_stats[surf]["brier"]   += (p_w - 1.0) ** 2
            if diff > 0:
                surf_stats[surf]["correct"] += 1

    # ── aggregate metrics ─────────────────────────────────────────────────────
    brier   = sum((p - 1.0) ** 2 for p in p_winner_list) / n
    logloss = -sum(math.log(max(p, 1e-7)) for p in p_winner_list) / n
    acc     = sum(1 for p in p_winner_list if p > 0.5) / n

    brier_s   = sum((p - 1.0) ** 2 for p in p_winner_s_list) / n
    logloss_s = -sum(math.log(max(p, 1e-7)) for p in p_winner_s_list) / n
    acc_s     = sum(1 for p in p_winner_s_list if p > 0.5) / n

    # ── print ─────────────────────────────────────────────────────────────────
    sep = "-" * 60

    print()
    print("=" * 60)
    print(f"  CALIBRATION REPORT -- ITF Women Elo")
    print(f"  Window: {CALIB_START[:4]}-{CALIB_END[:4]}    Matches: {n:,}")
    print("=" * 60)

    print()
    print("CALIBRATION CURVE (overall Elo)")
    print(f"  {'Pred%':<10} {'Actual%':>8} {'N':>7}   {'Gap':>6}   Interpretation")
    print(f"  {sep}")

    for label in _BIN_ORDER:
        outcomes = bins[label]
        if not outcomes:
            continue
        actual = sum(outcomes) / len(outcomes)
        lo     = int(label.split("-")[0])
        mid    = lo + 2.5
        gap    = actual - mid / 100
        interp = ("good" if abs(gap) <= 0.04
                  else ("overconfident" if gap < -0.04 else "underconfident"))
        print(f"  {label:<10} {actual*100:>7.1f}% {len(outcomes):>7,}   "
              f"{gap*100:>+5.1f}%   {interp}")

    print()
    print("OVERALL METRICS")
    print(f"  {'Metric':<22} {'Overall Elo':>12}  {'Surface Elo':>12}")
    print(f"  {sep}")
    print(f"  {'Accuracy (fav wins)':<22} {acc*100:>11.1f}%  {acc_s*100:>11.1f}%")
    print(f"  {'Brier Score':<22} {brier:>12.4f}  {brier_s:>12.4f}")
    print(f"  {'Log-Loss':<22} {logloss:>12.4f}  {logloss_s:>12.4f}")
    print()
    print(f"  Baseline (always 50%): Brier=0.2500  Log-Loss=0.6931  Acc=50.0%")
    print(f"  Good Elo model target: Brier<0.23    Log-Loss<0.65    Acc>65%")

    print()
    print("ACCURACY BY SURFACE")
    print(f"  {'Surface':<8} {'Matches':>8} {'Fav Acc':>9} {'Brier':>8}")
    print(f"  {sep}")
    for surf in SURFACES:
        ss = surf_stats[surf]
        if ss["total"] == 0:
            continue
        s_acc   = ss["correct"] / ss["total"]
        s_brier = ss["brier"]   / ss["total"]
        print(f"  {surf:<8} {ss['total']:>8,} {s_acc*100:>8.1f}%  {s_brier:>8.4f}")

    print()
    verdict = ("PREDICTIVE" if acc >= 0.60 and brier < 0.24
               else ("MARGINAL" if acc >= 0.55 else "WEAK"))
    verdict_note = {
        "PREDICTIVE": "Elo is well-calibrated. Proceed to backtest.",
        "MARGINAL":   "Elo has modest predictive power. Edge detection may work with tight thresholds.",
        "WEAK":       "Elo is barely better than random. Investigate before proceeding.",
    }[verdict]
    print(f"  VERDICT: [{verdict}]  {verdict_note}")
    print("=" * 60)


def print_matches_per_year(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT data_year, COUNT(*) FROM matches GROUP BY data_year ORDER BY data_year"
    ).fetchall()
    print()
    print("MATCHES PROCESSED PER YEAR")
    print(f"  {'Year':<6} {'Matches':>9}  {'Elo rows':>9}")
    print(f"  {'-'*30}")
    for year, cnt in rows:
        elo_cnt = conn.execute(
            "SELECT COUNT(*) FROM player_elo WHERE match_date LIKE ?",
            (f"{year}%",)
        ).fetchone()[0]
        print(f"  {year:<6} {cnt:>9,}  {elo_cnt:>9,}")
    total_m = sum(r[1] for r in rows)
    total_e = conn.execute("SELECT COUNT(*) FROM player_elo").fetchone()[0]
    print(f"  {'TOTAL':<6} {total_m:>9,}  {total_e:>9,}")


def print_top_players(state: dict, top_n: int = 20) -> None:
    """Print top N players by current (final) overall Elo."""
    if not state:
        return
    ranked = sorted(state.items(), key=lambda kv: kv[1].overall, reverse=True)[:top_n]

    print()
    print(f"TOP {top_n} PLAYERS BY CURRENT ELO (all-time ratings)")
    print(f"  {'Rank':<5} {'Player':<30} {'Overall':>8} {'Clay':>8} "
          f"{'Hard':>8} {'Grass':>8} {'Matches':>8}")
    print(f"  {'-'*80}")
    for rank, (name, s) in enumerate(ranked, 1):
        print(f"  {rank:<5} {name:<30} {s.overall:>8.1f} {s.clay:>8.1f} "
              f"{s.hard:>8.1f} {s.grass:>8.1f} {s.n:>8,}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Elo ratings and show calibration report"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop player_elo table before rebuilding",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip rating rebuild; just show report from existing player_elo",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache

    if args.report_only:
        # Re-derive calibration data from the DB (slower but avoids rebuild)
        log.info("--report-only: re-deriving calibration data from player_elo...")
        calib_data, state = _derive_calib_from_db(conn)
    else:
        if args.rebuild:
            log.warning("--rebuild: dropping player_elo")
            conn.execute("DROP TABLE IF EXISTS player_elo")
            conn.executescript(
                """
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
                CREATE INDEX IF NOT EXISTS idx_elo_lookup
                    ON player_elo (player_name, match_date);
                """
            )
            conn.commit()
        calib_data, state = build_ratings(conn)

    print_matches_per_year(conn)
    run_calibration(calib_data)
    print_top_players(state)
    conn.close()


def _derive_calib_from_db(conn: sqlite3.Connection):
    """
    Reconstruct calib_data and final state from the DB for --report-only mode.
    Slower (DB round-trips per match) but avoids a full rebuild.
    This is a light path for when ratings are already built.
    """
    log.info("Deriving calibration data from DB (this may take a minute)...")
    rows = conn.execute(
        """
        SELECT m.tourney_date, m.surface, m.winner_name, m.loser_name, m.score
        FROM   matches m
        WHERE  m.tourney_date BETWEEN ? AND ?
        ORDER  BY m.tourney_date ASC, m.id ASC
        """,
        (CALIB_START, CALIB_END),
    ).fetchall()

    calib_data = []
    for date, surface, wname, lname, score in rows:
        su = (score or "").upper()
        if "W/O" in su or "DEF" in su or "RET" in su:
            continue
        if surface not in SURFACE_KEY:
            surface = "Hard"
        w = get_player_elo(conn, wname, date)
        l = get_player_elo(conn, lname, date)
        s_key = f"{SURFACE_KEY[surface]}_elo"
        calib_data.append({
            "date":         date,
            "surface":      surface,
            "elo_diff":     w["overall_elo"] - l["overall_elo"],
            "surface_diff": w[s_key]         - l[s_key],
        })

    # For top-players: pull the latest rating per player
    log.info("Building final state from player_elo for top-N table...")
    latest = conn.execute(
        """
        SELECT player_name, overall_elo, clay_elo, hard_elo, grass_elo,
               matches_played, clay_matches, hard_matches, grass_matches
        FROM   player_elo pe1
        WHERE  id = (
            SELECT MAX(id) FROM player_elo pe2
            WHERE  pe2.player_name = pe1.player_name
        )
        """
    ).fetchall()

    state = {}
    for row in latest:
        nm, ov, cl, ha, gr, n, nc, nh, ng = row
        s = _PlayerState()
        s.overall = ov; s.clay = cl; s.hard = ha; s.grass = gr
        s.n = n;        s.n_clay = nc; s.n_hard = nh; s.n_grass = ng
        state[nm] = s

    return calib_data, state


if __name__ == "__main__":
    main()
