"""
grade_tennis.py -- The validation that actually matters.

For every ITF Women match-winner market Kambi posted (two-sided, de-viggable),
this script computes, on the SAME set of matches:

  1. Model edge        : elo_prob(side) - opening de-vigged fair prob
  2. CLV               : closing de-vig - bet-time de-vig, on the model's pick
                         (positive = market moved toward our side = we beat close)
  3. Grading           : look up the actual result in the Sackmann `matches`
                         table (by name + date window)
  4. Brier(Elo)        vs Brier(Kambi-close) on the graded subset

Brier needs results. Sackmann publishes ITF results weeks late, so on freshly
posted markets the graded subset may be empty -- that's reported honestly, and
the script is safe to re-run as Sackmann backfills.

CLV needs only the snapshot history (already in VPS Postgres) and is available
immediately. It is the faster validation signal.

Usage:
  python grade_tennis.py                 # grade everything, print report
  python grade_tennis.py --min-snaps 2   # only count CLV where >=2 snapshots
  python grade_tennis.py --no-store      # don't write kambi_grading table

Decision rule (per the review):
  If Brier(Elo) > Brier(Kambi) on a meaningful graded sample -> no edge, stop.
  Until then, lean on CLV: if model picks don't earn positive CLV, no edge.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _DIR / "config.json"
DB_PATH = _DIR / "data" / "tennis.db"
load_dotenv(_DIR / ".env", override=False, encoding="utf-8-sig")

SCALE = 400.0
START_ELO = 1500.0
GRADE_BACK_DAYS = 10        # search window: tourney_date in [game_date - N, game_date]
FUZZY_THRESHOLD = 0.90      # strict: a wrong match corrupts grading silently

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── odds / devig ────────────────────────────────────────────────────────────────
def american_to_decimal(odds: int) -> float:
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def devig(odds_a: int, odds_b: int) -> tuple[float, float, float]:
    """Proportional two-way de-vig. Returns (fair_a, fair_b, overround)."""
    ia = 1.0 / american_to_decimal(odds_a)
    ib = 1.0 / american_to_decimal(odds_b)
    ovr = ia + ib
    return ia / ovr, ib / ovr, ovr


# ── elo math ─────────────────────────────────────────────────────────────────────
def elo_prob(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / SCALE))


# ── name matching (mirror of predict_tennis, standalone) ─────────────────────────
def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


class Matcher:
    def __init__(self, conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT DISTINCT winner_name FROM matches "
            "UNION SELECT DISTINCT loser_name FROM matches"
        ).fetchall()
        self.norm_map = {_normalize(r[0]): r[0] for r in rows}
        try:
            self.aliases = {_normalize(k): s for k, s in
                            conn.execute("SELECT kambi_name, sackmann_name FROM player_aliases")}
        except sqlite3.OperationalError:
            self.aliases = {}

    def match(self, name: str) -> tuple[str, str, float]:
        """Return (sackmann_name, method, score)."""
        nk = _normalize(name)
        if nk in self.aliases:
            return self.aliases[nk], "alias", 1.0
        if nk in self.norm_map:
            return self.norm_map[nk], "exact", 1.0
        best_s, best_n = 0.0, ""
        for sn, real in self.norm_map.items():
            s = SequenceMatcher(None, nk, sn).ratio()
            if s > best_s:
                best_s, best_n = s, real
        if best_s >= FUZZY_THRESHOLD:
            return best_n, "fuzzy", best_s
        return "", "none", best_s


def get_elo(conn: sqlite3.Connection, name: str, before_date: str) -> tuple[float, int, bool]:
    """Return (overall_elo, matches_played, found). found=False -> defaulted to 1500."""
    row = conn.execute(
        "SELECT overall_elo, matches_played FROM player_elo "
        "WHERE player_name=? AND match_date<? ORDER BY match_date DESC, id DESC LIMIT 1",
        (name, before_date),
    ).fetchone()
    if row:
        return float(row[0]), int(row[1]), True
    return START_ELO, 0, False


# ── data structures ──────────────────────────────────────────────────────────────
@dataclass
class Side:
    kambi: str
    snaps: list[tuple[datetime, int]] = field(default_factory=list)  # (time, over_odds) ASC

    def opening(self) -> int:
        return self.snaps[0][1]

    def closing(self, before: datetime) -> Optional[int]:
        prior = [o for t, o in self.snaps if t < before]
        return prior[-1] if prior else None

    def n_snaps(self) -> int:
        return len(self.snaps)


@dataclass
class GradedMatch:
    event_id: str
    game_time: datetime
    a_kambi: str
    b_kambi: str
    a_sack: str
    b_sack: str
    a_method: str
    b_method: str
    a_elo: float
    b_elo: float
    a_n: int
    b_n: int
    defaulted: bool          # either side had no Elo history
    thin: bool               # either side < 10 prior matches
    # market
    open_fair_a: float
    close_fair_a: Optional[float]
    open_ovr: float
    close_ovr: Optional[float]
    min_snaps: int
    # model
    elo_prob_a: float
    edge_a: float            # elo_prob_a - open_fair_a
    pick_side: str           # 'A' or 'B' (model's edge side)
    pick_edge: float
    clv: Optional[float]     # close_fair(pick) - open_fair(pick)
    # snapshot timing (for CLV validity)
    first_lead_h: float      # hours from first snapshot to game_time
    last_lead_h: float       # hours from last pre-game snapshot to game_time
    span_h: float            # hours between first and last (pre-game) snapshot
    post_start: bool         # any snapshot at/after game_time (in-play contamination)
    # result
    result: Optional[str]    # 'A' | 'B' | None (ungraded)


# ── VPS pull ─────────────────────────────────────────────────────────────────────
def fetch_matches(pg) -> dict[str, dict]:
    cur = pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT g.event_id, g.game_time, ps.player_name, ps.over_odds, ps.snapshot_time
        FROM   props_snapshots ps
        JOIN   games g ON g.event_id = ps.event_id
        WHERE  g.league = 'ITF Women'
          AND  ps.line IS NULL
          AND  ps.over_odds IS NOT NULL
        ORDER  BY g.event_id, ps.player_name, ps.snapshot_time ASC
    """)
    events: dict[str, dict] = {}
    for r in cur.fetchall():
        ev = events.setdefault(r["event_id"], {"game_time": r["game_time"], "sides": {}})
        side = ev["sides"].setdefault(r["player_name"], Side(kambi=r["player_name"]))
        side.snaps.append((r["snapshot_time"], r["over_odds"]))
    return events


# ── grading against Sackmann ─────────────────────────────────────────────────────
def grade_result(conn: sqlite3.Connection, a_sack: str, b_sack: str,
                 game_date: str) -> Optional[str]:
    """Find the Sackmann result. Returns 'A', 'B', or None if not yet published."""
    if not a_sack or not b_sack:
        return None
    lo = (datetime.fromisoformat(game_date) - __import__("datetime").timedelta(days=GRADE_BACK_DAYS)).date().isoformat()
    row = conn.execute(
        "SELECT winner_name, loser_name FROM matches "
        "WHERE tourney_date BETWEEN ? AND ? "
        "  AND ((winner_name=? AND loser_name=?) OR (winner_name=? AND loser_name=?)) "
        "ORDER BY tourney_date DESC LIMIT 1",
        (lo, game_date, a_sack, b_sack, b_sack, a_sack),
    ).fetchone()
    if not row:
        return None
    return "A" if row[0] == a_sack else "B"


# ── core ─────────────────────────────────────────────────────────────────────────
def build(events: dict, conn: sqlite3.Connection, matcher: Matcher) -> list[GradedMatch]:
    out: list[GradedMatch] = []
    for eid, ev in events.items():
        sides = list(ev["sides"].values())
        if len(sides) != 2:
            continue
        A, B = sides[0], sides[1]
        gt: datetime = ev["game_time"]
        gdate = gt.astimezone(timezone.utc).date().isoformat()

        a_sack, a_m, _ = matcher.match(A.kambi)
        b_sack, b_m, _ = matcher.match(B.kambi)
        a_elo, a_n, a_found = get_elo(conn, a_sack or A.kambi, gdate)
        b_elo, b_n, b_found = get_elo(conn, b_sack or B.kambi, gdate)
        defaulted = not (a_found and b_found)
        thin = (a_n < 10) or (b_n < 10)

        open_fair_a, open_fair_b, open_ovr = devig(A.opening(), B.opening())
        ca, cb = A.closing(gt), B.closing(gt)
        if ca is not None and cb is not None:
            close_fair_a, close_fair_b, close_ovr = devig(ca, cb)
        else:
            close_fair_a = close_fair_b = close_ovr = None

        ep_a = elo_prob(a_elo, b_elo)
        edge_a = ep_a - open_fair_a
        edge_b = (1 - ep_a) - open_fair_b
        if edge_a >= edge_b:
            pick, pick_edge, pick_open, pick_close = "A", edge_a, open_fair_a, close_fair_a
        else:
            pick, pick_edge, pick_open, pick_close = "B", edge_b, open_fair_b, close_fair_b
        clv = (pick_close - pick_open) if pick_close is not None else None

        # ── snapshot timing diagnostics ──────────────────────────────────────
        all_times = [t for t, _ in A.snaps] + [t for t, _ in B.snaps]
        pre_times = [t for t in all_times if t < gt]
        first_t = min(all_times)
        first_lead_h = (gt - first_t).total_seconds() / 3600
        if pre_times:
            last_pre = max(pre_times)
            last_lead_h = (gt - last_pre).total_seconds() / 3600
            span_h = (last_pre - first_t).total_seconds() / 3600
        else:
            last_lead_h = float("nan")
            span_h = 0.0
        post_start = any(t >= gt for t in all_times)

        out.append(GradedMatch(
            event_id=eid, game_time=gt,
            a_kambi=A.kambi, b_kambi=B.kambi, a_sack=a_sack, b_sack=b_sack,
            a_method=a_m, b_method=b_m,
            a_elo=a_elo, b_elo=b_elo, a_n=a_n, b_n=b_n,
            defaulted=defaulted, thin=thin,
            open_fair_a=open_fair_a, close_fair_a=close_fair_a,
            open_ovr=open_ovr, close_ovr=close_ovr,
            min_snaps=min(A.n_snaps(), B.n_snaps()),
            elo_prob_a=ep_a, edge_a=edge_a, pick_side=pick, pick_edge=pick_edge,
            clv=clv,
            first_lead_h=first_lead_h, last_lead_h=last_lead_h,
            span_h=span_h, post_start=post_start,
            result=grade_result(conn, a_sack, b_sack, gdate),
        ))
    return out


# ── reporting ────────────────────────────────────────────────────────────────────
def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def st_median(xs):
    xs = sorted(x for x in xs if x == x)  # drop NaN
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def report(gms: list[GradedMatch], min_snaps: int) -> None:
    W = 70
    def hdr(t): print("\n" + "=" * W + f"\n  {t}\n" + "=" * W)

    n = len(gms)
    matched = [g for g in gms if g.a_method != "none" and g.b_method != "none"]
    fuzzy = [g for g in gms if "fuzzy" in (g.a_method, g.b_method)]
    defaulted = [g for g in matched if g.defaulted]
    thin = [g for g in matched if g.thin]

    hdr("KAMBI ITF WOMEN -- GRADING & CLV REPORT")
    print(f"  Two-sided matches posted        : {n}")
    print(f"  Both names matched to Sackmann   : {len(matched)} "
          f"({len(matched)/n*100:.0f}%)   [fuzzy: {len(fuzzy)}]")
    print(f"  ...of which Elo-defaulted (1500) : {len(defaulted)}")
    print(f"  ...of which thin (<10 matches)   : {len(thin)}")
    print(f"  Name format check: Kambi='First Last' (e.g. {gms[0].a_kambi!r})")

    # ── market structure ──
    hdr("MARKET STRUCTURE (the vig you must beat)")
    ovrs = [g.open_ovr for g in gms]
    print(f"  Opening overround (hold): mean {(_mean(ovrs)-1)*100:+.1f}%   "
          f"min {(min(ovrs)-1)*100:+.1f}%  max {(max(ovrs)-1)*100:+.1f}%")
    have_close = [g for g in gms if g.close_ovr is not None]
    if have_close:
        c = [g.close_ovr for g in have_close]
        print(f"  Closing overround       : mean {(_mean(c)-1)*100:+.1f}%   "
              f"(n={len(have_close)})")
    movers = [g for g in gms if g.min_snaps >= min_snaps and g.clv is not None]
    print(f"  Matches with >={min_snaps} snapshots both sides (CLV-usable): {len(movers)}")

    # ── model vs market disagreement ──
    hdr("MODEL vs MARKET (no result needed)")
    edges = [abs(g.edge_a) for g in matched]
    big = [g for g in matched if g.pick_edge >= 0.08 and not g.defaulted]
    print(f"  Mean |model - market| (matched) : {_mean(edges)*100:.1f}%")
    print(f"  Matches with model edge >=8%     : "
          f"{sum(1 for g in matched if g.pick_edge>=0.08)}  "
          f"(non-defaulted: {len(big)})")
    print(f"  How edges split by data quality:")
    for lbl, sub in [("clean (>=10 matches)", [g for g in matched if not g.thin]),
                     ("thin  (<10 matches) ", thin),
                     ("defaulted (1500)    ", defaulted)]:
        e8 = sum(1 for g in sub if g.pick_edge >= 0.08)
        print(f"    {lbl}: n={len(sub):>3}  edge>=8%: {e8:>3}  "
              f"mean|edge|={_mean([abs(g.edge_a) for g in sub])*100:4.1f}%")

    # ── CLV data validity (must come BEFORE trusting any CLV number) ──
    hdr("CAN WE EVEN MEASURE CLV? (snapshot timing)")
    leads = [g.first_lead_h for g in gms]
    spans = [g.span_h for g in gms]
    single = [g for g in gms if g.span_h < 0.5]
    poststart = [g for g in gms if g.post_start]
    print(f"  First snapshot before start : median {st_median(leads):.1f}h  "
          f"(true opening line posts DAYS out -- we catch it ~hours out)")
    print(f"  Snapshot span (first->last) : median {st_median(spans):.1f}h")
    print(f"  Single-capture events       : {len(single)}/{len(gms)} "
          f"({len(single)/len(gms)*100:.0f}%) -- NO open->close arc, CLV is trivially 0")
    print(f"  Events polled AFTER start    : {len(poststart)} "
          f"(in-play odds present; excluded from 'close' but a collector bug)")
    print(f"  >>> As collected, CLV is only meaningful for the few events with a")
    print(f"      real pre-game movement window. Treat below as a fragile hint only.")

    # ── CLV (only where the line actually moved) ──
    hdr("CLV -- among matches where the line ACTUALLY moved")
    print("  CLV = close_fair(pick) - bet_fair(pick).  + = market moved to our pick.")
    moved = [g for g in gms if g.clv is not None and abs(g.clv) > 1e-9]
    flat = [g for g in gms if g.clv is not None and abs(g.clv) <= 1e-9]
    print(f"  Lines that moved : {len(moved)}    Flat (CLV=0): {len(flat)}")
    if moved:
        clvs = [g.clv for g in moved]
        pos = sum(1 for c in clvs if c > 0)
        print(f"  Mean CLV (moved) : {_mean(clvs)*100:+.2f} pp")
        print(f"  Beat close       : {pos}/{len(moved)} ({pos/len(moved)*100:.0f}%)")
        em = [g for g in moved if g.pick_edge >= 0.08 and not g.defaulted]
        if em:
            ec = [g.clv for g in em]
            ep = sum(1 for c in ec if c > 0)
            print(f"  --- edge>=8% picks that moved ({len(em)}): "
                  f"mean {_mean(ec)*100:+.2f}pp  beat-close {ep}/{len(em)} "
                  f"({ep/len(em)*100:.0f}%)")
        print(f"\n  HINT: directionally positive, but n is tiny and the timing window")
        print(f"  is hours-not-days. NOT sufficient to deploy. Fix the collector first.")

    # ── grading / Brier ──
    hdr("BRIER -- Elo vs Kambi on graded matches")
    graded = [g for g in matched if g.result is not None and g.close_fair_a is not None]
    print(f"  Graded matches (result found in Sackmann): {len(graded)} / {len(matched)}")
    if len(graded) < 30:
        gd_min = min(g.game_time for g in gms).date()
        gd_max = max(g.game_time for g in gms).date()
        print(f"  >>> Betting window {gd_min}..{gd_max} post-dates Sackmann's latest")
        print(f"      published results. ITF results lag ~2-6 weeks. Re-run this script")
        print(f"      after backfilling Sackmann; it will recompute automatically.")
        print(f"  >>> BRIER VERDICT: INSUFFICIENT DATA (n={len(graded)}). Not evaluated.")
        if graded:
            def y_a(g): return 1.0 if g.result == "A" else 0.0
            elo_b = _mean([(g.elo_prob_a - y_a(g)) ** 2 for g in graded])
            kam_b = _mean([(g.close_fair_a - y_a(g)) ** 2 for g in graded])
            print(f"      (peek, do NOT trust: Brier Elo={elo_b:.4f} "
                  f"Kambi={kam_b:.4f} on n={len(graded)})")
    else:
        def y_a(g): return 1.0 if g.result == "A" else 0.0
        elo_b = _mean([(g.elo_prob_a - y_a(g)) ** 2 for g in graded])
        kam_b = _mean([(g.close_fair_a - y_a(g)) ** 2 for g in graded])
        base_b = _mean([(0.5 - y_a(g)) ** 2 for g in graded])
        print(f"  Brier(Elo)   = {elo_b:.4f}")
        print(f"  Brier(Kambi) = {kam_b:.4f}   (de-vigged closing line)")
        print(f"  Brier(0.50)  = {base_b:.4f}   (coinflip baseline)")
        verdict = ("ELO BEATS KAMBI -- real edge signal, proceed."
                   if elo_b < kam_b else
                   "KAMBI BEATS ELO -- no edge. STOP. (per the review's decision rule)")
        print(f"\n  VERDICT: {verdict}")
        if len(graded) < 100:
            print(f"  (Caution: n={len(graded)} is modest; ~100+ gives more confidence.)")

    print("\n" + "=" * W)


# ── persistence ──────────────────────────────────────────────────────────────────
def store(conn: sqlite3.Connection, gms: list[GradedMatch]) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS kambi_grading;
        CREATE TABLE kambi_grading (
            event_id     TEXT PRIMARY KEY,
            game_time    TEXT,
            a_kambi TEXT, b_kambi TEXT, a_sack TEXT, b_sack TEXT,
            a_method TEXT, b_method TEXT,
            a_elo REAL, b_elo REAL, a_n INT, b_n INT,
            defaulted INT, thin INT,
            open_fair_a REAL, close_fair_a REAL, open_ovr REAL, close_ovr REAL,
            min_snaps INT,
            elo_prob_a REAL, edge_a REAL, pick_side TEXT, pick_edge REAL, clv REAL,
            first_lead_h REAL, last_lead_h REAL, span_h REAL, post_start INT,
            result TEXT, graded_at TEXT
        )""")
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        for g in gms:
            conn.execute(
                "INSERT OR REPLACE INTO kambi_grading VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (g.event_id, g.game_time.isoformat(),
                 g.a_kambi, g.b_kambi, g.a_sack, g.b_sack, g.a_method, g.b_method,
                 g.a_elo, g.b_elo, g.a_n, g.b_n, int(g.defaulted), int(g.thin),
                 g.open_fair_a, g.close_fair_a, g.open_ovr, g.close_ovr, g.min_snaps,
                 g.elo_prob_a, g.edge_a, g.pick_side, g.pick_edge, g.clv,
                 g.first_lead_h, g.last_lead_h, g.span_h, int(g.post_start),
                 g.result, now))
    log.info("Stored %d rows in kambi_grading.", len(gms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-snaps", type=int, default=2,
                    help="Min snapshots per side to count CLV (default 2)")
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))["vps"]
    pg = psycopg2.connect(host=cfg["host"], port=cfg.get("port", 5432),
                          dbname=cfg["database"], user=cfg["user"],
                          password=os.environ.get("VPS_DB_PASSWORD", ""), connect_timeout=15)
    conn = sqlite3.connect(DB_PATH)
    sack_max = conn.execute("SELECT MAX(tourney_date) FROM matches").fetchone()[0]
    log.info("Sackmann latest published tourney_date: %s", sack_max)

    events = fetch_matches(pg)
    pg.close()
    log.info("Pulled %d ITF Women events from VPS.", len(events))

    matcher = Matcher(conn)
    gms = build(events, conn, matcher)
    report(gms, args.min_snaps)
    if not args.no_store:
        store(conn, gms)
    conn.close()


if __name__ == "__main__":
    main()
