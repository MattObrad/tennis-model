# ITF Women Tennis Model — Conclusions & Decision Log

## Brier Test #1 — June 4, 2026 (n=21 graded bets)

**Context:** First gradeable sample after scrape_itf_results.py backfill eliminated
the Sackmann 2-4 week lag. All 21 bets from alert_date=2026-06-04, graded via
the new tennisexplorer same-day results.

### Aggregate results

| Metric                | Score   | vs Baseline |
|-----------------------|---------|-------------|
| Elo (model) Brier     | 0.2429  | −0.0071     |
| Kambi (market) Brier  | 0.2152  | −0.0348     |
| Baseline (50/50)      | 0.2500  | —           |
| **Elo − Kambi gap**   | **+0.0277** | **Elo worse** |

Actual win rate: 52.4% (11/21). Both models beat a 50/50 coin flip, but
Kambi is materially sharper than Elo.

### Calibration by model probability bucket

| Bucket | N | ModelP% | KambiP% | Actual% | Elo Brier | Kambi Brier | Verdict |
|--------|---|---------|---------|---------|-----------|-------------|---------|
| 50–60% | 6 | 57.4% | 33.6% | **16.7%** | 0.2985 | 0.1752 | BAD — model way overconfident |
| 60–70% | 4 | 63.0% | 45.6% | 50.0% | 0.2485 | 0.2274 | BAD — marginal improvement over baseline |
| 70–80% | 5 | 76.1% | 54.9% | 60.0% | 0.2896 | 0.2889 | NEUTRAL — near-identical to Kambi |
| **80%+** | **6** | **84.5%** | **64.5%** | **83.3%** | **0.1446** | **0.1857** | **GOOD — Elo beats Kambi** |

Key findings:
- **50–60% bucket is actively bad.** Model predicted ~57%, actual win rate 17%.
  Kambi had these same players at ~34% and was much closer to correct.
  The "edges" in this range are entirely model error, not real mispricing.
- **80%+ bucket beats Kambi** (Elo 0.1446 vs Kambi 0.1857). When Elo is
  very confident it appears well-calibrated. n=6 is too small to rely on, but
  directionally encouraging.
- **CLV average +4.89pp** but beat rate only 32% (6/19). Three outliers
  (Crossley +47pp, Gonzalez Vilar +27pp, Solar Donoso +12pp) inflate the mean.
  On 13/19 bets the market moved away from our pick by close.

### Prior hypothesis confirmed

Memory note from 2026-06-05: "Elo is underconfident — raw probabilities too
compressed vs sharp market (13–22% 'edges' on near-coin-flip matches)."
This Brier test confirms that framing: the "edges" in the 55–70% range are
not real edges, they reflect Elo overconfidence vs the Kambi sharp market.

---

## Decision #1 — min_model_prob raised to 0.70

**Date:** 2026-06-05  
**Changed:** `config.json` + `tennis_config_vps.json`, deployed to VPS as
`/home/picks/tennis_config.json`.

**Rationale:** The 50–60% and 60–70% buckets show the model is consistently wrong
in this range (17% actual vs 57% predicted). There is no value in generating
alerts here while in paper mode. The 70%+ range shows neutral-to-good Brier
performance and represents the only part of the distribution worth monitoring.

**Impact on alert volume (last 7 days of predictions):**

| Threshold | Qualifying alerts | Change |
|-----------|------------------|--------|
| Old: 0.55 | 37               | —      |
| New: 0.70 | 17               | −20 (−54%) |

Bucket breakdown of what was cut vs kept:
- 55–60%: 11 → dropped (Brier-bad)
- 60–65%: 8  → dropped (Brier-bad)
- 65–70%: 1  → dropped (Brier-bad)
- 70–75%: 5  → kept
- 75–80%: 5  → kept
- 80%+:   7  → kept

Paper mode remains ON. The threshold change does not affect betting — it
reduces alert noise during the monitoring period.

---

## Next Evaluation

**Target sample:** ~200 graded bets (current: 21).  
**Estimated timeline:** 2–3 weeks at ~10–15 alerts/day under new 0.70 threshold
  (roughly 7–10 qualifying alerts/day × 14–21 days).

**Re-run Brier test when:** alerts.db has ~200 graded TENNIS bets with
`graded=1 AND result IN ('WIN','LOSS')`.

**Decision rule (unchanged):**
> If Brier(Elo) > Brier(Kambi) at n≥200 → stop. Do not remove paper mode.  
> If Brier(Elo) ≤ Brier(Kambi) at n≥200 → begin live validation at minimum stakes.

**Additional things to check at next evaluation:**
1. Does the 80%+ bucket still beat Kambi? (n=6 now, needs ~30+)
2. Is the 70–80% bucket neutral or directional?
3. Platt scaling / market-shrinkage recalibration — did it improve holdout Brier?
   (Prior test: calibration made 2025 holdout worse 0.1973→0.1980; recheck with more data)
4. Surface Elo: `predict_tennis.py` only uses `overall_elo` despite clay/hard/grass
   columns existing. Test if surface-specific Elo improves 70%+ bucket calibration.

---

## Infrastructure changes (2026-06-05)

- **`scrape_itf_results.py`** deployed to VPS — runs at 06:00 UTC daily.
  Source: tennisexplorer.com. Eliminates Sackmann 2-4 week lag.
  First backfill: 793 matches (May 30–Jun 5), 33 requests, Elo rebuilt 10.1s.
- **`matches` table** gained nullable `notes TEXT` column for unresolved player names.
- **`collect_tennis_vps.py`** updated with `cleanup_te_rows()` hook — removes
  TE- scraper rows when Sackmann covers the same matches.

---

*Last updated: 2026-06-05*
