"""
notify_tennis.py -- Discord notification layer for ITF Women tennis edge alerts.

Webhook URL from environment:
    DISCORD_WEBHOOK_TENNIS   edge alerts (green, 3066993)

Follows the same requests-library pattern as notify.py (MLB/WNBA).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False, encoding="utf-8-sig")

log = logging.getLogger(__name__)

_COLOR_TENNIS = 3066993   # green
_COLOR_WIN    = 3066993   # green
_COLOR_LOSS   = 15158332  # red
_FOOTER       = "ObServatory Tennis Model"


def _discord_post(webhook_url: str, payload: dict) -> bool:
    """POST JSON to a Discord webhook. Returns True on 2xx."""
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_TENNIS not set — notification skipped.")
        return False
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        ok   = resp.status_code in (200, 204)
        if not ok:
            log.warning("Discord returned HTTP %d: %s", resp.status_code, resp.text[:120])
        return ok
    except Exception as exc:
        log.error("Discord post failed: %s", exc)
        return False


def send_tennis_alert(alert: dict, config: dict) -> bool:
    """
    Send one match-winner edge alert to Discord.

    alert dict keys:
        player_name, opponent_name, tourney_name,
        model_prob, fair_prob, edge, odds (American int),
        event_id, extreme_flag (bool), game_time_ct (str, optional)
    """
    webhook  = os.environ.get("DISCORD_WEBHOOK_TENNIS", "").strip()
    player   = alert.get("player_name",   "Player")
    opponent = alert.get("opponent_name", "Opponent")
    tourney  = alert.get("tourney_name",  "ITF Women")
    prob     = alert.get("model_prob",    0.0)
    fair     = alert.get("fair_prob",     0.0)
    edge     = alert.get("edge",          0.0)
    odds     = alert.get("odds",          0)
    extreme  = alert.get("extreme_flag",  False)
    gt_ct    = alert.get("game_time_ct",  "")

    title = "🎾 Tennis Edge Alert"
    if extreme:
        title += "  ⚠️ EXTREME"

    odds_str = f"{odds:+d}" if odds else ""

    # "1:00 PM CT | ITF Women"  or just "ITF Women" if no game time
    context_line = f"{gt_ct} | {tourney}" if gt_ct else tourney

    desc = (
        f"**BET: {player} ({odds_str})**\n"
        f"vs {opponent}\n"
        f"{context_line}\n\n"
        f"Our P: {prob*100:.0f}%  |  Market: {fair*100:.0f}%  |  Edge: +{edge*100:.0f}%"
    )

    payload = {
        "embeds": [{
            "title":       title,
            "description": desc,
            "color":       _COLOR_TENNIS,
            "footer":      {"text": _FOOTER},
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        }]
    }

    ok = _discord_post(webhook, payload)
    if ok:
        log.info("Discord tennis alert sent: %s vs %s  edge=%.1f%%", player, opponent, edge * 100)
    return ok


def send_summary(n_alerts: int, config: dict) -> bool:
    """Count banner sent before individual alerts when multiple fire."""
    webhook = os.environ.get("DISCORD_WEBHOOK_TENNIS", "").strip()
    payload = {
        "embeds": [{
            "title":       "🎾 Tennis Edges Today",
            "description": f"**{n_alerts}** qualifying edge{'s' if n_alerts != 1 else ''} — alerts incoming.",
            "color":       _COLOR_TENNIS,
            "footer":      {"text": _FOOTER},
        }]
    }
    return _discord_post(webhook, payload)


def send_test(config: dict) -> bool:
    """Test webhook connectivity."""
    webhook = os.environ.get("DISCORD_WEBHOOK_TENNIS", "").strip()
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    payload = {
        "embeds": [{
            "title":       "🎾 Tennis webhook test",
            "description": f"ObServatory Tennis webhook OK — {ts}",
            "color":       _COLOR_TENNIS,
            "footer":      {"text": _FOOTER},
        }]
    }
    ok = _discord_post(webhook, payload)
    if ok:
        log.info("Tennis Discord test: OK")
    else:
        log.error("Tennis Discord test: FAILED — check DISCORD_WEBHOOK_TENNIS in .env")
    return ok
