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

    alert dict keys (all required):
        player_name, opponent_name, surface, tourney_name,
        model_prob, fair_prob, edge, odds (American int),
        event_id, extreme_flag (bool)
    """
    webhook  = os.environ.get("DISCORD_WEBHOOK_TENNIS", "").strip()
    player   = alert.get("player_name",   "Player")
    opponent = alert.get("opponent_name", "Opponent")
    surface  = alert.get("surface",       "Unknown")
    tourney  = alert.get("tourney_name",  "ITF Women")
    prob     = alert.get("model_prob",    0.0)
    fair     = alert.get("fair_prob",     0.0)
    edge     = alert.get("edge",          0.0)
    odds     = alert.get("odds",          0)
    extreme  = alert.get("extreme_flag",  False)

    title = "🎾 Tennis Edge Alert"
    if extreme:
        title += "  ⚠️ !EXTREME"

    # Decimal odds from American
    if odds < 0:
        decimal_odds = 1.0 + 100.0 / abs(odds)
    else:
        decimal_odds = 1.0 + odds / 100.0

    ev = prob * decimal_odds - 1.0

    payload = {
        "embeds": [{
            "title":  title,
            "color":  _COLOR_TENNIS,
            "fields": [
                {"name": "Player",          "value": player,                        "inline": True},
                {"name": "Opponent",        "value": opponent,                      "inline": True},
                {"name": "Surface",         "value": surface,                       "inline": True},
                {"name": "Predicted Win%",  "value": f"{prob*100:.1f}%",            "inline": True},
                {"name": "Market Implied%", "value": f"{fair*100:.1f}%",            "inline": True},
                {"name": "Edge",            "value": f"+{edge*100:.1f}%",           "inline": True},
                {"name": "Odds",            "value": f"{odds:+d}",                  "inline": True},
                {"name": "Decimal",         "value": f"{decimal_odds:.2f}x",        "inline": True},
                {"name": "EV",              "value": f"{ev*100:+.1f}%",             "inline": True},
                {"name": "Tournament",      "value": tourney,                       "inline": False},
            ],
            "footer":    {"text": _FOOTER},
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
