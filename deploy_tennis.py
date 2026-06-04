"""
deploy_tennis.py -- Deploy the ITF Women tennis pipeline to the VPS.

Files deployed (local -> remote):
    predict_tennis.py       -> /home/picks/predict_tennis.py
    notify_tennis.py        -> /home/picks/notify_tennis.py
    elo_engine.py           -> /home/picks/elo_engine.py
    collect_tennis_vps.py   -> /home/picks/collect_tennis_vps.py
    tennis_config_vps.json  -> /home/picks/tennis_config.json
    [data/tennis.db         -> /home/picks/tennis.db]  (--include-db only; 149 MB)

tennis.db is skipped by default because:
  - It is 149 MB and takes ~2 minutes over SCP
  - After initial deploy, collect_tennis_vps.py maintains it on the VPS
  - Use --include-db for the first deploy or after a local schema change

VPS credentials:  CLI args > env vars (VPS_HOST, VPS_USER, VPS_KEY_PATH) > config

After deploy the VPS cron should read:
    0  5 * * *  cd /home/picks && python collect_tennis_vps.py >> /home/picks/logs/collect_tennis.log 2>&1
    0 11 * * *  cd /home/picks && python predict_tennis.py --config /home/picks/tennis_config.json >> /home/picks/logs/predict_tennis.log 2>&1

VPS environment variables required (/home/picks/.env or crontab):
    VPS_DB_PASSWORD=password
    DISCORD_WEBHOOK_TENNIS=https://discord.com/api/webhooks/...

Usage:
    python deploy_tennis.py --dry-run           # show manifest, no transfer
    python deploy_tennis.py                     # deploy code files only
    python deploy_tennis.py --include-db        # first deploy: includes tennis.db (149 MB)
    python deploy_tennis.py --skip-scripts      # re-deploy db only
    python deploy_tennis.py --vps-key ~/.ssh/id_ed25519
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent

load_dotenv(_DIR / ".env", override=False, encoding="utf-8-sig")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt_bytes(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"

def _fmt_elapsed(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s" if m else f"{sec}s"

def _load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _DIR / "tennis_config_vps.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def _resolve_credentials(args, cfg: dict) -> tuple[str, str, str | None, str]:
    """CLI > env > config, three-layer precedence. Returns (host, user, key, remote_dir)."""
    host = (args.vps_host
            or os.environ.get("VPS_HOST")
            or cfg.get("deploy", {}).get("host", ""))
    user = (args.vps_user
            or os.environ.get("VPS_USER")
            or cfg.get("deploy", {}).get("user", "picks"))
    key  = args.vps_key or os.environ.get("VPS_KEY_PATH")
    rdir = cfg.get("deploy", {}).get("remote_dir", "/home/picks").rstrip("/")
    return host, user, key, rdir


# ── SCP ───────────────────────────────────────────────────────────────────────
def _scp(
    local:   str,
    remote:  str,
    host:    str,
    user:    str,
    key:     str | None,
    dry_run: bool = False,
) -> float:
    """SCP one file. Returns elapsed seconds. Raises RuntimeError on failure."""
    lp   = Path(local)
    size = lp.stat().st_size if lp.exists() else 0
    log.info("  %-32s  %8s  ->  %s:%s",
             lp.name, _fmt_bytes(size), host, remote)
    if dry_run:
        return 0.0
    cmd = ["scp"]
    if key:
        cmd += ["-i", str(key)]
    cmd += [str(local), f"{user}@{host}:{remote}"]
    t0     = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"scp failed (exit {result.returncode}): {local}")
    return elapsed


# ── manifest ──────────────────────────────────────────────────────────────────
def _build_manifest(rdir: str, include_db: bool, skip_scripts: bool) -> list[dict]:
    """
    Returns list of dicts: {local, remote, label, optional}.
    All remotes are absolute paths on the VPS.
    """
    scripts = [
        {"local": str(_DIR / "predict_tennis.py"),
         "remote": f"{rdir}/predict_tennis.py",
         "label": "prediction engine"},
        {"local": str(_DIR / "notify_tennis.py"),
         "remote": f"{rdir}/notify_tennis.py",
         "label": "Discord notifier"},
        {"local": str(_DIR / "elo_engine.py"),
         "remote": f"{rdir}/elo_engine.py",
         "label": "Elo engine (imported by collect_vps)"},
        {"local": str(_DIR / "collect_tennis_vps.py"),
         "remote": f"{rdir}/collect_tennis_vps.py",
         "label": "VPS daily collector"},
        {"local": str(_DIR / "tennis_config_vps.json"),
         "remote": f"{rdir}/tennis_config.json",
         "label": "VPS config (paths use /home/picks)"},
        {"local": str(_DIR / "grade_tennis.py"),
         "remote": f"{rdir}/grade_tennis.py",
         "label": "Kambi grader + CLV (re-run when Sackmann backfills)"},
        {"local": str(_DIR / "calibrate_tennis.py"),
         "remote": f"{rdir}/calibrate_tennis.py",
         "label": "Platt calibration monitor (not applied; raw Elo is better)"},
    ]

    db_entry = {
        "local":  str(_DIR / "data" / "tennis.db"),
        "remote": f"{rdir}/tennis.db",
        "label":  "SQLite DB — 149 MB, takes ~2 min",
        "optional": True,
    }

    manifest = []
    if not skip_scripts:
        manifest.extend(scripts)
    if include_db:
        manifest.append(db_entry)

    return manifest


def _verify(manifest: list[dict], dry_run: bool) -> bool:
    ok = True
    for item in manifest:
        p = Path(item["local"])
        if not p.exists():
            log.error("Missing local file: %s", p)
            if not item.get("optional"):
                ok = False
    return ok or dry_run


# ── post-deploy instructions ──────────────────────────────────────────────────
CRON_LINES = """\
  Cron lines to add on VPS  (crontab -e as root):

    # Tennis ITF Women — daily collect + predict
    0  5  * * *  cd /home/picks && python3 collect_tennis_vps.py >> /home/picks/logs/collect_tennis.log 2>&1
    0 11  * * *  cd /home/picks && python3 predict_tennis.py --config /home/picks/tennis_config.json >> /home/picks/logs/predict_tennis.log 2>&1
    # Tennis ITF Women — high-frequency Kambi snapshot (07:00–20:59 UTC)
    # Runs kambi_collector_tennis.py from collectors/ so kambi_shared is importable
    */10 7-20 * * *  cd /home/picks && python3 collectors/kambi_collector_tennis.py >> /home/picks/logs/tennis_hf.log 2>&1
"""

ENV_LINES = """\
  VPS environment variables (add to /home/picks/.env or crontab MAILTO line):

    VPS_DB_PASSWORD=password
    DISCORD_WEBHOOK_TENNIS=https://discord.com/api/webhooks/YOUR_TENNIS_WEBHOOK

  predict_tennis.py loads /home/picks/.env automatically via python-dotenv.
  Ensure python-dotenv and psycopg2 are installed on the VPS:
    pip install python-dotenv psycopg2-binary requests pandas
"""

FIRST_RUN = """\
  First-run checklist on VPS:

    1. mkdir -p /home/picks/logs
    2. Confirm tennis.db exists: ls -lh /home/picks/tennis.db
    3. Test collect:   python /home/picks/collect_tennis_vps.py --dry-run
    4. Test predict:   python /home/picks/predict_tennis.py \\
                              --config /home/picks/tennis_config.json --dry-run
    5. Add cron lines (see above)
    6. Tail logs:      tail -f /home/picks/logs/predict_tennis.log
"""


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deploy ITF Women tennis pipeline to VPS")
    p.add_argument("--dry-run",      action="store_true",
                   help="Show manifest without transferring anything")
    p.add_argument("--include-db",   action="store_true",
                   help="Also deploy tennis.db (149 MB; use on first deploy)")
    p.add_argument("--skip-scripts", action="store_true",
                   help="Skip script files; only deploy tennis.db (implies --include-db)")
    p.add_argument("--vps-host",     default=None)
    p.add_argument("--vps-user",     default=None)
    p.add_argument("--vps-key",      default=None, metavar="PATH",
                   help="SSH private key path (default: VPS_KEY_PATH env var)")
    p.add_argument("--config",       default=None, metavar="PATH",
                   help="Local config to use (default: tennis_config_vps.json)")
    args = p.parse_args(argv)

    if args.skip_scripts:
        args.include_db = True

    cfg  = _load_config(args.config)
    host, user, key, rdir = _resolve_credentials(args, cfg)

    if not host:
        log.error("No VPS host. Set VPS_HOST env var, --vps-host, or deploy.host in config.")
        return 1
    if not key and not args.dry_run:
        log.error("No SSH key. Set VPS_KEY_PATH env var or --vps-key.")
        return 1

    manifest = _build_manifest(rdir, args.include_db, args.skip_scripts)

    # ── manifest preview ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  TENNIS DEPLOY MANIFEST  ->  {user}@{host}:{rdir}")
    if args.dry_run:
        print("  *** DRY RUN -- no files will be transferred ***")
    print("=" * 70)
    print(f"  {'Local file':<32}  {'Size':>8}  {'Remote destination'}")
    print(f"  {'-'*66}")
    for item in manifest:
        lp   = Path(item["local"])
        size = _fmt_bytes(lp.stat().st_size) if lp.exists() else "MISSING"
        rname = Path(item["remote"]).name
        note  = f"  # {item['label']}" if "label" in item else ""
        print(f"  {lp.name:<32}  {size:>8}  /home/picks/{rname:<22}{note}")
    print(f"  {'-'*66}")
    print()

    if not args.include_db:
        print("  NOTE: tennis.db NOT included. Use --include-db for first deploy.")
        print()

    # ── cron and env preview (always shown) ───────────────────────────────────
    print(CRON_LINES)
    print(ENV_LINES)

    if args.dry_run:
        print(FIRST_RUN)
        print("  Dry run complete. Re-run without --dry-run to deploy.")
        return 0

    # ── pre-flight ────────────────────────────────────────────────────────────
    if not _verify(manifest, args.dry_run):
        return 1

    # ── transfer ──────────────────────────────────────────────────────────────
    total_bytes   = 0
    total_elapsed = 0.0
    failed: list[str] = []

    print("Transferring files...")
    for item in manifest:
        try:
            elapsed = _scp(item["local"], item["remote"], host, user, key,
                           dry_run=args.dry_run)
            total_elapsed += elapsed
            if Path(item["local"]).exists():
                total_bytes += Path(item["local"]).stat().st_size
        except RuntimeError as exc:
            log.error("%s", exc)
            failed.append(item["local"])

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DEPLOY SUMMARY")
    print("=" * 70)
    for item in manifest:
        lp     = Path(item["local"])
        size   = _fmt_bytes(lp.stat().st_size) if lp.exists() else "?"
        status = "FAILED" if item["local"] in failed else "OK"
        print(f"  {lp.name:<32}  {size:>8}  [{status}]")

    print(f"  {'-'*50}")
    print(f"  Total: {len(manifest)} file(s)  "
          f"{_fmt_bytes(total_bytes)}  in {_fmt_elapsed(total_elapsed)}")
    print()
    print(FIRST_RUN)
    print("=" * 70)

    if failed:
        log.error("%d file(s) failed.", len(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
