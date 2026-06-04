from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


BACKUP_DIR = Path("session_backups")
DEFAULT_PREFIX = "session"


def _safe_prefix(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return safe.strip("_") or DEFAULT_PREFIX


def _derive_prefix_from_payload(payload: dict[str, Any]) -> str:
    ansys_file = str(payload.get("ansys_file", "") or "").strip()
    if not ansys_file:
        return DEFAULT_PREFIX
    return _safe_prefix(Path(ansys_file).stem)


def _next_backup_path(prefix: str, timestamp: datetime) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    candidate = BACKUP_DIR / f"{prefix}_{ts}.json"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = BACKUP_DIR / f"{prefix}_{ts}_{i:02d}.json"
        if not candidate.exists():
            return candidate
        i += 1


def save_session_backup(payload: dict[str, Any], path: Path | None = None) -> Path:
    now = datetime.now()
    data = dict(payload)
    data["timestamp"] = now.isoformat(timespec="seconds")
    target = path if path is not None else _next_backup_path(_derive_prefix_from_payload(data), now)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _log.info("Session backup saved: %s", target)
    return target


def _latest_backup_path() -> Path | None:
    if not BACKUP_DIR.exists():
        return None
    files = sorted(BACKUP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_session_backup(path: Path | None = None) -> dict[str, Any] | None:
    target = path if path is not None else _latest_backup_path()
    if target is None or not target.exists():
        return None
    _log.info("Loading session backup: %s", target)
    return json.loads(target.read_text(encoding="utf-8"))
