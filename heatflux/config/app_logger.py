from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_path: Path | None = None) -> None:
    """Configure rotating file logging for the application.

    Silent to the console; all output goes to heatflux.log.
    Safe to call multiple times — skips setup if handlers already exist.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    target = log_path or Path("heatflux.log")
    handler = logging.handlers.RotatingFileHandler(
        target, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(handler)
