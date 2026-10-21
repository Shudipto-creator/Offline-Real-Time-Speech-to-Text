"""
Logging setup for the speech-to-text application.

Technical diagnostics only. NEVER logs microphone audio, transcript text,
or copied/exported text. Uses rotating file handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Sensitive data patterns that must never appear in logs
_SENSITIVE_PATTERNS = [
    "transcript",
    "audio",
    "recognized_text",
    "copied",
    "exported",
]


class SensitiveDataFilter(logging.Filter):
    """Filter that blocks any record whose message contains sensitive content."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        for pattern in _SENSITIVE_PATTERNS:
            if pattern in msg:
                return False
        return True


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    name: str = "speech2text",
) -> logging.Logger:
    """Configure rotating file logging and a console handler.

    Args:
        level: Logging level (default INFO).
        log_dir: Directory for log files (default: project/logs).
        name: Logger name.

    Returns:
        Configured logger.
    """
    if log_dir is None:
        log_dir = LOG_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Sensitive data filter
    sensitive_filter = SensitiveDataFilter()

    # File handler with rotation
    file_handler = logging.FileHandler(
        log_dir / "speech2text.log",
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.addFilter(sensitive_filter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings+ to console
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )
    console_handler.addFilter(sensitive_filter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "speech2text") -> logging.Logger:
    """Get or create a child logger."""
    return logging.getLogger(name)