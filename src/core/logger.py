"""
Centralized logging setup.
Writes to both console and rotating log files in data/logs/.

Usage:
    from src.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello!")
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Will be initialized on first call
_initialized = False


def setup_logging(log_dir: str = "./data/logs", level=logging.INFO):
    """Configure root logger with console + file handlers."""
    global _initialized
    if _initialized:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (rotating, 5MB max, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_path / "rag_app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _initialized = True
    logging.getLogger(__name__).info("Logging initialized → %s", log_path)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Auto-initializes logging on first call."""
    setup_logging()
    return logging.getLogger(name)
