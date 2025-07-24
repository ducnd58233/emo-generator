import logging
import sys
from datetime import datetime

from ..common.constants import (
    DATE_FORMAT,
    DEFAULT_LOGGER_NAME,
    LOG_EXT,
    LOG_FORMAT,
    LOGS_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_CONFIGURED_LOGGERS: set[str] = set()


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME, level: int = logging.INFO
) -> logging.Logger:
    """Set up and return a logger instance with file and console handlers.


    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    if name in _CONFIGURED_LOGGERS:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # File handler with date-based naming
    today = datetime.now().strftime(DATE_FORMAT)
    log_file = LOGS_DIR / f"{today}{LOG_EXT}"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Mark as configured
    _CONFIGURED_LOGGERS.add(name)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger by name.

    If the logger hasn't been configured yet, it will be set up automatically.

    Args:
        name: Logger name (defaults to default logger if None)

    Returns:
        Logger instance
    """
    logger_name = name or DEFAULT_LOGGER_NAME

    if logger_name not in _CONFIGURED_LOGGERS:
        setup_logger(logger_name)

    return logging.getLogger(logger_name)
