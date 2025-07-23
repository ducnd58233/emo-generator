import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Set up logs directory at project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_DEFAULT_LOGGER_NAME = "emo_generator"
_CONFIGURED_LOGGERS = set()


def setup_logger(
    name: str = _DEFAULT_LOGGER_NAME, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up and return a logger instance with file and console handlers.
    Ensures handlers are not duplicated.
    Args:
        name: Logger name.
        level: Logging level.
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    logger.handlers.clear()
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f"{today}.log"), encoding="utf-8"
    )
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


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger by name
    Args:
        name: Logger name (defaults to root logger if None).
    Returns:
        Logger instance.
    """
    logger_name = name or _DEFAULT_LOGGER_NAME

    if logger_name not in _CONFIGURED_LOGGERS:
        setup_logger(logger_name)

    return logging.getLogger(logger_name)


setup_logger()
