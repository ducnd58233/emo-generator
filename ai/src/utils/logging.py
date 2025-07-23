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
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        today = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"{today}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger by name, using the global setup.
    Args:
        name: Logger name (defaults to root logger if None).
    Returns:
        Logger instance.
    """
    return logging.getLogger(name or _DEFAULT_LOGGER_NAME)


# Optionally, set up the root logger at import time
setup_logger()
