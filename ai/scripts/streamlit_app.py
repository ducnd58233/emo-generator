from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_model_files() -> None:
    """Check if model files are available and provide guidance."""
    model_path = Path("models/mlflow_registry/data/model.pth")

    if not model_path.exists():
        logger.error("Model file not found - you can load one through the UI")
        logger.error("To download from MLflow: python -m scripts.download_mlflow_model")


def main() -> None:
    """Main application entry point."""
    logger.info("Starting Emoji Generator...")

    check_model_files()

    from src.ui.app import run_app

    run_app()


if __name__ == "__main__":
    main()
