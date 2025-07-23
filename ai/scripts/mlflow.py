import subprocess
import sys

from src.utils.logging import get_logger

logger = get_logger(__name__)


def start_mlflow_server():
    """Start MLflow tracking server"""
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "./mlruns",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
    ]

    logger.info("Starting MLflow server...")
    subprocess.run(cmd)


if __name__ == "__main__":
    start_mlflow_server()
