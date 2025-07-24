from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from src.inference.generator import EmojiGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

from ai.src.common.constants import MLFLOW_CANDIDATE_ALIAS

sys.path.append(str(Path(__file__).parent.parent))
logger = get_logger(__name__)


def register_model_to_mlflow(
    checkpoint_path: str,
    config_path: str,
    model_name: str,
    alias: str = MLFLOW_CANDIDATE_ALIAS,
    device: str = "cpu",
    mlflow_tracking_uri: str = "http://localhost:5000",
) -> str | None:
    """Register a local checkpoint to MLflow Model Registry.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        config_path: Path to model configuration file
        model_name: Name for the registered model in MLflow
        alias: Model alias to set (candidate, champion, etc.)
        device: Device to load the model on
        mlflow_tracking_uri: MLflow tracking server URI

    Returns:
        Model version string if successful, None otherwise
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Load configuration and model
        config = load_config(config_path)
        generator = EmojiGenerator.from_pretrained(checkpoint_path, config, device)
        model = generator.diffusion_model

        # Create MLflow run and log model
        with mlflow.start_run(run_name=f"register_{model_name}") as run:
            # Log the model with metadata
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            # Log configuration as artifact
            mlflow.log_artifact(config_path, artifact_path="config")
            logger.info(f"Logged model to MLflow run: {run.info.run_id}")

        # Get model version and set alias
        client = MlflowClient()

        try:
            # Get latest model version
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                logger.error("No model versions found after registration")
                return None

            model_version = latest_versions[0].version
            logger.info(f"Registered model: {model_name}, version: {model_version}")

            # Set alias using modern approach
            client.set_registered_model_alias(model_name, alias, model_version)
            logger.info(
                f"Set alias '{alias}' for model {model_name} version {model_version}"
            )

            return model_version

        except Exception as e:
            logger.warning(f"Failed to set alias using latest_versions: {e}")

            # Fallback: search for model versions
            try:
                all_versions = client.search_model_versions(f"name='{model_name}'")
                if not all_versions:
                    logger.error("No model versions found in fallback search")
                    return None

                # Get the latest version by number
                latest_version = max(all_versions, key=lambda x: int(x.version))
                model_version = latest_version.version

                logger.info(f"Found model version in fallback: {model_version}")

                # Set alias
                client.set_registered_model_alias(model_name, alias, model_version)
                logger.info(
                    f"Set alias '{alias}' for model {model_name} version {model_version}"
                )

                return model_version

            except Exception as fallback_e:
                logger.error(f"Fallback registration also failed: {fallback_e}")
                return None

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return None


def main() -> None:
    """Main function for MLflow model registration script."""
    parser = argparse.ArgumentParser(
        description="Register local checkpoint to MLflow Model Registry"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Path to model config YAML file",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name for MLflow registered model",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=MLFLOW_CANDIDATE_ALIAS,
        choices=["candidate", "champion", "challenger", "archived"],
        help="Model alias to set (modern approach instead of stages)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to load model (cpu/cuda)"
    )

    args = parser.parse_args()

    # Validate inputs
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Register model
    logger.info(f"Registering model '{args.model_name}' from {checkpoint_path}")

    version = register_model_to_mlflow(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        model_name=args.model_name,
        alias=args.alias,
        device=args.device,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )

    if version:
        logger.info(f"✅ Model registration completed successfully! Version: {version}")
    else:
        logger.error("❌ Model registration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
