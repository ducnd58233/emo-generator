import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from src.inference.generator import EmojiGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

sys.path.append(str(Path(__file__).parent.parent))
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Register local checkpoint to MLflow Model Registry"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Path to model config yaml",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name for MLflow registered model",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="candidate",
        choices=["candidate", "champion", "archived"],
        help="Model alias to set (replaces deprecated stages)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to load model"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Load config
    config = load_config(args.config)

    # Build model (same as inference)
    generator = EmojiGenerator.from_pretrained(
        args.checkpoint_path, config, args.device
    )
    model = generator.diffusion_model

    with mlflow.start_run(run_name=f"register_{args.model_name}") as run:
        mlflow.pytorch.log_model(
            model, name=args.model_name, registered_model_name=args.model_name
        )

        mlflow.log_artifact(args.config, artifact_path="config")
        logger.info(f"Logged model to MLflow run: {run.info.run_id}")

    # Register model to Model Registry
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(args.model_name)
        if latest_versions:
            model_version = latest_versions[0].version
            logger.info(
                f"Registered model: {args.model_name}, version: {model_version}"
            )

            # Set alias using modern approach
            if args.alias:
                client.set_registered_model_alias(
                    args.model_name, args.alias, model_version
                )
                logger.info(
                    f"Set alias '{args.alias}' for model {args.model_name} version {model_version}"
                )
        else:
            logger.error("Model registration failed or not found.")

    except Exception as e:
        logger.error(f"Failed to set model alias: {e}")
        try:
            all_versions = client.search_model_versions(f"name='{args.model_name}'")
            if all_versions:
                latest_version = max(all_versions, key=lambda x: int(x.version))
                logger.info(f"Found model version: {latest_version.version}")
                if args.alias:
                    client.set_registered_model_alias(
                        args.model_name, args.alias, latest_version.version
                    )
                    logger.info(
                        f"Set alias '{args.alias}' for model {args.model_name} version {latest_version.version}"
                    )
        except Exception as fallback_e:
            logger.error(f"Fallback also failed: {fallback_e}")


if __name__ == "__main__":
    main()
