import argparse
import os
import sys

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from src.inference.generator import EmojiGenerator
from src.utils.config import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        "--stage",
        type=str,
        default="None",
        help="Stage to promote (e.g. Staging, Production)",
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

    # Log model to MLflow
    with mlflow.start_run(run_name=f"register_{args.model_name}") as run:
        artifact_path = "model"
        mlflow.pytorch.log_model(
            model, artifact_path=artifact_path, registered_model_name=args.model_name
        )
        # Log config yaml as artifact for reproducibility
        mlflow.log_artifact(args.config, artifact_path="config")
        print(f"Logged model to MLflow run: {run.info.run_id}")

    # Register model to Model Registry
    client = MlflowClient()
    latest_version = client.get_latest_versions(args.model_name, stages=["None"])
    if latest_version:
        model_version = latest_version[0].version
        print(f"Registered model: {args.model_name}, version: {model_version}")
        if args.stage and args.stage.lower() != "none":
            client.transition_model_version_stage(
                args.model_name, model_version, args.stage
            )
            print(
                f"Promoted model {args.model_name} version {model_version} to stage {args.stage}"
            )
    else:
        print("Model registration failed or not found.")


if __name__ == "__main__":
    main()
