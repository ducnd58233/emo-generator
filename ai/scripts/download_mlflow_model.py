from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_mlflow_model(
    model_name: str,
    output_dir: str,
    alias: str,
    version: int | None = None,
    tracking_uri: str | None = None,
) -> str:
    """
    Download a model from the MLflow Model Registry.

    Args:
        model_name: Name of the registered model.
        alias_or_version: Alias (e.g. 'champion', 'challenger', 'candidate') or version number.
        output_dir: Directory to save the downloaded model artifacts.
        tracking_uri: MLflow tracking URI (optional).

    Returns:
        Path to the downloaded model directory.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}@{alias}"

    print(f"Downloading model from MLflow Registry: {model_uri}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Output directory: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download artifacts
    local_path = mlflow.artifacts.download_artifacts(
        model_uri, dst_path=str(output_path)
    )
    print(f"Model downloaded to: {local_path}")
    return str(local_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download a model from the MLflow Model Registry."
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Registered model name in MLflow."
    )
    parser.add_argument(
        "--alias",
        type=str,
        required=True,
        help="Model alias (e.g. 'champion', 'challenger', 'candidate').",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=False,
        help="Model version number.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/mlflow_registry",
        help="Directory to save the downloaded model artifacts.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (optional).",
    )

    args = parser.parse_args()

    try:
        download_mlflow_model(
            model_name=args.model_name,
            alias=args.alias,
            version=args.version,
            output_dir=args.output_dir,
            tracking_uri=args.tracking_uri,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
