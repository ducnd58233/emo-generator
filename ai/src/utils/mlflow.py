from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import mlflow.pytorch
from huggingface_hub import HfApi
from mlflow.tracking import MlflowClient

from .logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Unified model registry for MLflow and HuggingFace"""

    def __init__(self, mlflow_uri: str):
        self.mlflow_client = MlflowClient(mlflow_uri)
        self.hf_api = HfApi()

    def load_model_from_registry(self, model_name: str, alias: str = "champion"):
        """Load model from MLflow registry using model alias (modern approach)"""
        try:
            model_uri = f"models:/{model_name}@{alias}"
            logger.info(f"Loading model from URI: {model_uri}")
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}@{alias}: {e}")
            # Fallback to version-based loading if alias not found
            try:
                latest_version = self.mlflow_client.get_latest_versions(model_name)[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                logger.info(f"Fallback: Loading model from URI: {model_uri}")
                return mlflow.pytorch.load_model(model_uri)
            except Exception as fallback_e:
                logger.error(f"Fallback failed: {fallback_e}")
                raise fallback_e

    def register_model(
        self,
        model,
        model_name: str,
        signature=None,
        tags: dict[str, str] | None = None,
        alias: str = "candidate",
    ) -> str:
        """Register model to MLflow Model Registry"""
        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                signature=signature,
                registered_model_name=model_name,
            )

            # Get model version
            model_version = model_info.registered_model_version

            # Set alias for the model version
            self.mlflow_client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=model_version,
            )

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        name=model_name,
                        version=model_version,
                        key=key,
                        value=value,
                    )

            logger.info(
                f"Model registered: {model_name} v{model_version} with alias '{alias}'"
            )
            return model_version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def download_model_from_registry(
        self,
        model_name: str,
        alias: str = "champion",
        dst_path: str | None = "models/mlflow_registry",
    ) -> Path:
        """Download model from MLflow registry to local path"""
        try:
            model_uri = f"models:/{model_name}@{alias}"
            downloaded_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=dst_path
            )
            logger.info(f"Model downloaded to: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def get_model_info(self, model_name: str, alias: str = "champion") -> dict:
        """Get model information from registry"""
        try:
            model_version = self.mlflow_client.get_model_version_by_alias(
                name=model_name, alias=alias
            )
            return {
                "name": model_version.name,
                "version": model_version.version,
                "alias": alias,
                "run_id": model_version.run_id,
                "status": model_version.status,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise

    def upload_to_huggingface(
        self,
        model_name: str,
        hf_repo_id: str,
        alias: str = "champion",
        token: str | None = None,
    ) -> None:
        """Push model from MLflow registry to HuggingFace Hub"""
        try:
            # Download model from MLflow
            local_path = self.download_model_from_registry(model_name, alias)

            # Push to HuggingFace Hub
            self.hf_api.upload_folder(
                folder_path=local_path,
                repo_id=hf_repo_id,
                token=token,
                commit_message=f"Upload {model_name} v{alias} from MLflow",
            )

            # Clean up local files
            if local_path.exists():
                shutil.rmtree(local_path)

            logger.info(f"Model pushed to HuggingFace: {hf_repo_id}")

        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
            raise
