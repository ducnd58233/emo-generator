import shutil
from pathlib import Path
from typing import Any, Dict, Optional

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
        model: Any,
        model_name: str,
        signature: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        alias: str = "candidate",
    ) -> str:
        """Register model in MLflow with modern alias approach"""
        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                signature=signature,
                registered_model_name=model_name,
            )

            version = model_info.registered_model_version
            logger.info(f"Registered model {model_name} version {version}")

            if tags:
                for key, value in tags.items():
                    self.mlflow_client.set_model_version_tag(
                        model_name, version, key, value
                    )

            if alias:
                self.mlflow_client.set_registered_model_alias(
                    model_name, alias, version
                )
                logger.info(
                    f"Set alias '{alias}' for model {model_name} version {version}"
                )

            return version

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise

    def download_artifacts_from_registry(
        self,
        model_name: str,
        alias: str = "champion",
        dst_path: Optional[str] = "models/mlflow_registry",
    ) -> str:
        """Download model artifacts from MLflow registry using model alias"""
        try:
            model_uri = f"models:/{model_name}@{alias}"
            logger.info(f"Downloading artifacts from URI: {model_uri}")
            return mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=dst_path
            )
        except Exception as e:
            logger.error(f"Failed to download artifacts for {model_name}@{alias}: {e}")
            try:
                latest_version = self.mlflow_client.get_latest_versions(model_name)[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                logger.info(f"Fallback: Downloading artifacts from URI: {model_uri}")
                return mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri, dst_path=dst_path
                )
            except Exception as fallback_e:
                logger.error(f"Fallback download failed: {fallback_e}")
                raise fallback_e

    def upload_to_huggingface(
        self,
        model_name: str,
        model_path: str,
        hf_repo_id: str,
        alias: str = "champion",
        token: Optional[str] = None,
    ) -> None:
        """Upload MLflow model to HuggingFace Hub using alias"""
        try:
            if not model_path:
                model_path = self.download_artifacts_from_registry(model_name, alias)

            logger.info(f"Uploading {model_path} to HuggingFace repo '{hf_repo_id}'")

            self.hf_api.upload_folder(
                folder_path=str(model_path),
                repo_id=hf_repo_id,
                repo_type="model",
                token=token,
                commit_message=f"Upload {model_name}@{alias} from MLflow",
            )

            logger.info(f"Successfully uploaded to https://huggingface.co/{hf_repo_id}")

        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            raise
        finally:
            if "model_path" in locals() and model_path and Path(model_path).exists():
                shutil.rmtree(model_path)

    def get_model_by_alias(self, model_name: str, alias: str = "champion"):
        """Get model version info by alias"""
        try:
            return self.mlflow_client.get_model_version_by_alias(model_name, alias)
        except Exception as e:
            logger.warning(f"Alias '{alias}' not found for model {model_name}: {e}")
            return None

    def set_model_alias(self, model_name: str, alias: str, version: str) -> None:
        """Set an alias for a specific model version"""
        try:
            self.mlflow_client.set_registered_model_alias(model_name, alias, version)
            logger.info(f"Set alias '{alias}' for model {model_name} version {version}")
        except Exception as e:
            logger.error(
                f"Failed to set alias '{alias}' for {model_name} version {version}: {e}"
            )
            raise
