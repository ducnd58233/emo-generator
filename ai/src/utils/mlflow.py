import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def load_model_from_registry(model_name: str, stage: str = "Production"):
    """Load model from MLflow registry"""
    client = MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=[stage])[0]
    model_uri = f"models:/{model_name}/{model_version.version}"
    return mlflow.pytorch.load_model(model_uri)


def list_model_versions(model_name: str):
    """List all versions of a registered model"""
    client = MlflowClient()
    return client.search_model_versions(f"name='{model_name}'")


def promote_model(model_name: str, version: str, stage: str):
    """Promote model to different stage"""
    client = MlflowClient()
    client.transition_model_version_stage(model_name, version, stage)
