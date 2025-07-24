from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import streamlit as st
import torch
from PIL import Image

from ..common.constants import (
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
)
from ..inference.generator import EmojiGenerator
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.model import get_device

logger = get_logger(__name__)


@st.cache_resource
def load_model_from_mlflow_registry(
    model_name: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> tuple[EmojiGenerator, dict[str, Any]]:
    """Load model from MLflow registry with Streamlit caching."""
    try:
        device = get_device()

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"Loading model {model_name}@{alias} from MLflow registry...")

        config_path = Path("config/model.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_config(config_path)
        logger.info("Loaded model configuration")

        local_model_path = Path("models/mlflow_registry")
        if not local_model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {local_model_path}")

        model_file = local_model_path / "data" / "model.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        generator = EmojiGenerator.from_pretrained(
            model_path=str(model_file),
            config=config,
            device=device,
        )

        logger.info("Model loaded successfully!")
        return generator, config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@st.cache_resource
def load_model_from_checkpoint(
    checkpoint_path: str,
) -> tuple[EmojiGenerator, dict[str, Any]]:
    """Load model from local checkpoint with Streamlit caching."""
    try:
        device = get_device()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load configuration
        config_path = Path("config/model.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_config(config_path)
        logger.info("Loaded model configuration")

        generator = EmojiGenerator.from_pretrained(
            model_path=str(checkpoint_path),
            config=config,
            device=device,
        )

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        return generator, config

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


class StreamlitModelManager:
    """Manages model loading and caching for Streamlit applications."""

    def __init__(self):
        self.device = get_device()
        self.generator: EmojiGenerator | None = None
        self.config: dict[str, Any] | None = None

    def load_model_from_registry(
        self,
        model_name: str,
        alias: str = "champion",
        tracking_uri: str | None = None,
    ) -> EmojiGenerator:
        """Load model from MLflow registry using cached function."""
        try:
            self.generator, self.config = load_model_from_mlflow_registry(
                model_name, alias, tracking_uri
            )
            return self.generator
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            raise

    def load_model_from_checkpoint(self, checkpoint_path: str) -> EmojiGenerator:
        """Load model from local checkpoint using cached function."""
        try:
            self.generator, self.config = load_model_from_checkpoint(checkpoint_path)
            return self.generator
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            raise

    def generate_emoji(
        self,
        prompt: str,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        latent_height: int = DEFAULT_LATENT_HEIGHT,
        latent_width: int = DEFAULT_LATENT_WIDTH,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """Generate emoji image from text prompt."""
        if self.generator is None:
            raise RuntimeError("Model not loaded. Please load a model first.")

        try:
            with st.spinner("Generating emoji..."):
                image = self.generator.generate(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    seed=seed,
                )
            return image

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            st.error(f"Generation failed: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.generator is not None

    def get_device_info(self) -> str:
        """Get device information for display."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"CUDA - {gpu_name}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        else:
            return "CPU"
