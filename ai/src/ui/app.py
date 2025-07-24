from __future__ import annotations

import io
import os
from typing import Any

import streamlit as st

from ..common.constants import (
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    MLFLOW_CHAMPION_ALIAS,
)
from ..utils.logging import get_logger
from .streamlit_utils import StreamlitModelManager

logger = get_logger(__name__)

EMO_GENERATOR_DEV_MODE = os.environ.get("EMO_GENERATOR_DEV_MODE", "1") == "1"
EMO_GENERATOR_MODEL_PATH = os.environ.get(
    "EMO_GENERATOR_MODEL_PATH", "models/mlflow_registry/data/model.pth"
)


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Emoji Generator",
        page_icon=":art:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the application header."""
    st.title("Emoji Generator")
    st.markdown(
        """
        Generate custom emoji images from text prompts using **Stable Diffusion**.

        Simply enter a description of the emoji you want to create, adjust the parameters if needed,
        and click **Generate** to create your custom emoji!
        """
    )
    st.divider()


def render_sidebar(model_manager: StreamlitModelManager) -> dict[str, Any]:
    """Render the sidebar with model loading and generation parameters."""
    with st.sidebar:
        st.header("Settings")

        # Model Loading Section
        st.subheader("Model Configuration")

        if EMO_GENERATOR_DEV_MODE:
            model_source = st.radio(
                "Model Source",
                ["MLflow Registry", "Local Checkpoint"],
                help="Choose whether to load from MLflow Model Registry or a local checkpoint file",
            )
        else:
            model_source = "Local Checkpoint"

        if model_source == "MLflow Registry":
            col1, col2 = st.columns(2)
            with col1:
                model_name = st.text_input(
                    "Model Name",
                    value="emoji-stable-diffusion",
                    help="Name of the registered model in MLflow",
                )
            with col2:
                alias = st.selectbox(
                    "Model Alias",
                    [MLFLOW_CHAMPION_ALIAS, "challenger", "candidate"],
                    help="Model version alias",
                )

            tracking_uri = st.text_input(
                "MLflow Tracking URI",
                value="http://localhost:5000",
                help="MLflow tracking server URI (optional)",
            )

            if st.button("Load Model from Registry", type="primary"):
                with st.spinner("Loading model from MLflow registry..."):
                    try:
                        model_manager.load_model_from_registry(
                            model_name=model_name,
                            alias=alias,
                            tracking_uri=tracking_uri,
                        )
                        st.success("Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")

        else:  # Local Checkpoint
            if EMO_GENERATOR_DEV_MODE:
                checkpoint_path = st.text_input(
                    "Checkpoint Path",
                    value=EMO_GENERATOR_MODEL_PATH,
                    help="Path to the local model checkpoint file",
                )
                if st.button("Load Checkpoint", type="primary"):
                    with st.spinner("Loading model from checkpoint..."):
                        try:
                            model_manager.load_model_from_checkpoint(checkpoint_path)
                            st.success("Model loaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")
            else:
                checkpoint_path = EMO_GENERATOR_MODEL_PATH
                if not st.session_state.get("model_loaded_prod", False):
                    with st.spinner("Loading model from checkpoint..."):
                        try:
                            model_manager.load_model_from_checkpoint(checkpoint_path)
                            st.session_state["model_loaded_prod"] = True
                            st.success("Model loaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")

        # Display model status
        if model_manager.is_model_loaded():
            st.success("Model Ready")
            st.info(f"**Device:** {model_manager.get_device_info()}")
        else:
            st.warning("No Model Loaded")

        st.divider()

        # Generation Parameters Section
        st.subheader("Generation Parameters")

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=DEFAULT_SEED,
            help="Random seed for reproducible generation",
        )

        num_inference_steps = st.slider(
            "Inference Steps",
            min_value=10,
            max_value=200,
            value=DEFAULT_NUM_INFERENCE_STEPS,
            step=10,
            help="Number of denoising steps (more steps = higher quality, slower)",
        )

        # Advanced parameters (collapsible)
        with st.expander("Advanced Parameters"):
            latent_height = st.slider(
                "Latent Height",
                min_value=2,
                max_value=16,
                value=DEFAULT_LATENT_HEIGHT,
                help="Height of the latent space (affects output resolution)",
            )

            latent_width = st.slider(
                "Latent Width",
                min_value=2,
                max_value=16,
                value=DEFAULT_LATENT_WIDTH,
                help="Width of the latent space (affects output resolution)",
            )

    return {
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "latent_height": latent_height,
        "latent_width": latent_width,
    }


def render_generation_interface(
    model_manager: StreamlitModelManager, params: dict[str, Any]
) -> None:
    """Render the main generation interface."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Prompt")
        prompt = st.text_area(
            "Describe your emoji:",
            value="a happy blob emoji with sparkling eyes",
            height=100,
            help="Describe the emoji you want to generate. Be specific for better results!",
            placeholder="Example: a confused cat emoji with big eyes, a sleeping panda emoji, a rocket emoji with flames...",
        )

        # Prompt suggestions
        st.markdown("**Suggestions:**")
        suggestion_cols = st.columns(3)
        suggestions = [
            "a confused blob emoji with wide eyes",
            "a sleepy cat emoji with droopy eyes",
            "a magical unicorn emoji with rainbow mane",
            "a robot emoji with glowing blue eyes",
            "a pizza emoji with melted cheese",
            "a dancing banana emoji with arms",
        ]

        for i, suggestion in enumerate(suggestions):
            col = suggestion_cols[i % 3]
            if col.button(f"{suggestion[:20]}...", key=f"suggestion_{i}"):
                st.session_state.prompt = suggestion
                st.rerun()

        if st.button(
            "Generate Emoji",
            type="primary",
            disabled=not model_manager.is_model_loaded(),
        ):
            if not prompt.strip():
                st.error("Please enter a prompt to generate an emoji.")
                return

            try:
                generated_image = model_manager.generate_emoji(
                    prompt=prompt,
                    **params,
                )

                # Store in session state for persistence
                st.session_state.generated_image = generated_image
                st.session_state.last_prompt = prompt

                st.success("Emoji generated successfully!")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                logger.error(f"Generation error: {e}")

    with col2:
        st.subheader("Generated Emoji")

        if (
            hasattr(st.session_state, "generated_image")
            and st.session_state.generated_image
        ):
            # Display the generated image
            st.image(
                st.session_state.generated_image,
                caption=f"Prompt: {st.session_state.last_prompt}",
                use_container_width=True,
            )

            # Download button
            img_buffer = io.BytesIO()
            st.session_state.generated_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            st.download_button(
                label="Download Emoji",
                data=img_buffer.getvalue(),
                file_name="generated_emoji.png",
                mime="image/png",
                type="secondary",
            )

            # Image info
            with st.expander("Image Info"):
                st.write(f"**Size:** {st.session_state.generated_image.size}")
                st.write(f"**Mode:** {st.session_state.generated_image.mode}")
                st.write(f"**Prompt:** {st.session_state.last_prompt}")

        else:
            st.info("Your generated emoji will appear here")


def render_footer() -> None:
    """Render the application footer."""
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 0.5rem;'>
            <p>Built with Streamlit, Stable Diffusion, and MLflow</p>
            <p><small>Emoji Generator v1.0 | Generate unique emojis from text prompts</small></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_app():
    """Main application entry point."""
    setup_page_config()

    # Initialize model manager
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = StreamlitModelManager()

    model_manager = st.session_state.model_manager

    # Render UI components
    render_header()

    # Render sidebar and get parameters
    params = render_sidebar(model_manager)

    # Render main generation interface
    render_generation_interface(model_manager, params)

    # Render footer
    render_footer()
