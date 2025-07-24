from __future__ import annotations

import mlflow
import mlflow.models
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common.constants import (
    CLIP_MAX_LENGTH,
    DEFAULT_CONTEXT_DIM,
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
)
from ..models.encoders.clip import CLIPTextEncoder
from ..models.encoders.vae import VAEEncoder
from ..models.stable_diffusion.diffusion import StableDiffusion
from ..models.stable_diffusion.scheduler import DDPMScheduler
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import get_logger
from ..utils.mlflow import ModelRegistry

logger = get_logger(__name__)


class StableDiffusionTrainer:
    """Stable Diffusion trainer with MLflow integration"""

    def __init__(
        self,
        config: dict[str, dict[str, str | int | float | bool]],
        device: str = "cuda",
        is_validation: bool = False,
    ):
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.resume_batch_idx = 0

        self._init_models()
        self._init_training_components()

        self.model_registry = ModelRegistry(config["mlflow"]["tracking_uri"])
        self.checkpoint_manager = CheckpointManager(
            save_dir=config["experiment"]["save_dir"],
            max_checkpoints=config["experiment"]["max_checkpoints"],
        )

        if not is_validation:
            self._init_mlflow()

    def _init_models(self) -> None:
        """Initialize all models"""
        model_config = self.config["model"]

        self.diffusion_model = StableDiffusion(
            h_dim=model_config["stable_diffusion"]["h_dim"],
            n_head=model_config["stable_diffusion"]["n_head"],
            time_dim=model_config["stable_diffusion"]["time_dim"],
        ).to(self.device)

        self.text_encoder = CLIPTextEncoder(
            config=model_config["clip"], device=self.device
        )

        self.vae_encoder = VAEEncoder(config=model_config["vae"]).to(self.device)

        self.generator = torch.Generator(device=self.device)
        self.scheduler = DDPMScheduler(
            random_generator=self.generator,
            train_timesteps=model_config["stable_diffusion"]["num_train_timesteps"],
            beta_start=model_config["stable_diffusion"]["beta_start"],
            beta_end=model_config["stable_diffusion"]["beta_end"],
        )

    def _init_training_components(self) -> None:
        """Initialize optimizer, scheduler, etc."""
        training_config = self.config["training"]

        self.optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(training_config["epochs"]),
            eta_min=float(training_config["eta_min"]),
        )

        self.scaler = GradScaler() if training_config["mixed_precision"] else None
        self.criterion = nn.MSELoss()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["experiment"]["name"])
            logger.info(f"MLflow experiment: {self.config['experiment']['name']}")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")

    def train_epoch(self, train_loader: DataLoader, start_batch: int = 0) -> float:
        """Train for one epoch"""
        self.diffusion_model.train()
        epoch_loss = 0.0

        remaining_batches = len(train_loader) - start_batch

        train_iter = iter(train_loader)
        for _ in range(start_batch):
            next(train_iter)
        progress_iter = enumerate(train_iter, start=start_batch)

        progress_bar = tqdm(
            progress_iter,
            desc=f"Epoch {self.current_epoch + 1}",
            total=remaining_batches,
        )

        for batch_idx, (images, prompts) in progress_bar:
            loss = self._training_step(images, prompts)
            epoch_loss += loss

            # Log metrics
            if self.global_step % self.config["experiment"]["log_every"] == 0:
                mlflow.log_metrics(
                    {
                        "train_loss": loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": self.current_epoch + 1,
                        "step": self.global_step,
                    },
                    step=self.global_step,
                )

            # Save checkpoint
            if self.global_step % self.config["experiment"]["save_every"] == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.diffusion_model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    best_loss=self.best_loss,
                    scaler=self.scaler,
                    batch_idx=batch_idx + 1,  # Save next batch index
                )

            progress_bar.set_postfix(
                {
                    "batch_loss": f"{loss:.5f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        return epoch_loss / remaining_batches if remaining_batches > 0 else 0.0

    def _training_step(self, images: torch.Tensor, prompts: list[str]) -> float:
        """Single training step"""
        images = images.to(self.device)

        # Encode to latent space
        with torch.no_grad():
            latents = self.vae_encoder.encode(images)
            text_embeddings = self.text_encoder(prompts)

        # Sample timesteps and add noise
        timesteps = torch.randint(
            0,
            self.scheduler.total_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        )
        noisy_latents, noise = self.scheduler.add_noise(latents, timesteps)

        # Forward pass
        if self.scaler:
            with autocast("cuda", dtype=torch.float16):
                noise_pred = self.diffusion_model(
                    noisy_latents, text_embeddings, timesteps
                )
                loss = self.criterion(noise_pred, noise)
        else:
            noise_pred = self.diffusion_model(noisy_latents, text_embeddings, timesteps)
            loss = self.criterion(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.global_step += 1
        return loss.item()

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.diffusion_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, prompts in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device)

                latents = self.vae_encoder.encode(images)
                text_embeddings = self.text_encoder(prompts)

                timesteps = torch.randint(
                    0,
                    self.scheduler.total_train_timesteps,
                    (latents.shape[0],),
                    device=self.device,
                )
                noisy_latents, noise = self.scheduler.add_noise(latents, timesteps)

                noise_pred = self.diffusion_model(
                    noisy_latents, text_embeddings, timesteps
                )
                loss = self.criterion(noise_pred, noise)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model, optimizer, scheduler, scaler, and training state from checkpoint for resume training."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            filepath,
            self.diffusion_model,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            device=self.device,
        )
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)
        self.resume_batch_idx = checkpoint.get("batch_idx", 0)
        logger.info(
            f"Checkpoint loaded: epoch={self.current_epoch}, global_step={self.global_step}, "
            f"best_loss={self.best_loss}, batch_idx={self.resume_batch_idx}"
        )

    def save_model(self, filepath: str) -> None:
        """Save only the model state dict to local file"""
        torch.save(self.diffusion_model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def register_model_to_mlflow(
        self, model_name: str, is_best: bool = False, is_final: bool = False
    ) -> str | None:
        """Register model to MLflow Model Registry with proper versioning and aliases.

        Args:
            model_name: Name for the registered model
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model after training

        Returns:
            Model version string if successful, None otherwise
        """
        try:
            # Create model signature for MLflow
            signature = self._create_model_signature()

            # Prepare tags
            tags = {
                "epoch": str(self.current_epoch + 1),
                "step": str(self.global_step),
                "best_loss": str(self.best_loss),
                "model_type": "candidate",  # default
            }

            if is_best:
                tags["model_type"] = "best"
            elif is_final:
                tags["model_type"] = "final"

            alias = "candidate"  # default
            if is_best:
                alias = "challenger"
            elif is_final:
                alias = "champion"

            version = self.model_registry.register_model(
                model=self.diffusion_model,
                model_name=model_name,
                signature=signature,
                tags=tags,
                alias=alias,
            )

            logger.info(
                f"Model registered to MLflow: {model_name} v{version} with alias '{alias}'"
            )
            return version

        except Exception as e:
            logger.error(f"Failed to register model to MLflow: {e}")
            return None

    def _create_model_signature(self):
        """Create MLflow model signature for the diffusion model."""
        try:
            sample_latent = torch.randn(
                1, DEFAULT_LATENT_CHANNELS, DEFAULT_LATENT_HEIGHT, DEFAULT_LATENT_WIDTH
            ).to(self.device)
            sample_context = torch.randn(1, CLIP_MAX_LENGTH, DEFAULT_CONTEXT_DIM).to(
                self.device
            )
            sample_timestep = torch.randint(0, 1000, (1,)).to(self.device)

            # Get sample output
            with torch.no_grad():
                sample_output = self.diffusion_model(
                    sample_latent, sample_context, sample_timestep
                )

            # Create signature
            input_schema = {
                "latent": sample_latent.cpu().numpy(),
                "context": sample_context.cpu().numpy(),
                "timestep": sample_timestep.cpu().numpy(),
            }

            output_schema = sample_output.cpu().numpy()

            return mlflow.models.infer_signature(input_schema, output_schema)

        except Exception as e:
            logger.warning(f"Failed to create model signature: {e}")
            return None

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> None:
        """Run the full training loop"""
        logger.info(
            f"Starting training for {self.config['training']['epochs']} epochs..."
        )

        with mlflow.start_run(run_name=f"{self.config['experiment']['name']}_run"):
            try:
                mlflow.log_params(
                    {
                        "learning_rate": self.config["training"]["learning_rate"],
                        "epochs": self.config["training"]["epochs"],
                        "batch_size": self.config["training"].get(
                            "batch_size", "unknown"
                        ),
                        "mixed_precision": self.config["training"]["mixed_precision"],
                        "weight_decay": self.config["training"].get(
                            "weight_decay", 0.01
                        ),
                        "eta_min": self.config["training"]["eta_min"],
                        "model_h_dim": self.config["model"]["stable_diffusion"][
                            "h_dim"
                        ],
                        "model_n_head": self.config["model"]["stable_diffusion"][
                            "n_head"
                        ],
                        "num_train_timesteps": self.config["model"]["stable_diffusion"][
                            "num_train_timesteps"
                        ],
                        "beta_start": self.config["model"]["stable_diffusion"][
                            "beta_start"
                        ],
                        "beta_end": self.config["model"]["stable_diffusion"][
                            "beta_end"
                        ],
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log MLflow parameters: {e}")

            total_epochs = self.config["training"]["epochs"]

            for epoch in range(self.current_epoch, total_epochs):
                self.current_epoch = epoch

                # Train epoch
                start_batch = (
                    self.resume_batch_idx if epoch == self.current_epoch else 0
                )
                train_loss = self.train_epoch(train_loader, start_batch)
                self.resume_batch_idx = 0

                # Validation
                val_loss = None
                if val_loader is not None:
                    val_loss = self.validate(val_loader)

                    try:
                        mlflow.log_metrics(
                            {
                                "val_loss": val_loss,
                                "train_epoch_loss": train_loss,
                            },
                            step=self.global_step,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log validation metrics: {e}")

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.checkpoint_manager.save_checkpoint(
                            model=self.diffusion_model,
                            optimizer=self.optimizer,
                            lr_scheduler=self.lr_scheduler,
                            epoch=epoch,
                            global_step=self.global_step,
                            best_loss=self.best_loss,
                            scaler=self.scaler,
                            batch_idx=0,
                            is_best=True,
                        )

                        model_name = self.config["experiment"].get(
                            "model_name", "emoji-stable-diffusion"
                        )
                        self.register_model_to_mlflow(
                            model_name=model_name, is_best=True
                        )
                        logger.info(
                            f"New best model registered! Validation loss: {val_loss:.6f}"
                        )

                self.lr_scheduler.step()

                logger.info(
                    f"Epoch {epoch + 1} completed: train_loss={train_loss:.6f}"
                    + (f", val_loss={val_loss:.6f}" if val_loss else "")
                )

        # Register final model to MLflow
        model_name = self.config["experiment"].get(
            "model_name", "emoji-stable-diffusion"
        )
        self.register_model_to_mlflow(model_name=model_name, is_final=True)

        logger.info("Training completed!")
        return self.global_step
