from typing import Any, Dict

import mlflow
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
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

        self._init_mlflow()

    def _init_models(self):
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

    def _init_training_components(self):
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

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["experiment"]["name"])
        logger.info(f"MLflow tracking URI: {self.config['mlflow']['tracking_uri']}")
        logger.info(f"MLflow experiment: {self.config['experiment']['name']}")

    def train_epoch(self, train_loader: DataLoader, start_batch: int = 0) -> float:
        """Train for one epoch."""
        self.diffusion_model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']}",
        )

        progress_iter = iter(progress_bar)
        for _ in range(start_batch):
            try:
                next(progress_iter)
            except StopIteration:
                break

        for batch_idx, (images, prompts) in enumerate(progress_iter, start=start_batch):
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

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _training_step(self, images: torch.Tensor, prompts: list) -> float:
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
        """
        Load model, optimizer, scheduler, scaler, and training state from checkpoint for resume training.
        Args:
            filepath: Path to checkpoint file.
        """
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
            f"Checkpoint loaded: epoch={self.current_epoch}, global_step={self.global_step}, best_loss={self.best_loss}, batch_idx={self.resume_batch_idx}"
        )

    def save_model(self, is_best: bool = False, is_final: bool = False) -> None:
        """Save model to registry"""
        signature = self._create_model_signature()

        tags = {
            "epoch": str(self.current_epoch + 1),
            "step": str(self.global_step),
            "best_loss": str(self.best_loss),
        }
        if is_best:
            tags.update({"model_type": "best"})
        if is_final:
            tags.update({"model_type": "final"})

        alias = "candidate"
        if is_best:
            alias = "challenger"
        elif is_final:
            alias = "champion"

        try:
            version = self.model_registry.register_model(
                model=self.diffusion_model,
                model_name=self.config["experiment"]["model_name"],
                signature=signature,
                tags=tags,
                alias=alias,
            )
            logger.info(f"Registered model version {version} with alias '{alias}'")
        except Exception as e:
            logger.error(f"Failed to save model to MLflow: {e}")

    def _create_model_signature(self):
        """Create MLflow model signature"""
        sample_input = torch.randn(1, 4, 4, 4).to(self.device)
        sample_context = torch.randn(1, 77, 512).to(self.device)
        sample_time = torch.randint(0, 1000, (1,)).to(self.device)

        with torch.no_grad():
            sample_output = self.diffusion_model(
                sample_input, sample_context, sample_time
            )

        return mlflow.models.infer_signature(
            {
                "latent": sample_input.cpu().numpy(),
                "context": sample_context.cpu().numpy(),
                "timestep": sample_time.cpu().numpy(),
            },
            sample_output.cpu().numpy(),
        )

    def train(self, train_loader: DataLoader, val_loader: Any = None) -> None:
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

            batches_per_epoch = len(train_loader)
            start_epoch = self.current_epoch
            start_batch = getattr(self, "resume_batch_idx", 0)

            self.config["training"]["epochs"]
            logger.info(
                f"Resuming from epoch {start_epoch + 1}, batch {start_batch + 1}/{batches_per_epoch}, step {self.global_step}"
            )

            for epoch in range(self.current_epoch, self.config["training"]["epochs"]):
                self.current_epoch = epoch

                if epoch == start_epoch:
                    train_loss = self.train_epoch(train_loader, start_batch=start_batch)
                else:
                    train_loss = self.train_epoch(train_loader, start_batch=0)
                self.resume_batch_idx = 0

                if val_loader:
                    val_loss = self.validate(val_loader)

                    try:
                        mlflow.log_metrics(
                            {
                                "val_loss": val_loss,
                                "train_epoch_loss": train_loss,
                            },
                            step=epoch,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log validation metrics: {e}")

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_model(is_best=True)
                        logger.info(f"New best model! Validation loss: {val_loss:.5f}")

                    logger.info(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}"
                    )
                else:
                    try:
                        mlflow.log_metrics({"train_epoch_loss": train_loss}, step=epoch)
                    except Exception as e:
                        logger.warning(f"Failed to log training metrics: {e}")
                    logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.5f}")

                self.lr_scheduler.step()

            self.save_model(is_final=True)

        logger.info("Training completed!")
