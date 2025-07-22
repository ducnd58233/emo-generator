import os
from typing import Any, Dict, Optional

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
from ..utils.checkpoint import save_checkpoint


class StableDiffusionTrainer:
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = device

        # Initialize models
        self.diffusion_model = StableDiffusion(
            h_dim=config["model"]["stable_diffusion"]["h_dim"],
            n_head=config["model"]["stable_diffusion"]["n_head"],
        ).to(device)

        self.text_encoder = CLIPTextEncoder(
            config=config["model"]["clip"], device=device
        )

        self.vae_encoder = VAEEncoder(config=config["model"]["vae"]).to(device)

        # Initialize scheduler
        self.generator = torch.Generator(device=device)
        self.scheduler = DDPMScheduler(
            random_generator=self.generator,
            train_timesteps=config["model"]["stable_diffusion"]["num_train_timesteps"],
            beta_start=config["model"]["stable_diffusion"]["beta_start"],
            beta_end=config["model"]["stable_diffusion"]["beta_end"],
        )

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["optimization"]["weight_decay"]),
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(config["training"]["eta_min"]),
        )

        # Mixed precision training
        self.scaler = GradScaler() if config["training"]["mixed_precision"] else None

        # Loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["experiment"]["name"])

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.diffusion_model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['training']['epochs']}",
        )

        for batch_idx, (images, prompts) in enumerate(progress_bar):
            images = images.to(self.device)

            # Encode images to latent space
            with torch.no_grad():
                latents = self.vae_encoder.encode(images)

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                self.scheduler.total_train_timesteps,
                (latents.shape[0],),
                device=self.device,
            )

            # Add noise to latents
            noisy_latents, noise = self.scheduler.add_noise(latents, timesteps)

            # Encode text
            with torch.no_grad():
                text_embeddings = self.text_encoder(prompts)

            # Forward pass with mixed precision
            if self.scaler:
                with autocast("cuda", dtype=torch.float16):
                    noise_pred = self.diffusion_model(
                        noisy_latents, text_embeddings, timesteps
                    )
                    loss = self.criterion(noise_pred, noise)
            else:
                noise_pred = self.diffusion_model(
                    noisy_latents, text_embeddings, timesteps
                )
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

            # Update progress
            batch_loss = loss.item()
            epoch_loss += batch_loss
            self.global_step += 1

            # Log to MLflow
            if self.global_step % self.config["experiment"]["log_every"] == 0:
                mlflow.log_metrics(
                    {
                        "train_loss": batch_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": self.current_epoch,
                        "step": self.global_step,
                    },
                    step=self.global_step,
                )

            # Save checkpoint (local file, for resuming training)
            if self.global_step % self.config["experiment"]["save_every"] == 0:
                save_checkpoint(
                    self.diffusion_model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.current_epoch,
                    batch_loss,
                    os.path.join(
                        self.config["experiment"]["save_dir"],
                        f"checkpoint_epoch_{self.current_epoch + 1}_step_{self.global_step}.pt",
                    ),
                    self.scaler,
                )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{batch_loss:.5f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        return epoch_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        self.diffusion_model.eval()
        val_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for images, prompts in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)

                # Encode images to latent space
                latents = self.vae_encoder.encode(images)

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    self.scheduler.total_train_timesteps,
                    (latents.shape[0],),
                    device=self.device,
                )

                # Add noise to latents
                noisy_latents, noise = self.scheduler.add_noise(latents, timesteps)

                # Encode text
                text_embeddings = self.text_encoder(prompts)

                # Forward pass
                noise_pred = self.diffusion_model(
                    noisy_latents, text_embeddings, timesteps
                )
                loss = self.criterion(noise_pred, noise)

                val_loss += loss.item()

        return val_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        print(f"Starting training for {self.config['training']['epochs']} epochs...")

        with mlflow.start_run(run_name=f"{self.config['experiment']['name']}_run"):
            for epoch in range(self.current_epoch, self.config["training"]["epochs"]):
                self.current_epoch = epoch

                # Train epoch
                train_loss = self.train_epoch(train_loader)

                # Validation
                if (
                    val_loader
                    and epoch % self.config["experiment"]["validate_every"] == 0
                ):
                    val_loss = self.validate(val_loader)

                    # Log validation metrics
                    mlflow.log_metrics(
                        {
                            "val_loss": val_loss,
                            "train_epoch_loss": train_loss,
                        },
                        step=epoch,
                    )

                    # Save best model to registry
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_model_to_registry(is_best=True)

                    print(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}"
                    )
                else:
                    mlflow.log_metrics(
                        {
                            "train_epoch_loss": train_loss,
                        },
                        step=epoch,
                    )
                    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.5f}")

                # Update learning rate
                self.lr_scheduler.step()

            # Log final model
            self.save_model_to_registry(is_final=True)

        print("Training completed!")

    def save_model_to_registry(self, is_best: bool = False, is_final: bool = False):
        """Save model to MLflow registry with versioning"""
        model_name = self.config["experiment"]["model_name"]

        # Create model signature
        sample_input = torch.randn(1, 4, 4, 4).to(self.device)
        sample_context = torch.randn(1, 77, 512).to(self.device)
        sample_time = torch.randint(0, 1000, (1,)).to(self.device)

        with torch.no_grad():
            sample_output = self.diffusion_model(
                sample_input, sample_context, sample_time
            )

        signature = mlflow.models.infer_signature(
            {
                "latent": sample_input.cpu().numpy(),
                "context": sample_context.cpu().numpy(),
                "timestep": sample_time.cpu().numpy(),
            },
            sample_output.cpu().numpy(),
        )

        mlflow.pytorch.log_model(
            pytorch_model=self.diffusion_model,
            artifact_path="stable_diffusion_model",
            signature=signature,
            registered_model_name=model_name,
        )

        tags = {"epoch": str(self.current_epoch)}
        if is_best:
            tags.update({"stage": "best", "validation_loss": str(self.best_loss)})
        if is_final:
            tags.update({"stage": "final"})

        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name)[0]

        for key, value in tags.items():
            client.set_model_version_tag(model_name, model_version.version, key, value)

        if is_best:
            client.transition_model_version_stage(
                model_name, model_version.version, "Staging"
            )
        if is_final:
            client.transition_model_version_stage(
                model_name, model_version.version, "Production"
            )
