# Emoji Generator: Text-to-Emoji with Stable Diffusion

A modern, modular, and extensible framework for generating custom emoji images from text prompts using Stable Diffusion, CLIP, and VAE. Designed for researchers, developers, and creative users who want to train, fine-tune, or deploy state-of-the-art text-to-image models for emoji and small icon generation.

---

## 🚀 Features

- **Text-to-Emoji Generation**: Generate unique emoji images from any text prompt using a custom Stable Diffusion pipeline.
- **Modular Architecture**: Built with extensibility in mind—swap out encoders, diffusion models, or datasets easily.
- **Training, Inference, and Evaluation**: Full support for model training, inference, and evaluation with simple commands.
- **MLflow Integration**: Track experiments, log metrics, and manage model versions with MLflow.
- **Data Management**: Download and extract datasets from Google Drive with a single script.
- **Reproducible Environments**: Use Conda, Poetry, or Docker for robust, reproducible setups.
- **GPU Acceleration**: Out-of-the-box support for CUDA and mixed-precision training.

---

## 📑 Table of Contents

- [Features](#-features)
- [Quickstart](#-quickstart)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [MLflow Tracking](#mlflow-tracking)
- [Data Management](#-data-management)
- [Directory Structure](#-directory-structure)
- [Docker](#-docker)
- [Troubleshooting & Tips](#-troubleshooting--tips)
- [Contributing](#-contributing)

---

## ⚡ Quickstart

### 1. Clone the repository

```bash
git clone git@github.com:ducnd58233/emo-generator.git
cd emo-generator/ai
```

### 2. Set up the environment

#### Using Conda & Poetry (recommended)

```bash
conda env create -f environment.yml
conda activate emo-generator-ai
poetry install
```

#### Export conda env

```bash
conda env export --from-history --no-builds > environment.yml
```

#### Create .env file

```bash
cp .env.example .env
```

- And add your HuggingFace token
- ***

## 🛠️ Usage

### MLflow Tracking

Start the MLflow tracking server:

```bash
poetry run python -m scripts.mlflow
```

### Training

Train a Stable Diffusion model for emoji generation:

```bash
poetry run python -m scripts.train \
  --config config/model.yaml \
  --train_config config/training.yaml \
  --data_config config/data.yaml
```

### Inference

Generate emojis from text prompts:

```bash
poetry run python -m scripts.inference \
  --model_path <path_to_checkpoint> \
  --config config/model.yaml \
  --prompt "a confused blob emoji with wide eyes" \
  --num_images 1 \
  --num_steps 100 \
  --latent_height 4 \
  --latent_width 4 \
  --seed 42 \
  --device cuda
```

### Evaluation

Evaluate your model and generate qualitative samples:

```bash
poetry run python -m scripts.evaluate \
  --model_path <path_to_checkpoint> \
  --config config/model.yaml \
  --data_config config/data.yaml \
  --training_config config/training.yaml
```

Skip sample generation for faster evaluation:

```bash
poetry run python -m scripts.evaluate \
  --model_path <path_to_checkpoint> \
  --config config/model.yaml \
  --data_config config/data.yaml \
  --training_config config/training.yaml \
  --skip_samples
```

### Model Management

Register a trained model to MLflow Model Registry:

```bash
poetry run python -m scripts.register_to_mlflow \
  --checkpoint_path models/best_model.pt \
  --model_name emoji-stable-diffusion \
  --alias candidate
```

## Configuration

The project uses three separate YAML configuration files for modularity and flexibility:

- `config/model.yaml`: Model and training hyperparameters
- `config/data.yaml`: Data and transform settings
- `config/training.yaml`: Experiment, MLflow, and optimization settings

All scripts expect these files to be present and kept in sync.

---

## 📦 Data Management

Download and extract datasets from Google Drive:

```bash
poetry run python scripts/download_and_extract_gdrive.py <gdrive_link1> [<gdrive_link2> ...]
```

Example

```bash
poetry run python scripts/download_and_extract_gdrive.py \
        https://drive.google.com/file/d/15Z_F4Dwgb3NLqEGnVMUEJqyxXgW7Gx-h/view?usp=sharing \
        https://drive.google.com/file/d/15g5wrEEIXcxmlekgFTnIwIlew3okBSsj/view?usp=sharing
```

All files will be downloaded to `ai/datasets`. Zip archives are automatically extracted.

---

## 🗂️ Directory Structure

```
ai/
  config/           # YAML configs for model, data, training
  datasets/         # Downloaded datasets (images, metadata)
  models/           # Saved checkpoints
  scripts/          # Training, inference, evaluation, utils
  src/              # Source code (data, models, utils, etc.)
  outputs/          # Generated images (created at runtime)
  mlruns/           # MLflow experiment logs
  environment.yml   # Conda environment
  pyproject.toml    # Poetry project config
```

---

## 🧩 Troubleshooting & Tips

- For best performance and lower memory usage, ensure `accelerate` is installed (see [PyPI](https://pypi.org/project/accelerate/)).
- If you encounter CUDA or memory errors, try reducing batch size or using CPU mode (`--device cpu`).
- All configs are in YAML—customize model, data, and training easily.
- For more help, check issues or open a discussion.

---

## 🤝 Contributing

We welcome contributions! Please:

- Fork the repo and create a feature branch
- Follow code style (see `.pre-commit-config.yaml`)
- Add tests and docs where appropriate
- Open a pull request with a clear description

---
