# Emoji Generator: Text-to-Emoji with Stable Diffusion

A modern, modular, and extensible framework for generating custom emoji images from text prompts using Stable Diffusion, CLIP, and VAE. Designed for researchers, developers, and creative users who want to train, fine-tune, or deploy state-of-the-art text-to-image models for emoji and small icon generation.

---

## 🚀 Features

- **Text-to-Emoji Generation**: Generate unique emoji images from any text prompt using a custom Stable Diffusion pipeline.
- **Modular Architecture**: Built with extensibility in mind—swap out encoders, diffusion models, or datasets easily.
- **Training, Inference, and Evaluation**: Full support for model training, inference, and evaluation with simple commands.
- **Streamlit Web UI**: Interactive web interface for easy emoji generation with real-time parameter control.
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
  - [Streamlit Web UI](#streamlit-web-ui)
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
poetry run python -m scripts.mlflow_client
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
  --training_config config/training.yaml \
  --output_dir <path_to_output> \
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

### Streamlit Web UI

Launch an interactive web interface for emoji generation with real-time parameter control and model management. The Streamlit app provides a user-friendly way to generate emojis without command-line interaction.

#### Quick Start

1. **Install dependencies**:

```bash
poetry install
```

2. **Download a trained model** (if you don't have one):

```bash
poetry run python -m scripts.download_mlflow_model \
  --model-name emoji-stable-diffusion \
  --alias champion \
  --output-dir models/mlflow_registry
```

3. **Launch the Streamlit app**:

```bash
poetry run streamlit run scripts/streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

#### Features

- **🎨 Interactive Generation**: Generate emojis from text prompts with real-time preview
- **⚙️ Parameter Control**: Adjust inference steps, guidance scale, seed, and latent dimensions
- **🔧 Model Management**: Load models from MLflow Registry or local checkpoints
- **💾 Download Results**: Save generated emojis as PNG files
- **🚀 Auto-caching**: Models cached for improved performance with Streamlit's `@st.cache_resource`
- **📱 Responsive UI**: Modern, mobile-friendly interface with sidebar controls
- **💡 Prompt Suggestions**: Quick-start with predefined example prompts
- **📊 Device Detection**: Automatic GPU/CPU detection and usage display

#### Model Loading Options

**MLflow Registry** (Recommended):

- Load models by name and alias (champion/challenger/candidate)
- Automatic model versioning and metadata retrieval
- Support for remote MLflow tracking servers

**Local Checkpoints**:

- Load from any local `.pth` checkpoint file
- Useful for newly trained models not yet registered

#### Generation Parameters

- **Inference Steps** (10-200): Number of denoising steps (higher = better quality, slower)
- **Guidance Scale** (1.0-20.0): Text prompt adherence (higher = closer to prompt)
- **Seed**: Random seed for reproducible generation
- **Latent Dimensions**: Height/width of latent space (affects output resolution)

#### Usage Tips

**Prompt Writing:**

- Be specific and descriptive: "a confused blob emoji with wide eyes"
- Use emotion words: happy, sad, confused, excited, surprised
- Include style descriptors: sparkly, glowing, cartoon-like, pixel-art
- Mention facial features: wide eyes, small mouth, raised eyebrows
- Add context: "a sleepy cat emoji with droopy eyes and a yawn"

**Parameter Tuning:**

- **More inference steps**: Higher quality but slower generation (50-100 recommended)
- **Higher guidance scale**: Closer adherence to prompt (7.5-15.0 typical range)
- **Consistent seed**: Use same seed for reproducible results across runs
- **Larger latent dimensions**: Higher resolution output (use 8x8 for detailed emojis)

**Example Prompts:**

- "a confused blob emoji with wide eyes"
- "a sleepy cat emoji with droopy eyes"
- "a magical unicorn emoji with rainbow mane"
- "a robot emoji with glowing blue eyes"
- "a pizza emoji with melted cheese and pepperoni"
- "a dancing banana emoji with stick arms"

#### Troubleshooting

**Model Loading Issues:**

- Ensure model files exist in `models/mlflow_registry/data/model.pth`
- Verify `config/model.yaml` exists and is properly formatted
- Check that all dependencies are installed: `poetry install` (remember to activate conda env first)

**Performance Issues:**

- Reduce inference steps for faster generation
- Use smaller latent dimensions if running out of memory
- Close other applications to free up GPU memory

**Common Errors:**

- `"Config file not found"`: Run from the `ai/` directory
- `"Model file not found"`: Download a model first using the download script
- Import errors: Install missing dependencies

### Model Management

Register a trained model to MLflow Model Registry:

```bash
poetry run python -m scripts.register_to_mlflow \
  --checkpoint_path models/best_model.pt \
  --model_name emoji-stable-diffusion \
  --alias candidate
```

Download models from MLflow Registry:

```bash
python -m scripts.download_mlflow_model \
  --model-name emoji-stable-diffusion \
  --alias champion \
  --output-dir models/mlflow_registry
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
  models/           # Saved checkpoints and MLflow registry
    mlflow_registry/  # Downloaded MLflow models
  scripts/          # Training, inference, evaluation, UI apps
    streamlit_app.py  # Streamlit web interface with environment setup
  src/              # Source code (data, models, utils, etc.)
    ui/             # UI components and utilities
  outputs/          # Generated images (created at runtime)
  mlruns/           # MLflow experiment logs
  environment.yml   # Conda environment
  pyproject.toml    # Poetry project config
```

---

## 🧩 Troubleshooting & Tips

- If you encounter CUDA or memory errors, try reducing batch size or using CPU mode (`--device cpu`).
- All configs are in YAML—customize model, data, and training easily.
- **Streamlit UI Issues**: Ensure you're running from the `ai/` directory and have downloaded a model first.
- For more help, check issues or open a discussion.

---

## 🤝 Contributing

We welcome contributions! Please:

- Fork the repo and create a feature branch
- Follow code style (see `.pre-commit-config.yaml`)
- Add tests and docs where appropriate
- Open a pull request with a clear description

---
