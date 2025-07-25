# Emoji Generator: Text-to-Emoji with Stable Diffusion

A modern, modular, and extensible framework for generating custom emoji images from text prompts using Stable Diffusion, CLIP, and VAE. Designed for researchers, developers, and creative users who want to train, fine-tune, or deploy state-of-the-art text-to-image models for emoji and small icon generation.

---

## 🚀 Features

- **Text-to-Emoji Generation**: Generate unique emoji images from any text prompt using a custom Stable Diffusion pipeline.
- **Modular Architecture**: Easily swap encoders, diffusion models, or datasets.
- **Training, Inference, and Evaluation**: Full support with simple commands.
- **Streamlit Web UI**: Interactive emoji generation with real-time parameter control.
- **MLflow Integration**: Track experiments, log metrics, manage model versions.
- **Hugging Face Hub Integration**: Upload/download models for sharing and deployment.
- **Data Management**: Download/extract datasets from Google Drive with a single script.
- **Reproducible Environments**: Use Conda, Poetry, or Docker for robust, reproducible setups.
- **GPU Acceleration**: Out-of-the-box support for CUDA and mixed-precision training.

---

## 📑 Table of Contents

- [Features](#-features)
- [Quickstart](#-quickstart)
- [Directory Structure](#-directory-structure)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Streamlit Web UI](#streamlit-web-ui)
  - [MLflow Tracking](#mlflow-tracking)
  - [Model Management](#model-management)
  - [Hugging Face Model Management](#hugging-face-model-management)
- [Data Management](#-data-management)
- [Docker](#-docker)
- [Troubleshooting & Tips](#-troubleshooting--tips)
- [Contributing](#-contributing)

---

## ⚡ Quickstart

### 1. Clone the repository

Clone the repo and enter the project directory:

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

Export your current conda environment for reproducibility:

```bash
conda env export --from-history --no-builds > environment.yml
```

#### Create .env file

Copy the example environment file:

```bash
cp .env.example .env
```

---

## 🗂️ Directory Structure

```text
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

## 🛠️ Usage

### MLflow Tracking

Start the MLflow tracking server (for experiment tracking and model registry):

```bash
poetry run python -m scripts.mlflow_client
```

### Training

Train a Stable Diffusion model for emoji generation:

```bash
poetry run python -m scripts.train \
  --config config/model.yaml \
  --train-config config/training.yaml \
  --data-config config/data.yaml
```

### Inference

Generate emojis from text prompts:

```bash
poetry run python -m scripts.inference \
  --model_path <path_to_checkpoint> \
  --config config/model.yaml \
  --prompt "a confused blob emoji with wide eyes" \
  --num-images 1 \
  --num-steps 100 \
  --latent-height 4 \
  --latent-width 4 \
  --seed 42 \
  --device cuda
```

### Evaluation

Evaluate your model and generate qualitative samples:

```bash
poetry run python -m scripts.evaluate \
  --model-path <path_to_checkpoint> \
  --config config/model.yaml \
  --data-config config/data.yaml \
  --training-config config/training.yaml \
  --output-dir <path_to_output>
```

Skip sample generation for faster evaluation:

```bash
poetry run python -m scripts.evaluate \
  --model-path <path_to_checkpoint> \
  --config config/model.yaml \
  --data-config config/data.yaml \
  --training-config config/training.yaml \
  --skip-samples
```

---

## 🌐 Streamlit Web UI

Launch an interactive web interface for emoji generation with real-time parameter control and model management.

#### Quick Start

1. **Install dependencies**:

```bash
poetry install
```

2. **Download a trained model** (if you don't have one):

```bash
poetry run python -m scripts.download_from_huggingface
```

3. **Configure environment variables**:

Create a `.env` file in the `ai/` directory with the following content:

```ini
# .env
EMO_GENERATOR_DEV_MODE=1  # Set to 0 for production mode
EMO_GENERATOR_MODEL_PATH=models/mlflow_registry/data/model.pth  # Change to your model path if needed
```

- `EMO_GENERATOR_DEV_MODE=1` (default): Development mode. The sidebar UI allows you to choose the model source (MLflow Registry or Local Checkpoint) and specify the model path interactively.
- `EMO_GENERATOR_DEV_MODE=0`: Production mode. The app will load the model from the specified path and will not show the model source/path selection UI.
- `EMO_GENERATOR_MODEL_PATH`: Sets the model checkpoint path for production mode.

4. **Launch the Streamlit app**:

```bash
poetry run streamlit run scripts/streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

- In production mode, the app will load the model from the specified path and will not show the model source/path selection UI.
- In development mode, you have full control over model source and path in the sidebar.

#### Features

- 🎨 Interactive Generation: Generate emojis from text prompts with real-time preview
- ⚙️ Parameter Control: Adjust inference steps, guidance scale, seed, and latent dimensions
- 🔧 Model Management: Load models from MLflow Registry or local checkpoints
- 💾 Download Results: Save generated emojis as PNG files
- 🚀 Auto-caching: Models cached for improved performance
- 📱 Responsive UI: Modern, mobile-friendly interface
- 💡 Prompt Suggestions: Quick-start with predefined example prompts
- 📊 Device Detection: Automatic GPU/CPU detection and usage display

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

> **Prompt Writing:**
>
> - Be specific and descriptive: "a confused blob emoji with wide eyes"
> - Use emotion words: happy, sad, confused, excited, surprised
> - Include style descriptors: sparkly, glowing, cartoon-like, pixel-art
> - Mention facial features: wide eyes, small mouth, raised eyebrows
> - Add context: "a sleepy cat emoji with droopy eyes and a yawn"

> **Parameter Tuning:**
>
> - More inference steps: Higher quality but slower generation (50-100 recommended)
> - Higher guidance scale: Closer adherence to prompt (7.5-15.0 typical range)
> - Consistent seed: Use same seed for reproducible results
> - Larger latent dimensions: Higher resolution output (use 8x8 for detailed emojis)

**Example Prompts:**

- "a confused blob emoji with wide eyes"
- "a sleepy cat emoji with droopy eyes"
- "a magical unicorn emoji with rainbow mane"
- "a robot emoji with glowing blue eyes"
- "a pizza emoji with melted cheese and pepperoni"
- "a dancing banana emoji with stick arms"

#### Troubleshooting

> **Model Loading Issues:**
>
> - Ensure model files exist in `models/mlflow_registry/data/model.pth`
> - Verify `config/model.yaml` exists and is properly formatted
> - Check that all dependencies are installed: `poetry install` (remember to activate conda env first)

> **Performance Issues:**
>
> - Reduce inference steps for faster generation
> - Use smaller latent dimensions if running out of memory
> - Close other applications to free up GPU memory

> **Common Errors:**
>
> - "Config file not found": Run from the `ai/` directory
> - "Model file not found": Download a model first using the download script
> - Import errors: Install missing dependencies

---

## 📦 Model Management

Register a trained model to MLflow Model Registry:

```bash
poetry run python -m scripts.register_to_mlflow \
  --checkpoint-path models/best_model.pt \
  --model-name emoji-stable-diffusion \
  --alias candidate
```

Download models from MLflow Registry:

```bash
python -m scripts.download_mlflow_model \
  --model-name emoji-stable-diffusion \
  --alias champion \
  --output-dir models/mlflow_registry
```

---

## 🤗 Hugging Face Model Management

Easily upload and download models to/from the Hugging Face Hub for sharing, deployment, or collaboration.

### Prerequisites

- [Create a Hugging Face account](https://huggingface.co/join)
- [Create a repository](https://huggingface.co/new) (or use an existing one)
- Add your Hugging Face token to your `.env` file as `HF_TOKEN` (see Quickstart)
- Install dependencies: `poetry install` (includes `huggingface_hub`)

### Upload a Model to Hugging Face Hub

Upload a model file or an entire folder (including config, tokenizer, etc.):

```bash
poetry run python -m scripts.upload_to_huggingface \
  --model-path <path_to_model_or_folder> \
  --repo-id <username/repo-name> \
  [--token <your_hf_token>] \
  [--commit-message "Add my model"]
```

- `--model-path`: Path to the model file or directory to upload
- `--repo-id`: Your Hugging Face repo (e.g. `your-username/emoji-model`)
- `--token`: (Optional) Hugging Face token (defaults to `HF_TOKEN` in `.env`)
- `--commit-message`: (Optional) Custom commit message

**Example:**

```bash
poetry run python -m scripts.upload_to_huggingface \
  --model-path models/mlflow_registry/data \
  --repo-id your-username/emoji-model
```

### Download a Model from Hugging Face Hub

Download a specific file (e.g. `model.pth`) or the entire repository snapshot:

```bash
poetry run python -m scripts.download_from_huggingface \
  --repo-id <username/repo-name> \
  [--filename <file_name>] \
  [--output-dir <output_directory>] \
  [--token <your_hf_token>]
```

- `--repo-id`: The Hugging Face repo to download from (default: `ducnd58233/emo-gen`)
- `--filename`: (Optional) Specific file to download (default: `model.pth`). Omit to download the full repo.
- `--output-dir`: (Optional) Where to save the file(s) (default: `models/mlflow_registry/data`)
- `--token`: (Optional) Hugging Face token (defaults to `HF_TOKEN` in `.env`)

**Example: Download a single file**

```bash
poetry run python -m scripts.download_from_huggingface \
  --repo-id your-username/emoji-model \
  --filename model.pth
```

**Example: Download the entire repo**

```bash
poetry run python -m scripts.download_from_huggingface \
  --repo-id your-username/emoji-model
```

> **Tips:**
>
> - For large models, uploading/downloading the entire folder is recommended to preserve all assets (config, tokenizer, etc.).
> - Make sure your Hugging Face token has write access for uploads.
> - You can manage your tokens at https://huggingface.co/settings/tokens
> - For more details, see the [Hugging Face Hub documentation](https://huggingface.co/docs/hub/models-uploading)

---

## ⚙️ Configuration

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

**Example:**

```bash
poetry run python scripts/download_and_extract_gdrive.py \
        https://drive.google.com/file/d/15Z_F4Dwgb3NLqEGnVMUEJqyxXgW7Gx-h/view?usp=sharing \
        https://drive.google.com/file/d/15g5wrEEIXcxmlekgFTnIwIlew3okBSsj/view?usp=sharing
```

All files will be downloaded to `ai/datasets`. Zip archives are automatically extracted.

---

## Running the Streamlit UI with Docker (CPU & GPU)

### Prerequisites

- Docker (and NVIDIA Container Toolkit for GPU)
- Hugging Face access token (if required by the model)

### Providing Environment Variables (e.g., HF_TOKEN)

You can provide secrets like `HF_TOKEN` to Docker Compose in several ways:

1. **Using a custom env file:**

    - Run Compose with the `--env-file` flag:
    
    ```bash
    docker compose -f deployments/docker/docker-compose.yml --env-file .env up streamlit-cpu -d
    ```
    
    - For gpu
    
    ```bash
    docker compose -f deployments/docker/docker-compose.yml --env-file .env up streamlit-gpu -d
    ```

2. **From your shell environment:**

   - Export the variable in your shell before running Compose:

   ```bash
     export HF_TOKEN=your_hf_token_here
     docker compose -f deployments/docker/docker-compose.yml up streamlit-cpu -d
     # Or for gpu
     docker compose -f deployments/docker/docker-compose.yml up streamlit-gpu -d
   ```

   - This will override the value in `.env` if both are set.

3. **Inline with `-e` (for `docker run`):**
   - When running a container directly:
     ```bash
     docker run --rm -p 8501:8501 -e HF_TOKEN=your_hf_token_here emo-streamlit-cpu
     ```

**Precedence:** Shell > `--env-file` > `.env` file > Compose defaults.

For more, see the [Docker Compose env file docs](https://docs.docker.com/compose/environment-variables/env-file/).

### 1. CPU-only (no GPU required)

Build the image:

```bash
docker build -f deployments/docker/Dockerfile.cpu -t emo-streamlit-cpu .
```

Run the container:

```bash
docker run --rm -p 8501:8501 -e HF_TOKEN=your_hf_token_here emo-streamlit-cpu
```


### 2. GPU (NVIDIA GPU required)

Build the image:

```bash
docker build -f deployments/docker/Dockerfile.gpu -t emo-streamlit-gpu .
```

Run the container (with NVIDIA runtime):

```bash
docker run --rm --gpus all -p 8501:8501 -e HF_TOKEN=your_hf_token_here emo-streamlit-gpu
```


- The app will be available at http://localhost:8501
- The model will be downloaded at build time and placed in the correct directory.
- If you want to persist the model outside the container, mount a volume:

  docker run --rm -p 8501:8501 -e HF_TOKEN=your_hf_token_here -v $(pwd)/models/mlflow_registry/data:/app/models/mlflow_registry/data emo-streamlit-cpu

or (for GPU):

    docker run --rm --gpus all -p 8501:8501 -e HF_TOKEN=your_hf_token_here -v $(pwd)/models/mlflow_registry/data:/app/models/mlflow_registry/data emo-streamlit-gpu

- If your model is public, you can omit HF_TOKEN.
- For more advanced options, see the Dockerfiles in deployments/docker.

---

## 🧩 Troubleshooting & Tips

> **CUDA or memory errors?** Try reducing batch size or using CPU mode (`--device cpu`).
>
> **All configs are in YAML**—customize model, data, and training easily.
>
> **Streamlit UI Issues:** Ensure you're running from the `ai/` directory and have downloaded a model first.
>
> For more help, check issues or open a discussion.

---

## 🤝 Contributing

We welcome contributions! Please:

- Fork the repo and create a feature branch
- Follow code style (see `.pre-commit-config.yaml`)
- Add tests and docs where appropriate
- Open a pull request with a clear description

---
