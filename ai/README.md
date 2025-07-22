## Setup

```bash
conda env create -f environment.yml
conda activate emo-generator-ai

conda install setuptools poetry
```

## Install dependencies

```bash
poetry install --no-root
```

## Download and extract data

To download and extract data from Google Drive (for example, public datasets), use the provided script:

```bash
poetry run python scripts/download_and_extract_gdrive.py <gdrive_link1> [<gdrive_link2> ...]
```

Example:

```bash
poetry run python scripts/download_and_extract_gdrive.py \
  https://drive.google.com/file/d/15Z_F4Dwgb3NLqEGnVMUEJqyxXgW7Gx-h/view?usp=sharing \
  https://drive.google.com/file/d/15g5wrEEIXcxmlekgFTnIwIlew3okBSsj/view?usp=sharing
```

- All files will be downloaded to `ai/data`. If a file is a zip archive, it will be automatically extracted there.

## Export conda environment

```bash
conda env export --from-history --no-builds > environment.yml
```

## Docker

- Run docker compose

```bash
docker compose -f ./deployments/docker/docker-compose.yml up -d
```

- Stop docker compose

```bash
docker compose -f ./deployments/docker/docker-compose.yml down
```

## Run project

```bash
poetry run python src/main.py
```
