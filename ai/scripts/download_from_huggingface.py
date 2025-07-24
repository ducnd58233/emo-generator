from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from src.utils.logging import get_logger

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a model (file or full repo) from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ducnd58233/emo-gen",
        help="Hugging Face Hub repository ID (e.g. username/model-repo)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model.pth",
        help="Specific filename to download (optional, downloads all if not set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/mlflow_registry/data",
        help="Directory to save the downloaded model or file.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (optional, can use HF_TOKEN env variable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("download_from_huggingface")

    repo_id: str = args.repo_id
    filename: Optional[str] = args.filename
    output_dir = Path(args.output_dir)
    token: Optional[str] = args.token

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if filename:
            logger.info(
                f"Downloading file '{filename}' from Hugging Face Hub repo {repo_id}"
            )
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(output_dir),
                token=token,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded file to: {file_path}")
        else:
            logger.info(
                f"Downloading entire repo snapshot from Hugging Face Hub repo {repo_id}"
            )
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(output_dir),
                token=token,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded repo snapshot to: {snapshot_path}")
    except Exception as e:
        logger.error(f"Failed to download from Hugging Face Hub: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
