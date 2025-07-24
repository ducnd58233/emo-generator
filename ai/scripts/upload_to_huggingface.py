from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi
from src.utils.logging import get_logger

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a model (folder or file) to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model folder or file to upload.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Hub repository ID (e.g. username/model-repo)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (optional, can use HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for the upload (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("upload_to_huggingface")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    repo_id: str = args.repo_id
    token: Optional[str] = args.token
    commit_message: str = (
        args.commit_message
        or f"Upload {model_path.name} to Hugging Face Hub via script"
    )

    api = HfApi()

    try:
        if model_path.is_dir():
            logger.info(
                f"Uploading folder {model_path} to Hugging Face Hub repo {repo_id}"
            )
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )
        elif model_path.is_file():
            logger.info(
                f"Uploading file {model_path} to Hugging Face Hub repo {repo_id}"
            )
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
            )
        else:
            logger.error(f"Model path is neither a file nor a directory: {model_path}")
            sys.exit(1)
        logger.info(f"Successfully uploaded to Hugging Face Hub: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face Hub: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
