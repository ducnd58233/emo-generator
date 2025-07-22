#!/usr/bin/env python3
"""
Download and (optionally) extract files from Google Drive links.

Usage:
    python download_and_extract_gdrive.py <gdrive_link1> [<gdrive_link2> ...]

Example:
    python download_and_extract_gdrive.py \
        https://drive.google.com/file/d/15Z_F4Dwgb3NLqEGnVMUEJqyxXgW7Gx-h/view?usp=sharing \
        https://drive.google.com/file/d/15g5wrEEIXcxmlekgFTnIwIlew3okBSsj/view?usp=sharing

Requirements:
    - gdown (install via pip or add to environment.yml)
"""
import argparse
import logging
import re
import zipfile
from pathlib import Path

import gdown

DATA_DIR = Path(__file__).parent.parent / "datasets"

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def extract_file_id(gdrive_url: str) -> str:
    """Extract the file ID from a Google Drive share link."""
    match = re.search(r"/d/([\w-]+)", gdrive_url)
    if match:
        return match.group(1)
    # Alternative format: open?id=FILEID
    match = re.search(r"id=([\w-]+)", gdrive_url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract file ID from URL: {gdrive_url}")


def safe_extract_zip(zip_path: Path, extract_to: Path):
    """Safely extract a zip file to the target directory, preventing path traversal."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = extract_to / member.filename
            try:
                # Python 3.9+: use is_relative_to
                if not member_path.resolve().is_relative_to(extract_to.resolve()):
                    raise Exception(f"Unsafe zip file: {member.filename}")
            except AttributeError:
                # For Python <3.9, fallback to string comparison
                resolved_member = str(member_path.resolve())
                resolved_extract_to = str(extract_to.resolve())
                if not resolved_member.startswith(resolved_extract_to):
                    raise Exception(f"Unsafe zip file: {member.filename}")
        zf.extractall(extract_to)
    logging.info(f"Extracted zip: {zip_path.name} -> {extract_to}")


def download_gdrive_file(gdrive_url: str, output_dir: Path) -> Path:
    """Download a file from Google Drive and return the local file path."""
    file_id = extract_file_id(gdrive_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    # gdown will name the file as the original name by default
    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading: {gdrive_url}")
    output_path = gdown.download(
        url, output=None, quiet=False, fuzzy=True, use_cookies=False
    )
    if output_path is None:
        raise Exception(f"Failed to download file from: {gdrive_url}")
    output_path = Path(output_path)
    # Move to output_dir if not already there
    if output_path.parent != output_dir:
        target_path = output_dir / output_path.name
        output_path.replace(target_path)
        output_path = target_path
    logging.info(f"Downloaded to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract files from Google Drive links."
    )
    parser.add_argument("links", nargs="+", help="Google Drive share links")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for link in args.links:
        if not link.strip() or link.strip() == "\\":
            logging.warning(f"Skipping invalid argument: {link!r}")
            continue
        try:
            file_path = download_gdrive_file(link, DATA_DIR)
            if zipfile.is_zipfile(file_path):
                safe_extract_zip(file_path, DATA_DIR)
                # Remove the zip file after successful extraction
                try:
                    file_path.unlink()
                    logging.info(f"Removed zip file: {file_path.name}")
                except Exception as del_err:
                    logging.warning(
                        f"Could not remove zip file {file_path.name}: {del_err}"
                    )
            else:
                logging.info(f"File is not a zip: {file_path.name}")
        except Exception as e:
            logging.error(f"Error processing {link}: {e}")


if __name__ == "__main__":
    main()
