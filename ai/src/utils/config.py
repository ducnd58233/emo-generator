import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ..common.constants import CONFIG_EXT

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix != CONFIG_EXT:
        raise ValueError(f"Configuration file must have {CONFIG_EXT} extension")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}") from e


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration

    Raises:
        OSError: If unable to create directory or write file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.


    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}

    result = {}

    for config in configs:
        for key, value in config.items():
            if (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result
