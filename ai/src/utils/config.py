import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    merged = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
    return merged
