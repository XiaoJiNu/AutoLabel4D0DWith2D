"""Load repo path constants from configs/paths.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATHS_YAML = _REPO_ROOT / "configs" / "paths.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def load_paths(yaml_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(yaml_path) if yaml_path else _DEFAULT_PATHS_YAML
    if not path.is_file():
        raise FileNotFoundError(f"paths.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"paths.yaml must be a mapping: {path}")
    return data


def get_path(key: str, yaml_path: str | Path | None = None) -> Path:
    cfg = load_paths(yaml_path)
    if key not in cfg:
        raise KeyError(f"path key '{key}' missing in {yaml_path or _DEFAULT_PATHS_YAML}")
    return Path(cfg[key])
