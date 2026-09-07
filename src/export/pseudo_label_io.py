"""Write / read / validate AutoLabel4D pseudo-label JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA = _REPO_ROOT / "configs" / "label_schema.yaml"

REQUIRED_TOP = [
    "version",
    "sample_token",
    "scene_name",
    "timestamp",
    "lidar_sd_token",
    "objects",
    "meta",
]
REQUIRED_OBJECT = [
    "track_id",
    "category",
    "translation",
    "size",
    "rotation",
    "score",
    "quality",
    "valid_fields",
    "ignore",
    "source",
]
REQUIRED_META = ["generator", "is_fake", "notes"]
QUALITY_VALUES = {"A", "B", "C"}


def load_schema(schema_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_label(path: str | Path, label: Dict[str, Any], *, indent: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_label(label)
    with out.open("w", encoding="utf-8") as f:
        json.dump(label, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    return out


def read_label(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        label = json.load(f)
    if not isinstance(label, dict):
        raise ValueError(f"label must be a JSON object: {path}")
    return label


def _check_float_list(name: str, value: Any, n: int) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{name} must be length-{n} list, got {value!r}")
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)):
            raise ValueError(f"{name}[{i}] must be numeric, got {v!r}")


def validate_label(label: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:
    """Raise ValueError if required fields are missing or malformed."""
    _ = schema  # schema yaml is documentation; validation is hard-coded for stability
    missing = [k for k in REQUIRED_TOP if k not in label]
    if missing:
        raise ValueError(f"missing top-level fields: {missing}")

    if not isinstance(label["version"], str):
        raise ValueError("version must be str")
    if not isinstance(label["sample_token"], str):
        raise ValueError("sample_token must be str")
    if not isinstance(label["scene_name"], str):
        raise ValueError("scene_name must be str")
    if not isinstance(label["timestamp"], int):
        raise ValueError("timestamp must be int")
    if not isinstance(label["lidar_sd_token"], str):
        raise ValueError("lidar_sd_token must be str")

    objects = label["objects"]
    if not isinstance(objects, list):
        raise ValueError("objects must be a list")
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"objects[{i}] must be a dict")
        miss = [k for k in REQUIRED_OBJECT if k not in obj]
        if miss:
            raise ValueError(f"objects[{i}] missing fields: {miss}")
        if not isinstance(obj["track_id"], str):
            raise ValueError(f"objects[{i}].track_id must be str")
        if not isinstance(obj["category"], str):
            raise ValueError(f"objects[{i}].category must be str")
        _check_float_list(f"objects[{i}].translation", obj["translation"], 3)
        _check_float_list(f"objects[{i}].size", obj["size"], 3)
        _check_float_list(f"objects[{i}].rotation", obj["rotation"], 4)
        if not isinstance(obj["score"], (int, float)):
            raise ValueError(f"objects[{i}].score must be numeric")
        if obj["quality"] not in QUALITY_VALUES:
            raise ValueError(f"objects[{i}].quality must be one of {sorted(QUALITY_VALUES)}")
        if not isinstance(obj["valid_fields"], list) or not all(isinstance(x, str) for x in obj["valid_fields"]):
            raise ValueError(f"objects[{i}].valid_fields must be list[str]")
        if not isinstance(obj["ignore"], bool):
            raise ValueError(f"objects[{i}].ignore must be bool")
        if not isinstance(obj["source"], str):
            raise ValueError(f"objects[{i}].source must be str")

    meta = label["meta"]
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    miss_m = [k for k in REQUIRED_META if k not in meta]
    if miss_m:
        raise ValueError(f"meta missing fields: {miss_m}")
    if not isinstance(meta["generator"], str):
        raise ValueError("meta.generator must be str")
    if not isinstance(meta["is_fake"], bool):
        raise ValueError("meta.is_fake must be bool")
    if not isinstance(meta["notes"], str):
        raise ValueError("meta.notes must be str")


def validate_label_file(path: str | Path) -> Dict[str, Any]:
    label = read_label(path)
    validate_label(label)
    return label


def list_label_files(directory: str | Path) -> List[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))
