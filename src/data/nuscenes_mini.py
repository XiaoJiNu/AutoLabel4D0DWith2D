"""Thin helpers for nuScenes mini sample / LIDAR_TOP access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes


def load_nuscenes(dataroot: str | Path, version: str = "v1.0-mini", verbose: bool = False) -> NuScenes:
    return NuScenes(version=version, dataroot=str(dataroot), verbose=verbose)


def find_scene(nusc: NuScenes, scene_name: str) -> Dict[str, Any]:
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene
    available = [s["name"] for s in nusc.scene]
    raise ValueError(f"scene '{scene_name}' not found; available={available}")


def list_sample_tokens(nusc: NuScenes, scene_name: str, max_samples: Optional[int] = None) -> List[str]:
    scene = find_scene(nusc, scene_name)
    tokens: List[str] = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        if max_samples is not None and len(tokens) >= max_samples:
            break
        sample = nusc.get("sample", token)
        token = sample["next"]
    return tokens


def get_lidar_sd(nusc: NuScenes, sample_token: str, channel: str = "LIDAR_TOP") -> Dict[str, Any]:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][channel]
    return nusc.get("sample_data", sd_token)


def lidar_filepath(nusc: NuScenes, sample_token: str, channel: str = "LIDAR_TOP") -> Path:
    sd = get_lidar_sd(nusc, sample_token, channel=channel)
    return Path(nusc.dataroot) / sd["filename"]


def cam_filepath(nusc: NuScenes, sample_token: str, channel: str = "CAM_FRONT") -> Tuple[Path, Dict[str, Any]]:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][channel]
    sd = nusc.get("sample_data", sd_token)
    return Path(nusc.dataroot) / sd["filename"], sd


def sample_meta(nusc: NuScenes, sample_token: str, scene_name: str) -> Dict[str, Any]:
    sample = nusc.get("sample", sample_token)
    lidar_sd = get_lidar_sd(nusc, sample_token)
    return {
        "sample_token": sample_token,
        "scene_name": scene_name,
        "timestamp": int(sample["timestamp"]),
        "lidar_sd_token": lidar_sd["token"],
    }
