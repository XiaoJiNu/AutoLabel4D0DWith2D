#!/usr/bin/env python3
"""Re-print teacher_weights_v3.yaml status (path existence + sizes). No downloads."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

REPO = Path(__file__).resolve().parents[1]
YAML_PATH = REPO / "configs" / "teacher_weights_v3.yaml"
MANIFEST = Path("/data/data/automomous/autolabel4d/manifests/teacher_weight_probe.json")


def human(n: int) -> str:
    if n < 50 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f}MB"
    return f"{n / (1024 ** 3):.2f}GB"


def check_path(p: str) -> str:
    path = Path(p)
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        return "DIR"
    sz = path.stat().st_size
    return f"OK {human(sz)} ({sz} bytes)"


def main() -> int:
    print(f"yaml: {YAML_PATH} exists={YAML_PATH.is_file()}")
    print(f"manifest: {MANIFEST} exists={MANIFEST.is_file()}")
    if not YAML_PATH.is_file():
        return 1
    text = YAML_PATH.read_text()
    if yaml is None:
        print("--- raw yaml (PyYAML not installed) ---")
        print(text)
        return 0
    data = yaml.safe_load(text)
    teachers = (data or {}).get("teachers") or {}
    for name, block in teachers.items():
        status = block.get("status")
        print(f"\n[{name}] status={status} pipeline_id={block.get('pipeline_id')}")
        for w in block.get("weights_found") or []:
            p = w.get("path")
            print(f"  FOUND  {check_path(p)}  :: {p}")
        for c in block.get("code_trees_found") or []:
            p = c.get("path") if isinstance(c, dict) else c
            print(f"  CODE   {check_path(p)}  :: {p}")
        for m in block.get("missing") or []:
            slot = m.get("slot")
            hid = m.get("intended_hf_model_id") or m.get("intended") or m.get("intended_repo")
            print(f"  MISSING slot={slot} intended={hid}")
    ckpt = Path("/data/data/automomous/autolabel4d/checkpoints")
    print(f"\ncheckpoints_root empty={ckpt.is_dir() and not any(ckpt.iterdir())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
