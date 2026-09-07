"""Real SAM 2.1 Small teacher via ultralytics (local hiera-small weights).

Ultralytics maps weights by basename (sam2.1_s.pt). We symlink/copy the
facebook sam2.1_hiera_small.pt into a cache path with that name.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from teachers.base import SerialTeacherGuard

DEFAULT_SAM_PT = Path(
    "/data/data/automomous/autolabel4d/checkpoints/sam/sam2.1-hiera-small/sam2.1_hiera_small.pt"
)
DEFAULT_SAM_ALIAS = Path(
    "/data/data/automomous/autolabel4d/checkpoints/sam/sam2.1-hiera-small/sam2.1_s.pt"
)


def _ensure_ultralytics_alias(src: Path, alias: Path = DEFAULT_SAM_ALIAS) -> Path:
    src = Path(src)
    alias = Path(alias)
    if alias.is_file() or alias.is_symlink():
        return alias
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.symlink(str(src.resolve()), str(alias))
    except OSError:
        # fallback: hard copy is too big; try relative symlink
        if alias.exists() or alias.is_symlink():
            try:
                alias.unlink()
            except OSError:
                pass
        os.symlink(src.name, str(alias))
    return alias


class SamTeacherReal:
    """SAM 2.1 Small — box-prompted masks from DINO detections."""

    name = "sam_real"

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_SAM_PT,
        *,
        device: str = "cuda:0",
        alias_path: str | Path = DEFAULT_SAM_ALIAS,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.alias_path = Path(alias_path)
        self.device = device
        self._loaded = False
        self._model = None
        self._load_notes = ""

    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        try:
            from ultralytics import SAM

            alias = _ensure_ultralytics_alias(self.checkpoint, self.alias_path)
            self._model = SAM(str(alias))
            # move to device if possible
            try:
                if hasattr(self._model, "to"):
                    self._model.to(self.device)
            except Exception:
                pass
            self._load_notes = (
                f"ultralytics SAM from alias={alias} src={self.checkpoint}"
            )
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            SerialTeacherGuard.release(self)
            raise RuntimeError(
                f"SAM load failed: {type(exc).__name__}: {exc}; "
                f"ckpt={self.checkpoint}"
            ) from exc

    def unload(self) -> None:
        import torch

        self._model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        SerialTeacherGuard.release(self)

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded or self._model is None:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")

        import numpy as np

        image_path = frame.get("image_path") or frame.get("cam_path")
        if not image_path:
            raise ValueError("frame.image_path required")
        image_path = str(image_path)
        dets = list(frame.get("detections_2d") or [])
        masks_out: List[Dict[str, Any]] = []

        if not dets:
            return {
                "teacher": self.name,
                "sample_token": frame.get("sample_token"),
                "image_path": image_path,
                "camera": frame.get("camera", "CAM_FRONT"),
                "masks": [],
                "meta": {
                    "is_fake": False,
                    "is_stub": False,
                    "success": True,
                    "notes": self._load_notes + "; no DINO boxes → empty masks",
                    "n_masks": 0,
                },
            }

        # Batch all boxes in one predict call
        bboxes = [list(map(float, d["bbox_xyxy"])) for d in dets if "bbox_xyxy" in d]
        if not bboxes:
            return {
                "teacher": self.name,
                "sample_token": frame.get("sample_token"),
                "image_path": image_path,
                "masks": [],
                "meta": {
                    "is_fake": False,
                    "is_stub": False,
                    "success": True,
                    "notes": self._load_notes + "; dets missing bbox_xyxy",
                    "n_masks": 0,
                },
            }

        results = self._model.predict(
            image_path, bboxes=bboxes, verbose=False, device=self.device
        )
        r0 = results[0]
        mask_tensor = None if r0.masks is None else r0.masks.data
        n_masks = 0 if mask_tensor is None else int(mask_tensor.shape[0])

        for i, det in enumerate(dets):
            if "bbox_xyxy" not in det:
                continue
            entry: Dict[str, Any] = {
                "bbox_xyxy": list(map(float, det["bbox_xyxy"])),
                "score": float(det.get("score", 0.0)),
                "category": str(det.get("category", "object")),
                "camera": frame.get("camera", det.get("camera", "CAM_FRONT")),
                "dino_idx": i,
            }
            if mask_tensor is not None and i < n_masks:
                m = mask_tensor[i].detach().float().cpu().numpy()
                # Compact RLE-ish: store bbox + area + downsample optional; keep full mask as bool packed
                # For cache size: store uint8 mask only if small; else store bbox-crop.
                ys, xs = np.where(m > 0.5)
                if ys.size:
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    crop = (m[y0:y1, x0:x1] > 0.5).astype(np.uint8)
                    entry["mask_bbox_xyxy"] = [x0, y0, x1, y1]
                    entry["mask_area"] = int(crop.sum())
                    entry["mask_h"] = int(crop.shape[0])
                    entry["mask_w"] = int(crop.shape[1])
                    # store as packed bits for compactness
                    flat = np.packbits(crop.reshape(-1))
                    entry["mask_packbits_b64"] = __import__("base64").b64encode(
                        flat.tobytes()
                    ).decode("ascii")
                else:
                    entry["mask_area"] = 0
            masks_out.append(entry)

        return {
            "teacher": self.name,
            "sample_token": frame.get("sample_token"),
            "image_path": image_path,
            "camera": frame.get("camera", "CAM_FRONT"),
            "masks": masks_out,
            "meta": {
                "is_fake": False,
                "is_stub": False,
                "success": True,
                "notes": self._load_notes,
                "n_masks": len(masks_out),
                "n_mask_tensors": n_masks,
            },
        }


__all__ = ["SamTeacherReal", "DEFAULT_SAM_PT", "DEFAULT_SAM_ALIAS", "_ensure_ultralytics_alias"]
