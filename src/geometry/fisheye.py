"""Fisheye → virtual pinhole interface (Stage C stub).

RoboSense RS_4F_1L uses four OV fisheye cameras. Full undistort / remap to a
virtual pinhole will be filled when RS calibration (K, D, xi / Kannala, etc.)
is available. Until then:

- `fisheye_to_virtual_pinhole` raises NotImplementedError by default, or
- call with `passthrough=True` to return the input unchanged (documented stub).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class VirtualPinhole:
    """Result of a (future) fisheye unwrap."""

    image: np.ndarray
    K: np.ndarray
    # Optional map for reuse / caching (C3)
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


def fisheye_to_virtual_pinhole(
    image: np.ndarray,
    K_fisheye: np.ndarray,
    D: np.ndarray,
    *,
    K_virtual: Optional[np.ndarray] = None,
    image_size_virtual: Optional[Tuple[int, int]] = None,
    passthrough: bool = False,
    model: str = "kannala_brandt",
) -> VirtualPinhole:
    """Convert a fisheye image to a virtual pinhole view.

    Args:
        image: HxWxC uint8 (or similar) fisheye frame.
        K_fisheye: 3x3 fisheye intrinsics.
        D: distortion coefficients (model-dependent).
        K_virtual: optional target pinhole K; if None, to be derived later.
        image_size_virtual: optional (W, H) of virtual image.
        passthrough: if True, return image/K unchanged with stub meta (no raise).
        model: reserved for "kannala_brandt" / "equidistant" / "mei" etc.

    Returns:
        VirtualPinhole with remapped image and virtual K.

    Raises:
        NotImplementedError: unless passthrough=True.

    TODO(RS):
      - Implement cv2.fisheye.initUndistortRectifyMap / custom Mei model.
      - Cache map1/map2 per camera channel under outputs or autolabel4d cache.
      - Validate FOV crop so projected LiDAR lands on sensible pixels.
    """
    K = np.asarray(K_fisheye, dtype=np.float64).reshape(3, 3)
    if passthrough:
        Kv = np.asarray(K_virtual, dtype=np.float64).reshape(3, 3) if K_virtual is not None else K.copy()
        return VirtualPinhole(
            image=image,
            K=Kv,
            map1=None,
            map2=None,
            meta={
                "stub": True,
                "passthrough": True,
                "model": model,
                "note": (
                    "fisheye→virtual pinhole not implemented; returning input as-is. "
                    "Enable real unwrap when RS calib arrives."
                ),
                "D_shape": list(np.asarray(D).shape),
                "image_size_virtual": image_size_virtual,
            },
        )

    raise NotImplementedError(
        "fisheye_to_virtual_pinhole: Stage C stub. Pass passthrough=True for no-op, "
        "or implement when RoboSense fisheye calibration is available "
        f"(model={model!r})."
    )


def is_fisheye_stub_ready() -> bool:
    """False until real RS unwrap is implemented."""
    return False


__all__ = [
    "VirtualPinhole",
    "fisheye_to_virtual_pinhole",
    "is_fisheye_stub_ready",
]
