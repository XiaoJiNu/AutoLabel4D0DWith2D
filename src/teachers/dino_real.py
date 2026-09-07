"""Real Grounding DINO Tiny teacher (transformers API, serial load→infer→unload)."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from teachers.base import SerialTeacherGuard

DEFAULT_DINO_DIR = Path(
    "/data/data/automomous/autolabel4d/checkpoints/dino/grounding-dino-tiny"
)
# Full nuScenes detection/tracking 10-class prompt set
DEFAULT_PROMPTS = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic cone",
    "barrier",
)
# Map free-text labels → nuScenes schema categories (10-class only)
_LABEL_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "construction vehicle": "construction_vehicle",
    "construction": "construction_vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "traffic_cone": "traffic_cone",
    "traffic cone": "traffic_cone",
    "cone": "traffic_cone",
    "barrier": "barrier",
}


class DinoTeacherReal:
    """Grounding DINO Tiny via HuggingFace transformers (local weights)."""

    name = "dino_real"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_DINO_DIR,
        *,
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
        self._loaded = False
        self._model = None
        self._processor = None
        self._load_mode: Optional[str] = None
        self._load_notes: str = ""

    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        import torch

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self._model.to(self.device)
            self._model.eval()
            self._load_mode = "transformers"
            self._load_notes = (
                f"transformers AutoModelForZeroShotObjectDetection from {self.model_dir}"
            )
            self._loaded = True
            return
        except Exception as exc:  # noqa: BLE001
            # Best-effort torch load of safetensors (document failure; still raise)
            notes = f"transformers API failed: {type(exc).__name__}: {exc}"
            try:
                from safetensors.torch import load_file

                st = self.model_dir / "model.safetensors"
                if st.is_file():
                    _ = load_file(str(st), device="cpu")
                    notes += (
                        "; safetensors torch load of model.safetensors succeeded "
                        "but no usable detection head without transformers; "
                        "cannot run real DINO inference in best-effort mode."
                    )
                else:
                    notes += "; model.safetensors missing for best-effort torch load."
            except Exception as exc2:  # noqa: BLE001
                notes += f"; best-effort torch load also failed: {type(exc2).__name__}: {exc2}"
            self._load_notes = notes
            SerialTeacherGuard.release(self)
            raise RuntimeError(notes) from exc

    def unload(self) -> None:
        import torch

        self._model = None
        self._processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        SerialTeacherGuard.release(self)

    def _text_prompt(self) -> str:
        # Grounding DINO expects lowercase phrases separated by periods.
        parts = [p.strip().lower() for p in self.prompts if p and str(p).strip()]
        return ". ".join(parts) + "."

    @staticmethod
    def _normalize_label(lab: str) -> str:
        s = str(lab or "").strip().lower().rstrip(".")
        if s in _LABEL_MAP:
            return _LABEL_MAP[s]
        s2 = s.replace(" ", "_")
        if s2 in _LABEL_MAP:
            return _LABEL_MAP[s2]
        # Keep only if already a nuScenes class; else mark for drop downstream
        allowed = {
            "car", "truck", "bus", "trailer", "construction_vehicle",
            "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
        }
        return s2 if s2 in allowed else "object"

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")

        import torch
        from PIL import Image

        image_path = frame.get("image_path") or frame.get("cam_path")
        if not image_path:
            raise ValueError("frame.image_path required")
        image_path = str(image_path)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        text = frame.get("text_prompt") or self._text_prompt()

        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Prefer post_process_grounded_object_detection when available
        target_sizes = torch.tensor([(h, w)], device=self.device)
        detections_2d: List[Dict[str, Any]] = []
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
            boxes = results.get("boxes")
            scores = results.get("scores")
            labels = results.get("labels") or results.get("text_labels") or []
            n = int(boxes.shape[0]) if boxes is not None else 0
            for i in range(n):
                box = boxes[i].detach().float().cpu().tolist()  # xyxy
                score = float(scores[i].detach().cpu()) if scores is not None else 0.0
                raw_lab = labels[i] if i < len(labels) else ""
                if hasattr(raw_lab, "item"):
                    raw_lab = str(raw_lab)
                cat = self._normalize_label(str(raw_lab))
                x1, y1, x2, y2 = [float(v) for v in box]
                detections_2d.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "score": score,
                        "category": cat,
                        "label_raw": str(raw_lab),
                        "camera": frame.get("camera", "CAM_FRONT"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            # Fallback: empty with note
            return {
                "teacher": self.name,
                "sample_token": frame.get("sample_token"),
                "image_path": image_path,
                "camera": frame.get("camera", "CAM_FRONT"),
                "detections_2d": [],
                "meta": {
                    "is_fake": False,
                    "is_stub": False,
                    "success": False,
                    "notes": f"post_process failed: {type(exc).__name__}: {exc}; load={self._load_notes}",
                    "image_size": [w, h],
                    "text_prompt": text,
                    "box_threshold": self.box_threshold,
                    "text_threshold": self.text_threshold,
                    "load_mode": self._load_mode,
                },
            }

        return {
            "teacher": self.name,
            "sample_token": frame.get("sample_token"),
            "image_path": image_path,
            "camera": frame.get("camera", "CAM_FRONT"),
            "detections_2d": detections_2d,
            "meta": {
                "is_fake": False,
                "is_stub": False,
                "success": True,
                "notes": self._load_notes,
                "image_size": [w, h],
                "text_prompt": text,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "load_mode": self._load_mode,
                "n_dets": len(detections_2d),
            },
        }


__all__ = ["DinoTeacherReal", "DEFAULT_DINO_DIR", "DEFAULT_PROMPTS"]
