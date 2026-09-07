"""Stage F SimpleTrack hook stub (no full tracking run).

Records the tracker config path for later DetZero / SimpleTrack integration.
Reference (DetZero / community): https://github.com/tusen-ai/SimpleTrack
(Commit pin TBD when submodule is vendored — do not clone/download in this smoke.)

Real evaluation flow (later):
  student PointPillar boxes → SimpleTrack → track_id metrics vs fusion pseudo.
This stub only stores config metadata and returns a no-op plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# Upstream reference recorded for Stage F4 docs / future vendor pin.
SIMPLETRACK_REPO = "https://github.com/tusen-ai/SimpleTrack"
SIMPLETRACK_NOTE = (
    "DetZero and related AutoLabel pipelines commonly wrap tusen-ai/SimpleTrack "
    "as a fixed tracker for comparing pseudo-label quality. Exact DetZero commit "
    "is environment-specific; pin when third_party/SimpleTrack is added."
)

DEFAULT_TRACKER_CONFIG_NAME = "simpletrack_nuscenes_smoke.yaml"


class SimpleTrackHook:
    """Stub: record tracker config path; never runs full multi-frame track."""

    name = "simpletrack_hook_stub"

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        repo_hint: str = SIMPLETRACK_REPO,
    ):
        self.config_path = Path(config_path) if config_path else None
        self.repo_hint = repo_hint
        self._recorded: Optional[Dict[str, Any]] = None

    def record_config(self, config_path: str | Path | None = None) -> Dict[str, Any]:
        """Remember where the SimpleTrack YAML will live; do not execute tracker."""
        if config_path is not None:
            self.config_path = Path(config_path)
        path = self.config_path
        exists = bool(path and path.is_file())
        self._recorded = {
            "hook": self.name,
            "tracker": "simpletrack",
            "config_path": str(path) if path else None,
            "config_exists": exists,
            "repo_reference": self.repo_hint,
            "notes": SIMPLETRACK_NOTE,
            "ran_full_track": False,
            "is_stub": True,
        }
        return dict(self._recorded)

    def ensure_placeholder_config(self, out_dir: str | Path) -> Path:
        """Write a minimal placeholder YAML path record (not a full SimpleTrack cfg)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / DEFAULT_TRACKER_CONFIG_NAME
        if not cfg.is_file():
            cfg.write_text(
                "# Placeholder SimpleTrack config path record (Stage F stub).\n"
                f"# Upstream: {SIMPLETRACK_REPO}\n"
                "# Do not run full tracking from this smoke.\n"
                "tracker: simpletrack\n"
                "dataset: nuscenes_style\n"
                "is_stub: true\n"
                "notes: >\n"
                "  Replace with a real SimpleTrack YAML when third_party is vendored.\n",
                encoding="utf-8",
            )
        self.config_path = cfg
        return cfg

    @property
    def last_record(self) -> Optional[Dict[str, Any]]:
        return dict(self._recorded) if self._recorded else None
