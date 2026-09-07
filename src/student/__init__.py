"""Student training stubs (Stage F PointPillar + SimpleTrack)."""
from .pointpillar_smoke import FakeStudent, validate_dataloader_inputs
from .simpletrack_hook import SimpleTrackHook

__all__ = ["FakeStudent", "SimpleTrackHook", "validate_dataloader_inputs"]
