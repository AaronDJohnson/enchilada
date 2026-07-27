from dataclasses import replace  # re-exported: every segment needs it
from importlib.metadata import PackageNotFoundError, version

from turntable.orbits import NumericOrbit, Orbit
from turntable.residuals import Residuals
from turntable.segment import NoiseSegment, Segment
from turntable.wheel import ModelWithdrawnWarning, Wheel

try:
    __version__ = version("turntable")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
    "ModelWithdrawnWarning",
    "NoiseSegment",
    "NumericOrbit",
    "Orbit",
    "Residuals",
    "Segment",
    "Wheel",
    "__version__",
    "replace",
]
