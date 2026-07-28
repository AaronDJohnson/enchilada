from dataclasses import replace  # re-exported: every block needs it
from importlib.metadata import PackageNotFoundError, version

from turntable.block import Block, NoiseBlock
from turntable.orbits import NumericOrbit, Orbit
from turntable.residuals import Residuals
from turntable.wheel import (
    ModelWithdrawnWarning,
    NoiseOverwrittenWarning,
    Wheel,
)

try:
    __version__ = version("turntable")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Block",
    "ModelWithdrawnWarning",
    "NoiseBlock",
    "NoiseOverwrittenWarning",
    "NumericOrbit",
    "Orbit",
    "Residuals",
    "Wheel",
    "__version__",
    "replace",
]
