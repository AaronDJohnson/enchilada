from dataclasses import replace  # re-exported: every block needs it
from importlib.metadata import PackageNotFoundError, version

from enchilada.block import Block, NoiseBlock
from enchilada.orbits import NumericOrbit, Orbit
from enchilada.residuals import Residuals
from enchilada.wheel import (
    ModelWithdrawnWarning,
    NoiseOverwrittenWarning,
    Wheel,
)

try:
    __version__ = version("enchilada")
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
