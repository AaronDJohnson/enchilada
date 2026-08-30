from dataclasses import replace  # re-exported: every block needs it
from importlib.metadata import PackageNotFoundError, version

from enchilada.block import Block, NoiseBlock
from enchilada.data import L1Data
from enchilada.orbits import NumericOrbit, Orbit
from enchilada.template import Template
from enchilada.wheel import NoiseOverwrittenWarning, Wheel

try:
    __version__ = version("enchilada")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Block",
    "NoiseBlock",
    "NoiseOverwrittenWarning",
    "NumericOrbit",
    "Orbit",
    "L1Data",
    "Template",
    "Wheel",
    "__version__",
    "replace",
]
