from dataclasses import replace  # convenience for frozen data/result snapshots
from importlib.metadata import PackageNotFoundError, version

from enchilada.block import Block
from enchilada.block_result import BlockResult
from enchilada.covariance import DataCovariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid
from enchilada.ledger import Ledger
from enchilada.orbits import NumericOrbit, Orbit
from enchilada.translated_covariance import TranslatedCovariance
from enchilada.translation import transform
from enchilada.wheel import NoiseOverwrittenWarning, Wheel

try:
    __version__ = version("enchilada")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Block",
    "DataCovariance",
    "Ledger",
    "NoiseOverwrittenWarning",
    "NumericOrbit",
    "Orbit",
    "L1Data",
    "BlockResult",
    "Wheel",
    "WDMGrid",
    "TranslatedCovariance",
    "__version__",
    "replace",
    "transform",
]
