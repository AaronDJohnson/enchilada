from dataclasses import replace  # convenience for frozen data/state snapshots
from importlib.metadata import PackageNotFoundError, version

from enchilada.block import Block
from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid
from enchilada.ledger import Ledger
from enchilada.translation import transform
from enchilada.wheel import NoiseOverwrittenWarning, Wheel

try:
    __version__ = version("enchilada")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Block",
    "Covariance",
    "Ledger",
    "NoiseOverwrittenWarning",
    "L1Data",
    "BlockState",
    "Wheel",
    "WDMGrid",
    "__version__",
    "replace",
    "transform",
]
