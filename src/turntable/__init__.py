from importlib.metadata import PackageNotFoundError, version

from turntable.orbits import NumericOrbit, Orbit
from turntable.residuals import Residuals
from turntable.segment import Catalog, NoiseSegment, Segment, State
from turntable.wheel import Wheel

try:
    __version__ = version("turntable")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = [
    "Catalog",
    "NoiseSegment",
    "NumericOrbit",
    "Orbit",
    "Residuals",
    "Segment",
    "State",
    "Wheel",
    "__version__",
]
