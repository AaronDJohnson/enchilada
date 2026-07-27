"""The py.typed promise: consumer typos are static errors, not silent Any.

turntable ships py.typed, so its annotations are load-bearing for consumers.
`Residuals.__getattr__` is therefore hidden from type checkers -- any
`__getattr__` would tell a checker that every attribute name exists. These
tests pin that, because the guarantee is documented and easy to regress
(annotating the method `-> Never` looks equivalent and is not: Never is
assignable to everything, so it suppresses the error instead of raising it).
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

CONSUMER = """
import numpy as np
from turntable import Residuals

r = Residuals(tdi={"A": np.zeros(8)}, sample_rate=1.0, channels=("A",),
              tdi_generation="2.0", observable="strain")
print(r.Tobs)      # correct spelling: must NOT error
print(r.Tobbs)     # typo in expression position
y: str = r.T_obs   # typo in assignment position
"""


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_consumer_typos_are_static_errors(tmp_path):
    f = tmp_path / "consumer.py"
    f.write_text(textwrap.dedent(CONSUMER))
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(f)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert 'has no attribute "Tobbs"' in out, out
    assert 'has no attribute "T_obs"' in out, out
    # exactly the two typos -- the correct `r.Tobs` on the preceding line must
    # not be flagged (an over-broad ban would break legitimate access)
    assert out.count("[attr-defined]") == 2, out


def test_runtime_attribute_hints_still_work():
    import numpy as np

    from turntable import Residuals

    r = Residuals(
        tdi={"A": np.zeros(8)},
        sample_rate=1.0,
        channels=("A",),
        tdi_generation="2.0",
        observable="strain",
    )
    assert hasattr(r, "Tobs") and not hasattr(r, "Tobbs")
    with pytest.raises(AttributeError, match="did you mean 'Tobs'"):
        _ = r.T_obs


def test_segment_protocol_is_runtime_checkable():
    from turntable import Segment
    from turntable.testing import EchoSegment

    assert isinstance(EchoSegment("e"), Segment)

    class NotASegment:
        pass

    assert not isinstance(NotASegment(), Segment)


def test_replace_is_re_exported():
    import dataclasses

    import turntable

    assert turntable.replace is dataclasses.replace
    assert "replace" in turntable.__all__


def test_public_surface_is_pinned():
    """__all__ governs re-export for consumers running mypy --strict, so a
    dropped entry silently becomes a type error downstream while CI stays
    green. Pin it literally."""
    import turntable

    assert turntable.__all__ == [
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
    for name in turntable.__all__:
        assert hasattr(turntable, name), name
