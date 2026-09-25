"""The py.typed promise: consumer typos are static errors, not silent Any.

enchilada ships py.typed, so its annotations are load-bearing for consumers.
`L1Data.__getattr__` is therefore hidden from type checkers -- any
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
from dataclasses import dataclass
import numpy as np
from typing import assert_type
from enchilada import (
    Block, L1Data, DataCovariance, TranslatedCovariance, BlockState, Wheel, WDMGrid,
    transform,
)
from enchilada.testing import EchoBlock, check_block

r = L1Data(channel_data={"A": np.zeros(8)}, sample_rate_hz=1.0, channel_names=("A",),
              tdi_generation="2.0", physical_observable="strain")
assert_type(r.channel_names, tuple[str, ...])
inferred = L1Data({"E": np.zeros(8), "A": np.zeros(8)}, sample_rate_hz=1.0,
                  tdi_generation="2.0", physical_observable="strain")
assert_type(inferred.channel_names, tuple[str, ...])
explicit_none = L1Data(channel_data={"A": np.zeros(8)}, channel_names=None,
                       sample_rate_hz=1.0, tdi_generation="2.0",
                       physical_observable="strain")
assert_type(explicit_none.channel_names, tuple[str, ...])
print(r.Tobs)      # correct spelling: must NOT error
print(r.Tobbs)     # typo in expression position
y: str = r.T_obs   # typo in assignment position

assert_type(transform(r, "frequency"), L1Data)
assert_type(transform(r, "wdm", num_frequency_divisions=2).wdm_grid, WDMGrid | None)
noise = DataCovariance.from_variance(r, 1.0)
assert_type(transform(noise, "frequency"), DataCovariance | TranslatedCovariance)
assert_type(transform(BlockState(), "time"), BlockState)
wheel = Wheel(r, transform(noise, "frequency"))
initial = BlockState()
wheel.add(EchoBlock("wdm"), initial_block_state=initial,
          data_domain="wdm", num_time_divisions=2)
wheel.add(EchoBlock("seeded"), initial_block_state=initial)
check_block(EchoBlock("checked"), r, initial_block_state=initial)

class SampleOnly:
    name = "sample_only"

    def sample(
        self, conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

sample_only: Block = SampleOnly()
wheel.add(sample_only, initial_block_state=BlockState())
check_block(sample_only, r, initial_block_state=BlockState())

@dataclass(frozen=True)
class FrozenBlock:
    name: str = "frozen"

    def sample(
        self, conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

frozen: Block = FrozenBlock()
wheel.add(FrozenBlock(), initial_block_state=BlockState())
check_block(FrozenBlock(), r, initial_block_state=BlockState())

@dataclass(frozen=True)
class ExplicitBlock(Block):
    name: str

    def sample(
        self, conditional_residual: L1Data,
        noise_covariance: DataCovariance | TranslatedCovariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

explicit: Block = ExplicitBlock("explicit")
wheel.add(explicit, initial_block_state=BlockState())
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
    assert out.count("error:") == 2, out


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_l1data_positional_settings_are_static_errors(tmp_path):
    consumer = tmp_path / "positional_consumer.py"
    consumer.write_text(
        "import numpy as np\nfrom enchilada import L1Data\n"
        'L1Data({"A": np.zeros(8)}, 1.0, ("A",), "2.0", "strain")\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(consumer)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert 'Too many positional arguments for "L1Data"' in out, out


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_l1data_wdm_grid_constructor_argument_is_a_static_error(tmp_path):
    consumer = tmp_path / "wdm_grid_consumer.py"
    consumer.write_text(
        "import numpy as np\nfrom enchilada import L1Data, WDMGrid\n"
        'L1Data({"A": np.zeros((3, 2))}, sample_rate_hz=1.0,\n'
        '       tdi_generation="2.0", physical_observable="strain",\n'
        '       data_domain="wdm", wdm_grid=WDMGrid(\n'
        "           num_frequency_divisions=2, num_time_divisions=2))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(consumer)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert 'Unexpected keyword argument "wdm_grid" for "L1Data"' in out, out


def test_runtime_attribute_hints_still_work():
    import numpy as np

    from enchilada import L1Data

    r = L1Data(
        channel_data={"A": np.zeros(8)},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="strain",
    )
    assert hasattr(r, "Tobs") and not hasattr(r, "Tobbs")
    with pytest.raises(AttributeError, match="did you mean 'Tobs'"):
        _ = r.T_obs


def test_block_protocol_is_runtime_checkable():
    from enchilada import Block
    from enchilada.testing import EchoBlock

    assert isinstance(EchoBlock("e"), Block)

    class NotABlock:
        pass

    assert not isinstance(NotABlock(), Block)


def test_frozen_dataclass_can_explicitly_inherit_block():
    from dataclasses import dataclass

    import numpy as np

    from enchilada import Block, BlockState, L1Data, Wheel

    @dataclass(frozen=True)
    class ExplicitBlock(Block):
        name: str

        def sample(
            self, conditional_residual, noise_covariance, current_block_state, *, rng
        ):
            return current_block_state

    block = ExplicitBlock("explicit")
    assert isinstance(block, Block)
    observations = L1Data(
        channel_data={"A": np.zeros(4)},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="strain",
    )
    wheel = Wheel(observations)
    wheel.add(block, initial_block_state=BlockState(parameters={"value": 1}))
    wheel.run(1)
    assert wheel.ledger["explicit"].parameters == {"value": 1}


def test_replace_is_re_exported():
    import dataclasses

    import enchilada

    assert enchilada.replace is dataclasses.replace
    assert "replace" in enchilada.__all__


def test_public_surface_is_pinned():
    """__all__ governs re-export for consumers running mypy --strict, so a
    dropped entry silently becomes a type error downstream while CI stays
    green. Pin it literally."""
    import enchilada

    assert enchilada.__all__ == [
        "Block",
        "DataCovariance",
        "Ledger",
        "NoiseOverwrittenWarning",
        "NumericOrbit",
        "Orbit",
        "L1Data",
        "BlockState",
        "Wheel",
        "WDMGrid",
        "TranslatedCovariance",
        "__version__",
        "replace",
        "transform",
    ]
    for name in enchilada.__all__:
        assert hasattr(enchilada, name), name
