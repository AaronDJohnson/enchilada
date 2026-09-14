"""The block-author tutorial is executable and its sampler targets the posterior."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).resolve().parents[1] / "examples" / "create_block.ipynb"


def test_block_tutorial_runs_and_checks_its_sampler(tmp_path):
    notebook = json.loads(TUTORIAL.read_text())
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    code += (
        "\nif plt is not None:\n"
        "    for number in plt.get_fignums():\n"
        "        plt.figure(number).canvas.draw()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "Known-injection start checks passed" in result.stdout
    assert "Posterior checks passed" in result.stdout
    assert "Continuation and isolation checks passed" in result.stdout
    assert "Domain handoff checks passed" in result.stdout
    assert "Noise-only block checks passed" in result.stdout


@pytest.fixture
def tutorial_definitions():
    """Load the actual taught class without rerunning plots or campaigns."""
    notebook = json.loads(TUTORIAL.read_text())
    module_name = "block_tutorial_for_test"
    import types

    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        for cell in notebook["cells"]:
            if "definitions" in cell.get("metadata", {}).get("tags", []):
                exec("".join(cell["source"]), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_tutorial_sampler_refreshes_target_when_other_blocks_change_it(
    tutorial_definitions,
):
    import numpy as np

    from enchilada import DataCovariance, L1Data

    module = tutorial_definitions
    times = np.arange(64)
    basis = np.sin(2 * np.pi * 0.125 * times)
    data = L1Data(
        channel_data={"A": 2.0 * basis},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = DataCovariance.from_variance(data, 0.2)
    block = module.SineAmplitudeBlock(
        name="signal", frequency_hz=0.125, proposal_std=0.4, steps_per_call=20
    )
    initial_result = block._render(data, 0.0, 0, 0)
    # Identical continuation state and RNG, but the other blocks change the
    # conditional data and covariance. A cached old target would repeat a draw.
    positive = block.sample(data, noise, initial_result, rng=np.random.default_rng(1))
    changed_data = replace(data, channel_data={"A": -2.0 * basis})
    changed_noise = DataCovariance.from_variance(changed_data, 0.4)
    negative = block.sample(
        changed_data, changed_noise, initial_result, rng=np.random.default_rng(1)
    )
    assert positive.model_parameters["amplitude"] > 0
    assert negative.model_parameters["amplitude"] < 0
    assert initial_result.sampler_state["num_proposals"] == 0


def test_tutorial_rejection_returns_retained_signal_and_complete_state(
    tutorial_definitions,
):
    import numpy as np

    from enchilada import DataCovariance, L1Data

    data = L1Data(
        channel_data={"A": np.zeros(64)},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = DataCovariance.from_variance(data, 0.2)
    block = tutorial_definitions.SineAmplitudeBlock(
        name="signal", frequency_hz=0.125, proposal_std=1e12, steps_per_call=8
    )
    initial_result = block._render(data, 0.0, 0, 0)
    result = block.sample(data, noise, initial_result, rng=np.random.default_rng(1))
    assert result.model_parameters == initial_result.model_parameters
    np.testing.assert_array_equal(
        result.tdi_signal_contribution["A"], initial_result.tdi_signal_contribution["A"]
    )
    assert result.sampler_state == {"num_proposals": 8, "num_acceptances": 0}
    assert initial_result.sampler_state == {"num_proposals": 0, "num_acceptances": 0}
