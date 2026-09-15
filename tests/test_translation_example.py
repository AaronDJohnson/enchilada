"""The optional translation walkthrough preserves data and likelihood weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wdm", reason="translation example needs the local WDM backend")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples/domain_translation.py"


def test_translation_example_preserves_data_and_noise_weight_across_domains():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["wdm_shapes"] == {
        "frequency_selector": [9, 16],
        "time_selector": [17, 8],
    }
    assert report["frequency_roundtrip_max_error"] < 1e-12
    assert report["wdm_roundtrip_max_error"] < 1e-12
    assert report["result_roundtrip_max_error"] < 1e-12
    assert report["result_model_parameters"] == {"amplitude": 0.1}
    np.testing.assert_allclose(
        list(report["quadratic_forms"].values()),
        report["quadratic_forms"]["time"],
        rtol=1e-12,
    )
    assert report["block_domains"] == ["time", "frequency", "wdm"]
    assert report["canonical_domain"] == "time"
    assert report["residual_max_error"] < 1e-12
    assert report["sample_calls"] == {"time": 2, "frequency": 2, "wdm": 2}


def test_translation_notebook_runs_all_cells_with_real_backend(tmp_path):
    """Execute the shipped walkthrough, including its numerical assertions."""
    pytest.importorskip("matplotlib", reason="translation notebook draws WDM grids")
    notebook_path = EXAMPLE.with_suffix(".ipynb")
    notebook = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    # Finish drawing every figure so plotting failures cannot hide behind Agg's
    # deferred rendering. The cells' own checks cover transforms and Wheel state.
    code += "\nfor number in plt.get_fignums():\n    plt.figure(number).canvas.draw()\n"
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=EXAMPLE.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
