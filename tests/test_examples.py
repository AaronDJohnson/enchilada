"""The shipped examples run, and the toy fit actually converges to truth."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "demo.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "residual RMS" in proc.stdout


def test_toy_fit_converges_to_truth():
    toy = load_example("toy_fit")
    results = toy.run_toy_fit(n_cycles=200, burn_in=80, seed=0)
    truth = {
        "slow": toy.TRUTH["slow"],
        "fast": toy.TRUTH["fast"],
        "noise": toy.TRUTH["sigma"],
    }
    for name, (mean, std) in results.items():
        assert std > 0.0
        assert mean == pytest.approx(truth[name], abs=max(5 * std, 0.05)), (
            f"{name}: posterior {mean} +/- {std} vs truth {truth[name]}"
        )


def test_notebook_code_cells_execute():
    import json

    nb = json.loads((EXAMPLES / "demo.ipynb").read_text())
    code = "\n\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    pytest.importorskip("scipy", reason="notebook orbit cells need scipy")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
