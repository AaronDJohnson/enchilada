"""The README is the first code a new user copies, and nothing else in the
suite touches it.

Three levels of check, weakest to strongest:

* every fenced ``python`` block must parse -- catches syntax rot;
* every call to a turntable API in any block must use keyword arguments that
  actually exist -- catches the rename class of rot (``n_iterations`` ->
  ``full_turns``) even in illustrative fragments that cannot be executed;
* blocks tagged ``<!-- runnable -->`` must run to completion in a fresh
  interpreter -- the full check, for blocks that are self-contained.
"""

import ast
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import turntable
from turntable import Residuals, Wheel

README = Path(__file__).resolve().parent.parent / "README.md"

# Fenced python blocks, each with a flag for whether the line above tagged it
# runnable. Fragments (`MyBlock(...)`, an undefined `time_series`) deliberately
# are not: they illustrate a call, they are not programs.
_BLOCK = re.compile(r"(<!-- runnable -->\n)?```python\n(.*?)```", re.S)


def _blocks():
    return [
        (i, bool(tag), textwrap.dedent(body))
        for i, (tag, body) in enumerate(_BLOCK.findall(README.read_text()))
    ]


def test_the_readme_has_blocks_to_check():
    """Guard the regex itself: a fence-syntax change must not silently turn
    this whole module into a no-op that passes."""
    blocks = _blocks()
    assert len(blocks) >= 3
    assert sum(runnable for _, runnable, _ in blocks) >= 1


@pytest.mark.parametrize("index", [i for i, _, _ in _blocks()])
def test_every_block_parses(index):
    _, _, body = _blocks()[index]
    ast.parse(body)  # SyntaxError is the failure


# Public callables a README block may invoke, by the name it is called under.
_CALLABLES = {
    "Residuals": Residuals,
    "Wheel": Wheel,
    "check_block": None,  # filled below; testing imports lazily
}


def _keyword_targets():
    from turntable.testing import check_block

    targets = dict(_CALLABLES)
    targets["check_block"] = check_block
    # bound-method names are unambiguous across the two public classes
    for cls in (Wheel, Residuals):
        for name, member in vars(cls).items():
            if not name.startswith("_") and callable(member):
                targets.setdefault(name, member)
    return targets


@pytest.mark.parametrize("index", [i for i, _, _ in _blocks()])
def test_calls_into_turntable_use_real_keywords(index):
    """A renamed parameter leaves the README compiling but wrong. Check the
    keywords against the live signatures instead of trusting prose."""
    _, _, body = _blocks()[index]
    targets = _keyword_targets()

    for node in ast.walk(ast.parse(body)):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        target = targets.get(name)
        if target is None:
            continue
        params = inspect.signature(target).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs at the call site
                continue
            assert kw.arg in params, (
                f"README block {index}: {name}(...) passes {kw.arg!r}, which is "
                f"not a parameter of {target!r}. Signature: {list(params)}"
            )


@pytest.mark.parametrize(
    "index", [i for i, runnable, _ in _blocks() if runnable] or [pytest.param(0)]
)
def test_runnable_blocks_actually_run(index):
    _, runnable, body = _blocks()[index]
    assert runnable, "no block is tagged <!-- runnable --> in README.md"
    result = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"README block {index} failed:\n{result.stderr}"


def test_readme_names_only_real_exports():
    """Every `turntable.X` the README mentions must exist -- a dropped export
    should fail here, not in a new user's first session."""
    # Strip URLs first: the clone line ends in "turntable.git", which is not
    # an attribute lookup.
    prose = re.sub(r"https?://\S+", "", README.read_text())
    mentioned = set(re.findall(r"\bturntable\.([A-Za-z_][A-Za-z0-9_]*)", prose))
    submodules = {"testing", "orbits", "residuals", "block", "wheel"}
    for name in mentioned - submodules:
        assert hasattr(turntable, name), (
            f"README references turntable.{name}, which does not exist"
        )
