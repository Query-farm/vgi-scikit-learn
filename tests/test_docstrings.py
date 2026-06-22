"""Docstring-consistency gate.

Runs pydoclint over the worker (the ``vgi_sklearn`` package plus the two
top-level entry modules) as part of the test suite. pydoclint complements
ruff's ``D`` rules: ruff checks docstring *shape*, while pydoclint verifies that
documented arguments, return values, and dataclass attributes actually match
the code. Configuration lives in ``[tool.pydoclint]`` in ``pyproject.toml`` —
this test invokes the same CLI, so there is no duplicated rule set.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGETS = ["vgi_sklearn/", "sklearn_worker.py", "serve.py"]

# A real violation line looks like ``    42: DOC101: ...``. If pydoclint exits
# non-zero without emitting any such code, it failed to *run* (e.g. import
# error) rather than finding violations.
_VIOLATION_RE = re.compile(r"\bDOC\d{3}\b")


def test_pydoclint_clean() -> None:
    """The worker must pass the pydoclint docstring gate (config in pyproject.toml)."""
    pydoclint = shutil.which("pydoclint")
    if pydoclint is None:  # pragma: no cover - dev dependency should always be present
        pytest.skip("pydoclint is not installed")

    result = subprocess.run(
        [pydoclint, *_TARGETS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    if result.returncode == 0:
        return

    # pydoclint depends on ``docstring-parser-fork`` while some VGI deps pull the
    # upstream ``docstring-parser``; both claim the ``docstring_parser`` import
    # namespace, so on some interpreter/OS combinations pydoclint crashes on
    # import. That's a broken tool environment, not a docstring problem — skip
    # rather than fail.
    if not _VIOLATION_RE.search(output):  # pragma: no cover - env-dependent
        pytest.skip(f"pydoclint could not run in this environment:\n{output}")

    pytest.fail(f"pydoclint found docstring violations:\n\n{output}")
