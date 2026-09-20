"""Guard against version drift between pyproject.toml and __init__.py.

See torch-multioutput-alias-guard's identical test for the fleet-wide
rationale (a real released bug in reboot-safety-check where these two
values drifted).
"""
from __future__ import annotations

import re
from pathlib import Path

from torch_memory_budget_rng_guard import __version__

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def test_init_version_matches_pyproject_version():
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject_path.read_text()
    match = _VERSION_RE.search(text)
    assert match, 'Could not find `version = "..."` in pyproject.toml'
    pyproject_version = match.group(1)
    assert __version__ == pyproject_version, (
        f"__init__.py __version__ ({__version__!r}) does not match "
        f"pyproject.toml's [project].version ({pyproject_version!r}). "
        "These must be bumped together on every release."
    )
