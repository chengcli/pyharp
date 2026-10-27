"""pyharp declares the pydisort versions it runs with: 1.7.x. With pydisort 1.8.13 the DISORT test of
test_rayleigh.py passes its assertions and then the process aborts with "double free or corruption" (pydisort 1.8
changed the layout of DisortImpl, which pyharp's library allocates), so the requirement excludes 1.8 until that is
fixed."""

import re
from pathlib import Path

import pytest

packaging = pytest.importorskip("packaging")
from packaging.requirements import Requirement  # noqa: E402

_CONFIGURED = "@CMAKE_CURRENT_SOURCE_DIR@"
ROOT = (Path(_CONFIGURED) if not _CONFIGURED.startswith("@") else Path(__file__).resolve().parent).parent


def pydisort_requirements():
    text = (ROOT / "pyproject.toml").read_text()
    reqs = [Requirement(r) for r in re.findall(r'"(pydisort[^"]*)"', text)]
    assert len(reqs) == 2, "pydisort appears in build-system.requires and in project.dependencies"
    return reqs


@pytest.mark.parametrize("version, allowed", [("1.7.1", True), ("1.7.9", True), ("1.8.0", False),
                                              ("1.8.13", False), ("1.7.0", False)])
def test_requirement_excludes_pydisort_1_8(version, allowed):
    for req in pydisort_requirements():
        assert req.specifier.contains(version) == allowed, f"{req} and pydisort {version}"
