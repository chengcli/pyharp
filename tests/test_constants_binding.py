#!/usr/bin/env python3
"""pyharp.constants exposes every name its stub (python/constants.pyi) declares.

The Boltzmann constant was bound as ``kBtolz``, so ``pyharp.constants.kBoltz``
raised AttributeError although the stub declares ``kBoltz``.
"""

import re
from pathlib import Path

import pyharp
from pyharp import constants

STUB = Path(pyharp.__file__).with_name("constants.pyi")


def stub_names():
    names = re.findall(r"^(\w+)\s*:\s*float", STUB.read_text(), flags=re.M)
    assert names, f"no names found in {STUB}"
    return names


def test_kboltz_is_bound():
    assert isinstance(constants.kBoltz, float)
    assert 1.38e-23 < constants.kBoltz < 1.39e-23


def test_every_stub_name_is_bound():
    missing = [n for n in stub_names() if not hasattr(constants, n)]
    assert missing == [], f"declared in constants.pyi but not bound: {missing}"
