"""pyharp allocates pydisort's DisortImpl, so it must refuse any pydisort but the one it was built against."""

import subprocess
import sys
from importlib.metadata import requires


def test_wheel_pins_pydisort_exactly():
    pins = [r for r in requires("pyharp") if r.startswith("pydisort")]
    assert len(pins) == 1 and pins[0].startswith("pydisort=="), pins


def test_extension_itself_refuses_a_different_pydisort():
    # report a different pydisort, then load the extension directly: the package __init__ is skipped, as it
    # is for any direct load of the .so, so the extension's module init must refuse the version on its own
    code = ("import glob, importlib.metadata as m, importlib.util, os, torch\n"
            "real = m.version\n"
            "m.version = lambda n: '0.0.0' if n == 'pydisort' else real(n)\n"
            "pkg = os.path.dirname(importlib.util.find_spec('pyharp').origin)\n"
            "spec = importlib.util.spec_from_file_location('pyharp', glob.glob(pkg + '/pyharp.*.so')[0])\n"
            "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "pyharp was built against pydisort" in run.stderr, (run.returncode, run.stderr[-500:])
