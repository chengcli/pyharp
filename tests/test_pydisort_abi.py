"""pyharp's library allocates pydisort's DisortImpl, so it must run with the pydisort it was built against:
with a different layout (488 vs 496 bytes between 1.7 and 1.8) the constructor overruns the allocation and the
process aborts at exit. A mismatched install must be refused cleanly instead."""

import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

CONFIG = Path(__file__).with_name("rayleigh_test.yaml")
SCRIPT = f"""
import pyharp
op = pyharp.RadiationOptions.from_yaml({str(CONFIG)!r})
band = op.bands()[0]
band.wavenumber([20000.0]); band.weight([1.0])
band.set_wave_lower([19999.5]); band.set_wave_upper([20000.5])
pyharp.disort_config(band.disort(), 4, 2, 1, 1)
rad = pyharp.Radiation(op)
"""


def test_disort_band_process_exits_cleanly_or_is_refused():
    run = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert run.returncode == 0 or "pyharp was built against pydisort" in run.stderr, (run.returncode, run.stderr[-500:])


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
