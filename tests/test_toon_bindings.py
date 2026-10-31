"""Python surface of the Toon solver: btop_factor is bound, pyharp.pyi declares ToonMcKay89 and its options, and
the unimplemented "twostr" solver is refused with NotImplementedError when a band is configured."""

import ast
import math
from pathlib import Path

import pytest
import torch

import pyharp

_CONFIGURED = "@CMAKE_CURRENT_SOURCE_DIR@"
ROOT = (Path(_CONFIGURED) if not _CONFIGURED.startswith("@") else Path(__file__).resolve().parent).parent


def test_btop_factor_is_bound_with_its_default():
    op = pyharp.ToonMcKay89Options()
    assert op.btop_factor() == pytest.approx(math.exp(-1.0), rel=1e-15)
    op.btop_factor(0.5)
    assert op.btop_factor() == 0.5


def test_btop_factor_sets_the_top_boundary():
    def toa_down(factor):
        op = pyharp.ToonMcKay89Options()
        op.wave_lower([300.0])
        op.wave_upper([600.0])
        op.flags("planck")
        op.top_emission_flag(-1)   # tau_top = dtau[0] * btop_factor
        op.btop_factor(factor)
        prop = torch.zeros(1, 1, 4, 3, dtype=torch.float64)
        prop[..., 0] = 0.5
        flux = pyharp.ToonMcKay89(op)(prop, temf=torch.full((1, 5), 250.0, dtype=torch.float64),
                                      albedo=torch.zeros(1, 1, dtype=torch.float64))
        return flux[0, 0, -1, 1].item()

    assert toa_down(1.0) > toa_down(0.1) > 0.0


def test_pyi_declares_toon():
    tree = ast.parse((ROOT / "python" / "pyharp.pyi").read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ToonMcKay89Options" in classes and "ToonMcKay89" in classes
    methods = {f.name for f in classes["ToonMcKay89Options"].body if isinstance(f, ast.FunctionDef)}
    assert {"wave_lower", "wave_upper", "flags", "top_emission_flag", "btop_factor", "hard_surface"} <= methods
    assert "forward" in {f.name for f in classes["ToonMcKay89"].body if isinstance(f, ast.FunctionDef)}


TWOSTR_YAML = """species:
  - name: H2
    composition: {H: 2}
geometry:
  cells:
    nx1: 2
opacities:
  rayleigh:
    type: rayleigh
    species: [H2]
    nmom: 4
bands:
  - name: lw
    range: [100., 200.]
    opacities: [rayleigh]
    solver: twostr
    nwave: 1
"""


def test_twostr_from_yaml_is_not_implemented(tmp_path):
    cfg = tmp_path / "twostr.yaml"
    cfg.write_text(TWOSTR_YAML)
    with pytest.raises(NotImplementedError, match="twostr"):
        pyharp.RadiationOptions.from_yaml(str(cfg))


def test_twostr_band_is_not_implemented():
    op = pyharp.RadiationBandOptions()
    op.solver_name("twostr")
    with pytest.raises(NotImplementedError, match="twostr"):
        pyharp.RadiationBand(op)
