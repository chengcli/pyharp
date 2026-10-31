"""The Toon option zenith_correction is not implemented (one umu0 per column, no per-level spherical correction),
so setting it raises NotImplementedError naming the option, from the solver and from a YAML configuration; without
it the solver behaves as before."""

import pytest
import torch

import pyharp


def toon_options(flags):
    op = pyharp.ToonMcKay89Options()
    op.wave_lower([300.0])
    op.wave_upper([600.0])
    op.flags(flags)
    return op


@pytest.mark.parametrize("flags", ["zenith_correction", "planck,zenith_correction", "zenith_correction,hard_surface"])
def test_solver_with_zenith_correction_is_not_implemented(flags):
    with pytest.raises(NotImplementedError, match="zenith_correction"):
        pyharp.ToonMcKay89(toon_options(flags))


@pytest.mark.parametrize("flags", ["", "planck", "planck,hard_surface"])
def test_solver_without_zenith_correction_runs(flags):
    solver = pyharp.ToonMcKay89(toon_options(flags))
    prop = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    prop[..., 0] = 0.5
    kw = {"albedo": torch.zeros(1, 1, dtype=torch.float64)}
    if "planck" in flags:
        kw["temf"] = torch.full((1, 4), 250.0, dtype=torch.float64)
    flux = solver(prop, **kw)
    assert torch.isfinite(flux).all()


YAML = """species:
  - name: H2
    composition: {{H: 2}}
geometry:
  cells:
    nx1: 2
opacities:
  rayleigh:
    type: rayleigh
    species: [H2]
    nmom: 4
bands:
  - name: sw
    range: [10000., 20000.]
    opacities: [rayleigh]
    solver: toon
    nwave: 1
{extra}
"""


@pytest.mark.parametrize("extra", ["    toon:\n      zenith_correction: true", "    flags: zenith_correction"])
def test_yaml_with_zenith_correction_is_not_implemented(tmp_path, extra):
    cfg = tmp_path / "zc.yaml"
    cfg.write_text(YAML.format(extra=extra))
    with pytest.raises(NotImplementedError, match="zenith_correction"):
        pyharp.RadiationOptions.from_yaml(str(cfg))


@pytest.mark.parametrize("extra", ["", "    toon:\n      zenith_correction: false"])
def test_yaml_without_zenith_correction_configures(tmp_path, extra):
    cfg = tmp_path / "zc.yaml"
    cfg.write_text(YAML.format(extra=extra))
    op = pyharp.RadiationOptions.from_yaml(str(cfg))
    assert not op.bands()[0].toon().zenith_correction()
