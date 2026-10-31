"""Toon longwave in thick layers at and near the resonance lambda mu = 1 of each Gauss angle, against the
high-precision reference of tests/reference/toon_lw_reference.py (250 digits, exact exponentials; table written by
tests/reference/make_toon_lw_resonance.py): float64 within 1e-10 relative, float32 within 1e-5, no inf or nan."""

import json
from pathlib import Path

import pytest
import torch

import pyharp

_CONFIGURED = "@CMAKE_CURRENT_SOURCE_DIR@"
TESTS = Path(_CONFIGURED) if not _CONFIGURED.startswith("@") else Path(__file__).resolve().parent
CASES = json.loads((TESTS / "reference" / "toon_lw_resonance.json").read_text())
TOL = {"float64": 1e-10, "float32": 1e-5}
DTYPE = {"float64": torch.float64, "float32": torch.float32}


def solve(case, device="cpu"):
    dt = DTYPE[case["dtype"]]
    op = pyharp.ToonMcKay89Options()
    op.wave_lower([1.0])
    op.wave_upper([1.0e5])
    op.flags("planck,hard_surface" if case["hard_surface"] else "planck")
    op.top_emission_flag(case["top"])
    solver = pyharp.ToonMcKay89(op)
    solver.to(torch.device(device))
    rows = case["rows"]
    prop = torch.tensor(rows, dtype=dt, device=device).view(1, 1, len(rows), 3)
    temf = torch.tensor(case["temf"], dtype=dt, device=device).view(1, -1)
    return solver(prop, temf=temf, albedo=torch.full((1, 1), case["albedo"], dtype=dt, device=device))[0, 0].cpu()


def check(case, flux):
    ref = torch.tensor(case["ref"], dtype=torch.float64)
    got = flux.to(torch.float64)
    assert torch.isfinite(got).all(), f"{case['label']} ({case['dtype']}): non-finite fluxes {got.tolist()}"
    scale = ref.abs().max()
    tol = TOL[case["dtype"]]
    # relative to each value; values below tol of the largest flux are compared on that scale
    err = ((got - ref).abs() / torch.maximum(ref.abs(), tol * scale)).max().item()
    assert err < tol, f"{case['label']} ({case['dtype']}): max relative error {err:.3e}"


@pytest.mark.parametrize("case", CASES, ids=[f"{c['dtype']}-{c['label']}" for c in CASES])
def test_cpu(case):
    check(case, solve(case))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("case", CASES, ids=[f"{c['dtype']}-{c['label']}" for c in CASES])
def test_cuda(case):
    check(case, solve(case, "cuda"))
