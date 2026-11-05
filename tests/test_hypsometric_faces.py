"""calc_dz_hypsometric with layer-centre pressures and the grid's face positions zf: on a stretched column the
layer thicknesses must be the grid's own widths (isothermal, so ln p is linear in z and the answer is exact)."""

import pytest
import torch

from pyharp import calc_dz_hypsometric

R, T, G, P0, L = 3600.0, 300.0, 24.0, 1.0e5, 3.6e5  # L = 8 scale heights
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def faces(n, q, device):
    w = q ** torch.arange(n, dtype=torch.float64)
    return torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(w, 0)]).to(device) * (L / w.sum().item())


def column(zf):
    zc = 0.5 * (zf[1:] + zf[:-1])
    pres = P0 * torch.exp(-zc * G / (R * T))
    temp = torch.full_like(zc, T)
    return pres, temp, torch.tensor(G / R, dtype=torch.float64, device=zf.device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1.058, 1.0 / 1.058, 1.0], ids=["g+1.058", "g-1.058", "uniform"])
@pytest.mark.parametrize("n", [32, 64])
def test_centre_pressures_with_faces_give_the_grid_widths(n, q, device):
    zf = faces(n, q, device)
    pres, temp, g_ov_r = column(zf)
    dz = calc_dz_hypsometric(pres, temp, g_ov_r, zf=zf)
    err = ((dz - zf.diff()) / zf.diff()).abs().max().item()
    # dz comes from ln p differences: round-off bound 64 eps max|ln p| / min d(ln p)
    tol = 64 * 2.2e-16 * pres.log().abs().max().item() / (zf.diff().min().item() * G / (R * T))
    assert err < tol, f"max relative error {err:.3e} > {tol:.1e} (n={n}, q={q}, {device})"


@pytest.mark.parametrize("device", DEVICES)
def test_faces_must_have_nlyr_plus_one_entries(device):
    zf = faces(8, 1.058, device)
    pres, temp, g_ov_r = column(zf)
    with pytest.raises(RuntimeError):
        calc_dz_hypsometric(pres, temp, g_ov_r, zf=zf[:-1])
