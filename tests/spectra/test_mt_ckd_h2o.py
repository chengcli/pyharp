from pathlib import Path

import numpy as np
import xarray as xr

from pyharp.spectra.mt_ckd_h2o import default_mt_ckd_h2o_data_path
from pyharp.spectra.mt_ckd_h2o import _radiation_term, compute_mt_ckd_h2o_continuum_cross_section


def test_default_mt_ckd_h2o_data_path_uses_local_external_directory(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)

    assert default_mt_ckd_h2o_data_path() == (
        repo_root / "external" / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"
    )


def test_default_mt_ckd_h2o_data_path_searches_parent_directories(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / "python" / "spectra")

    assert default_mt_ckd_h2o_data_path() == (
        repo_root / "external" / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"
    )


def test_compute_mt_ckd_h2o_continuum_includes_self_and_foreign_terms(tmp_path) -> None:
    reference_grid = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    self_absco_ref = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    foreign_absco_ref = np.array([7.0, 11.0, 13.0, 17.0], dtype=np.float64)
    self_texp = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    data_path = tmp_path / "mt_ckd.nc"
    xr.Dataset(
        data_vars={
            "self_absco_ref": ("wavenumbers", self_absco_ref),
            "for_absco_ref": ("wavenumbers", foreign_absco_ref),
            "self_texp": ("wavenumbers", self_texp),
            "ref_press": xr.DataArray(1000.0),
            "ref_temp": xr.DataArray(250.0),
        },
        coords={"wavenumbers": reference_grid},
    ).to_netcdf(data_path)

    result = compute_mt_ckd_h2o_continuum_cross_section(
        wavenumber_grid_cm1=reference_grid,
        temperature_k=200.0,
        pressure_pa=1.2e5,
        h2o_vmr=0.25,
        foreign_vmr=0.75,
        data_path=data_path,
    )

    rho_ratio = (1200.0 / 1000.0) * (250.0 / 200.0)
    expected = (
        self_absco_ref * (250.0 / 200.0) ** self_texp * 0.25
        + foreign_absco_ref * 0.75
    ) * rho_ratio * _radiation_term(reference_grid, 200.0)
    assert np.allclose(result, expected)


def test_compute_mt_ckd_h2o_continuum_defaults_foreign_to_remaining_fraction(tmp_path) -> None:
    reference_grid = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    data_path = tmp_path / "mt_ckd.nc"
    xr.Dataset(
        data_vars={
            "self_absco_ref": ("wavenumbers", np.ones(4, dtype=np.float64)),
            "for_absco_ref": ("wavenumbers", np.full(4, 2.0, dtype=np.float64)),
            "self_texp": ("wavenumbers", np.zeros(4, dtype=np.float64)),
            "ref_press": xr.DataArray(1000.0),
            "ref_temp": xr.DataArray(250.0),
        },
        coords={"wavenumbers": reference_grid},
    ).to_netcdf(data_path)

    inferred = compute_mt_ckd_h2o_continuum_cross_section(
        wavenumber_grid_cm1=reference_grid,
        temperature_k=250.0,
        pressure_pa=1.0e5,
        h2o_vmr=0.2,
        data_path=data_path,
    )
    explicit = compute_mt_ckd_h2o_continuum_cross_section(
        wavenumber_grid_cm1=reference_grid,
        temperature_k=250.0,
        pressure_pa=1.0e5,
        h2o_vmr=0.2,
        foreign_vmr=0.8,
        data_path=data_path,
    )

    assert np.allclose(inferred, explicit)
