"""Python wrapper for the HITRAN-linked MT_CKD_H2O water continuum model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


_MT_CKD_H2O_RELATIVE_PATH = Path("external") / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"


def _find_repo_root(start: Path) -> Path | None:
    """Return the nearest parent directory that looks like the repository root."""
    for root in (start, *start.parents):
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_mt_ckd_h2o_data_path() -> Path:
    """Return the local MT_CKD_H2O coefficient file path."""
    cwd = Path.cwd().resolve()
    repo_root = _find_repo_root(cwd)
    root = repo_root if repo_root is not None else cwd
    return root / _MT_CKD_H2O_RELATIVE_PATH


def _radiation_term(wavenumber_cm1: np.ndarray, temperature_k: float) -> np.ndarray:
    """Return the MT_CKD radiation term used to convert continuum coefficients into cross sections."""
    radcn2 = 1.4387752
    xkt = float(temperature_k) / radcn2
    wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=np.float64)
    xviokt = wavenumber_cm1 / xkt
    radiation = np.array(wavenumber_cm1, dtype=np.float64, copy=True)

    small = xviokt <= 0.01
    mid = (~small) & (xviokt <= 10.0)
    radiation[small] = 0.5 * xviokt[small] * wavenumber_cm1[small]
    expvkt = np.exp(-xviokt[mid])
    radiation[mid] = wavenumber_cm1[mid] * (1.0 - expvkt) / (1.0 + expvkt)
    return radiation


def compute_mt_ckd_h2o_continuum_cross_section(
    *,
    wavenumber_grid_cm1: np.ndarray,
    temperature_k: float,
    pressure_pa: float,
    h2o_vmr: float = 1.0,
    foreign_vmr: float | None = None,
    data_path: Path | None = None,
) -> np.ndarray:
    """Return MT_CKD H2O continuum absorption cross section in cm^2/molecule."""
    if not (0.0 <= h2o_vmr <= 1.0):
        raise ValueError("h2o_vmr must be between 0 and 1.")
    if foreign_vmr is None:
        foreign_vmr = max(0.0, 1.0 - float(h2o_vmr))
    if not (0.0 <= foreign_vmr <= 1.0):
        raise ValueError("foreign_vmr must be between 0 and 1.")

    resolved_path = default_mt_ckd_h2o_data_path() if data_path is None else Path(data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"MT_CKD_H2O coefficient file not found at {resolved_path}. "
            "Fetch the MT_CKD_H2O repository into external/MT_CKD_H2O first."
        )

    target_grid = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    pressure_mbar = float(pressure_pa) / 100.0

    with xr.open_dataset(resolved_path) as dataset:
        reference_grid = np.asarray(dataset["wavenumbers"].values, dtype=np.float64)
        self_absco_ref = np.asarray(dataset["self_absco_ref"].values, dtype=np.float64)
        foreign_absco_ref = np.asarray(dataset["for_absco_ref"].values, dtype=np.float64)
        self_texp = np.asarray(dataset["self_texp"].values, dtype=np.float64)
        ref_press_mbar = float(dataset["ref_press"].values)
        ref_temp_k = float(dataset["ref_temp"].values)

    rho_ratio = (pressure_mbar / ref_press_mbar) * (ref_temp_k / float(temperature_k))
    sigma_self = self_absco_ref * (ref_temp_k / float(temperature_k)) ** self_texp
    sigma_self = sigma_self * float(h2o_vmr) * rho_ratio
    sigma_foreign = foreign_absco_ref * float(foreign_vmr) * rho_ratio
    continuum_ref = (sigma_self + sigma_foreign) * _radiation_term(reference_grid, float(temperature_k))

    interpolator = interp1d(
        reference_grid,
        continuum_ref,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    continuum = np.asarray(interpolator(target_grid), dtype=np.float64)
    continuum[continuum < 0.0] = 0.0
    return continuum
