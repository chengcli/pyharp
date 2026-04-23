"""Shared xarray dataset helpers for spectroscopy products."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_NETCDF_ENGINE = "scipy"
WAVENUMBER_ATTRS = {"long_name": "wavenumber", "units": "cm^-1"}
HDF5_USE_FILE_LOCKING_ENV = "HDF5_USE_FILE_LOCKING"
HDF5_USE_FILE_LOCKING_DISABLED = "FALSE"


def clean_var_token(value: object) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")
    return "_".join(part for part in token.split("_") if part)


def add_wavenumber_attrs(dataset: xr.Dataset) -> xr.Dataset:
    if "wavenumber" in dataset:
        dataset["wavenumber"].attrs = dict(WAVENUMBER_ATTRS)
    return dataset


def build_state_attrs(
    *,
    species_name: str,
    temperature_k: float,
    pressure_pa: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "species_name": str(species_name),
        "temperature_k": float(temperature_k),
        "pressure_pa": float(pressure_pa),
        "pressure_bar": float(pressure_pa) / 1.0e5,
    }
    if extra:
        attrs.update(extra)
    return attrs


def combine_band_datasets(datasets: list[xr.Dataset], *, wn_ranges: list[tuple[float, float]]) -> xr.Dataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    max_points = max(int(dataset.sizes["wavenumber"]) for dataset in datasets)
    band_count = len(datasets)
    coords: dict[str, object] = {
        "band": ("band", np.arange(band_count, dtype=np.int64)),
    }
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    wavenumber_values = np.full((band_count, max_points), np.nan, dtype=np.float64)
    band_sizes = np.zeros(band_count, dtype=np.int64)
    band_min = np.zeros(band_count, dtype=np.float64)
    band_max = np.zeros(band_count, dtype=np.float64)

    variable_names = tuple(datasets[0].data_vars)
    for name in variable_names:
        data_vars[name] = (("band", "wavenumber"), np.full((band_count, max_points), np.nan, dtype=np.float64))

    for band_index, (dataset, wn_range) in enumerate(zip(datasets, wn_ranges, strict=True)):
        size = int(dataset.sizes["wavenumber"])
        band_sizes[band_index] = size
        band_min[band_index], band_max[band_index] = wn_range
        wavenumber_values[band_index, :size] = np.asarray(dataset["wavenumber"].values, dtype=np.float64)
        for name in variable_names:
            data_vars[name][1][band_index, :size] = np.asarray(dataset[name].values, dtype=np.float64)

    data_vars["band_size"] = (("band",), band_sizes)
    data_vars["band_wavenumber_min"] = (("band",), band_min)
    data_vars["band_wavenumber_max"] = (("band",), band_max)
    coords["wavenumber"] = (("band", "wavenumber"), wavenumber_values)

    combined = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(datasets[0].attrs))
    combined.attrs["num_bands"] = band_count
    for name in variable_names:
        combined[name].attrs = dict(datasets[0][name].attrs)
    if "wavenumber" in datasets[0]:
        combined["wavenumber"].attrs = dict(datasets[0]["wavenumber"].attrs)
    combined["band_size"].attrs = {"long_name": "band sample count", "units": "1"}
    combined["band_wavenumber_min"].attrs = {"long_name": "band minimum wavenumber", "units": "cm^-1"}
    combined["band_wavenumber_max"].attrs = {"long_name": "band maximum wavenumber", "units": "cm^-1"}
    return combined


@contextmanager
def hdf5_file_locking_disabled():
    previous_value = os.environ.get(HDF5_USE_FILE_LOCKING_ENV)
    os.environ[HDF5_USE_FILE_LOCKING_ENV] = HDF5_USE_FILE_LOCKING_DISABLED
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(HDF5_USE_FILE_LOCKING_ENV, None)
        else:
            os.environ[HDF5_USE_FILE_LOCKING_ENV] = previous_value


def write_dataset(dataset: xr.Dataset, output_path: Path, *, engine: str = DEFAULT_NETCDF_ENGINE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with hdf5_file_locking_disabled():
        dataset.to_netcdf(output_path, engine=engine)
