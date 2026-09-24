"""Vectorized Voigt line-by-line cross sections that reproduce HAPI's results.

HAPI's ``absorptionCoefficient_Voigt`` loops over lines in Python (~50-100 us per
line). This module evaluates the same model on numpy arrays, one block of
(line, grid point) pairs at a time, following HAPI's conventions exactly:

* ``Sw(T) = sw Q(296)/Q(T) exp(-c2 E''/T)(1 - exp(-c2 nu/T)) / [same at 296 K]``;
  lines with ``Sw(T)`` below the intensity threshold are skipped.
* Doppler HWHM ``sqrt(2 k T ln2 / (m c^2)) nu``, CGS constants from HAPI.
* Lorentz HWHM ``sum_d x_d gamma_d (296/T)^n p`` with ``n = n_d`` when the table
  has it and ``n_air`` otherwise; shift ``sum_d x_d delta_d p`` (0 when missing).
* The profile is evaluated on grid points in ``(nu - wing, nu + wing]`` around the
  unshifted line center and centered at ``nu + Delta0``, with HAPI's Faddeeva
  approximation ``hum1_wei``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .hitemp_lines import C2_CM_K, T_REF_K, _float_field, _int_field, _iter_par_records, _local_iso_ids


LINE_DTYPE = np.dtype(
    [
        ("molec_id", "i2"),
        ("local_iso_id", "i2"),
        ("nu", "f8"),
        ("sw", "f8"),
        ("gamma_air", "f8"),
        ("gamma_self", "f8"),
        ("elower", "f8"),
        ("n_air", "f8"),
        ("delta_air", "f8"),
    ]
)
# Column spans in the 160-character HITRAN record.
_FLOAT_COLUMNS = {
    "nu": (3, 15),
    "sw": (15, 25),
    "gamma_air": (35, 40),
    "gamma_self": (40, 45),
    "elower": (45, 55),
    "n_air": (55, 59),
    "delta_air": (59, 67),
}
_STANDARD_ORDER = [
    "molec_id", "local_iso_id", "nu", "sw", "a", "gamma_air", "gamma_self", "elower", "n_air",
    "delta_air", "global_upper_quanta", "global_lower_quanta", "local_upper_quanta",
    "local_lower_quanta", "ierr", "iref", "line_mixing_flag", "gp", "gpp",
]
_SQRT_LN2 = np.sqrt(np.log(2.0))
_SQRT_PI = np.sqrt(np.pi)
DEFAULT_MAX_PAIRS = 1 << 23

PartitionSum = Callable[[int, int, float], float]
MolecularMass = Callable[[int, int], float]
Faddeeva = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def load_line_table(table_dir: Path, table_name: str) -> np.ndarray:
    """Return a HAPI ``.data`` table as a structured array, cached as ``.npy`` beside it.

    The cache is rebuilt whenever the ``.data`` file changes and is memory-mapped,
    so parallel workers share one copy through the page cache.
    """
    table_dir = Path(table_dir)
    data_path = table_dir / f"{table_name}.data"
    header = json.loads((table_dir / f"{table_name}.header").read_text())
    if header.get("order") != _STANDARD_ORDER or header.get("extra"):
        raise ValueError(f"{table_name} is not a standard 160-character HITRAN table")
    stat = data_path.stat()
    source = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_path = table_dir / f".{table_name}.lines.npy"
    meta_path = table_dir / f".{table_name}.lines.json"
    if cache_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == source:
        return np.load(cache_path, mmap_mode="r")

    blocks = []
    for records in _iter_par_records(data_path):
        block = np.empty(records.shape[0], dtype=LINE_DTYPE)
        block["molec_id"] = _int_field(records, 0, 2)
        block["local_iso_id"] = _local_iso_ids(records)
        for name, (start, stop) in _FLOAT_COLUMNS.items():
            block[name] = _float_field(records, start, stop)
        blocks.append(block)
    lines = np.concatenate(blocks) if blocks else np.empty(0, dtype=LINE_DTYPE)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, lines)
    os.replace(tmp_path, cache_path)
    meta_path.write_text(json.dumps(source))
    return np.load(cache_path, mmap_mode="r")


def voigt_cross_section(
    lines: np.ndarray,
    wavenumber_grid_cm1: np.ndarray,
    *,
    temperature_k: float,
    pressure_atm: float,
    diluent: Mapping[str, float],
    intensity_threshold: float,
    wing_cm1: float,
    partition_sum: PartitionSum,
    molecular_mass: MolecularMass,
    faddeeva: Faddeeva,
    boltzmann_cgs: float,
    speed_of_light_cgs: float,
    subtract_wing_pedestal: bool = False,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> np.ndarray:
    """Return the Voigt cross section in cm^2/molecule on a sorted wavenumber grid.

    ``subtract_wing_pedestal`` removes each line's value at the wing cutoff and clips
    at zero, as pyharp does for H2O so that MT_CKD supplies the far wings.
    """
    grid = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("wavenumber grid must be sorted")
    xsect = np.zeros(grid.shape, dtype=np.float64)
    if lines.shape[0] == 0 or grid.size == 0:
        return xsect
    temperature = float(temperature_k)
    pressure = float(pressure_atm)

    nu = np.asarray(lines["nu"], dtype=np.float64)
    strength = _line_intensity(lines, temperature, partition_sum)
    keep = strength >= intensity_threshold
    if not keep.any():
        return xsect
    lines, nu, strength = lines[keep], nu[keep], strength[keep]

    gamma_doppler = nu * _doppler_factor(lines, temperature, molecular_mass, boltzmann_cgs, speed_of_light_cgs)
    gamma_lorentz, shift = _pressure_parameters(lines, temperature, pressure, diluent)
    cte = _SQRT_LN2 / gamma_doppler
    y = gamma_lorentz * cte
    amplitude = strength * cte / _SQRT_PI
    if subtract_wing_pedestal:
        # pyharp's H2O pedestal is taken at (nu - Delta0) + wing, i.e. 25 - 2*Delta0 from the shifted center.
        pedestal = amplitude * faddeeva((wing_cm1 - 2.0 * shift) * cte, y)[0]

    lower = np.searchsorted(grid, nu - wing_cm1, side="right")
    upper = np.searchsorted(grid, nu + wing_cm1, side="right")
    counts = upper - lower
    ends = np.cumsum(counts)
    start = 0
    while start < nu.size:
        # Take lines until the block holds about max_pairs (line, grid point) pairs.
        stop = max(start + 1, int(np.searchsorted(ends, (ends[start - 1] if start else 0) + max_pairs, side="right")))
        stop = min(stop, nu.size)
        block_counts = counts[start:stop]
        total = int(block_counts.sum())
        if total:
            line_index = np.repeat(np.arange(start, stop), block_counts)
            offsets = np.cumsum(block_counts) - block_counts
            point_index = lower[line_index] + np.arange(total) - np.repeat(offsets, block_counts)
            x = (grid[point_index] - nu[line_index] - shift[line_index]) * cte[line_index]
            values = amplitude[line_index] * faddeeva(x, y[line_index])[0]
            if subtract_wing_pedestal:
                values = np.maximum(values - pedestal[line_index], 0.0)
            xsect += np.bincount(point_index, weights=values, minlength=grid.size)
        start = stop
    return xsect


def volume_concentration_cm3(pressure_atm: float, temperature_k: float, boltzmann_cgs: float) -> float:
    """Return the number density in molecules/cm^3, as HAPI's ``volumeConcentration``."""
    return (pressure_atm / 9.869233e-7) / (boltzmann_cgs * temperature_k)


def _line_intensity(lines: np.ndarray, temperature: float, partition_sum: PartitionSum) -> np.ndarray:
    nu = np.asarray(lines["nu"], dtype=np.float64)
    elower = np.asarray(lines["elower"], dtype=np.float64)
    ratio = np.empty(nu.shape)
    for (molec, iso), sel in _groups(lines):
        ratio[sel] = partition_sum(molec, iso, T_REF_K) / partition_sum(molec, iso, temperature)
    boltzmann_t = np.exp(-C2_CM_K * elower / temperature) * (1.0 - np.exp(-C2_CM_K * nu / temperature))
    boltzmann_ref = np.exp(-C2_CM_K * elower / T_REF_K) * (1.0 - np.exp(-C2_CM_K * nu / T_REF_K))
    return np.asarray(lines["sw"], dtype=np.float64) * ratio * boltzmann_t / boltzmann_ref


def _doppler_factor(
    lines: np.ndarray, temperature: float, molecular_mass: MolecularMass, boltzmann: float, speed_of_light: float
) -> np.ndarray:
    factor = np.empty(lines.shape[0])
    for (molec, iso), sel in _groups(lines):
        mass_g = molecular_mass(molec, iso) * 1.66053873e-27 * 1000.0
        factor[sel] = np.sqrt(2.0 * boltzmann * temperature * np.log(2.0) / mass_g / speed_of_light**2)
    return factor


def _pressure_parameters(
    lines: np.ndarray, temperature: float, pressure: float, diluent: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.zeros(lines.shape[0])
    shift = np.zeros(lines.shape[0])
    names = set(lines.dtype.names)
    for broadener, fraction in diluent.items():
        if f"gamma_{broadener}" not in names:
            raise ValueError(f"Table has no gamma_{broadener}; resolve the diluent to air/self first.")
        exponent = lines[f"n_{broadener}"] if f"n_{broadener}" in names else lines["n_air"]
        gamma += fraction * lines[f"gamma_{broadener}"] * (T_REF_K / temperature) ** exponent * pressure
        if f"delta_{broadener}" in names:
            shift += fraction * lines[f"delta_{broadener}"] * pressure
    return gamma, shift


def _groups(lines: np.ndarray):
    keys = lines["molec_id"].astype(np.int64) * 1000 + lines["local_iso_id"].astype(np.int64)
    for key in np.unique(keys):
        yield (int(key // 1000), int(key % 1000)), keys == key
