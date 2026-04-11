"""Blackbody helper functions for spectroscopy plots."""

from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZMANN = 1.380649e-23


def compute_normalized_blackbody_curve(
    *,
    wavenumber_cm1: np.ndarray,
    temperature_k: float,
) -> np.ndarray:
    """Return a normalized Planck curve on a wavenumber grid."""
    grid_cm1 = np.asarray(wavenumber_cm1, dtype=np.float64)
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    grid_m1 = np.clip(grid_cm1, 0.0, None) * 100.0
    exponent = (H_PLANCK * C_LIGHT * grid_m1) / (K_BOLTZMANN * float(temperature_k))
    denominator = np.expm1(np.clip(exponent, 0.0, 700.0))
    numerator = 2.0 * H_PLANCK * C_LIGHT**2 * grid_m1**3
    radiance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(grid_m1, dtype=np.float64),
        where=denominator > 0.0,
    )
    peak = float(np.max(radiance)) if radiance.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(grid_m1, dtype=np.float64)
    return radiance / peak
