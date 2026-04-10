"""Tools for single-state spectroscopy calculations."""

from .config import SpectroscopyConfig, SpectralBandConfig
from .spectrum import AbsorptionSpectrum, compute_absorption_spectrum

__all__ = [
    "AbsorptionSpectrum",
    "SpectralBandConfig",
    "SpectroscopyConfig",
    "compute_absorption_spectrum",
]
