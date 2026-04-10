"""Runtime configuration for HITRAN spectroscopy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SpectralBandConfig:
    """Configuration for one spectral band on a regular wavenumber grid."""

    name: str
    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    resolution_cm1: float = 1.0

    def grid(self) -> np.ndarray:
        """Return the inclusive wavenumber grid."""
        if self.resolution_cm1 <= 0:
            raise ValueError("resolution_cm1 must be positive")
        span_cm1 = self.wavenumber_max_cm1 - self.wavenumber_min_cm1
        if span_cm1 < 0:
            raise ValueError("wavenumber_max_cm1 must be >= wavenumber_min_cm1")
        count = int(round(span_cm1 / self.resolution_cm1))
        if not np.isclose(
            span_cm1,
            count * self.resolution_cm1,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "wavenumber range must be an integer multiple of resolution_cm1 "
                "to produce an inclusive grid"
            )
        return self.wavenumber_min_cm1 + np.arange(count + 1, dtype=np.float64) * self.resolution_cm1


@dataclass(frozen=True)
class HitranSpecies:
    """Resolved HITRAN metadata for one molecular species."""

    name: str
    molecule_id: int
    isotopologue_ids: tuple[int, ...]
    cia_filename: str | None = None

    @property
    def line_table_prefix(self) -> str:
        """Return the cache prefix used for the species line table."""
        return f"{self.name.lower()}_lines"

    @property
    def cia_pair(self) -> str:
        """Return the self-broadening CIA pair name."""
        return f"{self.name}-{self.name}"


@dataclass(frozen=True)
class HitranCiaPair:
    """Resolved HITRAN CIA metadata for one collisional pair."""

    pair: str
    filename: str


def _normalize_cia_pair_name(pair: str) -> str:
    return "-".join(part.strip().upper() for part in str(pair).split("-") if part.strip())


_HITRAN_CIA_PAIRS_BY_NAME: dict[str, HitranCiaPair] = {
    "CO2-CO2": HitranCiaPair(pair="CO2-CO2", filename="CO2-CO2_2024.cia"),
    "H2-H2": HitranCiaPair(pair="H2-H2", filename="H2-H2_2011.cia"),
    "CH4-CH4": HitranCiaPair(pair="CH4-CH4", filename="CH4-CH4_2011.cia"),
    "N2-N2": HitranCiaPair(pair="N2-N2", filename="N2-N2_2021.cia"),
    "CO2-CH4": HitranCiaPair(pair="CO2-CH4", filename="CO2-CH4_2024.cia"),
    "CO2-H2": HitranCiaPair(pair="CO2-H2", filename="CO2-H2_2024.cia"),
    "H2-HE": HitranCiaPair(pair="H2-He", filename="H2-He_2011.cia"),
    "N2-CH4": HitranCiaPair(pair="N2-CH4", filename="N2-CH4_2024.cia"),
}


_HITRAN_SPECIES_BY_NAME: dict[str, HitranSpecies] = {
    "CO2": HitranSpecies(
        name="CO2",
        molecule_id=2,
        isotopologue_ids=(1, 2, 3, 4, 5, 6, 7),
        cia_filename="CO2-CO2_2024.cia",
    ),
    "H2O": HitranSpecies(
        name="H2O",
        molecule_id=1,
        isotopologue_ids=(1, 2, 3, 4, 5, 6, 7),
        cia_filename=None,
    ),
    "H2": HitranSpecies(
        name="H2",
        molecule_id=45,
        isotopologue_ids=(1, 2),
        cia_filename="H2-H2_2011.cia",
    ),
    "CH4": HitranSpecies(
        name="CH4",
        molecule_id=6,
        isotopologue_ids=(1, 2, 3, 4),
        cia_filename="CH4-CH4_2011.cia",
    ),
    "N2": HitranSpecies(
        name="N2",
        molecule_id=22,
        isotopologue_ids=(1, 2),
        cia_filename="N2-N2_2021.cia",
    ),
}

_USER_BROADENER_CANONICAL_NAMES: dict[str, str] = {
    "AIR": "air",
    "SELF": "self",
    "H2": "H2",
    "HE": "He",
    "CO2": "CO2",
}

_HAPI_BROADENER_KEYS: dict[str, str] = {
    "air": "air",
    "self": "self",
    "H2": "h2",
    "He": "he",
    "CO2": "co2",
}


def resolve_hitran_cia_pair(pair: str) -> HitranCiaPair:
    """Resolve a supported HITRAN CIA pair by name, allowing reversed order."""
    key = _normalize_cia_pair_name(pair)
    resolved = _HITRAN_CIA_PAIRS_BY_NAME.get(key)
    if resolved is not None:
        return resolved
    parts = key.split("-")
    if len(parts) == 2:
        reversed_key = f"{parts[1]}-{parts[0]}"
        resolved = _HITRAN_CIA_PAIRS_BY_NAME.get(reversed_key)
        if resolved is not None:
            return resolved
    supported = ", ".join(sorted(metadata.pair for metadata in _HITRAN_CIA_PAIRS_BY_NAME.values()))
    raise ValueError(f"Unsupported CIA pair {pair!r}. Supported pairs: {supported}.")


def supported_hitran_cia_pairs() -> tuple[HitranCiaPair, ...]:
    """Return the built-in HITRAN CIA pair metadata."""
    return tuple(_HITRAN_CIA_PAIRS_BY_NAME[key] for key in sorted(_HITRAN_CIA_PAIRS_BY_NAME))


def supported_hitran_species_names() -> tuple[str, ...]:
    """Return the built-in HITRAN line-species names."""
    return tuple(sorted(_HITRAN_SPECIES_BY_NAME))


def parse_broadening_composition(value: str | Mapping[str, float] | None) -> dict[str, float] | None:
    """Parse and normalize a line-broadening composition specification."""
    if value is None:
        return None
    entries = value.items() if isinstance(value, Mapping) else _split_broadening_string(value)
    totals: dict[str, float] = {}
    for name, fraction_value in entries:
        broadener_name = _canonical_broadener_name(name)
        try:
            fraction = float(fraction_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid broadening fraction {fraction_value!r} for {name!r}.") from exc
        if fraction < 0.0:
            raise ValueError("broadening fractions must be non-negative")
        totals[broadener_name] = totals.get(broadener_name, 0.0) + fraction
    if not totals:
        raise ValueError("broadening composition must contain at least one BROADENER:FRACTION entry")
    total_fraction = sum(totals.values())
    if total_fraction <= 0.0:
        raise ValueError("broadening fractions must sum to a positive value")
    return {name: fraction / total_fraction for name, fraction in totals.items()}


def resolve_broadening_diluent(
    absorber_species: str,
    composition: str | Mapping[str, float] | None,
) -> dict[str, float]:
    """Resolve a user-facing broadening composition into HAPI Diluent keys."""
    parsed = parse_broadening_composition(composition)
    if parsed is None:
        return {"self": 1.0}
    absorber = resolve_hitran_species(absorber_species).name
    diluent: dict[str, float] = {}
    for broadener_name, fraction in parsed.items():
        if broadener_name == absorber:
            key = "self"
        else:
            key = _HAPI_BROADENER_KEYS.get(broadener_name, str(broadener_name).strip().lower())
        diluent[key] = diluent.get(key, 0.0) + float(fraction)
    return diluent


def _split_broadening_string(value: str) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for chunk in str(value).split(","):
        piece = chunk.strip()
        if not piece:
            continue
        name, sep, fraction_text = piece.partition(":")
        if not sep:
            raise ValueError("broadening entries must have the form BROADENER:FRACTION")
        entries.append((name.strip(), fraction_text.strip()))
    return tuple(entries)


def _canonical_broadener_name(name: object) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("broadening species names must be non-empty")
    upper = text.upper()
    if upper in _USER_BROADENER_CANONICAL_NAMES:
        return _USER_BROADENER_CANONICAL_NAMES[upper]
    try:
        return resolve_hitran_species(text).name
    except ValueError:
        return text


def resolve_hitran_species(name: str) -> HitranSpecies:
    """Resolve a supported HITRAN species by name."""
    key = str(name).upper()
    try:
        return _HITRAN_SPECIES_BY_NAME[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_HITRAN_SPECIES_BY_NAME))
        raise ValueError(f"Unsupported species {name!r}. Supported species: {supported}.") from exc


@dataclass(frozen=True)
class SpectroscopyConfig:
    """Configuration shared by single-state spectroscopy calculations."""

    output_path: Path
    hitran_cache_dir: Path
    species_name: str = "CO2"
    isotopologue_ids: tuple[int, ...] | None = None
    broadening_composition: dict[str, float] | str | None = None
    cia_index_url: str = "https://hitran.org/cia/"
    refresh_hitran: bool = False
    min_line_strength: float = 1.0e-27

    @property
    def hitran_species(self) -> HitranSpecies:
        """Return the resolved HITRAN species metadata."""
        return resolve_hitran_species(self.species_name)

    @property
    def molecule_id(self) -> int:
        """Return the HITRAN molecule id for the selected species."""
        return self.hitran_species.molecule_id

    @property
    def cia_pair(self) -> str:
        """Return the configured self-CIA pair name."""
        return self.hitran_species.cia_pair

    @property
    def cia_filename(self) -> str:
        """Return the configured CIA filename for the selected species."""
        cia_filename = self.hitran_species.cia_filename
        if cia_filename is None:
            raise ValueError(f"No CIA file is configured for species {self.hitran_species.name}.")
        return cia_filename

    def resolved_isotopologue_ids(self) -> tuple[int, ...]:
        """Return the active local isotopologue ids for the selected species."""
        if self.isotopologue_ids is None:
            return self.hitran_species.isotopologue_ids
        return tuple(int(value) for value in self.isotopologue_ids)

    def resolved_broadening_composition(self) -> dict[str, float] | None:
        """Return the normalized user-facing broadening composition."""
        return parse_broadening_composition(self.broadening_composition)

    def resolved_line_diluent(self) -> dict[str, float]:
        """Return the HAPI Diluent mapping for the selected absorber."""
        return resolve_broadening_diluent(self.hitran_species.name, self.broadening_composition)

    def resolved_line_table_name(self, band: SpectralBandConfig) -> str:
        """Return a cache-stable HITRAN table name keyed by spectral bounds."""
        lower = band.wavenumber_min_cm1
        upper = band.wavenumber_max_cm1
        lower_tag = str(int(round(lower)))
        upper_tag = str(int(round(upper)))
        return f"{self.hitran_species.line_table_prefix}_{lower_tag}_{upper_tag}"

    def ensure_directories(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.hitran_cache_dir.mkdir(parents=True, exist_ok=True)
