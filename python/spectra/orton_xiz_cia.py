"""Download and parse legacy Orton/Xiz H2 CIA tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

import numpy as np

_ORTON_XIZ_ARCHIVE_URL = "https://www.dropbox.com/s/czbzu9hglxty4rj/H2-He-cia.tar.gz?dl=1"
_LOSCHMIDT_M3 = 2.68719e25
_FILES_BY_PAIR_MODEL_STATE: dict[tuple[str, str, str], str] = {
    ("H2-H2", "xiz", "eq"): "H2-H2-eq.xiz.txt",
    ("H2-H2", "xiz", "nm"): "H2-H2-nm.xiz.txt",
    ("H2-H2", "orton", "eq"): "H2-H2-eq.orton.txt",
    ("H2-H2", "orton", "nm"): "H2-H2-nm.orton.txt",
    ("H2-He", "xiz", "eq"): "H2-He-eq.xiz.txt",
    ("H2-He", "xiz", "nm"): "H2-He-nm.xiz.txt",
    ("H2-He", "orton", "eq"): "H2-He-eq.orton.txt",
    ("H2-He", "orton", "nm"): "H2-He-nm.orton.txt",
}
_EXPECTED_SHA256: dict[str, str] = {
    "H2-H2-eq.orton.txt": "2dd91f9897ba0f3a0e5877a88ff259b40dcbba14e6bad4790f6f93a00ba51609",
    "H2-H2-eq.xiz.txt": "63c7f2a3aa7661948a094309fe461b3db7221f89ec8bf1b36682890c13d361fc",
    "H2-H2-nm.orton.txt": "1318d794226b5b737874cbdb4d9a5de5b932c3477a401f5b70b6b9f25877817b",
    "H2-H2-nm.xiz.txt": "1b581fa35c000b707f3a7900a3f539c562267c578755f52bd5f068fc7f183aa8",
    "H2-He-eq.orton.txt": "7eda8cb1e875d0712b156217f8f226157ad0d0133fe159747050c0b486d1bf32",
    "H2-He-eq.xiz.txt": "f522657383a3142d881ffc18814cc5b32b908068a77c28b3e6823292d1b624fd",
    "H2-He-nm.orton.txt": "0f124edb384b24d84ab4c67f26cc5756130e6cbab2d804aa302168cd9c44dc19",
    "H2-He-nm.xiz.txt": "d086c58045aeabd17f3d77efa78f477805785e7fd420e1f488fed8043f17b927",
}


def _normalize_pair_name(pair: str) -> str:
    normalized = "-".join(part.strip() for part in str(pair).split("-") if part.strip())
    upper = normalized.upper()
    if upper == "H2-H2":
        return "H2-H2"
    if upper in {"H2-HE", "HE-H2"}:
        return "H2-He"
    raise ValueError("Orton/Xiz CIA currently supports only H2-H2 and H2-He pairs.")


def resolve_orton_xiz_cia_filename(*, pair: str, model: str = "xiz", state: str = "eq") -> str:
    key = (_normalize_pair_name(pair), str(model).strip().lower(), str(state).strip().lower())
    try:
        return _FILES_BY_PAIR_MODEL_STATE[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported Orton/Xiz CIA selection: pair={pair!r}, model={model!r}, state={state!r}.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid_cached_file(path: Path) -> bool:
    expected = _EXPECTED_SHA256.get(path.name)
    if not path.exists():
        return False
    if expected is None:
        return True
    return _sha256(path) == expected


def download_orton_xiz_cia_archive(
    *,
    cache_dir: Path,
    refresh: bool = False,
    archive_url: str = _ORTON_XIZ_ARCHIVE_URL,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not refresh and all(_is_valid_cached_file(cache_dir / filename) for filename in _EXPECTED_SHA256):
        return
    req = Request(archive_url, headers={"User-Agent": "spectra/0.1"})
    with urlopen(req) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(cache_dir)


def download_orton_xiz_cia_file_by_name(
    *,
    cache_dir: Path,
    filename: str,
    refresh: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if _is_valid_cached_file(target) and not refresh:
        return target
    download_orton_xiz_cia_archive(cache_dir=cache_dir, refresh=refresh)
    if not _is_valid_cached_file(target):
        raise ValueError(f"Downloaded Orton/Xiz CIA file {filename!r} is missing or failed checksum validation.")
    return target


@dataclass(frozen=True)
class OrtonXizCiaDataset:
    """Legacy H2 CIA data represented as binary-equivalent coefficients."""

    pair: str
    model: str
    state: str
    wavenumber_cm1: np.ndarray
    temperatures_k: np.ndarray
    raw_table: np.ndarray
    source_path: Path

    def _legacy_absorption_coefficient_cm_1_amagat_2(self) -> np.ndarray:
        if self.model == "xiz":
            return np.exp(-self.raw_table)
        if self.model == "orton":
            return np.power(10.0, self.raw_table)
        raise ValueError(f"Unsupported Orton/Xiz model {self.model!r}.")

    def interpolate_to_grid(
        self,
        temperature_k: float,
        wavenumber_grid_cm1: np.ndarray,
    ) -> np.ndarray:
        """Return a HITRAN-style binary coefficient in cm^5 molecule^-2."""
        wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
        legacy_coeff = self._legacy_absorption_coefficient_cm_1_amagat_2()
        temperature_interp = np.empty(self.wavenumber_cm1.shape, dtype=np.float64)
        for wave_idx in range(self.wavenumber_cm1.size):
            temperature_interp[wave_idx] = np.interp(
                float(temperature_k),
                self.temperatures_k,
                legacy_coeff[wave_idx],
                left=legacy_coeff[wave_idx, 0],
                right=legacy_coeff[wave_idx, -1],
            )
        binary_coeff = temperature_interp * 1.0e12 / (_LOSCHMIDT_M3 ** 2)
        return np.interp(wavenumber_grid_cm1, self.wavenumber_cm1, binary_coeff, left=0.0, right=0.0)


def parse_orton_xiz_cia_file(
    path: Path,
    *,
    pair: str,
    model: str,
    state: str,
) -> OrtonXizCiaDataset:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(f"Malformed Orton/Xiz CIA table in {path}.")
    temperatures_k = np.asarray(matrix[0, 1:], dtype=np.float64)
    wavenumber_cm1 = np.asarray(matrix[1:, 0], dtype=np.float64)
    raw_table = np.asarray(matrix[1:, 1:], dtype=np.float64)
    return OrtonXizCiaDataset(
        pair=_normalize_pair_name(pair),
        model=str(model).strip().lower(),
        state=str(state).strip().lower(),
        wavenumber_cm1=wavenumber_cm1,
        temperatures_k=temperatures_k,
        raw_table=raw_table,
        source_path=path,
    )


def load_orton_xiz_cia_dataset(
    *,
    cache_dir: Path,
    pair: str,
    model: str = "xiz",
    state: str = "eq",
    refresh: bool = False,
) -> OrtonXizCiaDataset:
    filename = resolve_orton_xiz_cia_filename(pair=pair, model=model, state=state)
    path = download_orton_xiz_cia_file_by_name(cache_dir=cache_dir, filename=filename, refresh=refresh)
    return parse_orton_xiz_cia_file(path, pair=pair, model=model, state=state)
