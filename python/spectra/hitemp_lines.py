"""Build HAPI line tables from locally downloaded HITEMP line lists."""

from __future__ import annotations

import bz2
import contextlib
import copy
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Sequence

import numpy as np


PAR_RECORD_LENGTH = 160
C2_CM_K = 1.4388028496642257  # same second radiation constant as HAPI
T_REF_K = 296.0
# The per-band parent table read from the HITEMP files is screened over (at least)
# this range, so runs at different temperatures re-screen it instead of the raw files.
DEFAULT_HITEMP_TEMPERATURE_RANGE_K = (10.0, 2000.0)
FILTER_TEMPERATURE_SAMPLES = 16
_READ_CHUNK_BYTES = 32 * 1024 * 1024
_HEADER_METADATA_KEY = "pyharp_hitemp"

# Official names: 01_00000-00050_HITEMP2010.zip, 02_HITEMP2024.par.bz2, 06_HITEMP2020.par.bz2
HITEMP_FILE_PATTERN = re.compile(
    r"^(?P<molec>\d{2})_(?:(?P<lo>\d+)-(?P<hi>\d+)_)?HITEMP(?P<year>\d{4})(?P<ext>\.zip|\.par\.bz2|\.par)$"
)

PartitionSum = Callable[[int, int, float], float]


@dataclass(frozen=True)
class HitempFile:
    """One HITEMP source file and the wavenumber range it covers."""

    path: Path
    molecule_id: int
    year: int
    wavenumber_min_cm1: float
    wavenumber_max_cm1: float


def find_hitemp_files(hitemp_dir: Path, molecule_id: int) -> tuple[HitempFile, ...]:
    """Return the newest HITEMP edition files for one molecule, sorted by wavenumber."""
    hitemp_dir = Path(hitemp_dir)
    candidates: list[HitempFile] = []
    for path in sorted(hitemp_dir.rglob("*")):
        match = HITEMP_FILE_PATTERN.match(path.name)
        if match is None or not path.is_file() or int(match["molec"]) != int(molecule_id):
            continue
        lower = float(match["lo"]) if match["lo"] is not None else 0.0
        upper = float(match["hi"]) if match["hi"] is not None else float("inf")
        candidates.append(HitempFile(path, int(molecule_id), int(match["year"]), lower, upper))
    if not candidates:
        raise FileNotFoundError(f"No HITEMP files for molecule {molecule_id:02d} under {hitemp_dir}.")
    newest = max(item.year for item in candidates)
    return tuple(sorted((item for item in candidates if item.year == newest), key=lambda item: item.wavenumber_min_cm1))


def has_hitemp_files(hitemp_dir: Path, molecule_id: int) -> bool:
    """Return whether ``hitemp_dir`` holds HITEMP files for the molecule."""
    try:
        find_hitemp_files(hitemp_dir, molecule_id)
    except FileNotFoundError:
        return False
    return True


def filter_temperatures_k(temperatures_k: Sequence[float]) -> np.ndarray:
    """Return the temperatures at which line strengths are compared with the threshold.

    These are the given temperatures plus samples spanning their range, so a table
    screened on them keeps every line HAPI would keep at those temperatures.
    """
    values = np.asarray(sorted(float(value) for value in temperatures_k))
    if values.size == 0 or values[0] <= 0.0:
        raise ValueError("HITEMP screening temperatures must be positive")
    if values[0] == values[-1]:
        return values[:1]
    return np.unique(np.concatenate([values, np.geomspace(values[0], values[-1], FILTER_TEMPERATURE_SAMPLES)]))


def max_line_strength(
    *,
    nu: np.ndarray,
    sw: np.ndarray,
    elower: np.ndarray,
    local_iso_id: np.ndarray,
    molecule_id: int,
    temperatures_k: Sequence[float],
    partition_sum: PartitionSum,
) -> np.ndarray:
    """Return the largest line intensity over the temperatures, scaled from 296 K like HAPI."""
    strongest = np.zeros_like(sw)
    for iso in np.unique(local_iso_id):
        sel = local_iso_id == iso
        q_ref = partition_sum(molecule_id, int(iso), T_REF_K)
        stim_ref = -np.expm1(-C2_CM_K * nu[sel] / T_REF_K)
        for temperature in temperatures_k:
            q_t = partition_sum(molecule_id, int(iso), float(temperature))
            strength = (
                sw[sel]
                * (q_ref / q_t)
                * np.exp(-C2_CM_K * elower[sel] * (1.0 / temperature - 1.0 / T_REF_K))
                * (-np.expm1(-C2_CM_K * nu[sel] / temperature))
                / stim_ref
            )
            strongest[sel] = np.maximum(strongest[sel], strength)
    return strongest


def prepare_hitemp_table(
    *,
    hitemp_dir: Path,
    table_dir: Path,
    table_name: str,
    molecule_id: int,
    local_iso_ids: Sequence[int],
    wavenumber_min_cm1: float,
    wavenumber_max_cm1: float,
    temperature_range_k: tuple[float, float],
    min_line_strength: float,
    default_header: dict,
    partition_sum: PartitionSum,
    refresh: bool = False,
) -> int:
    """Write a filtered HAPI table (``.data`` + ``.header``) from HITEMP files and return its line count.

    Lines are kept when they fall in the wavenumber range, belong to the requested
    isotopologues, and exceed ``min_line_strength`` at any temperature in the range.
    An existing table built from the same inputs over a range covering this one is
    reused, so a table widened once for a hot run is not rebuilt for cooler runs.
    """
    sources = [
        item
        for item in find_hitemp_files(hitemp_dir, molecule_id)
        if item.wavenumber_max_cm1 >= wavenumber_min_cm1 and item.wavenumber_min_cm1 <= wavenumber_max_cm1
    ]
    header_path = Path(table_dir) / f"{table_name}.header"
    if not refresh and header_path.exists() and (Path(table_dir) / f"{table_name}.data").exists():
        existing = json.loads(header_path.read_text()).get(_HEADER_METADATA_KEY, {})
        tmin, tmax = sorted(float(value) for value in temperature_range_k)
        etmin, etmax = existing.get("temperature_range_k", (np.inf, -np.inf))
        same_inputs = existing.get("sources") == [item.path.name for item in sources] and all(
            existing.get(key) == value
            for key, value in {
                "molecule_id": int(molecule_id),
                "local_iso_ids": sorted(int(value) for value in local_iso_ids),
                "wavenumber_range_cm1": [float(wavenumber_min_cm1), float(wavenumber_max_cm1)],
                "min_line_strength": float(min_line_strength),
            }.items()
        )
        if same_inputs and etmin <= tmin and etmax >= tmax:
            return int(json.loads(header_path.read_text())["number_of_rows"])
    return _write_filtered_table(
        sources=[item.path for item in sources],
        source_metadata={"sources": [item.path.name for item in sources]},
        table_dir=table_dir,
        table_name=table_name,
        molecule_id=molecule_id,
        local_iso_ids=local_iso_ids,
        wavenumber_min_cm1=wavenumber_min_cm1,
        wavenumber_max_cm1=wavenumber_max_cm1,
        temperatures_k=temperature_range_k,
        min_line_strength=min_line_strength,
        default_header=default_header,
        partition_sum=partition_sum,
        refresh=refresh,
    )


def derive_hitemp_table(
    *,
    parent_dir: Path,
    parent_name: str,
    table_dir: Path,
    table_name: str,
    temperatures_k: Sequence[float],
    partition_sum: PartitionSum,
    refresh: bool = False,
) -> int:
    """Re-screen a table from :func:`prepare_hitemp_table` at the temperatures of one run.

    The parent table is plain, pre-filtered text, so this is much cheaper than
    re-reading the compressed HITEMP files, and the child is rebuilt whenever the
    parent changes.
    """
    parent_header = json.loads((Path(parent_dir) / f"{parent_name}.header").read_text())
    parent = parent_header[_HEADER_METADATA_KEY]
    tmin, tmax = min(temperatures_k), max(temperatures_k)
    ptmin, ptmax = parent["temperature_range_k"]
    if tmin < ptmin or tmax > ptmax:
        raise ValueError(f"Temperature range {tmin:g}-{tmax:g} K lies outside the parent table's {ptmin:g}-{ptmax:g} K.")
    lower, upper = parent["wavenumber_range_cm1"]
    return _write_filtered_table(
        sources=[Path(parent_dir) / f"{parent_name}.data"],
        source_metadata={"parent": parent},
        table_dir=table_dir,
        table_name=table_name,
        molecule_id=parent["molecule_id"],
        local_iso_ids=parent["local_iso_ids"],
        wavenumber_min_cm1=lower,
        wavenumber_max_cm1=upper,
        temperatures_k=temperatures_k,
        min_line_strength=parent["min_line_strength"],
        default_header={key: value for key, value in parent_header.items() if key != _HEADER_METADATA_KEY},
        partition_sum=partition_sum,
        refresh=refresh,
    )


def _write_filtered_table(
    *,
    sources: Sequence[Path],
    source_metadata: dict,
    table_dir: Path,
    table_name: str,
    molecule_id: int,
    local_iso_ids: Sequence[int],
    wavenumber_min_cm1: float,
    wavenumber_max_cm1: float,
    temperatures_k: Sequence[float],
    min_line_strength: float,
    default_header: dict,
    partition_sum: PartitionSum,
    refresh: bool,
) -> int:
    table_dir = Path(table_dir)
    data_path = table_dir / f"{table_name}.data"
    header_path = table_dir / f"{table_name}.header"
    temperatures = filter_temperatures_k(temperatures_k)
    metadata = {
        **source_metadata,
        "molecule_id": int(molecule_id),
        "local_iso_ids": sorted(int(value) for value in local_iso_ids),
        "wavenumber_range_cm1": [float(wavenumber_min_cm1), float(wavenumber_max_cm1)],
        "temperature_range_k": [float(temperatures[0]), float(temperatures[-1])],
        "screening_temperatures_k": [float(value) for value in temperatures],
        "min_line_strength": float(min_line_strength),
    }
    if not refresh and data_path.exists() and header_path.exists():
        existing = json.loads(header_path.read_text())
        if existing.get(_HEADER_METADATA_KEY) == metadata:
            return int(existing["number_of_rows"])

    table_dir.mkdir(parents=True, exist_ok=True)
    tmp_data = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    tmp_header = header_path.with_name(f".{header_path.name}.{os.getpid()}.tmp")
    isos = np.asarray(metadata["local_iso_ids"], dtype=np.int64)
    newline = np.full((1, 1), 0x0A, dtype=np.uint8)
    n_lines = 0
    with tmp_data.open("wb") as out:
        for source in sources:
            for records in _iter_par_records(source):
                nu = _float_field(records, 3, 15)
                if nu.size and nu.min() > wavenumber_max_cm1:
                    break  # .par files are sorted by wavenumber
                iso = _local_iso_ids(records)
                keep = (nu >= wavenumber_min_cm1) & (nu <= wavenumber_max_cm1) & np.isin(iso, isos)
                keep &= _int_field(records, 0, 2) == int(molecule_id)
                if not keep.any():
                    continue
                records, nu, iso = records[keep], nu[keep], iso[keep]
                strongest = max_line_strength(
                    nu=nu,
                    sw=_float_field(records, 15, 25),
                    elower=_float_field(records, 45, 55),
                    local_iso_id=iso,
                    molecule_id=molecule_id,
                    temperatures_k=temperatures,
                    partition_sum=partition_sum,
                )
                records = records[strongest >= min_line_strength]
                if records.shape[0] == 0:
                    continue
                out.write(np.hstack([records, np.repeat(newline, records.shape[0], axis=0)]).tobytes())
                n_lines += records.shape[0]

    header = copy.deepcopy(default_header)
    header["table_name"] = table_name
    header["number_of_rows"] = n_lines
    header["size_in_bytes"] = tmp_data.stat().st_size
    header[_HEADER_METADATA_KEY] = metadata
    tmp_header.write_text(json.dumps(header, indent=2))
    # The header is replaced last, so a table is only reused once its data is complete.
    os.replace(tmp_data, data_path)
    os.replace(tmp_header, header_path)
    return n_lines


@contextlib.contextmanager
def _open_par_stream(path: Path) -> Iterator[BinaryIO]:
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".par")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one .par file in {path}, found {members}.")
            with archive.open(members[0]) as stream:
                yield stream
    elif path.name.endswith(".bz2"):
        with bz2.open(path, "rb") as stream:
            yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def _iter_par_records(path: Path) -> Iterator[np.ndarray]:
    """Yield blocks of 160-byte .par records as ``(n, 160)`` uint8 arrays."""
    with _open_par_stream(path) as stream:
        remainder = b""
        while True:
            block = stream.read(_READ_CHUNK_BYTES)
            if not block:
                break
            data = remainder + block
            cut = data.rfind(b"\n") + 1
            remainder = data[cut:]
            if cut:
                yield _records_from_bytes(data[:cut])
        if remainder.strip():
            yield _records_from_bytes(remainder + b"\n")


def _records_from_bytes(data: bytes) -> np.ndarray:
    width = PAR_RECORD_LENGTH + 1
    if len(data) % width == 0:
        records = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
        if np.all(records[:, -1] == 0x0A):
            return records[:, :PAR_RECORD_LENGTH]
    lines = [line.rstrip(b"\r") for line in data.split(b"\n")]
    lines = [line for line in lines if line.strip()]
    for line in lines:
        if len(line) != PAR_RECORD_LENGTH:
            raise ValueError(f"Unexpected .par record length {len(line)}; expected {PAR_RECORD_LENGTH}.")
    return np.frombuffer(b"".join(lines), dtype=np.uint8).reshape(-1, PAR_RECORD_LENGTH)


def _raw_field(records: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.ascontiguousarray(records[:, start:stop]).view(f"S{stop - start}").ravel()


def _float_field(records: np.ndarray, start: int, stop: int) -> np.ndarray:
    raw = _raw_field(records, start, stop)
    try:
        return raw.astype(np.float64)
    except ValueError:
        return np.asarray([_parse_par_float(value) for value in raw], dtype=np.float64)


def _parse_par_float(value: bytes) -> float:
    text = value.decode("ascii").strip().replace("D", "E")
    try:
        return float(text)
    except ValueError:
        # Intensities below 1e-99 lose the "E" in the fixed-width field, e.g. "1.000-100".
        match = re.fullmatch(r"([+-]?\d*\.?\d+)([+-]\d+)", text)
        if match is None:
            raise
        return float(f"{match[1]}E{match[2]}")


def _int_field(records: np.ndarray, start: int, stop: int) -> np.ndarray:
    return _raw_field(records, start, stop).astype(np.int64)


def _local_iso_ids(records: np.ndarray) -> np.ndarray:
    """Decode the isotopologue column: '1'-'9', then '0' -> 10, 'A' -> 11, 'B' -> 12, ..."""
    code = records[:, 2].astype(np.int64)
    iso = code - ord("0")
    iso = np.where(code == ord("0"), 10, iso)
    return np.where(code >= ord("A"), 11 + code - ord("A"), iso)
