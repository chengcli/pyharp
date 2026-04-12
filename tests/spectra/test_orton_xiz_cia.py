from pathlib import Path

import numpy as np

from pyharp.spectra.orton_xiz_cia import (
    _LOSCHMIDT_M3,
    download_orton_xiz_cia_file_by_name,
    load_orton_xiz_cia_dataset,
    parse_orton_xiz_cia_file,
    resolve_orton_xiz_cia_filename,
)


XIZ_TEXT = """0 100 200
10 1.0 2.0
20 2.0 3.0
30 3.0 4.0
"""


def test_resolve_orton_xiz_cia_filename_supports_pairs_models_and_states() -> None:
    assert resolve_orton_xiz_cia_filename(pair="H2-H2", model="xiz", state="eq") == "H2-H2-eq.xiz.txt"
    assert resolve_orton_xiz_cia_filename(pair="He-H2", model="orton", state="nm") == "H2-He-nm.orton.txt"


def test_parse_orton_xiz_cia_file_interpolates_xiz_to_binary_equivalent(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2-eq.xiz.txt"
    path.write_text(XIZ_TEXT, encoding="utf-8")
    dataset = parse_orton_xiz_cia_file(path, pair="H2-H2", model="xiz", state="eq")

    interpolated = dataset.interpolate_to_grid(150.0, np.array([15.0, 25.0]))
    expected_legacy = np.array([np.exp(-1.5), np.exp(-2.5)], dtype=np.float64)
    expected_binary = expected_legacy * 1.0e12 / (_LOSCHMIDT_M3 ** 2)
    assert np.allclose(interpolated, expected_binary)


def test_download_orton_xiz_cia_file_by_name_uses_valid_cache(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "custom.txt"
    target.write_text("cached", encoding="utf-8")

    def _fail(*args, **kwargs):
        raise AssertionError("network should not be used when cache exists")

    monkeypatch.setattr("pyharp.spectra.orton_xiz_cia.urlopen", _fail)
    path = download_orton_xiz_cia_file_by_name(cache_dir=tmp_path, filename="custom.txt")
    assert path == target


def test_load_orton_xiz_cia_dataset_uses_default_filename(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "H2-H2-eq.xiz.txt"
    path.write_text(XIZ_TEXT, encoding="utf-8")
    monkeypatch.setattr(
        "pyharp.spectra.orton_xiz_cia.download_orton_xiz_cia_file_by_name",
        lambda *, cache_dir, filename, refresh=False: path,
    )
    dataset = load_orton_xiz_cia_dataset(cache_dir=tmp_path, pair="H2-H2", model="xiz", state="eq")

    assert dataset.source_path == path
    assert dataset.pair == "H2-H2"
    assert dataset.model == "xiz"
    assert dataset.state == "eq"
