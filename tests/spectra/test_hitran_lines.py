import numpy as np
from pyharp.spectra.config import SpectralBandConfig, SpectroscopyConfig
from pyharp.spectra.hitran_lines import (
    HapiLineProvider,
    HitranLineList,
    build_line_provider,
    download_hitran_lines,
    load_hitran_line_list,
    plot_hitran_line_positions,
)


class FakeHapi:
    def __init__(self):
        self.db_dir = None
        self.calls = []
        self.storage_calls = []
        self.absorption_calls = []
        self.LOCAL_TABLE_CACHE = {}
        self.ISO_INDEX = {"id": 0}
        self.ISO = {
            (2, 1): (7,),
            (2, 2): (8,),
            (2, 3): (9,),
            (2, 4): (10,),
            (2, 5): (11,),
            (2, 6): (12,),
            (2, 7): (13,),
        }

    def db_begin(self, path):
        self.db_dir = path

    def fetch_by_ids(self, table_name, iso_ids, numin, numax):
        self.calls.append((table_name, tuple(iso_ids), numin, numax))

    def storage2cache(self, table_name):
        self.storage_calls.append(table_name)

    def absorptionCoefficient_Voigt(self, **kwargs):
        self.absorption_calls.append(kwargs)
        grid = np.asarray(kwargs["OmegaGrid"], dtype=np.float64)
        return grid, np.ones_like(grid)


def test_download_hitran_lines_uses_band_bounds(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)
    config = SpectroscopyConfig(output_path=tmp_path / "out.nc", hitran_cache_dir=tmp_path / "cache")
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)

    db = download_hitran_lines(config, band)

    assert db.table_name == config.resolved_line_table_name(band)
    assert fake.db_dir == str(config.hitran_cache_dir)
    assert fake.calls == [
        (
            config.resolved_line_table_name(band),
            (7, 8, 9, 10, 11, 12, 13),
            band.wavenumber_min_cm1,
            band.wavenumber_max_cm1,
        )
    ]


def test_download_hitran_lines_skips_fetch_when_cache_exists(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)
    config = SpectroscopyConfig(output_path=tmp_path / "out.nc", hitran_cache_dir=tmp_path / "cache")
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)
    config.hitran_cache_dir.mkdir(parents=True, exist_ok=True)
    table_name = config.resolved_line_table_name(band)
    (config.hitran_cache_dir / f"{table_name}.data").write_text("x", encoding="utf-8")
    (config.hitran_cache_dir / f"{table_name}.header").write_text("x", encoding="utf-8")
    fake.LOCAL_TABLE_CACHE[table_name] = {"data": {"molec_id": [2, 2, 2]}}

    download_hitran_lines(config, band)

    assert fake.calls == []
    assert fake.storage_calls == [table_name]


def test_download_hitran_lines_refetches_contaminated_cache(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)
    config = SpectroscopyConfig(output_path=tmp_path / "out.nc", hitran_cache_dir=tmp_path / "cache")
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)
    config.hitran_cache_dir.mkdir(parents=True, exist_ok=True)
    table_name = config.resolved_line_table_name(band)
    (config.hitran_cache_dir / f"{table_name}.data").write_text("x", encoding="utf-8")
    (config.hitran_cache_dir / f"{table_name}.header").write_text("x", encoding="utf-8")
    fake.LOCAL_TABLE_CACHE[table_name] = {"data": {"molec_id": [1, 2]}}

    download_hitran_lines(config, band)

    assert fake.storage_calls == [table_name]
    assert fake.calls == [
        (
            table_name,
            (7, 8, 9, 10, 11, 12, 13),
            band.wavenumber_min_cm1,
            band.wavenumber_max_cm1,
        )
    ]


def test_load_hitran_line_list_filters_by_min_line_strength(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    table_name = "co2_lines_25_2500"
    fake.LOCAL_TABLE_CACHE[table_name] = {
        "data": {
            "molec_id": [2, 2, 2],
            "nu": [30.0, 40.0, 50.0],
            "sw": [1.0e-30, 1.0e-27, 1.0e-25],
        }
    }
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)
    monkeypatch.setattr(
        "pyharp.spectra.hitran_lines.download_hitran_lines",
        lambda config, band: type("LineDb", (), {"table_name": table_name, "cache_dir": config.hitran_cache_dir})(),
    )
    config = SpectroscopyConfig(output_path=tmp_path / "out.nc", hitran_cache_dir=tmp_path / "cache")
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)

    line_list = load_hitran_line_list(config, band)

    assert np.allclose(line_list.wavenumber_cm1, [40.0, 50.0])
    assert np.allclose(line_list.line_intensity, [1.0e-27, 1.0e-25])


def test_hapi_line_provider_passes_min_line_strength_to_hapi(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)
    fake.LOCAL_TABLE_CACHE["mock"] = {"data": {"gamma_self": [0.1]}}

    provider = HapiLineProvider("mock", cache_dir=tmp_path, min_line_strength=1.0e-27)
    provider.cross_section_cm2_molecule(np.array([25.0, 26.0]), temperature_k=300.0, pressure_pa=1.0e5)

    assert fake.absorption_calls[-1]["IntensityThreshold"] == 1.0e-27
    assert fake.absorption_calls[-1]["HITRAN_units"] is True


def test_hapi_line_provider_falls_back_missing_broadener_to_air(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    fake.LOCAL_TABLE_CACHE["mock"] = {"data": {"gamma_air": [0.1], "gamma_self": [0.2]}}
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)

    provider = HapiLineProvider("mock", cache_dir=tmp_path, diluent={"h2": 0.7, "self": 0.3})

    assert provider.diluent == {"air": 0.7, "self": 0.3}
    assert provider.diluent_fallbacks == {"h2": "air"}


def test_hapi_line_provider_raises_when_air_fallback_unavailable(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    fake.LOCAL_TABLE_CACHE["mock"] = {"data": {"gamma_self": [0.2]}}
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)

    try:
        HapiLineProvider("mock", cache_dir=tmp_path, diluent={"h2": 1.0})
    except ValueError as exc:
        assert "air broadening parameters are not available for fallback" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_line_provider_uses_config_broadening_composition(monkeypatch, tmp_path) -> None:
    fake = FakeHapi()
    fake.LOCAL_TABLE_CACHE["mock"] = {"data": {"gamma_air": [0.1], "gamma_self": [0.2], "gamma_h2": [0.3]}}
    monkeypatch.setattr("pyharp.spectra.hitran_lines._import_hapi", lambda: fake)

    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path / "cache",
        species_name="CO2",
        broadening_composition="CO2:0.25,H2:0.75",
    )
    line_db = type("LineDb", (), {"table_name": "mock", "cache_dir": tmp_path})()

    provider = build_line_provider(config, line_db)

    assert provider.diluent == {"self": 0.25, "h2": 0.75}


def test_plot_hitran_line_positions_writes_png(tmp_path):
    line_list = HitranLineList(
        species_name="H2",
        table_name="mock",
        wavenumber_cm1=np.array([20.0, 100.0, 1000.0]),
        line_intensity=np.array([1.0e-30, 1.0e-25, 1.0e-20]),
    )
    figure_path = tmp_path / "h2_lines.png"
    plot_hitran_line_positions(line_list, figure_path)
    assert figure_path.exists()
