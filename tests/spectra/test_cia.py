from pathlib import Path

import numpy as np

from pyharp.spectra.hitran_cia_plot import plot_cia_attenuation_coefficient, plot_cia_cross_section, plot_cia_transmission
from pyharp.spectra.hitran_cia_utils import compute_cia_attenuation_m1, compute_cia_transmission, download_cia_file, download_cia_file_by_name, find_cia_download_url, parse_cia_file
from pyharp.spectra.config import SpectroscopyConfig


CIA_TEXT = """# CO2-CO2 0 4 200.0 200.0 3
1.0 1.0e-4
2.0 2.0e-4
3.0 3.0e-4
# CO2-CO2 0 4 300.0 300.0 3
1.0 2.0e-4
2.0 3.0e-4
3.0 4.0e-4
"""

REALISTIC_CIA_TEXT = """            CO2-CO2   1.000000 750.00000    750 200.00 1.961e-43 -.999                            3
    1.0000  1.000E-04
    2.0000  2.000E-04
    3.0000  3.000E-04
            CO2-CO2   1.000000 750.00000    750 300.00 1.961e-43 -.999                            3
    1.0000  2.000E-04
    2.0000  3.000E-04
    3.0000  4.000E-04
"""


def test_parse_and_interpolate_cia_file(tmp_path: Path) -> None:
    path = tmp_path / "CO2-CO2_2024.cia"
    path.write_text(CIA_TEXT, encoding="utf-8")
    dataset = parse_cia_file(path, "CO2-CO2")

    interp = dataset.interpolate_to_grid(250.0, np.array([1.5, 2.5]))
    assert np.allclose(interp, np.array([2.0e-4, 3.0e-4]))


def test_parse_realistic_hitran_cia_header_format(tmp_path: Path) -> None:
    path = tmp_path / "CO2-CO2_2024.cia"
    path.write_text(REALISTIC_CIA_TEXT, encoding="utf-8")
    dataset = parse_cia_file(path, "CO2-CO2")
    interp = dataset.interpolate_to_grid(250.0, np.array([1.5, 2.5]))
    assert np.allclose(interp, np.array([2.0e-4, 3.0e-4]))


def test_interpolate_uses_only_temperatures_with_spectral_coverage(tmp_path: Path) -> None:
    path = tmp_path / "CO2-CO2_2024.cia"
    path.write_text(
        """            CO2-CO2   1.000000 10.00000    10 298.00 1.000e-45 -.999                            3
    1.0000  1.000E-45
    2.0000  2.000E-45
    3.0000  3.000E-45
            CO2-CO2   1.000000 10.00000    10 350.00 1.000e-45 -.999                            3
    20.0000  9.000E-45
    21.0000  9.000E-45
    22.0000  9.000E-45
""",
        encoding="utf-8",
    )
    dataset = parse_cia_file(path, "CO2-CO2")
    interp = dataset.interpolate_to_grid(300.0, np.array([2.0]))
    assert np.allclose(interp, np.array([2.0e-45]))


def test_find_cia_download_url_prefers_matching_link(monkeypatch) -> None:
    html = '<html><body><a href="/files/CO2-CO2_2024.cia">file</a></body></html>'
    monkeypatch.setattr("pyharp.spectra.hitran_cia_utils._download_text", lambda _: html)
    url = find_cia_download_url("https://hitran.org/cia/", "CO2-CO2_2024.cia")
    assert url == "https://hitran.org/files/CO2-CO2_2024.cia"


def test_download_cia_file_uses_cache(monkeypatch, tmp_path: Path) -> None:
    config = SpectroscopyConfig(output_path=tmp_path / "out.nc", hitran_cache_dir=tmp_path / "cache")
    target = config.hitran_cache_dir / config.cia_filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("cached", encoding="utf-8")

    called = {"value": False}

    def _fail(*args, **kwargs):
        called["value"] = True
        raise AssertionError("network should not be used when cache exists")

    monkeypatch.setattr("pyharp.spectra.hitran_cia_utils.urlopen", _fail)
    path = download_cia_file(config)
    assert path == target
    assert called["value"] is False


def test_download_cia_file_by_name_uses_cache(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "H2-H2_2011.cia"
    target.write_text("cached", encoding="utf-8")

    called = {"value": False}

    def _fail(*args, **kwargs):
        called["value"] = True
        raise AssertionError("network should not be used when cache exists")

    monkeypatch.setattr("pyharp.spectra.hitran_cia_utils.urlopen", _fail)
    path = download_cia_file_by_name(cache_dir=tmp_path, filename="H2-H2_2011.cia")
    assert path == target
    assert called["value"] is False


def test_plot_cia_cross_section_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2_2011.cia"
    path.write_text("""# H2-H2 0 4 200.0 200.0 3
1.0 1.0e-46
2.0 2.0e-46
3.0 3.0e-46
# H2-H2 0 4 300.0 300.0 3
1.0 2.0e-46
2.0 3.0e-46
3.0 4.0e-46
""", encoding="utf-8")
    dataset = parse_cia_file(path, "H2-H2")
    coeff = plot_cia_cross_section(
        dataset,
        temperature_k=250.0,
        wavenumber_grid_cm1=np.array([1.0, 2.0, 3.0]),
        figure_path=tmp_path / "h2_h2_cia.png",
    )
    assert np.all(coeff > 0.0)
    assert (tmp_path / "h2_h2_cia.png").exists()


def test_compute_cia_attenuation_m1_matches_expected_density_scaling(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2_2011.cia"
    path.write_text("""# H2-H2 0 4 300.0 300.0 1
2.0 2.0e-46
""", encoding="utf-8")
    dataset = parse_cia_file(path, "H2-H2")
    attenuation = compute_cia_attenuation_m1(
        dataset,
        temperature_k=300.0,
        pressure_pa=1.0e5,
        wavenumber_grid_cm1=np.array([2.0]),
    )
    number_density_cm3 = 1.0e5 / (1.380649e-23 * 300.0) / 1.0e6
    expected = 2.0e-46 * number_density_cm3**2 * 100.0
    assert np.allclose(attenuation, np.array([expected]))


def test_plot_cia_attenuation_coefficient_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2_2011.cia"
    path.write_text("""# H2-H2 0 4 300.0 300.0 3
1.0 2.0e-46
2.0 3.0e-46
3.0 4.0e-46
""", encoding="utf-8")
    dataset = parse_cia_file(path, "H2-H2")
    attenuation = plot_cia_attenuation_coefficient(
        dataset,
        temperature_k=300.0,
        pressure_pa=1.0e5,
        wavenumber_grid_cm1=np.array([1.0, 2.0, 3.0]),
        figure_path=tmp_path / "h2_h2_cia_attenuation.png",
    )
    assert np.all(attenuation > 0.0)
    assert (tmp_path / "h2_h2_cia_attenuation.png").exists()


def test_compute_cia_transmission_matches_beer_lambert(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2_2011.cia"
    path.write_text("""# H2-H2 0 4 300.0 300.0 1
2.0 2.0e-46
""", encoding="utf-8")
    dataset = parse_cia_file(path, "H2-H2")
    transmission = compute_cia_transmission(
        dataset,
        temperature_k=300.0,
        pressure_pa=1.0e5,
        path_length_m=1000.0,
        wavenumber_grid_cm1=np.array([2.0]),
    )
    attenuation = compute_cia_attenuation_m1(
        dataset,
        temperature_k=300.0,
        pressure_pa=1.0e5,
        wavenumber_grid_cm1=np.array([2.0]),
    )
    assert np.allclose(transmission, np.exp(-attenuation * 1000.0))


def test_plot_cia_transmission_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "H2-H2_2011.cia"
    path.write_text("""# H2-H2 0 4 300.0 300.0 3
1.0 2.0e-46
2.0 3.0e-46
3.0 4.0e-46
""", encoding="utf-8")
    dataset = parse_cia_file(path, "H2-H2")
    transmission = plot_cia_transmission(
        dataset,
        temperature_k=300.0,
        pressure_pa=1.0e5,
        path_length_m=1000.0,
        wavenumber_grid_cm1=np.array([1.0, 2.0, 3.0]),
        figure_path=tmp_path / "h2_h2_cia_transmission.png",
    )
    assert np.all(transmission > 0.0)
    assert np.all(transmission <= 1.0)
    assert (tmp_path / "h2_h2_cia_transmission.png").exists()
