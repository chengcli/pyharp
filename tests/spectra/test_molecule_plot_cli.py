from types import SimpleNamespace

import numpy as np

from pyharp.spectra.hitran_lines import LineDatabase
from pyharp.spectra.molecule_plot_cli import build_molecule_overview_batch_parser, build_overview_parser, run_xsection
from pyharp.spectra.molecule_plot_cli import _compute_overview_products


def test_overview_parser_accepts_optional_cia_and_pdf_output(tmp_path) -> None:
    parser = build_overview_parser()
    args = parser.parse_args(
        [
            "--species",
            "H2O",
            "--temperature-k",
            "325",
            "--pressure-bar",
            "2.5",
            "--broadening-composition",
            "air:0.8,self:0.2",
            "--wn-range",
            "1000,1500",
            "--path-length-km",
            "3",
            "--cia-filename",
            "H2-H2_2011.cia",
            "--cia-pair",
            "H2-H2",
            "--figure",
            str(tmp_path / "overview.pdf"),
        ]
    )

    assert args.species == "H2O"
    assert args.temperature_k == 325.0
    assert args.pressure_bar == 2.5
    assert args.broadening_composition == "air:0.8,self:0.2"
    assert args.wn_range == (1000.0, 1500.0)
    assert args.path_length_km == 3.0
    assert args.cia_filename == "H2-H2_2011.cia"
    assert args.cia_pair == "H2-H2"
    assert args.figure == tmp_path / "overview.pdf"


def test_batch_overview_parser_accepts_repeated_ranges_and_default_species(tmp_path) -> None:
    parser = build_molecule_overview_batch_parser()
    args = parser.parse_args(
        [
            "--wn-range=25,2500",
            "--wn-range=2501,20000",
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--broadening-composition",
            "H2:0.85,He:0.15",
            "--path-length-km",
            "1",
            "--figure",
            str(tmp_path / "combined.pdf"),
        ]
    )

    assert args.species == ["H2", "CO2", "H2O", "CH4", "N2"]
    assert args.wn_ranges == [(25.0, 2500.0), (2501.0, 20000.0)]
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.broadening_composition == "H2:0.85,He:0.15"
    assert args.path_length_km == 1.0
    assert args.figure == tmp_path / "combined.pdf"


def test_batch_overview_parser_accepts_comma_separated_species(tmp_path) -> None:
    parser = build_molecule_overview_batch_parser()
    args = parser.parse_args(
        [
            "--species",
            "H2,CO2,H2O",
            "--wn-range=25,2500",
            "--figure",
            str(tmp_path / "combined.pdf"),
        ]
    )

    assert args.species == ["H2", "CO2", "H2O"]


def test_overview_products_reuses_downloaded_line_database(monkeypatch, tmp_path) -> None:
    parser = build_overview_parser()
    args = parser.parse_args(
        [
            "--species",
            "CO2",
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--wn-range",
            "20,22",
            "--path-length-km",
            "1",
            "--hitran-dir",
            str(tmp_path / "hitran"),
            "--figure",
            str(tmp_path / "overview.pdf"),
            "--refresh-hitran",
        ]
    )

    line_db = LineDatabase(
        table_name="co2_lines_20_22",
        cache_dir=tmp_path / "hitran",
        wavenumber_min_cm1=20.0,
        wavenumber_max_cm1=22.0,
    )
    calls = {"download": 0}

    def fake_download(config, band):
        calls["download"] += 1
        return line_db

    def fake_load_line_list(config, band, *, line_db=None):
        assert line_db is not None
        assert line_db.table_name == "co2_lines_20_22"
        return SimpleNamespace(wavenumber_cm1=np.array([20.0]), line_intensity=np.array([1.0e-25]))

    def fake_compute_absorption_spectrum(*, config, band, temperature_k, pressure_pa, line_db=None):
        assert line_db is not None
        assert line_db.table_name == "co2_lines_20_22"
        grid = band.grid()
        return SimpleNamespace(
            species_name="CO2",
            wavenumber_cm1=grid,
            attenuation_line_m1=np.ones_like(grid),
            attenuation_cia_m1=np.zeros_like(grid),
            attenuation_total_m1=np.ones_like(grid),
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
        )

    monkeypatch.setattr("pyharp.spectra.molecule_plot_cli.download_hitran_lines", fake_download)
    monkeypatch.setattr("pyharp.spectra.molecule_plot_cli.load_hitran_line_list", fake_load_line_list)
    monkeypatch.setattr("pyharp.spectra.molecule_plot_cli._load_requested_cia_dataset", lambda args, config: None)
    monkeypatch.setattr(
        "pyharp.spectra.molecule_plot_cli.build_line_provider",
        lambda config, line_db: SimpleNamespace(broadening_summary=lambda: "requested=self:1.000 -> effective=self:1.000"),
    )
    monkeypatch.setattr("pyharp.spectra.molecule_plot_cli.compute_absorption_spectrum", fake_compute_absorption_spectrum)

    _compute_overview_products(args)

    assert calls["download"] == 1


def test_run_xsection_reports_broadening_summary(monkeypatch, tmp_path, capsys) -> None:
    args = SimpleNamespace(
        hitran_dir=tmp_path / "hitran",
        species="CO2",
        temperature_k=300.0,
        pressure_bar=1.0,
        broadening_composition="H2:0.9,He:0.1",
        wn_range=(20.0, 22.0),
        resolution=1.0,
        refresh_hitran=False,
        cia_filename=None,
        cia_pair=None,
        cia_index_url="https://hitran.org/cia/",
        refresh_cia=False,
        figure=tmp_path / "co2.png",
    )

    fake_provider = SimpleNamespace(broadening_summary=lambda: "requested=h2:0.900,he:0.100 -> effective=air:1.000 (fallback: h2->air, he->air)")
    fake_spectrum = SimpleNamespace(
        species_name="CO2",
        wavenumber_cm1=np.array([20.0, 21.0, 22.0]),
        sigma_line_cm2_molecule=np.ones(3),
        sigma_cia_cm2_molecule=np.zeros(3),
        sigma_total_cm2_molecule=np.ones(3),
    )

    monkeypatch.setattr(
        "pyharp.spectra.molecule_plot_cli._compute_requested_absorption_spectrum",
        lambda *call_args, **call_kwargs: (None, None, fake_spectrum, fake_provider),
    )
    monkeypatch.setattr("pyharp.spectra.molecule_plot_cli.plot_absorption_spectrum", lambda spectrum, figure: None)

    run_xsection(args)

    out = capsys.readouterr().out
    assert "Broadening: requested=h2:0.900,he:0.100 -> effective=air:1.000" in out
