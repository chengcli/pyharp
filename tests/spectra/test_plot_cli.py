import pytest

from pyharp.spectra import plot_cli


def test_plot_parser_accepts_cia_binary_with_wn_range(tmp_path) -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "binary",
            "--pair",
            "H2-H2",
            "--temperature-k",
            "300",
            "--wn-range",
            "20,10000",
            "--figure",
            str(tmp_path / "cia.png"),
        ]
    )

    assert args.command == "binary"
    assert args.pair == "H2-H2"
    assert args.wn_range == (20.0, 10000.0)
    assert args.figure == tmp_path / "cia.png"


def test_plot_parser_accepts_molecule_xsection_with_wn_range(tmp_path) -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "CO2",
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--wn-range",
            "20,2500",
            "--figure",
            str(tmp_path / "co2.png"),
        ]
    )

    assert args.command == "xsection"
    assert args.species == "CO2"
    assert args.wn_range == (20.0, 2500.0)
    assert args.figure == tmp_path / "co2.png"


def test_plot_parser_accepts_atm_overview_ranges(tmp_path) -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "overview",
            "--composition",
            "H2O:0.1,H2:0.9",
            "--wn-range=25,2500",
            "--wn-range=2501,20000",
            "--manifest",
            str(tmp_path / "sources.json"),
            "--figure",
            str(tmp_path / "atm.pdf"),
        ]
    )

    assert args.command == "overview"
    assert args.composition == "H2O:0.1,H2:0.9"
    assert args.wn_ranges == [(25.0, 2500.0), (2501.0, 20000.0)]
    assert args.manifest == tmp_path / "sources.json"
    assert args.figure == tmp_path / "atm.pdf"


def test_plot_main_dispatches_pair_attenuation(monkeypatch) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.cia_plot_cli.run_attenuation", fake_run)

    plot_cli.main(["attenuation", "--pair", "H2-He", "--wn-range", "25,30"])

    assert len(calls) == 1
    assert calls[0].pair == "H2-He"
    assert calls[0].wn_range == (25.0, 30.0)
    assert calls[0].figure.name == "h2_he_attenuation_300K_1bar_25_30.png"


def test_plot_main_dispatches_molecule_xsection_with_default_figure(monkeypatch) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)

    plot_cli.main(["xsection", "--species", "CO2", "--temperature-k", "275.5", "--pressure-bar", "0.25", "--wn-range", "25,30.5"])

    assert len(calls) == 1
    assert calls[0].figure.name == "co2_xsection_275p5K_0p25bar_25_30p5.png"


def test_plot_main_dispatches_composition_attenuation(monkeypatch) -> None:
    calls = []

    def fake_run(args, *, wn_range):
        calls.append((args, wn_range))

    monkeypatch.setattr("pyharp.spectra.plot_cli.atm_overview_cli.run_atm_attenuation", fake_run)

    plot_cli.main(["attenuation", "--composition", "H2:0.9,He:0.1", "--wn-range", "25,30"])

    assert len(calls) == 1
    assert calls[0][0].composition == "H2:0.9,He:0.1"
    assert calls[0][0].figure.name == "h2_0p9_he_0p1_attenuation_300K_1bar_25_30.png"
    assert calls[0][1] == (25.0, 30.0)


def test_plot_main_rejects_multiple_selectors() -> None:
    with pytest.raises(SystemExit):
        plot_cli.main(["attenuation", "--pair", "H2-H2", "--species", "H2O"])
