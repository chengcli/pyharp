import sys

from pyharp.spectra.dump_cli import build_parser, main
from pyharp.spectra.shared_cli import default_hitran_dir, default_output_path, project_root


def test_default_paths_are_inside_project_root() -> None:
    root = project_root()
    assert default_output_path().is_relative_to(root)
    assert default_output_path().name == "co2_xsection_300K_1bar_20_2500.nc"
    assert default_output_path().parent.name == "output"
    assert default_hitran_dir() == default_hitran_dir().parent / "hitran"
    assert default_hitran_dir().is_absolute() is False


def test_parser_does_not_expose_reference_column_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert "run" not in commands
    assert "status" not in commands


def test_xsection_parser_accepts_species_pressure_temperature_and_outputs(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--output",
            str(tmp_path / "xsection.nc"),
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--species",
            "co2",
            "--broadening-composition",
            "air:0.8,self:0.2",
            "--wn-range",
            "100,200",
        ]
    )
    assert args.command == "xsection"
    assert args.output == tmp_path / "xsection.nc"
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.species == "co2"
    assert args.broadening_composition == "air:0.8,self:0.2"
    assert args.wn_range == (100.0, 200.0)


def test_xsection_parser_accepts_cia_pair_selector(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--output",
            str(tmp_path / "pair.nc"),
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--pair",
            "H2-He",
            "--filename",
            "H2-He_2011.cia",
            "--wn-range",
            "50,150",
        ]
    )
    assert args.command == "xsection"
    assert args.output == tmp_path / "pair.nc"
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.pair == "H2-He"
    assert args.filename == "H2-He_2011.cia"
    assert args.wn_range == (50.0, 150.0)


def test_transmission_parser_accepts_path_length_and_outputs(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "transmission",
            "--output",
            str(tmp_path / "trans.nc"),
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--path-length-m",
            "1.5",
            "--species",
            "CO2",
            "--broadening-composition",
            "H2:0.85,He:0.15",
            "--wn-range",
            "50,150",
        ]
    )
    assert args.command == "transmission"
    assert args.output == tmp_path / "trans.nc"
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.path_length_m == 1.5
    assert args.species == "CO2"
    assert args.broadening_composition == "H2:0.85,He:0.15"
    assert args.wn_range == (50.0, 150.0)


def test_cli_xsection_reports_broadening_summary(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--species",
            "CO2",
            "--broadening-composition",
            "H2:0.9,He:0.1",
            "--wn-range",
            "20,22",
            "--output",
            str(tmp_path / "xsection.nc"),
        ],
    )
    monkeypatch.setattr("pyharp.spectra.dump_cli._resolve_species_cia", lambda args, config: None)
    monkeypatch.setattr("pyharp.spectra.dump_cli.download_hitran_lines", lambda config, band: object())
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.build_line_provider",
        lambda config, line_db: type(
            "Provider",
            (),
            {"broadening_summary": lambda self: "requested=h2:0.900,he:0.100 -> effective=air:1.000 (fallback: h2->air, he->air)"},
        )(),
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.compute_absorption_spectrum",
        lambda **kwargs: type("Spectrum", (), {})(),
    )
    monkeypatch.setattr("pyharp.spectra.dump_cli.write_spectrum_dataset", lambda spectrum, output_path: None)

    main()

    out = capsys.readouterr().out
    assert "Wrote NetCDF:" in out
    assert "Broadening: requested=h2:0.900,he:0.100 -> effective=air:1.000" in out


def test_cli_help_includes_examples_and_subcommands(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["-h"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")

    help_text = capsys.readouterr().out
    assert "xsection" in help_text
    assert "transmission" in help_text
    assert "pyharp-dump xsection --pair H2-He" in help_text


def test_cli_transmission_help_describes_broadening_and_path_length(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["transmission", "-h"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")

    help_text = capsys.readouterr().out
    assert "Compute line, CIA, and total transmission over a fixed path length and write a NetCDF dataset." in help_text
    assert "--broadening-composition BROADENER:FRACTION,..." in help_text
    assert "Propagation path length in meters." in help_text
    assert "pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004" in help_text
