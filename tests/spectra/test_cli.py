from pyharp.spectra.cli import build_parser, default_hitran_dir, default_output_path, project_root


def test_default_paths_are_inside_project_root() -> None:
    root = project_root()
    assert default_output_path().is_relative_to(root)
    assert default_hitran_dir().is_relative_to(root)
    assert default_output_path().name == "co2_spectrum_300K_1bar_20_2500.nc"
    assert default_output_path().parent.name == "output"
    assert default_hitran_dir().name == "hitran"


def test_parser_does_not_expose_reference_column_commands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert "run" not in commands
    assert "status" not in commands


def test_spectrum_parser_accepts_pressure_temperature_and_outputs(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "spectrum",
            "--output",
            str(tmp_path / "spec.nc"),
            "--figure",
            str(tmp_path / "spec.png"),
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--species",
            "co2",
            "--wn-range",
            "100,200",
        ]
    )
    assert args.command == "spectrum"
    assert args.output == tmp_path / "spec.nc"
    assert args.figure == tmp_path / "spec.png"
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.species == "co2"
    assert args.wn_range == (100.0, 200.0)


def test_transmittance_parser_accepts_path_length_and_outputs(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "transmittance",
            "--output",
            str(tmp_path / "trans.nc"),
            "--figure",
            str(tmp_path / "trans.png"),
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--path-length-m",
            "1.5",
            "--species",
            "CO2",
            "--wn-range",
            "50,150",
        ]
    )
    assert args.command == "transmittance"
    assert args.output == tmp_path / "trans.nc"
    assert args.figure == tmp_path / "trans.png"
    assert args.temperature_k == 300.0
    assert args.pressure_bar == 1.0
    assert args.path_length_m == 1.5
    assert args.species == "CO2"
    assert args.wn_range == (50.0, 150.0)
