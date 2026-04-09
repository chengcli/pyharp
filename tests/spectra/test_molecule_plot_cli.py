from pyharp.spectra.molecule_plot_cli import build_molecule_overview_batch_parser, build_overview_parser


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
            "--wn-min",
            "1000",
            "--wn-max",
            "1500",
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
    assert args.wn_min == 1000.0
    assert args.wn_max == 1500.0
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
