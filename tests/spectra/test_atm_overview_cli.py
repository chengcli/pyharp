from pyharp.spectra.atm_overview_cli import _find_binary_pairs, _parse_composition, build_atm_overview_parser


def test_parse_composition_normalizes_and_merges_duplicates() -> None:
    composition = _parse_composition("H2O:1,H2:8,H2O:1")

    assert composition == {"H2O": 0.2, "H2": 0.8}


def test_parse_composition_accepts_cia_only_species() -> None:
    composition = _parse_composition("H2:0.9,He:0.1")

    assert composition == {"H2": 0.9, "He": 0.1}


def test_find_binary_pairs_discovers_supported_cross_pairs() -> None:
    pairs = _find_binary_pairs({"CO2": 0.4, "CH4": 0.3, "H2": 0.3})

    assert ("CO2-CH4", "CO2-CH4_2024.cia", "CO2", "CH4") in pairs
    assert ("CO2-H2", "CO2-H2_2024.cia", "CO2", "H2") in pairs


def test_find_binary_pairs_supports_h2_he() -> None:
    pairs = _find_binary_pairs({"H2": 0.9, "He": 0.1})

    assert ("H2-He", "H2-He_2011.cia", "H2", "He") in pairs


def test_atm_overview_parser_accepts_manifest_path_and_ranges(tmp_path) -> None:
    parser = build_atm_overview_parser()
    args = parser.parse_args(
        [
            "--composition",
            "H2O:0.1,H2:0.9",
            "--wn-range=25,2500",
            "--wn-range=2501,20000",
            "--manifest",
            str(tmp_path / "sources.json"),
            "--figure",
            str(tmp_path / "overview.pdf"),
        ]
    )

    assert args.composition == "H2O:0.1,H2:0.9"
    assert args.wn_ranges == [(25.0, 2500.0), (2501.0, 20000.0)]
    assert args.manifest == tmp_path / "sources.json"
    assert args.figure == tmp_path / "overview.pdf"
