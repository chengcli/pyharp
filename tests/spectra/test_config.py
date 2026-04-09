from pyharp.spectra.config import SpectralBandConfig, SpectroscopyConfig, resolve_hitran_cia_pair, resolve_hitran_species


def test_resolve_hitran_species_is_case_insensitive() -> None:
    species = resolve_hitran_species("co2")
    assert species.name == "CO2"
    assert species.molecule_id == 2
    assert species.cia_pair == "CO2-CO2"


def test_config_derives_species_dependent_names(tmp_path) -> None:
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)
    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="co2",
    )
    assert config.hitran_species.name == "CO2"
    assert config.molecule_id == 2
    assert config.resolved_isotopologue_ids() == (1, 2, 3, 4, 5, 6, 7)
    assert config.cia_pair == "CO2-CO2"
    assert config.cia_filename == "CO2-CO2_2024.cia"
    assert config.resolved_line_table_name(band) == "co2_lines_25_2500"


def test_resolve_hitran_species_supports_h2o() -> None:
    species = resolve_hitran_species("H2O")
    assert species.name == "H2O"
    assert species.molecule_id == 1
    assert species.cia_filename is None


def test_resolve_hitran_species_supports_ch4() -> None:
    species = resolve_hitran_species("CH4")
    assert species.name == "CH4"
    assert species.molecule_id == 6
    assert species.isotopologue_ids == (1, 2, 3, 4)
    assert species.cia_filename == "CH4-CH4_2011.cia"


def test_ch4_config_derives_default_cia_metadata(tmp_path) -> None:
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)
    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="ch4",
    )
    assert config.hitran_species.name == "CH4"
    assert config.molecule_id == 6
    assert config.resolved_isotopologue_ids() == (1, 2, 3, 4)
    assert config.cia_pair == "CH4-CH4"
    assert config.cia_filename == "CH4-CH4_2011.cia"
    assert config.resolved_line_table_name(band) == "ch4_lines_25_2500"


def test_resolve_hitran_species_supports_n2() -> None:
    species = resolve_hitran_species("N2")
    assert species.name == "N2"
    assert species.molecule_id == 22
    assert species.isotopologue_ids == (1, 2)
    assert species.cia_filename == "N2-N2_2021.cia"


def test_n2_config_derives_default_cia_metadata(tmp_path) -> None:
    band = SpectralBandConfig("single_state", 25.0, 2500.0, 1.0)
    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="n2",
    )
    assert config.hitran_species.name == "N2"
    assert config.molecule_id == 22
    assert config.resolved_isotopologue_ids() == (1, 2)
    assert config.cia_pair == "N2-N2"
    assert config.cia_filename == "N2-N2_2021.cia"
    assert config.resolved_line_table_name(band) == "n2_lines_25_2500"


def test_resolve_hitran_cia_pair_supports_new_binary_pairs() -> None:
    assert resolve_hitran_cia_pair("CO2-CH4").filename == "CO2-CH4_2024.cia"
    assert resolve_hitran_cia_pair("CO2-H2").filename == "CO2-H2_2024.cia"
    assert resolve_hitran_cia_pair("H2-He").filename == "H2-He_2011.cia"
    assert resolve_hitran_cia_pair("N2-CH4").filename == "N2-CH4_2024.cia"


def test_resolve_hitran_cia_pair_accepts_reversed_order() -> None:
    assert resolve_hitran_cia_pair("He-H2").pair == "H2-He"
    assert resolve_hitran_cia_pair("CH4-CO2").pair == "CO2-CH4"
