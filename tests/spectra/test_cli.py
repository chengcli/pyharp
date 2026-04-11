import sys

import numpy as np
import xarray as xr

from pyharp.spectra.dump_cli import _combine_band_datasets, _xsection_dataset, build_parser, main
from pyharp.spectra.shared_cli import default_hitran_dir, default_output_path, project_root
from pyharp.spectra.spectrum import AbsorptionSpectrum


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
    assert args.wn_ranges == [(100.0, 200.0)]


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
    assert args.wn_ranges == [(50.0, 150.0)]


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
    assert args.wn_ranges == [(50.0, 150.0)]


def test_xsection_parser_accepts_repeated_wn_ranges(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output",
            str(tmp_path / "xsection.nc"),
        ]
    )

    assert args.wn_ranges == [(20.0, 2500.0), (2500.0, 10000.0)]


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
    monkeypatch.setattr("pyharp.spectra.dump_cli._resolve_continuum_sources", lambda **kwargs: (None, None))
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.compute_absorption_spectrum_from_sources",
        lambda **kwargs: type("Spectrum", (), {})(),
    )
    monkeypatch.setattr("pyharp.spectra.dump_cli._write_xsection_dataset", lambda spectrum, output_path: None)

    main()

    out = capsys.readouterr().out
    assert "Wrote NetCDF:" in out
    assert "Broadening: requested=h2:0.900,he:0.100 -> effective=air:1.000" in out


def test_cli_xsection_writes_multiple_ranges_to_one_file(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--species",
            "H2O",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output",
            str(tmp_path / "xsection.nc"),
        ],
    )
    calls = []
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._parallel_band_results",
        lambda tasks, *, worker: [
            (
                calls.append(task_args.wn_range)
                or xr.Dataset(
                    coords={"wavenumber_cm1": ("wavenumber_cm1", np.array([1.0, 2.0]))},
                    data_vars={"sigma_total_cm2_molecule": ("wavenumber_cm1", np.array([3.0, 4.0]))},
                ),
                "requested=self:1.000 -> effective=self:1.000",
            )
            for _, task_args in tasks
        ],
    )
    written = []
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._write_combined_dataset",
        lambda datasets, *, wn_ranges, output_path: written.append((len(datasets), wn_ranges, output_path)),
    )

    main()

    assert calls == [(20.0, 2500.0), (2500.0, 10000.0)]
    assert written == [(2, [(20.0, 2500.0), (2500.0, 10000.0)], tmp_path / "xsection.nc")]
    out = capsys.readouterr().out
    assert "Wrote NetCDF:" in out
    assert out.count("Broadening: requested=self:1.000 -> effective=self:1.000") == 2


def test_combine_band_datasets_preserves_band_metadata() -> None:
    first = xr.Dataset(
        coords={"wavenumber_cm1": ("wavenumber_cm1", np.array([20.0, 21.0]))},
        data_vars={"sigma_total_cm2_molecule": ("wavenumber_cm1", np.array([1.0, 2.0]))},
        attrs={"species_name": "H2O"},
    )
    second = xr.Dataset(
        coords={"wavenumber_cm1": ("wavenumber_cm1", np.array([30.0, 31.0, 32.0]))},
        data_vars={"sigma_total_cm2_molecule": ("wavenumber_cm1", np.array([3.0, 4.0, 5.0]))},
        attrs={"species_name": "H2O"},
    )

    combined = _combine_band_datasets([first, second], wn_ranges=[(20.0, 22.0), (30.0, 33.0)])
    try:
        assert combined.attrs["species_name"] == "H2O"
        assert combined.attrs["num_bands"] == 2
        assert np.allclose(combined["band_size"].values, [2, 3])
        assert np.allclose(combined["band_wavenumber_min_cm1"].values, [20.0, 30.0])
        assert np.allclose(combined["band_wavenumber_max_cm1"].values, [22.0, 33.0])
        assert np.allclose(combined["wavenumber_cm1"].values[0, :2], [20.0, 21.0])
        assert np.isnan(combined["wavenumber_cm1"].values[0, 2])
        assert np.allclose(combined["sigma_total_cm2_molecule"].values[1, :3], [3.0, 4.0, 5.0])
    finally:
        combined.close()


def test_xsection_dataset_keeps_only_sigma_fields() -> None:
    spectrum = AbsorptionSpectrum(
        species_name="H2O",
        wavenumber_cm1=np.array([20.0, 21.0]),
        sigma_line_cm2_molecule=np.array([1.0, 2.0]),
        sigma_cia_cm2_molecule=np.array([0.1, 0.2]),
        sigma_total_cm2_molecule=np.array([1.1, 2.2]),
        kappa_line_cm1=np.array([3.0, 4.0]),
        kappa_cia_cm1=np.array([0.3, 0.4]),
        kappa_total_cm1=np.array([3.3, 4.4]),
        attenuation_line_m1=np.array([5.0, 6.0]),
        attenuation_cia_m1=np.array([0.5, 0.6]),
        attenuation_total_m1=np.array([5.5, 6.6]),
        temperature_k=300.0,
        pressure_pa=1.0e5,
        number_density_cm3=2.4e19,
    )

    dataset = _xsection_dataset(spectrum)
    try:
        assert set(dataset.data_vars) == {
            "sigma_line_cm2_molecule",
            "sigma_cia_cm2_molecule",
            "sigma_total_cm2_molecule",
        }
    finally:
        dataset.close()


def test_cli_xsection_reuses_built_provider_for_species_spectrum(monkeypatch, tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--broadening-composition",
            "H2:0.9,He:0.1",
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--wn-range",
            "20,22",
            "--output",
            str(tmp_path / "xsection.nc"),
        ]
    )

    fake_provider = type(
        "Provider",
        (),
        {
            "broadening_summary": lambda self: "requested=h2:0.900,he:0.100 -> effective=air:1.000",
        },
    )()
    calls = {"resolve": 0, "compute": 0}

    monkeypatch.setattr("pyharp.spectra.dump_cli.download_hitran_lines", lambda config, band: object())
    monkeypatch.setattr("pyharp.spectra.dump_cli.build_line_provider", lambda config, line_db: fake_provider)
    monkeypatch.setattr("pyharp.spectra.dump_cli._resolve_species_cia", lambda args, config: None)

    def fake_resolve_continuum_sources(*, config, wavenumber_grid_cm1, temperature_k, pressure_pa):
        calls["resolve"] += 1
        assert config.hitran_species.name == "H2O"
        assert np.allclose(wavenumber_grid_cm1, np.array([20.0, 21.0, 22.0]))
        assert temperature_k == 300.0
        assert pressure_pa == 1.0e5
        return None, np.array([1.0e-24, 2.0e-24, 3.0e-24])

    def fake_compute_from_sources(
        *,
        species_name,
        wavenumber_grid_cm1,
        temperature_k,
        pressure_pa,
        line_provider,
        cia_dataset=None,
        cia_cross_section_cm2_molecule=None,
    ):
        calls["compute"] += 1
        assert species_name == "H2O"
        assert np.allclose(wavenumber_grid_cm1, np.array([20.0, 21.0, 22.0]))
        assert temperature_k == 300.0
        assert pressure_pa == 1.0e5
        assert line_provider is fake_provider
        assert cia_dataset is None
        assert np.allclose(cia_cross_section_cm2_molecule, np.array([1.0e-24, 2.0e-24, 3.0e-24]))
        return type("Spectrum", (), {})()

    monkeypatch.setattr("pyharp.spectra.dump_cli._resolve_continuum_sources", fake_resolve_continuum_sources)
    monkeypatch.setattr("pyharp.spectra.dump_cli.compute_absorption_spectrum_from_sources", fake_compute_from_sources)

    from pyharp.spectra.dump_cli import _args_for_wn_range, _compute_species_xsection

    _compute_species_xsection(_args_for_wn_range(args, args.wn_ranges[0]))

    assert calls == {"resolve": 1, "compute": 1}


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
