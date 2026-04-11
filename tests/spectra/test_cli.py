import sys

import numpy as np
from pathlib import Path

import xarray as xr

from pyharp.spectra.dataset_io import combine_band_datasets, write_dataset_via_tmp
from pyharp.spectra.dump_cli import _args_for_wn_range, _composition_transmission_dataset, _composition_xsection_dataset, _output_path_for_wn_range, _pair_xsection_dataset, _species_transmission_dataset, _stack_temperature_datasets, _xsection_dataset, build_parser, main
from pyharp.spectra.shared_cli import default_hitran_dir, default_output_path, project_root
from pyharp.spectra.spectrum import AbsorptionSpectrum


def test_default_paths_are_inside_project_root() -> None:
    root = project_root()
    assert default_output_path().is_relative_to(root)
    assert default_output_path().name == "co2_xsection_1bar_300K_20_2500cm1.nc"
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
            "--output-dir",
            str(tmp_path / "named"),
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
    assert args.output_dir == tmp_path / "named"
    assert args.temperature_k == [300.0]
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
    assert args.temperature_k == [300.0]
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
            "--path-length-km",
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
    assert args.temperature_k == [300.0]
    assert args.pressure_bar == 1.0
    assert args.path_length_km == 1.5
    assert args.species == "CO2"
    assert args.broadening_composition == "H2:0.85,He:0.15"
    assert args.wn_ranges == [(50.0, 150.0)]


def test_transmission_parser_defaults_path_length_to_one_km() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "transmission",
            "--species",
            "CO2",
            "--wn-range",
            "50,150",
        ]
    )

    assert args.path_length_km == 1.0


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


def test_xsection_parser_accepts_temperature_list(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--temperature-k",
            "300,400,500",
            "--output",
            str(tmp_path / "xsection.nc"),
        ]
    )

    assert args.temperature_k == [300.0, 400.0, 500.0]


def test_output_path_for_multiple_ranges_appends_band_suffix(tmp_path) -> None:
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

    assert _output_path_for_wn_range(args, wn_range=(20.0, 2500.0), suffix=".nc") == tmp_path / "xsection_20_2500.nc"
    assert _output_path_for_wn_range(args, wn_range=(2500.0, 10000.0), suffix=".nc") == tmp_path / "xsection_2500_10000.nc"


def test_output_path_for_multiple_ranges_normalizes_suffix_tokens(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--wn-range=-1,30.5",
            "--wn-range=30.5,100",
            "--output",
            str(tmp_path / "xsection.nc"),
        ]
    )

    assert _output_path_for_wn_range(args, wn_range=(-1.0, 30.5), suffix=".nc") == tmp_path / "xsection_m1_30p5.nc"
    assert _output_path_for_wn_range(args, wn_range=(30.5, 100.0), suffix=".nc") == tmp_path / "xsection_30p5_100.nc"


def test_output_path_for_wn_range_uses_output_dir_for_generated_names(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--wn-range=20,2500",
            "--output-dir",
            str(tmp_path / "products"),
        ]
    )

    assert _output_path_for_wn_range(args, wn_range=(20.0, 2500.0), suffix=".nc") == tmp_path / "products" / "h2o_xsection_1bar_300K_20_2500cm1.nc"


def test_output_path_for_wn_range_includes_temperature_list_token(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xsection",
            "--species",
            "H2O",
            "--temperature-k",
            "300,400,500",
            "--wn-range=20,2500",
            "--output-dir",
            str(tmp_path / "products"),
        ]
    )

    assert _output_path_for_wn_range(args, wn_range=(20.0, 2500.0), suffix=".nc") == tmp_path / "products" / "h2o_xsection_1bar_300_400_500K_20_2500cm1.nc"


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
    monkeypatch.setattr("pyharp.spectra.dump_cli._compute_xsection_band", lambda task: (xr.Dataset(), "requested=h2:0.900,he:0.100 -> effective=air:1.000 (fallback: h2->air, he->air)"))
    monkeypatch.setattr("pyharp.spectra.dump_cli.write_dataset_via_tmp", lambda dataset, output_path, *, engine: None)

    main()

    out = capsys.readouterr().out
    assert "Wrote NetCDF:" in out
    assert "Broadening: requested=h2:0.900,he:0.100 -> effective=air:1.000" in out


def test_cli_xsection_writes_one_file_per_range(monkeypatch, tmp_path, capsys) -> None:
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
    written = []
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._compute_xsection_band",
        lambda task: (
            xr.Dataset(
                coords={"wavenumber": ("wavenumber", np.array([1.0, 2.0]))},
                data_vars={"sigma_total": ("wavenumber", np.array([3.0, 4.0]))},
            ),
            "requested=self:1.000 -> effective=self:1.000",
        ),
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._parallel_band_results",
        lambda tasks, *, worker: [worker(task) for task in tasks],
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.write_dataset_via_tmp",
        lambda dataset, output_path, *, engine: written.append((tuple(dataset["wavenumber"].values), output_path, engine)),
    )

    main()

    assert written == [
        ((1.0, 2.0), tmp_path / "xsection_20_2500.nc", "scipy"),
        ((1.0, 2.0), tmp_path / "xsection_2500_10000.nc", "scipy"),
    ]
    out = capsys.readouterr().out
    assert "Wrote NetCDF: " + str(tmp_path / "xsection_20_2500.nc") in out
    assert "Wrote NetCDF: " + str(tmp_path / "xsection_2500_10000.nc") in out
    assert out.count("Broadening: requested=self:1.000 -> effective=self:1.000") == 2


def test_cli_xsection_composition_writes_one_file_with_component_fields(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--composition",
            "H2:0.9,He:0.1,CH4:0.004,H2O:0.002,NH3:0.0003",
            "--wn-range=20,2500",
            "--output",
            str(tmp_path / "mixture.nc"),
        ],
    )
    written = []

    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._compute_xsection_band",
        lambda task: (
            xr.Dataset(
                coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
                data_vars={
                    "sigma_total": ("wavenumber", np.array([1.0, 2.0])),
                    "sigma_line_h2": ("wavenumber", np.array([0.1, 0.2])),
                    "binary_absorption_coefficient_h2_he": ("wavenumber", np.array([0.01, 0.02])),
                },
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._parallel_band_results",
        lambda tasks, *, worker: [worker(task) for task in tasks],
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.write_dataset_via_tmp",
        lambda dataset, output_path, *, engine: written.append((sorted(dataset.data_vars), dataset["sigma_total"].dims, tuple(dataset["temperature"].values), output_path, engine)),
    )

    main()

    assert written == [
        (
            ["binary_absorption_coefficient_h2_he", "sigma_line_h2", "sigma_total"],
            ("temperature", "wavenumber"),
            (300.0,),
            tmp_path / "mixture.nc",
            "scipy",
        )
    ]
    out = capsys.readouterr().out
    assert "Wrote NetCDF:" in out


def test_cli_xsection_uses_output_dir_for_generated_path(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--species",
            "H2O",
            "--wn-range=20,2500",
            "--output-dir",
            str(tmp_path / "products"),
        ],
    )
    written = []
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._compute_xsection_band",
        lambda task: (xr.Dataset(), None),
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.write_dataset_via_tmp",
        lambda dataset, output_path, *, engine: written.append(output_path),
    )

    main()

    assert written == [tmp_path / "products" / "h2o_xsection_1bar_300K_20_2500cm1.nc"]
    out = capsys.readouterr().out
    assert "Wrote NetCDF: " + str(tmp_path / "products" / "h2o_xsection_1bar_300K_20_2500cm1.nc") in out


def test_stack_temperature_datasets_adds_temperature_dimension() -> None:
    first = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
        data_vars={"sigma_total": ("wavenumber", np.array([1.0, 2.0]))},
        attrs={"species_name": "H2O", "temperature_k": 300.0, "pressure_pa": 1.0e5, "number_density_cm3": 1.0},
    )
    second = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
        data_vars={"sigma_total": ("wavenumber", np.array([3.0, 4.0]))},
        attrs={"species_name": "H2O", "temperature_k": 400.0, "pressure_pa": 1.0e5, "number_density_cm3": 2.0},
    )

    stacked = _stack_temperature_datasets([first, second], temperatures=[300.0, 400.0])
    try:
        assert stacked["sigma_total"].dims == ("temperature", "wavenumber")
        assert np.allclose(stacked["temperature"].values, np.array([300.0, 400.0]))
        assert stacked["temperature"].attrs["units"] == "K"
        assert "temperature_k" not in stacked.attrs
        assert "number_density_cm3" not in stacked.attrs
    finally:
        stacked.close()
        first.close()
        second.close()


def test_cli_xsection_temperature_list_writes_temperature_stacked_dataset(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--species",
            "H2O",
            "--temperature-k",
            "300,400",
            "--wn-range",
            "20,22",
            "--output",
            str(tmp_path / "xsection.nc"),
        ],
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._parallel_band_results",
        lambda tasks, *, worker: [
            (
                xr.Dataset(
                    coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
                    data_vars={"sigma_total": ("wavenumber", np.array([float(task_args.temperature_k), float(task_args.temperature_k) + 1.0]))},
                    attrs={"species_name": "H2O", "temperature_k": float(task_args.temperature_k), "pressure_pa": 1.0e5},
                ),
                "requested=self:1.000 -> effective=self:1.000",
            )
            for _, task_args in tasks
        ],
    )
    written = []
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.write_dataset_via_tmp",
        lambda dataset, output_path, *, engine: written.append((dataset["sigma_total"].dims, tuple(dataset["temperature"].values), output_path, engine)),
    )

    main()

    assert written == [
        (("temperature", "wavenumber"), (300.0, 400.0), tmp_path / "xsection.nc", "scipy")
    ]
    out = capsys.readouterr().out
    assert "Wrote NetCDF: " + str(tmp_path / "xsection.nc") in out


def test_cli_xsection_temperature_list_and_ranges_flattens_all_tasks(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--species",
            "H2O",
            "--temperature-k",
            "300,400,500",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output",
            str(tmp_path / "xsection.nc"),
        ],
    )
    seen = []

    def fake_parallel(tasks, *, worker):
        for _, task_args in tasks:
            seen.append((task_args.wn_range, float(task_args.temperature_k)))
        return [
            (
                xr.Dataset(
                    coords={"wavenumber": ("wavenumber", np.array([1.0, 2.0]))},
                    data_vars={"sigma_total": ("wavenumber", np.array([3.0, 4.0]))},
                    attrs={"species_name": "H2O", "pressure_pa": 1.0e5},
                ),
                None,
            )
            for _ in tasks
        ]

    monkeypatch.setattr("pyharp.spectra.dump_cli._parallel_band_results", fake_parallel)
    monkeypatch.setattr("pyharp.spectra.dump_cli.write_dataset_via_tmp", lambda dataset, output_path, *, engine: None)

    main()

    assert seen == [
        ((20.0, 2500.0), 300.0),
        ((20.0, 2500.0), 400.0),
        ((20.0, 2500.0), 500.0),
        ((2500.0, 10000.0), 300.0),
        ((2500.0, 10000.0), 400.0),
        ((2500.0, 10000.0), 500.0),
    ]
    out = capsys.readouterr().out
    assert "Wrote NetCDF: " + str(tmp_path / "xsection_20_2500.nc") in out
    assert "Wrote NetCDF: " + str(tmp_path / "xsection_2500_10000.nc") in out


def test_combine_band_datasets_preserves_band_metadata() -> None:
    first = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
        data_vars={"sigma_total": ("wavenumber", np.array([1.0, 2.0]), {"long_name": "total absorption cross section", "units": "cm^2 molecule^-1"})},
        attrs={"species_name": "H2O"},
    )
    second = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.array([30.0, 31.0, 32.0]))},
        data_vars={"sigma_total": ("wavenumber", np.array([3.0, 4.0, 5.0]), {"long_name": "total absorption cross section", "units": "cm^2 molecule^-1"})},
        attrs={"species_name": "H2O"},
    )

    combined = combine_band_datasets([first, second], wn_ranges=[(20.0, 22.0), (30.0, 33.0)])
    try:
        assert combined.attrs["species_name"] == "H2O"
        assert combined.attrs["num_bands"] == 2
        assert np.allclose(combined["band_size"].values, [2, 3])
        assert np.allclose(combined["band_wavenumber_min"].values, [20.0, 30.0])
        assert np.allclose(combined["band_wavenumber_max"].values, [22.0, 33.0])
        assert np.allclose(combined["wavenumber"].values[0, :2], [20.0, 21.0])
        assert np.isnan(combined["wavenumber"].values[0, 2])
        assert np.allclose(combined["sigma_total"].values[1, :3], [3.0, 4.0, 5.0])
        assert combined["sigma_total"].attrs["units"] == "cm^2 molecule^-1"
    finally:
        combined.close()


def test_write_dataset_via_tmp_creates_missing_output_parent(tmp_path) -> None:
    dataset = xr.Dataset(coords={"wavenumber": ("wavenumber", np.array([1.0, 2.0]))}, data_vars={"sigma_total": ("wavenumber", np.array([3.0, 4.0]))})
    output_path = tmp_path / "nested" / "pair.nc"

    try:
        write_dataset_via_tmp(dataset, output_path, engine="scipy")
        assert output_path.exists()
    finally:
        dataset.close()


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

    dataset = _xsection_dataset(
        spectrum,
        species_name="H2O",
        secondary_component={"kind": "continuum", "label": "H2O continuum (MT_CKD)"},
        wn_range=(20.0, 22.0),
    )
    try:
        assert set(dataset.data_vars) == {
            "sigma_line_h2o",
            "sigma_continuum_h2o_continuum_mt_ckd",
            "sigma_total",
        }
        assert dataset.attrs["band_wavenumber_min_cm1"] == 20.0
        assert dataset.attrs["band_wavenumber_max_cm1"] == 22.0
    finally:
        dataset.close()


def test_xsection_dataset_adds_binary_cia_field_for_species_dump() -> None:
    spectrum = AbsorptionSpectrum(
        species_name="CH4",
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

    dataset = _xsection_dataset(
        spectrum,
        species_name="CH4",
        secondary_component={
            "kind": "binary_cia",
            "label": "CH4-CH4",
            "binary_absorption_coefficient": np.array([7.0, 8.0]),
        },
    )
    try:
        assert set(dataset.data_vars) == {
            "sigma_line_ch4",
            "sigma_cia_ch4_ch4",
            "binary_absorption_coefficient_ch4_ch4",
            "sigma_total",
        }
        assert dataset["binary_absorption_coefficient_ch4_ch4"].attrs["units"] == "cm^5 molecule^-2"
    finally:
        dataset.close()


def test_pair_xsection_dataset_uses_binary_absorption_units(monkeypatch, tmp_path) -> None:
    class FakeCia:
        def interpolate_to_grid(self, *, temperature_k, wavenumber_grid_cm1):
            return np.array([1.0e-46, 2.0e-46], dtype=np.float64)

    monkeypatch.setattr("pyharp.spectra.dump_cli.load_cia_dataset", lambda **kwargs: FakeCia())
    args = build_parser().parse_args(
        [
            "xsection",
            "--pair",
            "H2-He",
            "--temperature-k",
            "300",
            "--wn-range",
            "20,22",
            "--hitran-dir",
            str(tmp_path / "hitran"),
        ]
    )

    dataset = _pair_xsection_dataset(_args_for_wn_range(args, args.wn_ranges[0]))
    try:
        assert set(dataset.data_vars) == {"binary_absorption_coefficient"}
        assert dataset["binary_absorption_coefficient"].attrs["units"] == "cm^5 molecule^-2"
        assert "binary absorption coefficient" in dataset["binary_absorption_coefficient"].attrs["long_name"].lower()
        assert dataset.attrs["pair_name"] == "H2-He"
        assert dataset.attrs["band_wavenumber_min_cm1"] == 20.0
        assert dataset.attrs["band_wavenumber_max_cm1"] == 22.0
    finally:
        dataset.close()


def test_composition_xsection_dataset_uses_one_field_per_species_or_cia(monkeypatch, tmp_path) -> None:
    products = type(
        "Products",
        (),
        {
            "spectrum": type(
                "Spectrum",
                (),
                {
                    "wavenumber_cm1": np.array([20.0, 21.0]),
                    "sigma_total_cm2_molecule": np.array([1.0, 2.0]),
                    "temperature_k": 300.0,
                    "pressure_pa": 1.0e5,
                },
            )(),
            "species_terms": (
                type(
                    "SpeciesTerm",
                    (),
                    {"species_name": "H2O", "mole_fraction": 0.2, "sigma_line_cm2_molecule": np.array([3.0, 4.0])},
                )(),
            ),
            "secondary_sources": (
                type(
                    "Secondary",
                    (),
                    {"kind": "binary_cia", "label": "H2-He", "weight": 0.09, "sigma_cm2_molecule": np.array([5.0, 6.0])},
                )(),
                type(
                    "Secondary",
                    (),
                    {"kind": "continuum", "label": "H2O continuum (MT_CKD)", "weight": 0.2, "sigma_cm2_molecule": np.array([0.5, 0.6])},
                )(),
            ),
        },
    )()
    monkeypatch.setattr("pyharp.spectra.dump_cli._compute_composition_products", lambda args: products)
    args = build_parser().parse_args(
        [
            "xsection",
            "--composition",
            "H2:0.9,He:0.1,H2O:0.002",
            "--wn-range",
            "20,21",
            "--hitran-dir",
            str(tmp_path / "hitran"),
        ]
    )

    dataset = _composition_xsection_dataset(_args_for_wn_range(args, args.wn_ranges[0]))
    try:
        assert set(dataset.data_vars) == {
            "sigma_total",
            "sigma_line_h2o",
            "sigma_continuum_h2o_continuum_mt_ckd",
            "binary_absorption_coefficient_h2_he",
        }
        assert dataset.attrs["species_name"] == "H2,He,H2O"
        assert dataset.attrs["band_wavenumber_min_cm1"] == 20.0
        assert dataset.attrs["band_wavenumber_max_cm1"] == 21.0
        assert dataset["sigma_line_h2o"].attrs["units"] == "cm^2 molecule^-1"
        assert np.allclose(dataset["sigma_line_h2o"].values, np.array([3.0, 4.0]))
        assert dataset["sigma_continuum_h2o_continuum_mt_ckd"].attrs["units"] == "cm^2 molecule^-1"
        assert np.allclose(dataset["sigma_continuum_h2o_continuum_mt_ckd"].values, np.array([0.5, 0.6]) / 0.2)
        assert dataset["binary_absorption_coefficient_h2_he"].attrs["units"] == "cm^5 molecule^-2"
        assert dataset["binary_absorption_coefficient_h2_he"].attrs["long_name"].startswith("H2-He")
        assert np.allclose(dataset["binary_absorption_coefficient_h2_he"].values, np.array([5.0, 6.0]) / (0.09 * 1.0e5 / (1.380649e-16 * 300.0)))
    finally:
        dataset.close()


def test_cli_xsection_composition_multi_range_uses_composition_worker(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyharp-dump",
            "xsection",
            "--composition",
            "H2:0.9,He:0.1,H2O:0.002",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output",
            str(tmp_path / "mixture.nc"),
        ],
    )
    written = []

    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._compute_xsection_band",
        lambda task: (
            xr.Dataset(
                coords={"wavenumber": ("wavenumber", np.array([20.0, 21.0]))},
                data_vars={"sigma_total": ("wavenumber", np.array([1.0, 2.0]))},
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli._parallel_band_results",
        lambda tasks, *, worker: [worker(task) for task in tasks],
    )
    monkeypatch.setattr(
        "pyharp.spectra.dump_cli.write_dataset_via_tmp",
        lambda dataset, output_path, *, engine: written.append((tuple(dataset["wavenumber"].values), output_path, engine)),
    )

    main()

    assert written == [
        ((20.0, 21.0), tmp_path / "mixture_20_2500.nc", "scipy"),
        ((20.0, 21.0), tmp_path / "mixture_2500_10000.nc", "scipy"),
    ]
    out = capsys.readouterr().out
    assert "Wrote NetCDF: " + str(tmp_path / "mixture_20_2500.nc") in out
    assert "Wrote NetCDF: " + str(tmp_path / "mixture_2500_10000.nc") in out


def test_species_transmission_dataset_matches_xsection_naming() -> None:
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
    transmittance = type(
        "Trans",
        (),
        {
            "wavenumber_cm1": np.array([20.0, 21.0]),
            "transmittance_line": np.array([0.9, 0.8]),
            "transmittance_cia": np.array([0.99, 0.98]),
            "transmittance_total": np.array([0.89, 0.78]),
            "path_length_m": 10.0,
            "temperature_k": 300.0,
            "pressure_pa": 1.0e5,
        },
    )()

    dataset = _species_transmission_dataset(
        spectrum=spectrum,
        transmittance=transmittance,
        species_name="H2O",
        secondary_component={"kind": "continuum", "label": "H2O continuum (MT_CKD)"},
        wn_range=(20.0, 22.0),
    )
    try:
        assert set(dataset.data_vars) == {
            "transmittance_line_h2o",
            "attenuation_line_h2o",
            "transmittance_continuum_h2o_continuum_mt_ckd",
            "attenuation_continuum_h2o_continuum_mt_ckd",
            "transmittance_total",
            "attenuation_total",
        }
        assert dataset.attrs["species_name"] == "H2O"
        assert dataset.attrs["band_wavenumber_min_cm1"] == 20.0
        assert dataset.attrs["band_wavenumber_max_cm1"] == 22.0
        assert dataset["attenuation_continuum_h2o_continuum_mt_ckd"].attrs["units"] == "m^-1"
    finally:
        dataset.close()


def test_composition_transmission_dataset_uses_component_names_and_attrs(monkeypatch, tmp_path) -> None:
    products = type(
        "Products",
        (),
        {
            "spectrum": type(
                "Spectrum",
                (),
                {
                    "wavenumber_cm1": np.array([20.0, 21.0]),
                    "attenuation_total_m1": np.array([7.0, 8.0]),
                    "temperature_k": 300.0,
                    "pressure_pa": 1.0e5,
                    "number_density_cm3": 10.0,
                },
            )(),
            "transmittance": type(
                "Trans",
                (),
                {
                    "wavenumber_cm1": np.array([20.0, 21.0]),
                    "transmittance_total": np.array([0.7, 0.6]),
                    "path_length_m": 2.0,
                    "temperature_k": 300.0,
                    "pressure_pa": 1.0e5,
                },
            )(),
            "species_terms": (
                type(
                    "SpeciesTerm",
                    (),
                    {"species_name": "H2O", "mole_fraction": 0.2, "sigma_line_cm2_molecule": np.array([3.0, 4.0])},
                )(),
            ),
            "secondary_sources": (
                type(
                    "Secondary",
                    (),
                    {"kind": "continuum", "label": "H2O continuum (MT_CKD)", "weight": 0.2, "sigma_cm2_molecule": np.array([5.0, 6.0])},
                )(),
                type(
                    "Secondary",
                    (),
                    {"kind": "binary_cia", "label": "H2-He", "weight": 0.09, "sigma_cm2_molecule": np.array([7.0, 8.0])},
                )(),
            ),
        },
    )()
    monkeypatch.setattr("pyharp.spectra.dump_cli._compute_composition_products", lambda args: products)
    args = build_parser().parse_args(
        [
            "transmission",
            "--composition",
            "H2:0.9,He:0.1,H2O:0.002",
            "--path-length-km",
            "0.002",
            "--wn-range",
            "20,21",
            "--hitran-dir",
            str(tmp_path / "hitran"),
        ]
    )

    dataset = _composition_transmission_dataset(_args_for_wn_range(args, args.wn_ranges[0]))
    try:
        assert dataset.attrs["composition_input"] == "H2:0.9,He:0.1,H2O:0.002"
        assert dataset.attrs["species_name"] == "H2,He,H2O"
        assert dataset.attrs["band_wavenumber_min_cm1"] == 20.0
        assert dataset.attrs["band_wavenumber_max_cm1"] == 21.0
        assert set(dataset.data_vars) == {
            "transmittance_total",
            "attenuation_total",
            "attenuation_line_h2o",
            "transmittance_line_h2o",
            "attenuation_continuum_h2o_continuum_mt_ckd",
            "transmittance_continuum_h2o_continuum_mt_ckd",
            "attenuation_cia_h2_he",
            "transmittance_cia_h2_he",
        }
        assert np.allclose(dataset["attenuation_line_h2o"].values, np.array([600.0, 800.0]))
        assert np.allclose(dataset["attenuation_continuum_h2o_continuum_mt_ckd"].values, np.array([5000.0, 6000.0]))
        assert np.allclose(dataset["attenuation_cia_h2_he"].values, np.array([7000.0, 8000.0]))
    finally:
        dataset.close()


def test_composition_transmission_total_equals_product_of_weighted_components(monkeypatch, tmp_path) -> None:
    products = type(
        "Products",
        (),
        {
            "spectrum": type(
                "Spectrum",
                (),
                {
                    "wavenumber_cm1": np.array([20.0, 21.0]),
                    "attenuation_total_m1": np.array([18.0, 24.0]),
                    "temperature_k": 300.0,
                    "pressure_pa": 1.0e5,
                    "number_density_cm3": 10.0,
                },
            )(),
            "transmittance": type(
                "Trans",
                (),
                {
                    "wavenumber_cm1": np.array([20.0, 21.0]),
                    "transmittance_total": np.exp(-2.0 * np.array([18.0, 24.0])),
                    "path_length_m": 2.0,
                    "temperature_k": 300.0,
                    "pressure_pa": 1.0e5,
                },
            )(),
            "species_terms": (
                type(
                    "SpeciesTerm",
                    (),
                    {"species_name": "H2O", "mole_fraction": 0.2, "sigma_line_cm2_molecule": np.array([3.0, 4.0])},
                )(),
            ),
            "secondary_sources": (
                type(
                    "Secondary",
                    (),
                    {"kind": "continuum", "label": "H2O continuum (MT_CKD)", "sigma_cm2_molecule": np.array([5.0, 6.0])},
                )(),
                type(
                    "Secondary",
                    (),
                    {"kind": "binary_cia", "label": "H2-He", "sigma_cm2_molecule": np.array([7.0, 8.0])},
                )(),
            ),
        },
    )()
    monkeypatch.setattr("pyharp.spectra.dump_cli._compute_composition_products", lambda args: products)
    args = build_parser().parse_args(
        [
            "transmission",
            "--composition",
            "H2:0.9,He:0.1,H2O:0.002",
            "--path-length-km",
            "0.002",
            "--wn-range",
            "20,21",
            "--hitran-dir",
            str(tmp_path / "hitran"),
        ]
    )

    dataset = _composition_transmission_dataset(_args_for_wn_range(args, args.wn_ranges[0]))
    try:
        component_product = (
            dataset["transmittance_line_h2o"].values
            * dataset["transmittance_continuum_h2o_continuum_mt_ckd"].values
            * dataset["transmittance_cia_h2_he"].values
        )
        assert np.allclose(component_product, dataset["transmittance_total"].values)
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
        assert np.allclose(wavenumber_grid_cm1, np.array([20.0, 21.0]))
        assert temperature_k == 300.0
        assert pressure_pa == 1.0e5
        return None, np.array([1.0e-24, 2.0e-24])

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
        assert np.allclose(wavenumber_grid_cm1, np.array([20.0, 21.0]))
        assert temperature_k == 300.0
        assert pressure_pa == 1.0e5
        assert line_provider is fake_provider
        assert cia_dataset is None
        assert np.allclose(cia_cross_section_cm2_molecule, np.array([1.0e-24, 2.0e-24]))
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
    assert "Transmission path length in kilometers." in help_text
    assert "Output NetCDF path. Defaults to an auto-generated path" in help_text
    assert "under --output-dir." in help_text
    assert "pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004" in help_text
