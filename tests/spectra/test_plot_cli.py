import numpy as np
import pytest

from pyharp.spectra import plot_cli
from pyharp.spectra.dump_cli import _args_for_wn_range, _composition_transmission_dataset, build_parser as build_dump_parser


def test_plot_parser_accepts_cia_binary_with_wn_range(tmp_path) -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "binary",
            "--pair",
            "H2-H2",
            "--output-dir",
            str(tmp_path / "figures"),
            "--temperature-k",
            "300",
            "--wn-range",
            "20,10000",
            "--output",
            str(tmp_path / "cia.png"),
        ]
    )

    assert args.command == "binary"
    assert args.pair == "H2-H2"
    assert args.output_dir == tmp_path / "figures"
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
            "--broadening-composition",
            "air:0.8,self:0.2",
            "--wn-range",
            "20,2500",
            "--output",
            str(tmp_path / "co2.png"),
        ]
    )

    assert args.command == "xsection"
    assert args.species == "CO2"
    assert args.temperature_k == [300.0]
    assert args.pressure_bar == [1.0]
    assert args.broadening_composition == "air:0.8,self:0.2"
    assert args.wn_range == (20.0, 2500.0)
    assert args.figure == tmp_path / "co2.png"


def test_plot_parser_accepts_matched_temperature_pressure_vectors() -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "transmission",
            "--species",
            "CO2",
            "--temperature-k",
            "300,400",
            "--pressure-bar",
            "1,10",
        ]
    )

    assert args.temperature_k == [300.0, 400.0]
    assert args.pressure_bar == [1.0, 10.0]


def test_plot_parser_accepts_atm_overview_ranges(tmp_path) -> None:
    parser = plot_cli.build_parser()
    args = parser.parse_args(
        [
            "overview",
            "--composition",
            "H2O:0.1,H2:0.9",
            "--broadening-composition",
            "H2:0.85,He:0.15",
            "--wn-range=25,2500",
            "--wn-range=2501,20000",
            "--manifest",
            str(tmp_path / "sources.json"),
            "--output",
            str(tmp_path / "atm.pdf"),
        ]
    )

    assert args.command == "overview"
    assert args.composition == "H2O:0.1,H2:0.9"
    assert args.broadening_composition == "H2:0.85,He:0.15"
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
    assert calls[0].figure.name == "h2_he_attenuation_300K_1bar_25_30cm1.png"


def test_plot_main_dispatches_molecule_xsection_with_default_figure(monkeypatch) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)

    plot_cli.main(["xsection", "--species", "CO2", "--temperature-k", "275.5", "--pressure-bar", "0.25", "--wn-range", "25,30.5"])

    assert len(calls) == 1
    assert calls[0].figure.name == "co2_xsection_275p5K_0p25bar_25_30p5cm1.png"


def test_plot_main_dispatches_one_xsection_per_state_pair(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)
    monkeypatch.setattr(
        "pyharp.spectra.plot_cli._parallel_plot_results",
        lambda tasks, *, worker: [worker(task) for task in tasks],
    )

    plot_cli.main(
        [
            "xsection",
            "--species",
            "CO2",
            "--temperature-k",
            "275.5,300",
            "--pressure-bar",
            "0.25,1",
            "--wn-range",
            "25,30.5",
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert [(call.temperature_k, call.pressure_bar, call.figure.name) for call in calls] == [
        (275.5, 0.25, "co2_xsection_275p5K_0p25bar_25_30p5cm1.png"),
        (300.0, 1.0, "co2_xsection_300K_1bar_25_30p5cm1.png"),
    ]


def test_plot_main_appends_state_suffix_to_explicit_output_for_multiple_pairs(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)
    monkeypatch.setattr(
        "pyharp.spectra.plot_cli._parallel_plot_results",
        lambda tasks, *, worker: [worker(task) for task in tasks],
    )

    plot_cli.main(
        [
            "xsection",
            "--species",
            "CO2",
            "--temperature-k",
            "300,400",
            "--pressure-bar",
            "1,10",
            "--output",
            str(tmp_path / "co2.png"),
        ]
    )

    assert [call.figure for call in calls] == [
        tmp_path / "co2_300K_1bar.png",
        tmp_path / "co2_400K_10bar.png",
    ]


def test_plot_main_uses_output_dir_for_default_figure(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)

    plot_cli.main(
        [
            "xsection",
            "--species",
            "CO2",
            "--temperature-k",
            "275.5",
            "--pressure-bar",
            "0.25",
            "--wn-range",
            "25,30.5",
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert len(calls) == 1
    assert calls[0].figure == tmp_path / "figures" / "co2_xsection_275p5K_0p25bar_25_30p5cm1.png"


def test_plot_main_passes_broadening_composition_to_molecule_workflow(monkeypatch) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_xsection", fake_run)

    plot_cli.main(["xsection", "--species", "CO2", "--broadening-composition", "air:0.8,self:0.2"])

    assert calls[0].broadening_composition == "air:0.8,self:0.2"


def test_parallel_plot_results_uses_selected_process_context(monkeypatch) -> None:
    created = {}

    class DummyExecutor:
        def __init__(self, *, max_workers, mp_context):
            created["max_workers"] = max_workers
            created["mp_context"] = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, worker, tasks):
            return [worker(task) for task in tasks]

    monkeypatch.setattr("pyharp.spectra.plot_cli.process_pool_context", lambda: "ctx-token")
    monkeypatch.setattr("pyharp.spectra.plot_cli.ProcessPoolExecutor", DummyExecutor)

    result = plot_cli._parallel_plot_results([("x", 1), ("y", 2)], worker=lambda task: task[1] * 10)

    assert result == [10, 20]
    assert created == {"max_workers": 2, "mp_context": "ctx-token"}


def test_plot_main_dispatches_composition_attenuation(monkeypatch) -> None:
    calls = []

    def fake_run(args, *, wn_range):
        calls.append((args, wn_range))

    monkeypatch.setattr("pyharp.spectra.plot_cli.atm_overview_cli.run_atm_attenuation", fake_run)

    plot_cli.main(["attenuation", "--composition", "H2:0.9,He:0.1", "--wn-range", "25,30"])

    assert len(calls) == 1
    assert calls[0][0].composition == "H2:0.9,He:0.1"
    assert calls[0][0].figure.name == "h2_0p9_he_0p1_attenuation_300K_1bar_25_30cm1.png"
    assert calls[0][1] == (25.0, 30.0)


def test_plot_main_preserves_explicit_figure_over_output_dir(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.cia_plot_cli.run_binary", fake_run)

    plot_cli.main(
        [
            "binary",
            "--pair",
            "H2-He",
            "--output-dir",
            str(tmp_path / "figures"),
            "--output",
            str(tmp_path / "custom.png"),
        ]
    )

    assert len(calls) == 1
    assert calls[0].figure == tmp_path / "custom.png"


def test_plot_composition_transmission_matches_dump_total(monkeypatch, tmp_path) -> None:
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
    calls = []

    def fake_run(args, *, wn_range):
        calls.append((args, wn_range))

    monkeypatch.setattr("pyharp.spectra.plot_cli.atm_overview_cli.run_atm_transmission", fake_run)

    composition = "H2:0.9,He:0.1,H2O:0.002"
    plot_cli.main(
        [
            "transmission",
            "--composition",
            composition,
            "--path-length-km",
            "0.002",
            "--wn-range",
            "20,21",
            "--output",
            str(tmp_path / "mixture.png"),
        ]
    )

    dump_args = build_dump_parser().parse_args(
        [
            "transmission",
            "--composition",
            composition,
            "--path-length-km",
            "0.002",
            "--wn-range",
            "20,21",
            "--output",
            str(tmp_path / "mixture.nc"),
        ]
    )

    dataset = _composition_transmission_dataset(_args_for_wn_range(dump_args, dump_args.wn_ranges[0]))
    try:
        assert np.allclose(dataset["transmittance_total"].values, products.transmittance.transmittance_total)
    finally:
        dataset.close()

    assert len(calls) == 1
    assert calls[0][0].composition == composition
    assert calls[0][1] == (20.0, 21.0)
    assert calls[0][0].path_length_km == 0.002


def test_plot_main_rejects_multiple_selectors() -> None:
    with pytest.raises(SystemExit):
        plot_cli.main(["attenuation", "--pair", "H2-H2", "--species", "H2O"])


def test_plot_main_rejects_mismatched_temperature_pressure_vectors() -> None:
    with pytest.raises(SystemExit):
        plot_cli.main(["xsection", "--species", "CO2", "--temperature-k", "300,400", "--pressure-bar", "1"])


def test_plot_overview_parallelizes_over_state_pairs_and_wn_ranges(monkeypatch, tmp_path) -> None:
    inner_task_counts = []
    calls = []

    def fake_parallel_products(tasks):
        inner_task_counts.append(len(tasks))
        calls.append(tasks)
        return [
            type("Products", (), {"spectrum": type("Spectrum", (), {"temperature_k": 300.0, "pressure_pa": 1.0e5})()})()
            for _ in tasks
        ]

    class _DummyPdf:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def savefig(self, fig):
            return None

    dummy_figure = type("Figure", (), {})()
    dummy_axis = type("Axis", (), {})()

    monkeypatch.setattr("pyharp.spectra.atm_overview_cli._parallel_mixture_overview_products", fake_parallel_products)
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli._render_mixture_overview", lambda fig, axes, *, products: None)
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli._page_manifest", lambda products: {})
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.PdfPages", _DummyPdf)
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.plt.subplots", lambda **kwargs: (dummy_figure, np.array([[dummy_axis], [dummy_axis], [dummy_axis], [dummy_axis]])))
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.plt.close", lambda fig: None)

    plot_cli.main(
        [
            "overview",
            "--composition",
            "H2:0.9,He:0.1",
            "--temperature-k",
            "300,400,500",
            "--pressure-bar",
            "1,2,3",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output",
            str(tmp_path / "atm.pdf"),
        ]
    )

    assert inner_task_counts == [6]
    assert all(task_args.figure == tmp_path / "atm.pdf" for task_args, _ in calls[0])


def test_plot_overview_uses_single_output_pdf_for_multiple_state_pairs(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_overview", fake_run)

    plot_cli.main(
        [
            "overview",
            "--species",
            "CO2",
            "--temperature-k",
            "300,400",
            "--pressure-bar",
            "1,10",
            "--output",
            str(tmp_path / "overview.pdf"),
        ]
    )

    assert len(calls) == 1
    assert calls[0].figure == tmp_path / "overview.pdf"
    assert calls[0].temperature_k == [300.0, 400.0]
    assert calls[0].pressure_bar == [1.0, 10.0]


def test_plot_overview_uses_combined_default_name_without_duplicate_units(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(args):
        calls.append(args)

    monkeypatch.setattr("pyharp.spectra.plot_cli.molecule_plot_cli.run_overview_batch", fake_run)

    plot_cli.main(
        [
            "overview",
            "--species",
            "CO2",
            "H2O",
            "--temperature-k",
            "170,400,780",
            "--pressure-bar",
            "0.1,1,100",
            "--wn-range=20,2500",
            "--wn-range=2500,10000",
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )

    assert len(calls) == 1
    assert calls[0].figure == tmp_path / "figures" / "co2_h2o_overview_170_780K_0p1_100bar_20_10000cm1.pdf"


def test_plot_help_includes_subcommands_and_examples(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        plot_cli.main(["-h"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "binary" in help_text
    assert "Plot a HITRAN CIA binary coefficient." in help_text
    assert "Target selection:" in help_text
    assert "pyharp-plot transmission --composition H2O:0.1,H2:0.9" in help_text


def test_plot_subcommand_help_describes_selectors_and_outputs(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        plot_cli.main(["transmission", "-h"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "Plot transmission for exactly one target: --pair, --species, or --composition." in help_text
    assert "--composition SPECIES:FRACTION,..." in help_text
    assert "Transmission path length in kilometers." in help_text
    assert "Output PNG path. Defaults to an auto-generated path" in help_text
    assert "under --output-dir." in help_text
    assert "(default: None)" not in help_text
