import json

import numpy as np

from pyharp.spectra.atm_overview_cli import _find_binary_pairs, _parse_composition, compute_mixture_overview_products, build_atm_overview_parser, run_atm_overview


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
            "--broadening-composition",
            "H2:0.85,He:0.15",
            "--wn-range=25,2500",
            "--wn-range=2501,20000",
            "--manifest",
            str(tmp_path / "sources.json"),
            "--figure",
            str(tmp_path / "overview.pdf"),
        ]
    )

    assert args.composition == "H2O:0.1,H2:0.9"
    assert args.broadening_composition == "H2:0.85,He:0.15"
    assert args.wn_ranges == [(25.0, 2500.0), (2501.0, 20000.0)]
    assert args.manifest == tmp_path / "sources.json"
    assert args.figure == tmp_path / "overview.pdf"


def test_compute_mixture_overview_reports_broadening_fallback(monkeypatch, tmp_path, capsys) -> None:
    parser = build_atm_overview_parser()
    args = parser.parse_args(
        [
            "--composition",
            "CO2:0.1,H2:0.9",
            "--broadening-composition",
            "H2:0.9,He:0.1",
            "--wn-range=20,22",
            "--hitran-dir",
            str(tmp_path / "hitran"),
        ]
    )

    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli.download_hitran_lines",
        lambda config, band: type("LineDb", (), {"table_name": "co2_lines_20_22", "cache_dir": tmp_path})(),
    )
    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli.load_hitran_line_list",
        lambda config, band: type(
            "LineList",
            (),
            {"wavenumber_cm1": np.array([20.0]), "line_intensity": np.array([1.0e-25])},
        )(),
    )

    class FakeLineProvider:
        def broadening_summary(self):
            return "requested=h2:0.900,he:0.100 -> effective=air:1.000 (fallback: h2->air, he->air)"

        def cross_section_cm2_molecule(self, **kwargs):
            grid = np.asarray(kwargs["wavenumber_grid_cm1"], dtype=np.float64)
            return np.ones_like(grid)

    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.build_line_provider", lambda config, line_db: FakeLineProvider())
    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli.load_cia_dataset",
        lambda *args, **kwargs: type("Cia", (), {"source_path": tmp_path / "cia", "pair": "CO2-CO2", "interpolate_to_grid": lambda self, temperature_k, grid: np.zeros_like(grid)})(),
    )
    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli.compute_mt_ckd_h2o_continuum_cross_section",
        lambda **kwargs: np.zeros_like(kwargs["wavenumber_grid_cm1"]),
    )

    compute_mixture_overview_products(args, wn_range=(20.0, 22.0))

    out = capsys.readouterr().out
    assert "CO2 broadening: requested=h2:0.900,he:0.100 -> effective=air:1.000" in out


def test_run_atm_overview_manifest_always_uses_state_lists(monkeypatch, tmp_path) -> None:
    parser = build_atm_overview_parser()
    args = parser.parse_args(
        [
            "--composition",
            "H2:0.9,He:0.1",
            "--temperature-k",
            "300",
            "--pressure-bar",
            "1",
            "--wn-range=20,2500",
            "--figure",
            str(tmp_path / "overview.pdf"),
            "--manifest",
            str(tmp_path / "overview.manifest.json"),
        ]
    )
    written = {}

    class _DummyPdf:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def savefig(self, fig):
            return None

    products = type(
        "Products",
        (),
        {
            "band": type(
                "Band",
                (),
                {"wavenumber_min_cm1": 20.0, "wavenumber_max_cm1": 2500.0, "resolution_cm1": 1.0},
            )(),
            "spectrum": type(
                "Spectrum",
                (),
                {"wavenumber_cm1": np.array([20.0, 21.0]), "temperature_k": 300.0, "pressure_pa": 1.0e5},
            )(),
            "species_terms": (),
            "manifest_sources": (),
            "transmittance": object(),
        },
    )()

    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli._parallel_mixture_overview_products",
        lambda tasks: iter([products]),
    )
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli._render_mixture_overview", lambda fig, axes, *, products: None)
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.PdfPages", _DummyPdf)
    monkeypatch.setattr(
        "pyharp.spectra.atm_overview_cli.plt.subplots",
        lambda **kwargs: (object(), np.array([[object()], [object()], [object()], [object()]])),
    )
    monkeypatch.setattr("pyharp.spectra.atm_overview_cli.plt.close", lambda fig: None)
    monkeypatch.setattr(
        "pathlib.Path.write_text",
        lambda self, text: written.setdefault(str(self), text),
    )

    run_atm_overview(args)

    manifest = json.loads(written[str(tmp_path / "overview.manifest.json")])
    assert manifest["temperature_k"] == [300.0]
    assert manifest["pressure_bar"] == [1.0]
