from pathlib import Path

from pyharp.spectra.output_names import default_output_path


def test_default_output_path_uses_requested_pattern() -> None:
    path = default_output_path(
        target_name="CO2",
        plot_type="xsection",
        temperature_k=300.0,
        pressure_bar=1.0,
        wn_range=(20.0, 2500.0),
        suffix=".png",
    )

    assert path == Path("output/co2_xsection_1bar_300K_20_2500cm1.png")


def test_default_output_path_sanitizes_cia_and_composition_names() -> None:
    cia_path = default_output_path(
        target_name="H2-He",
        plot_type="attenuation",
        temperature_k=275.5,
        pressure_bar=0.25,
        wn_range=(25.0, 30.5),
        suffix=".png",
    )
    composition_path = default_output_path(
        target_name="H2O:0.1,H2:0.9",
        plot_type="overview",
        temperature_k=300.0,
        pressure_bar=1.0,
        wn_range=(25.0, 2500.0),
        suffix=".pdf",
    )

    assert cia_path == Path("output/h2_he_attenuation_0p25bar_275p5K_25_30p5cm1.png")
    assert composition_path == Path("output/h2o_0p1_h2_0p9_overview_1bar_300K_25_2500cm1.pdf")
