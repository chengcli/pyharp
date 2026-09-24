import contextlib
import copy
import io
import json

import numpy as np
import pytest

from pyharp.spectra.config import SpectralBandConfig, SpectroscopyConfig
from pyharp.spectra.hitran_molecule_utils import (
    FastLineProvider,
    HapiLineProvider,
    LineDatabase,
    build_line_provider,
    download_hitran_lines,
    load_hitran_line_list,
)
from pyharp.spectra.line_kernel import load_line_table, voigt_cross_section

hapi = pytest.importorskip("hapi")


def par_line(molec, iso, nu, sw, elower, gamma_air=0.07, gamma_self=0.3, n_air=0.7, delta_air=-0.002):
    gamma_air_field = f"{gamma_air:.4f}"[1:]  # HITRAN writes ".0700"
    line = (
        f"{molec:2d}{iso:1s}{nu:12.6f}{sw:10.3E}{1.0:10.3E}{gamma_air_field}{gamma_self:5.3f}"
        f"{elower:10.4f}{n_air:4.2f}{delta_air:8.5f}" + " " * 60 + f"{'000000':6s}{'0' * 12:12s} {1.0:7.1f}{1.0:7.1f}"
    )
    assert len(line) == 160
    return line


def write_table(table_dir, table_name, lines):
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / f"{table_name}.data").write_text("".join(line + "\n" for line in lines))
    header = copy.deepcopy(hapi.HITRAN_DEFAULT_HEADER)
    header.update(table_name=table_name, number_of_rows=len(lines))
    (table_dir / f"{table_name}.header").write_text(json.dumps(header))


LINES_CO2 = [
    par_line(2, "1", 660.0, 3e-19, 0.0),
    par_line(2, "1", 667.4, 1e-18, 100.0),
    par_line(2, "2", 668.1, 5e-20, 600.0, delta_air=0.003),
    par_line(2, "1", 690.0, 2e-21, 3000.0),
    par_line(2, "1", 700.0, 1e-30, 0.0),  # below the intensity threshold
]


def cross_sections(tmp_path, table_name, lines, grid, temperature_k, pressure_pa, diluent):
    table_dir = tmp_path / table_name
    write_table(table_dir, table_name, lines)
    kwargs = dict(cache_dir=table_dir, diluent=diluent, available_broadener_keys=("air", "self"))
    with contextlib.redirect_stdout(io.StringIO()):
        ref = HapiLineProvider(table_name, **kwargs).cross_section_cm2_molecule(grid, temperature_k, pressure_pa)
    got = FastLineProvider(table_name, **kwargs).cross_section_cm2_molecule(grid, temperature_k, pressure_pa)
    return ref, got


@pytest.mark.parametrize(
    "temperature_k, pressure_pa, diluent",
    [(296.0, 1.0e5, {"air": 1.0}), (1200.0, 2.0e7, {"air": 0.7, "self": 0.3}), (150.0, 1.0, {"self": 1.0})],
)
def test_fast_engine_matches_hapi_voigt(tmp_path, temperature_k, pressure_pa, diluent):
    # 694.9 and 635.0 put grid points exactly at nu +/- 25 cm^-1 of the 660/669.9 lines.
    grid = np.concatenate([np.arange(630.0, 700.0, 0.05), [635.0, 694.9]])
    grid.sort()
    ref, got = cross_sections(tmp_path, "co2_lines_630_700", LINES_CO2, grid, temperature_k, pressure_pa, diluent)
    assert ref.max() > 0.0
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * ref.max())


def test_fast_engine_matches_hapi_h2o_pedestal(tmp_path):
    lines = [par_line(1, "1", 1000.0, 1e-20, 200.0), par_line(1, "2", 1003.0, 3e-21, 50.0, delta_air=0.01)]
    grid = np.arange(970.0, 1035.0, 0.1)
    ref, got = cross_sections(tmp_path, "h2o_lines_970_1035", lines, grid, 500.0, 1.0e6, {"air": 0.8, "self": 0.2})
    assert (ref == 0.0).any() and ref.max() > 0.0  # the pedestal clips the far wings to zero
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * ref.max())


def test_fast_engine_absorption_coefficient_matches_hapi(tmp_path):
    table_dir = tmp_path / "co2_lines_630_700"
    write_table(table_dir, "co2_lines_630_700", LINES_CO2)
    grid = np.arange(640.0, 690.0, 0.5)
    kwargs = dict(cache_dir=table_dir, diluent={"air": 1.0}, available_broadener_keys=("air", "self"))
    with contextlib.redirect_stdout(io.StringIO()):
        ref = HapiLineProvider("co2_lines_630_700", **kwargs).absorption_coefficient_cm1(grid, 400.0, 5.0e4)
    got = FastLineProvider("co2_lines_630_700", **kwargs).absorption_coefficient_cm1(grid, 400.0, 5.0e4)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * ref.max())


@pytest.mark.parametrize("subtract_wing_pedestal", [False, True])
def test_blocking_does_not_change_the_result(tmp_path, subtract_wing_pedestal):
    table_dir = tmp_path / "tab"
    write_table(table_dir, "tab", LINES_CO2)
    lines = load_line_table(table_dir, "tab")
    grid = np.arange(630.0, 700.0, 0.05)

    def compute(max_pairs):
        return voigt_cross_section(
            lines,
            grid,
            temperature_k=800.0,
            pressure_atm=1.0,
            diluent={"air": 1.0},
            intensity_threshold=1e-27,
            wing_cm1=25.0,
            partition_sum=hapi.PYTIPS,
            molecular_mass=hapi.molecularMass,
            faddeeva=hapi.hum1_wei,
            boltzmann_cgs=hapi.cBolts,
            speed_of_light_cgs=hapi.cc,
            subtract_wing_pedestal=subtract_wing_pedestal,
            max_pairs=max_pairs,
        )

    whole = compute(1 << 30)
    assert whole.max() > 0.0
    # 7 pairs is smaller than any single line's window, so every block holds one line.
    for max_pairs in (7, 1000, 1500):
        np.testing.assert_allclose(compute(max_pairs), whole, rtol=1e-12, atol=1e-14 * whole.max())


def test_load_line_table_caches_and_refreshes(tmp_path):
    table_dir = tmp_path / "tab"
    write_table(table_dir, "tab", LINES_CO2[:2])
    lines = load_line_table(table_dir, "tab")
    assert lines.shape == (2,)
    np.testing.assert_allclose(lines["nu"], [660.0, 667.4])
    np.testing.assert_allclose(lines["delta_air"], [-0.002, -0.002])
    assert (table_dir / ".tab.lines.npy").exists()

    write_table(table_dir, "tab", LINES_CO2[:3])
    reloaded = load_line_table(table_dir, "tab")
    assert reloaded.shape == (3,) and list(reloaded["local_iso_id"]) == [1, 1, 2]


def test_load_line_table_falls_back_when_records_are_not_161_bytes(tmp_path):
    table_dir = tmp_path / "tab"
    write_table(table_dir, "tab", LINES_CO2[:3])
    data_path = table_dir / "tab.data"
    data_path.write_bytes(data_path.read_bytes().replace(b"\n", b"\r\n"))
    lines = load_line_table(table_dir, "tab")
    np.testing.assert_allclose(lines["nu"], [660.0, 667.4, 668.1])


def test_fast_engine_reads_the_hitemp_parent_and_matches_hapi_on_the_child(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    lines = [
        par_line(2, "1", 667.4, 1e-18, 100.0),
        par_line(2, "1", 680.0, 1e-30, 4000.0),  # strong only when hot
        par_line(2, "1", 690.0, 1e-28, 0.0),  # below the threshold at 1500 K, above it when cold
    ]
    (hitemp_dir / "02_HITEMP2024.par").write_text("".join(line + "\n" for line in lines))
    band = SpectralBandConfig(name="b", wavenumber_min_cm1=650.0, wavenumber_max_cm1=700.0, resolution_cm1=0.1)
    common = dict(
        output_path=tmp_path / "o.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="CO2",
        line_source="hitemp",
        hitemp_dir=hitemp_dir,
        hitemp_temperatures_k=(1500.0,),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        hapi_config = SpectroscopyConfig(**common)
        hapi_db = download_hitran_lines(hapi_config, band)
        ref = build_line_provider(hapi_config, hapi_db).cross_section_cm2_molecule(band.grid(), 1500.0, 1.0e5)
    fast_config = SpectroscopyConfig(**common, line_engine="fast")
    fast_db = download_hitran_lines(fast_config, band)
    got = build_line_provider(fast_config, fast_db).cross_section_cm2_molecule(band.grid(), 1500.0, 1.0e5)

    assert hapi_db.table_name == "co2_lines_625_725_hitemp_1500_1500K"
    assert fast_db.table_name == "co2_lines_625_725_hitemp"
    assert (fast_db.cache_dir / ".co2_lines_625_725_hitemp.lines.npy").exists()
    assert load_line_table(fast_db.cache_dir, fast_db.table_name).shape[0] == 3
    assert json.loads((hapi_db.cache_dir / f"{hapi_db.table_name}.header").read_text())["number_of_rows"] == 2
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * ref.max())


def test_fast_engine_writes_no_child_table(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    (hitemp_dir / "02_HITEMP2024.par").write_text(par_line(2, "1", 667.4, 1e-18, 100.0) + "\n")
    band = SpectralBandConfig(name="b", wavenumber_min_cm1=650.0, wavenumber_max_cm1=700.0, resolution_cm1=0.1)
    config = SpectroscopyConfig(
        output_path=tmp_path / "o.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="CO2",
        line_source="hitemp",
        hitemp_dir=hitemp_dir,
        hitemp_temperatures_k=(800.0, 1200.0),
        line_engine="fast",
    )
    download_hitran_lines(config, band)
    assert sorted(path.name for path in (tmp_path / "hitran" / "hitemp").iterdir()) == ["co2_lines_625_725_hitemp"]


def test_load_line_table_rejects_nonstandard_layout(tmp_path):
    table_dir = tmp_path / "tab"
    write_table(table_dir, "tab", LINES_CO2[:1])
    header = json.loads((table_dir / "tab.header").read_text())
    header["extra"] = ["gamma_h2"]
    (table_dir / "tab.header").write_text(json.dumps(header))
    with pytest.raises(ValueError, match="standard 160-character"):
        load_line_table(table_dir, "tab")


def test_build_line_provider_and_line_list_follow_line_engine(tmp_path):
    table_dir = tmp_path / "co2_lines_630_700"
    write_table(table_dir, "co2_lines_630_700", LINES_CO2)
    line_db = LineDatabase(
        table_name="co2_lines_630_700",
        cache_dir=table_dir,
        wavenumber_min_cm1=630.0,
        wavenumber_max_cm1=700.0,
        available_broadener_keys=("air", "self"),
    )
    band = SpectralBandConfig(name="b", wavenumber_min_cm1=655.0, wavenumber_max_cm1=675.0, resolution_cm1=1.0)
    fast = SpectroscopyConfig(output_path=tmp_path / "o.nc", hitran_cache_dir=tmp_path, line_engine="fast")
    assert isinstance(build_line_provider(fast, line_db), FastLineProvider)
    line_list = load_hitran_line_list(fast, band, line_db)
    np.testing.assert_allclose(line_list.wavenumber_cm1, [660.0, 667.4, 668.1, 690.0])

    hapi_config = SpectroscopyConfig(output_path=tmp_path / "o.nc", hitran_cache_dir=tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        assert isinstance(build_line_provider(hapi_config, line_db), HapiLineProvider)
    with pytest.raises(ValueError, match="line_engine"):
        SpectroscopyConfig(output_path=tmp_path / "o.nc", hitran_cache_dir=tmp_path, line_engine="gpu")
