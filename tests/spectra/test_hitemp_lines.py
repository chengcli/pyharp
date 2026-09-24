import bz2
import json
import zipfile

import numpy as np
import pytest

import argparse

from pyharp.spectra.atm_overview import line_source_options
from pyharp.spectra.config import SpectralBandConfig, SpectroscopyConfig
from pyharp.spectra.hitemp_lines import (
    _parse_par_float,
    derive_hitemp_table,
    filter_temperatures_k,
    find_hitemp_files,
    has_hitemp_files,
    max_line_strength,
    prepare_hitemp_table,
)
from pyharp.spectra.hitran_molecule_utils import build_line_provider, download_hitran_lines


DEFAULT_HEADER = {"table_type": "column-fixed", "number_of_rows": -1, "order": [], "format": {}, "default": {}}


def par_line(molec, iso, nu, sw, elower, gamma_air=0.07, gamma_self=0.3):
    gamma_air_field = f"{gamma_air:.4f}"[1:]  # HITRAN writes ".0700"
    line = (
        f"{molec:2d}{iso:1s}{nu:12.6f}{sw:10.3E}{1.0:10.3E}{gamma_air_field}{gamma_self:5.3f}"
        f"{elower:10.4f}{0.70:4.2f}{0.0:8.6f}" + " " * 60 + f"{'000000':6s}{'0' * 12:12s} {1.0:7.1f}{1.0:7.1f}"
    )
    assert len(line) == 160
    return line


def unit_partition_sum(molecule_id, iso, temperature):
    return 1.0


def write_par(path, lines):
    path.write_text("".join(line + "\n" for line in lines))


def prepare(tmp_path, hitemp_dir, **overrides):
    kwargs = dict(
        hitemp_dir=hitemp_dir,
        table_dir=tmp_path / "cache",
        table_name="tab",
        molecule_id=1,
        local_iso_ids=(1, 10),
        wavenumber_min_cm1=100.0,
        wavenumber_max_cm1=200.0,
        temperature_range_k=(296.0, 296.0),
        min_line_strength=1.0e-27,
        default_header=DEFAULT_HEADER,
        partition_sum=unit_partition_sum,
    )
    kwargs.update(overrides)
    return prepare_hitemp_table(**kwargs)


def table_lines(tmp_path):
    return (tmp_path / "cache" / "tab.data").read_text().splitlines()


def test_find_hitemp_files_uses_newest_edition_sorted_by_range(tmp_path):
    for name in [
        "02_00100-00200_HITEMP2010.zip",
        "02_00000-00100_HITEMP2010.zip",
        "02_HITEMP2024.par.bz2",
        "01_00000-00050_HITEMP2010.zip",
        "02_HITEMP2024.par.bz2.crdownload",
    ]:
        (tmp_path / name).write_bytes(b"")
    (co2,) = find_hitemp_files(tmp_path, 2)
    assert co2.path.name == "02_HITEMP2024.par.bz2"
    assert (co2.wavenumber_min_cm1, co2.wavenumber_max_cm1) == (0.0, float("inf"))

    (tmp_path / "02_HITEMP2024.par.bz2").unlink()
    names = [item.path.name for item in find_hitemp_files(tmp_path, 2)]
    assert names == ["02_00000-00100_HITEMP2010.zip", "02_00100-00200_HITEMP2010.zip"]

    with pytest.raises(FileNotFoundError):
        find_hitemp_files(tmp_path, 6)


def test_prepare_filters_range_isotopologue_and_molecule(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    lines = [
        par_line(1, "1", 50.0, 1e-20, 0.0),  # below range
        par_line(1, "1", 150.0, 1e-20, 0.0),
        par_line(1, "2", 160.0, 1e-20, 0.0),  # isotopologue not requested
        par_line(1, "0", 170.0, 1e-20, 0.0),  # isotopologue 10
        par_line(2, "1", 180.0, 1e-20, 0.0),  # other molecule
        par_line(1, "1", 250.0, 1e-20, 0.0),  # above range
    ]
    write_par(hitemp_dir / "01_HITEMP2010.par", lines)

    assert prepare(tmp_path, hitemp_dir) == 2
    assert table_lines(tmp_path) == [lines[1], lines[3]]
    header = json.loads((tmp_path / "cache" / "tab.header").read_text())
    assert header["number_of_rows"] == 2
    assert header["table_name"] == "tab"


def test_prepare_keeps_hot_lines_only_when_the_range_is_hot(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    cold = par_line(1, "1", 150.0, 1e-20, 0.0)
    hot_band = par_line(1, "1", 160.0, 1e-30, 5000.0)
    write_par(hitemp_dir / "01_HITEMP2010.par", [cold, hot_band])

    assert prepare(tmp_path, hitemp_dir) == 1
    assert prepare(tmp_path, hitemp_dir, temperature_range_k=(296.0, 1500.0)) == 2
    assert table_lines(tmp_path) == [cold, hot_band]


def test_derive_rescreens_parent_over_a_narrower_range(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    cold = par_line(1, "1", 150.0, 1e-20, 0.0)
    hot_band = par_line(1, "1", 160.0, 1e-30, 5000.0)
    write_par(hitemp_dir / "01_HITEMP2010.par", [cold, hot_band])
    assert prepare(tmp_path, hitemp_dir, temperature_range_k=(10.0, 2000.0)) == 2

    def derive(temperatures_k):
        return derive_hitemp_table(
            parent_dir=tmp_path / "cache",
            parent_name="tab",
            table_dir=tmp_path / "child",
            table_name="child",
            temperatures_k=temperatures_k,
            partition_sum=unit_partition_sum,
        )

    assert derive((250.0, 350.0)) == 1
    assert (tmp_path / "child" / "child.data").read_text().splitlines() == [cold]
    assert derive((1300.0, 1700.0)) == 2
    with pytest.raises(ValueError, match="outside the parent"):
        derive((1500.0, 2500.0))

    # A rebuilt parent invalidates the child built from it.
    child_header = tmp_path / "child" / "child.header"
    before = json.loads(child_header.read_text())["pyharp_hitemp"]
    prepare(tmp_path, hitemp_dir, temperature_range_k=(10.0, 3000.0))
    derive((1300.0, 1700.0))
    assert json.loads(child_header.read_text())["pyharp_hitemp"] != before


def test_filter_temperatures_include_the_run_temperatures():
    temperatures = filter_temperatures_k((800.0, 1000.0, 1200.0))
    assert {800.0, 1000.0, 1200.0} <= set(temperatures)
    assert len(temperatures) > 3
    assert list(filter_temperatures_k((300.0,))) == [300.0]


def test_prepare_reuses_a_parent_whose_range_covers_the_request(tmp_path):
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    write_par(hitemp_dir / "01_HITEMP2010.par", [par_line(1, "1", 150.0, 1e-20, 0.0)])
    prepare(tmp_path, hitemp_dir, temperature_range_k=(10.0, 2500.0))
    data_path = tmp_path / "cache" / "tab.data"
    mtime = data_path.stat().st_mtime_ns
    assert prepare(tmp_path, hitemp_dir, temperature_range_k=(10.0, 2000.0)) == 1
    assert data_path.stat().st_mtime_ns == mtime
    prepare(tmp_path, hitemp_dir, temperature_range_k=(10.0, 3000.0))
    assert data_path.stat().st_mtime_ns != mtime


def test_max_line_strength_matches_hitran_temperature_scaling():
    nu = np.asarray([1000.0])
    sw = np.asarray([1e-20])
    elower = np.asarray([2000.0])
    strength = max_line_strength(
        nu=nu,
        sw=sw,
        elower=elower,
        local_iso_id=np.asarray([1]),
        molecule_id=1,
        temperatures_k=[1000.0],
        partition_sum=lambda m, i, t: t / 296.0,
    )
    c2 = 1.4388028496642257
    expected = (
        1e-20
        * (296.0 / 1000.0)
        * np.exp(-c2 * 2000.0 / 1000.0)
        / np.exp(-c2 * 2000.0 / 296.0)
        * (1 - np.exp(-c2 * 1000.0 / 1000.0))
        / (1 - np.exp(-c2 * 1000.0 / 296.0))
    )
    np.testing.assert_allclose(strength, [expected], rtol=1e-12)


def test_prepare_reads_zip_and_bz2_and_reuses_cache(tmp_path):
    lines = [par_line(1, "1", 120.0, 1e-20, 0.0), par_line(1, "1", 180.0, 1e-20, 0.0)]
    zip_dir = tmp_path / "zip"
    zip_dir.mkdir()
    with zipfile.ZipFile(zip_dir / "01_00100-00150_HITEMP2010.zip", "w") as archive:
        archive.writestr("01_100-150_HITEMP2010.par", lines[0] + "\n")
    with zipfile.ZipFile(zip_dir / "01_00150-00200_HITEMP2010.zip", "w") as archive:
        archive.writestr("01_150-200_HITEMP2010.par", lines[1] + "\r\n")
    assert prepare(tmp_path, zip_dir) == 2
    assert table_lines(tmp_path) == lines

    bz2_dir = tmp_path / "bz2"
    bz2_dir.mkdir()
    (bz2_dir / "01_HITEMP2024.par.bz2").write_bytes(bz2.compress("".join(l + "\n" for l in lines).encode()))
    assert prepare(tmp_path, bz2_dir) == 2
    data_path = tmp_path / "cache" / "tab.data"
    mtime = data_path.stat().st_mtime_ns
    assert prepare(tmp_path, bz2_dir) == 2
    assert data_path.stat().st_mtime_ns == mtime
    assert prepare(tmp_path, bz2_dir, refresh=True) == 2
    assert data_path.stat().st_mtime_ns != mtime


def test_parse_par_float_handles_exponents_without_e():
    assert _parse_par_float(b" 1.000-100") == pytest.approx(1.0e-100)
    assert _parse_par_float(b"2.5D-03") == pytest.approx(2.5e-3)


def test_config_names_hitemp_tables_by_temperature_range(tmp_path):
    band = SpectralBandConfig(name="b", wavenumber_min_cm1=100.0, wavenumber_max_cm1=200.0, resolution_cm1=1.0)
    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path,
        species_name="H2O",
        line_source="hitemp",
        hitemp_dir=tmp_path,
        hitemp_temperatures_k=(1500.0, 300.0, 900.0),
    )
    # H2O tables must keep the h2o_ prefix: the line provider keys the MT_CKD pedestal on it.
    assert config.resolved_line_table_name(band) == "h2o_lines_100_200_hitemp_300_1500K"
    assert config.resolved_hitemp_parent_table_name(band) == "h2o_lines_100_200_hitemp"
    assert config.resolved_hitemp_parent_temperature_range_k() == (10.0, 2000.0)
    hot = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path,
        line_source="hitemp",
        hitemp_dir=tmp_path,
        hitemp_temperatures_k=(2000.0, 2500.0),
    )
    assert hot.resolved_hitemp_parent_temperature_range_k() == (10.0, 2500.0)
    with pytest.raises(ValueError, match="hitemp_dir"):
        SpectroscopyConfig(output_path=tmp_path / "o.nc", hitran_cache_dir=tmp_path, line_source="hitemp")
    with pytest.raises(ValueError, match="line_source"):
        SpectroscopyConfig(output_path=tmp_path / "o.nc", hitran_cache_dir=tmp_path, line_source="exomol")


def test_download_hitran_lines_builds_hapi_table_from_hitemp(tmp_path):
    pytest.importorskip("hapi")
    hitemp_dir = tmp_path / "hitemp"
    hitemp_dir.mkdir()
    write_par(
        hitemp_dir / "02_HITEMP2024.par",
        [par_line(2, "1", 667.0, 1e-19, 0.0), par_line(2, "1", 700.0, 1e-19, 0.0), par_line(2, "1", 900.0, 1e-19, 0.0)],
    )
    band = SpectralBandConfig(name="b", wavenumber_min_cm1=650.0, wavenumber_max_cm1=720.0, resolution_cm1=0.5)
    config = SpectroscopyConfig(
        output_path=tmp_path / "out.nc",
        hitran_cache_dir=tmp_path / "hitran",
        species_name="CO2",
        line_source="hitemp",
        hitemp_dir=hitemp_dir,
        hitemp_temperatures_k=(300.0, 1500.0),
    )
    line_db = download_hitran_lines(config, band)
    assert line_db.cache_dir == tmp_path / "hitran" / "hitemp" / line_db.table_name
    assert (line_db.wavenumber_min_cm1, line_db.wavenumber_max_cm1) == (625.0, 745.0)

    provider = build_line_provider(config, line_db)
    sigma = provider.cross_section_cm2_molecule(band.grid(), temperature_k=1500.0, pressure_pa=1.0e5)
    grid = band.grid()
    assert sigma[np.argmin(abs(grid - 667.0))] > 100 * sigma[np.argmin(abs(grid - 685.0))]
    assert sigma[np.argmin(abs(grid - 700.0))] > 0.0


def test_line_source_options_use_hitemp_only_for_species_with_files(tmp_path):
    (tmp_path / "01_00000-00050_HITEMP2010.zip").write_bytes(b"")
    (tmp_path / "02_HITEMP2024.par.bz2").write_bytes(b"")
    assert has_hitemp_files(tmp_path, 1) and not has_hitemp_files(tmp_path, 11)

    args = argparse.Namespace(hitemp_dir=tmp_path, refresh_hitran=True, line_temperatures_k=(25.0, 225.0, 425.0))
    assert line_source_options(args, "H2O", temperature_k=225.0) == {
        "line_engine": "hapi",
        "line_source": "hitemp",
        "hitemp_dir": tmp_path,
        "hitemp_temperatures_k": (25.0, 225.0, 425.0),
        "refresh_hitran": True,
    }
    assert line_source_options(args, "NH3", temperature_k=225.0) == {"refresh_hitran": True, "line_engine": "hapi"}

    args.hitemp_tables_ready = True
    assert line_source_options(args, "CO2", temperature_k=225.0)["refresh_hitran"] is False
    assert line_source_options(args, "NH3", temperature_k=225.0) == {"refresh_hitran": True, "line_engine": "hapi"}

    single = argparse.Namespace(hitemp_dir=tmp_path, refresh_hitran=False)
    assert line_source_options(single, "CO2", temperature_k=300.0)["hitemp_temperatures_k"] == (300.0,)
    assert line_source_options(argparse.Namespace(refresh_hitran=False), "CO2", temperature_k=300.0) == {
        "refresh_hitran": False,
        "line_engine": "hapi",
    }
    fast = argparse.Namespace(hitemp_dir=tmp_path, refresh_hitran=False, line_engine="fast")
    assert line_source_options(fast, "CO2", temperature_k=300.0)["line_engine"] == "fast"
    assert line_source_options(fast, "NH3", temperature_k=300.0)["line_engine"] == "fast"


def test_dump_cli_rejects_a_hitemp_dir_without_hitemp_files(tmp_path):
    from pyharp.spectra.dump_cli import _validate_hitemp_dir, build_parser

    parser = build_parser()
    (tmp_path / "empty").mkdir()
    args = parser.parse_args(["xsection", "--species", "H2O", "--wn-range=1000,1100", "--hitemp-dir", str(tmp_path / "empty")])
    with pytest.raises(SystemExit):
        _validate_hitemp_dir(args, parser)

    (tmp_path / "empty" / "01_00000-00050_HITEMP2010.zip").write_bytes(b"")
    _validate_hitemp_dir(args, parser)
