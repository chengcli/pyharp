from pathlib import Path

from pyharp.spectra.mt_ckd_h2o import default_mt_ckd_h2o_data_path


def test_default_mt_ckd_h2o_data_path_uses_local_external_directory(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)

    assert default_mt_ckd_h2o_data_path() == (
        repo_root / "external" / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"
    )


def test_default_mt_ckd_h2o_data_path_searches_parent_directories(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / "python" / "spectra")

    assert default_mt_ckd_h2o_data_path() == (
        repo_root / "external" / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"
    )
