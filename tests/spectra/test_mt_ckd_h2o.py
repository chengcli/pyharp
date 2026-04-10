from pathlib import Path

from pyharp.spectra.mt_ckd_h2o import default_mt_ckd_h2o_data_path


def test_default_mt_ckd_h2o_data_path_uses_repo_external_directory() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert default_mt_ckd_h2o_data_path() == (
        repo_root / "external" / "MT_CKD_H2O" / "data" / "absco-ref_wv-mt-ckd.nc"
    )
