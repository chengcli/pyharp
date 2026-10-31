"""CMakeLists.txt asks nvcc for sm_120 (RTX 50xx) only from CUDA 12.8, the first nvcc that accepts it; 12.0-12.7
reject it ("Unsupported gpu architecture 'compute_120'"). The architecture block of CMakeLists.txt is run under
`cmake -P` with each toolkit version."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_CONFIGURED = "@CMAKE_CURRENT_SOURCE_DIR@"
ROOT = (Path(_CONFIGURED) if not _CONFIGURED.startswith("@") else Path(__file__).resolve().parent).parent

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")


def arch_block():
    text = (ROOT / "CMakeLists.txt").read_text()
    start = text.index("if (CUDAToolkit_FOUND)")
    end = text.index("set(CMAKE_CUDA_ARCHITECTURES ${_cuda_arch_list})", start)
    return text[start:end] + "message(STATUS \"ARCHS=${_cuda_arch_list}\")\nendif()\n"


def architectures(tmp_path, toolkit=None, torch_cuda=None):
    lines = ["set(CUDAToolkit_FOUND ON)"]
    if toolkit:
        lines.append(f'set(CUDAToolkit_VERSION "{toolkit}")')
    if torch_cuda:
        lines.append(f'set(TORCH_CUDA_VERSION "{torch_cuda}")')
    script = tmp_path / "archs.cmake"
    script.write_text("\n".join(lines) + "\n" + arch_block())
    out = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, check=True)
    return [int(a) for a in re.search(r"ARCHS=([0-9;]*)", out.stdout + out.stderr).group(1).split(";") if a]


@pytest.mark.parametrize("toolkit, has_120", [("12.6", False), ("12.7", False), ("12.8", True), ("12.9", True),
                                              ("13.1", True)])
def test_sm120_only_from_cuda_12_8(tmp_path, toolkit, has_120):
    archs = architectures(tmp_path, toolkit=toolkit)
    assert (120 in archs) == has_120, f"CUDA {toolkit}: {archs}"
    assert 90 in archs and 75 in archs


@pytest.mark.parametrize("torch_cuda, has_120", [("12.6", False), ("12.8", True)])
def test_sm120_from_the_torch_cuda_version_as_well(tmp_path, torch_cuda, has_120):
    archs = architectures(tmp_path, torch_cuda=torch_cuda)
    assert (120 in archs) == has_120, f"torch CUDA {torch_cuda}: {archs}"


def test_cuda_13_drops_the_old_architectures(tmp_path):
    archs = architectures(tmp_path, toolkit="13.1")
    assert not {60, 61, 70} & set(archs)
