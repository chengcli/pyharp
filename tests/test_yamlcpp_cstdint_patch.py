"""The yaml-cpp patch gives src/emitterutils.cpp its <cstdint>.

yaml-cpp 0.8.0 declares EncodeUTF16SurrogatePair as std::pair<uint16_t, uint16_t> and uses uint32_t beneath it,
but src/emitterutils.cpp includes only <algorithm>, <iomanip> and <sstream> before its own headers. Whether
uint16_t is declared anyway depends on what those headers pull in: libstdc++ 13 still builds the file, libstdc++ 15
(GCC 15, the default of the ubuntu-26.04 runner) stops with

  src/emitterutils.cpp:221:11: error: 'uint16_t' was not declared in this scope

cmake/yamlpp.cmake applies patches/01.yamlpp.patch to the fetched yaml-cpp, so that patch must add the include.

  patch    the unified diff, parsed per file, adds `#include <cstdint>` to src/emitterutils.cpp (compiler
           independent; this is the check that fails without the fix)
  applies  the patch applies to the pristine yaml-cpp 0.8.0 files (git apply --check)
  compile  the real src/emitterutils.cpp as the patch leaves it compiles with the build's compiler; whether this
           toolchain also rejects the file without the include is printed, not asserted

The yaml-cpp source is the build's fetched copy (or $YAML_CPP_SRC); without it the last two are skipped.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SRC = "@CMAKE_CURRENT_SOURCE_DIR@"
_BIN = "@CMAKE_BINARY_DIR@"
_CXX = "@CMAKE_CXX_COMPILER@"
TESTS = Path(_SRC) if not _SRC.startswith("@") else Path(__file__).resolve().parent
PATCH = TESTS.parent / "patches" / "01.yamlpp.patch"
TARGET = "src/emitterutils.cpp"
INCLUDE = "#include <cstdint>"


def added_lines(patch_text, target):
    """Lines the unified diff adds to `target`."""
    added, in_target = [], False
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            in_target = line.endswith(f"b/{target}")
        elif line.startswith("+++ "):
            in_target = line.split("\t")[0].endswith(f"/{target}")
        elif in_target and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
    return added


def yaml_cpp_source():
    for candidate in (os.environ.get("YAML_CPP_SRC"),
                      None if _BIN.startswith("@") else str(Path(_BIN) / "_deps" / "yaml-cpp-src")):
        if candidate and (Path(candidate) / TARGET).is_file():
            return Path(candidate)
    return None


def pristine(source, path):
    """The file as yaml-cpp 0.8.0 ships it (the fetched checkout carries the patch in its work tree)."""
    return subprocess.run(["git", "-C", str(source), "show", f"HEAD:{path}"], capture_output=True, text=True,
                          check=True).stdout


def compiler():
    cxx = os.environ.get("CXX") or (None if _CXX.startswith("@") else _CXX) or "g++"
    return shutil.which(cxx)


def test_patch_adds_cstdint_to_emitterutils():
    added = added_lines(PATCH.read_text(), TARGET)
    assert INCLUDE in added, f"{PATCH.name} adds {added or 'nothing'} to {TARGET}, not '{INCLUDE}'"


def test_patch_applies_to_yaml_cpp_0_8_0(tmp_path):
    source = yaml_cpp_source()
    if source is None:
        pytest.skip("yaml-cpp source not found (build tree or $YAML_CPP_SRC)")
    for path in ("CMakeLists.txt", TARGET):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(pristine(source, path))
    r = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_patched_emitterutils_compiles(tmp_path):
    source = yaml_cpp_source()
    cxx = compiler()
    if source is None or cxx is None:
        pytest.skip("yaml-cpp source or C++ compiler not found")
    base = "".join(l for l in pristine(source, TARGET).splitlines(keepends=True) if l.strip() != INCLUDE)
    as_patched = (INCLUDE + "\n" + base) if INCLUDE in added_lines(PATCH.read_text(), TARGET) else base

    def build(text, name):
        unit = tmp_path / f"{name}.cpp"
        unit.write_text(text)
        return subprocess.run([cxx, "-std=c++11", f"-I{source / 'include'}", f"-I{source / 'src'}", "-c",
                               str(unit), "-o", str(tmp_path / f"{name}.o")], capture_output=True, text=True)

    version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    bare = build(base, "bare")
    print(f"{version}: {TARGET} without '{INCLUDE}' "
          f"{'compiles (the defect does not show on this toolchain)' if bare.returncode == 0 else 'fails (the defect reproduces)'}")
    r = build(as_patched, "as_patched")
    first = next((l for l in r.stderr.splitlines() if "uint16_t" in l), r.stderr[:200])
    assert r.returncode == 0, f"{version} cannot build {TARGET} as the patch leaves it: {first}"
