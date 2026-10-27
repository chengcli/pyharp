"""Every test in tests/ is built and run: C++ files registered with CTest, Python files run by CTest, all
spectra files run by CI, and every C++ test asserts something."""

import re
from pathlib import Path

# CMake copies this file into the build tree with @ONLY substitution; there it reads the source tree
_CONFIGURED = "@CMAKE_CURRENT_SOURCE_DIR@"
SOURCE_TESTS = Path(_CONFIGURED) if not _CONFIGURED.startswith("@") else Path(__file__).resolve().parent
ROOT = SOURCE_TESTS.parent


def cmake_text():
    return (SOURCE_TESTS / "CMakeLists.txt").read_text()


def active_lines(text):
    return [line.split("#", 1)[0] for line in text.splitlines()]


def test_every_cpp_test_is_registered():
    registered = set(re.findall(r"setup_test\((\w+)\)", "\n".join(active_lines(cmake_text()))))
    files = {p.stem for p in SOURCE_TESTS.glob("test_*.cpp")}
    assert files - registered == set(), f"unregistered C++ tests: {sorted(files - registered)}"


def test_python_tests_are_run_by_ctest():
    code = "\n".join(active_lines(cmake_text()))
    assert re.search(r"add_test\(\s*NAME\s+\$\{name\}", code), "the Python tests are copied but not added to CTest"


def test_ci_runs_every_spectra_test():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    runs_dir = re.search(r"pytest[^\n]*\btests/spectra/?(\s|$)", ci)
    listed = set(re.findall(r"tests/spectra/(test_\w+\.py)", ci))
    files = {p.name for p in (SOURCE_TESTS / "spectra").glob("test_*.py")}
    assert runs_dir or files <= listed, f"spectra tests CI does not run: {sorted(files - listed)}"


ASSERT_MACRO = re.compile(r"\b(EXPECT|ASSERT)_[A-Z_]+\s*\(")
# the tests the defect register names as asserting nothing: every TEST in them must assert a value
STRICT = ["test_bbflux.cpp", "test_hypsometric.cpp", "test_composition.cpp"]


def test_every_cpp_test_file_asserts_something():
    silent = sorted(p.name for p in SOURCE_TESTS.glob("test_*.cpp") if not ASSERT_MACRO.search(p.read_text()))
    assert silent == [], f"C++ test files without a single assertion: {silent}"


def test_the_named_tests_assert_in_every_test():
    silent = []
    for name in STRICT:
        bodies = re.split(r"\bTEST(?:_F)?\s*\(", (SOURCE_TESTS / name).read_text())[1:]
        if not bodies or any(not ASSERT_MACRO.search(b) for b in bodies):
            silent.append(name)
    assert silent == [], f"tests with a TEST that asserts nothing (or no TEST at all): {silent}"
