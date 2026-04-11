# Pyharp: Python-first High-performance Atmosphere Radiation Package

[![build](https://github.com/chengcli/pyharp/actions/workflows/ci.yml/badge.svg)](https://github.com/chengcli/pyharp/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)

Pyharp is the one-stop tool for calculating the radiation flux of planetary atmospheres,
from terrestrial to giant planets.
Detailed documentation and examples are available at [https://pyharp.readthedocs.io](https://pyharp.readthedocs.io).

## Installation

Pyharp can be installed via ``pip``:

```bash
pip install pyharp
```

We support Linux and Mac operation systems with Python version 3.10+.

---

## Spectroscopy workflow

Pyharp also includes `pyharp.spectra`, a pure-Python spectroscopy workflow for
HITRAN line data, HITRAN CIA data, single-state absorption spectra,
transmittance products, diagnostic plots, and gas-mixture overview figures.
The former standalone `spectra` library now lives under the `pyharp.spectra`
namespace.

The main library entry points are available from `pyharp.spectra`:

```python
from pathlib import Path

from pyharp.spectra import (
    AbsorptionSpectrum,
    SpectroscopyConfig,
    SpectralBandConfig,
    compute_absorption_spectrum,
)

band = SpectralBandConfig(
    name="single_state",
    wavenumber_min_cm1=20.0,
    wavenumber_max_cm1=2500.0,
    resolution_cm1=1.0,
)
config = SpectroscopyConfig(
    output_path=Path("output/h2o_absorption_300K_1bar.nc"),
    hitran_cache_dir=Path("hitran"),
    species_name="H2O",
)
spectrum: AbsorptionSpectrum = compute_absorption_spectrum(
    config=config,
    band=band,
    temperature_k=300.0,
    pressure_pa=1.0e5,
)
```

`SpectroscopyConfig.hitran_cache_dir` stores downloaded HITRAN line and CIA
files. `SpectroscopyConfig.output_path` controls where NetCDF products are
written by CLI helpers and is also used to create parent output directories.

H2O continuum calculations use the MT_CKD_H2O coefficient file at
`external/MT_CKD_H2O/data/absco-ref_wv-mt-ckd.nc` relative to the Pyharp
repository root. It is tracked as a Git submodule. Clone Pyharp with
submodules to fetch it immediately:

```bash
git clone --recurse-submodules https://github.com/chengcli/pyharp
```

If you already cloned Pyharp, initialize the submodule from the repository root:

```bash
git submodule update --init --recursive external/MT_CKD_H2O
```

If you are using command line argument outside of pyharp, clone MT_CKD directly:

```bash
mkdir -p external && cd external && git clone https://github.com/AER-RC/MT_CKD_H2O
```

The `pyharp-dump` CLI writes NetCDF spectroscopy products for single species,
CIA pairs, and gas mixtures:

```bash
pyharp-dump xsection --species H2O --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
pyharp-dump xsection --pair H2-H2 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
pyharp-dump transmission --species H2O --path-length-m 1 --wn-range=20,2500
```

Use repeated `--wn-range=min,max` values to store multiple bands in one file.
Across `pyharp.spectra`, wavenumber ranges are interpreted as
lower-inclusive and upper-exclusive: `--wn-range=20,22` with `1 cm^-1`
resolution samples `20` and `21`, not `22`.
See the [pyharp-dump CLI documentation](https://pyharp.readthedocs.io/en/latest/dump_cli.html)
for the full command reference, output naming conventions, and NetCDF schema.

Plotting diagnostics are available from one entry point, `pyharp-plot`. It
provides CIA binary coefficient, molecular cross-section, attenuation,
transmission, and overview plot subcommands:

```bash
pyharp-plot binary --pair H2-H2 --temperature-k 300 --wn-range=20,10000
pyharp-plot xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
pyharp-plot attenuation --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
pyharp-plot transmission --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=25,2500
pyharp-plot overview --species H2O CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500 --wn-range=2500,10000
```

Use `--pair` for CIA pairs, `--species` for molecules, and `--composition`
for gas mixtures such as `H2O:0.1,H2:0.9`. All plot commands accept
`--wn-range=min,max`; `overview` accepts multiple `--wn-range` values for
multi-page PDFs. These ranges are lower-inclusive and upper-exclusive. Use
`--figure` to choose the output path. Without `--figure`,
plots are written under `output/` with names derived from the target, plot
type, temperature, pressure, and wavenumber range.

Molecular line calculations also accept `--broadening-composition BROADENER:FRACTION,...`,
for example `air:0.8,self:0.2` or `H2:0.85,He:0.15`.
If a requested foreign broadener is unavailable in the HITRAN table for the
active absorber, Pyharp falls back to `air` for that fraction.

See the [pyharp-plot CLI documentation](https://pyharp.readthedocs.io/en/latest/plot_cli.html)
for command-specific options and more examples.

Supported built-in HITRAN line species are `CH4`, `CO2`, `H2`, `H2O`, `H2S`,
`N2`, and `NH3`. Built-in CIA pair resolution includes the self pairs for these species
where HITRAN CIA data is configured, plus `CO2-CH4`, `CO2-H2`, `H2-He`, and
`N2-CH4`.

`pyharp.spectra` does not provide a fixed reference-column radiative-transfer
experiment. Use the core Pyharp radiative-transfer APIs for column RT
calculations, and use `pyharp.spectra` for spectroscopy inputs, diagnostics,
single-state products, and overview plots.

---

## Development

If you want to further develop Pyharp, you will need to install it locally, which allows you
to modify the source code and test.
Open a Linux or Mac terminal and clone this repo using the following command:

```bash
git clone https://github.com/chengcli/pyharp
```

This will copy all source files into your local computer. You will need to install a few
system libraries before installing Pyharp. All following instructions are executed under
the `pyharp/` directory.

### System required for building locally
- Python 3.10+
- Linux or macOS
- netCDF
- python virtual environment (venv)

#### MacOS installation
```bash
brew install netcdf
```

#### RedHat installation
```bash
sudo yum install netcdf
```

#### Ubuntu installation
```bash
sudo apt-get install libnetcdf-dev
```

### Build C++ library
After you completed the installation steps, you can build the pyharp library.
We will build the package in-place, meaning that the build (binary files) are
located under `pyharp/build/bin`. To do so, make a new directory named `build`
```
mkdir build
```
All build files will be generated and placed under this directory. It is completely safe
to delete the whole directory if you want another build. `cd` to build and `cmake`

```
cd build
cmake ..
```
This command tells the cmake command to look for `CMakeFiles.txt` in the parent directory,
and start configuring the compile environment. Then compile the code by
```
make -j4
```
This comman will use 4 cores to compile the code in parallel. Once complete, all
executable files will be placed in `build/bin`.

### Build python package locally (dev mode)
The python library can be installed by running the following command in the root
directory:
```bash
pip install -e .
```

### Test the installation
To test the installation, import pyharp in a python shell:
```python
import pyharp
```
The build is successful if you do not see any error messages.

---

## Contributing
Contributions are welcome!
Please open an issue or PR if you’d like to:
- Find a bug
- Suggest new functions
- Add examples
- Improve documentation
- Expand test coverage

---

## Contact
Maintained by @chengcli — feel free to reach out with ideas, feedback, or collaboration proposals.
