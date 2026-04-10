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

## Supported opacities

Pyharp has built-in functionalities that work with various opacity sources.
The following table summaries off-the-shelf opacities.

| Opacity Name          | Tested    | Peer Reviewed | References    |
|-----------------------|-----------|---------------|---------------|
| Premix H2 molecule    | YES       | NO            | [1]           |
| H2-He continuum       | YES       | YES           | [2]           |
| CO2 molecule          | YES       | NO            |               |
| CO2 continuum         | YES       | NO            |               |
| H2O molecule          | YES       | NO            |               |
| H2O continuum         | YES       | NO            |               |
| N2 molecule           | YES       | NO            |               |
| N2 continuum          | YES       | NO            |               |
| Grey (user implement) | YES       | NO            |               |
| *(More coming)*       | ...       | ...           |               |

## Supported radiative transfer solvers

You can choose the backend radiative transfer solver to use by Pyharp.
Here are the available options:

| Radiative Transfer Solver | Tested    | Peer Reviewed | References |
|---------------------------|-----------|---------------|------------|
| DISORT                    | YES       | YES           | [1]        |
| Two-steam (Toon-McKay)    | NO        | NO            |            |

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

The top-level spectroscopy CLI computes one pressure-temperature state:

```bash
pyharp-spectra spectrum --species H2O --temperature-k 300 --pressure-bar 1
pyharp-spectra transmittance --species H2O --path-length-m 1
```

Additional installed plotting helpers:

- `pyharp-plot-cia-binary`
- `pyharp-plot-cia-attenuation`
- `pyharp-plot-cia-transmission`
- `pyharp-plot-molecule-xsection`
- `pyharp-plot-molecule-attenuation`
- `pyharp-plot-molecule-transmission`
- `pyharp-plot-molecule-plus-cia-attenuation`
- `pyharp-plot-molecule-plus-cia-transmission`
- `pyharp-plot-molecule-lines`
- `pyharp-plot-molecule-overview`
- `pyharp-gen-molecule-overview`
- `pyharp-gen-atm-overview`

Examples:

```bash
pyharp-plot-cia-binary --pair H2-H2 --temperature-k 300 --wn-min 20 --wn-max 10000
pyharp-plot-molecule-xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-min 20 --wn-max 2500
pyharp-gen-atm-overview --composition H2O:0.1,H2:0.9 --wn-range=25,2500 --temperature-k 300 --pressure-bar 1
```

Supported built-in HITRAN line species are `CH4`, `CO2`, `H2`, `H2O`, and
`N2`. Built-in CIA pair resolution includes the self pairs for these species
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
