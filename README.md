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

We support Linux and Mac operation systems with Python version 3.9+.

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
- Python 3.9+
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
