# Pyharp: High-performance Atmosphere Radiation Package in Python

[![build](https://github.com/chengcli/pyharp/actions/workflows/ci.yml/badge.svg)](https://github.com/chengcli/pyharp/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)

## Installation

Pyharp can be installed on either a Linux distribution or on MacOS via ``pip``:

```bash
pip install pyharp
```

You can also install it locally, which allows you to modify the source code and test it locally.
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
This comman will use 4 cores to compile the code in parallel. Once complete, all executable
files will be placed in `build/bin`.

### Build python package locally (dev mode)
The python library can be installed by running the following command:
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

## Supported Opacities
| Opacity Name  | Tested    | Peer Reviewed | Reference |
|---------------|-----------|---------------|-----------|
| rfm-lbl       | NO        | |           |
| rfm-ck        | YES       | |           |
| helios        | NO       | |           |
| simple-grey   | NO       | |           |
| freedman-mean | NO       | |           |
| jup-gas-vis   | NO       | |           |
| jup-gas-ir    | NO       | |           |
| s8-fuller     | YES       | |           |
| h2so4-simple  | YES       | |           |
| *(More coming)*| ...      | ...           |           |

---

## Contributing
Contributions are welcome!
Please open an issue or PR if you’d like to:
- Add new opacity sources
- Add command line tools or add GUI
- Expand test coverage

---

## Contact
Maintained by @chengcli — feel free to reach out with ideas, feedback, or collaboration proposals.
