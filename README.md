# Pyharp: High-performance Atmosphere Radiation Package in Python

[![build](https://github.com/chengcli/pyharp/actions/workflows/main.yml/badge.svg)](https://github.com/chengcli/canoe/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)

## Install system libraries and toolchain

Pyharp can be installed on either a Linux distribution or on MacOS. Open a Linux or Mac terminal,
you can clone this repo using the following command:
```
git clone https://github.com/chengcli/pyharp
```
This will copy all source files into your local computer. You will need to install a few
system libraries before installing pyharp. All following instructions are executed under
the `pyharp/` directory, which is referred to as the `root`.

## Install python libraries

The minimum python version is 3.8.

To install the required python libraries, you can use the following command:
```
pip install -r requirements.txt
```

## Build and test
After you completed the installation steps, you can build the pyharp library.
The easiest way is to build it in-place, meaning that the build (binary files) are
located under `root`. To do so, make a new directory named `build`
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

## Install python library

The python library can be installed by running the following command:
```
pip install .
```
