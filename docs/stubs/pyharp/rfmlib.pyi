from __future__ import annotations
from collections import OrderedDict
import numpy as np
import os as os
import shutil as shutil
import subprocess as subprocess
import torch as torch
__all__ = ['OrderedDict', 'create_netcdf_input', 'create_rfm_driver', 'np', 'os', 'read_rfm_atm', 'run_rfm', 'shutil', 'subprocess', 'torch', 'write_ktable', 'write_rfm_atm', 'write_rfm_drv']
def create_netcdf_input(fname: str, absorbers: typing.List[str], atm: typing.Dict[str, numpy.ndarray], wmin: float, wmax: float, wres: float, tnum: int, tmin: float, tmax: float) -> str:
    """

        Create an input file for writing kcoeff table to netCDF format

        Parameters
        ----------
        fname : str
            Name of the file.
        absorbers : list
            A list of absorbers.
        atm : Dict[str, np.ndarray]
            A dictionary containing the atmosphere.
        wmin : float
            Minimum wavenumber.
        wmax : float
            Maximum wavenumber.
        wres : float
            Wavenumber resolution.
        tnum : int
            Number of temperature points.
        tmin : float
            Minimum temperature.
        tmax : float
            Maximum temperature.

        Returns
        -------
        fname : str
            Name of the input file for netCDf

    """
def create_rfm_driver(wav_grid: typing.Tuple[float, float, float], tem_grid: typing.Tuple[int, float, float], absorbers: typing.List[str], hitran_file: str) -> typing.Dict[str, str]:
    """

        Create a RFM driver file.

        Parameters
        ----------
        wav_grid : Tuple[float, float, float]
            Wavenumber grid by minimum, maximum and resolution.
        tem_grid : Tuple[int, float, float]
            Temperature grid by number of points, minimum and maximum.
        absorbers : List
            A list of absorbers.
        hitran_file : str
            Path to HITRAN file.

        Returns
        -------
        driver : Dict[str, str]
            A dictionary containing the driver file content.

    """
def read_rfm_atm(filename):
    ...
def run_rfm(rundir: str = '.') -> None:
    """

        Call to run RFM.

        Parameters
        ----------
        None

        Returns
        -------
        None

    """
def write_ktable(fname: str, absorbers: typing.List[str], atm: typing.Dict[str, numpy.ndarray], wav_grid: typing.Tuple[float, float, float], tem_grid: typing.Tuple[int, float, float], basedir: str = '.') -> None:
    """

        Write kcoeff table to netCDF file.

        Parameters
        ----------
        fname : str
            Name of the file.
        absorbers : List
            A list of absorbers.
        atm : Dict[str, np.ndarray]
            A dictionary containing the atmosphere.
        wav_grid : Tuple[float, float, float]
            Wavenumber grid by minimum, maximum and resolution.
        tem_grid : Tuple[int, float, float]
            Temperature grid by number of points, minimum and maximum.

        Returns
        -------
        None

    """
def write_rfm_atm(atm: typing.Dict[str, numpy.ndarray], rundir: str = '.') -> None:
    """

        Write RFM atmosphere to file.

        Parameters
        ----------
        atm : Dict[str, np.ndarray]
            A dictionary containing the atmosphere
        rundir : str
            Directory to write the file. Default is current directory.

        Returns
        -------
        None

    """
def write_rfm_drv(driver: typing.Dict[str, str], rundir: str = '.') -> None:
    """

        Write RFM driver to file.

        Parameters
        ----------
        driver : Dict[str, str]
            A dictionary containing the driver file content.
        rundir : str
            Directory to write the file. Default is current directory

        Returns
        -------
        None

    """
