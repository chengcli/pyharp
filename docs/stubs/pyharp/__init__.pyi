from __future__ import annotations
from collections import OrderedDict
from importlib._bootstrap import H2SO4Simple
from importlib._bootstrap import RFM
from importlib._bootstrap import Radiation
from importlib._bootstrap import RadiationBand
from importlib._bootstrap import S8Fuller
import numpy as np
import os as os
import pydisort as pydisort
from pyharp.disort import disort_config
from pyharp.rfmlib import create_netcdf_input
from pyharp.rfmlib import create_rfm_driver
from pyharp.rfmlib import read_rfm_atm
from pyharp.rfmlib import run_rfm
from pyharp.rfmlib import write_ktable
from pyharp.rfmlib import write_rfm_atm
from pyharp.rfmlib import write_rfm_drv
import shutil as shutil
import subprocess as subprocess
import torch as torch
import typing
from . import constants
from . import cpp
from . import disort
from . import rfmlib
__all__ = ['AttenuatorOptions', 'H2SO4Simple', 'OrderedDict', 'RFM', 'Radiation', 'RadiationBand', 'RadiationBandOptions', 'RadiationOptions', 'S8Fuller', 'bbflux_wavelength', 'bbflux_wavenumber', 'calc_dz_hypsometric', 'constants', 'cpp', 'create_netcdf_input', 'create_rfm_driver', 'disort', 'disort_config', 'find_resource', 'interpn', 'np', 'os', 'pydisort', 'pyharp', 'read_rfm_atm', 'rfmlib', 'run_rfm', 'set_search_paths', 'shared', 'shutil', 'subprocess', 'torch', 'write_ktable', 'write_rfm_atm', 'write_rfm_drv']
class AttenuatorOptions:
    def __init__(self) -> None:
        """
        Set opacity band options

        Returns:
          pyharp.AttenuatorOptions: class object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().band_options(['band1', 'band2'])
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def bname(self) -> str:
        """
        Set or get the name of the band that the opacity is associated with

        Args:
          bname (str): name of the band that the opacity is associated with

        Returns:
          pyharp.AttenuatorOptions | str : class object if argument is not empty, otherwise the band name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().bname('band1')
        """
    @typing.overload
    def bname(self, arg0: str) -> AttenuatorOptions:
        """
        Set or get the name of the band that the opacity is associated with

        Args:
          bname (str): name of the band that the opacity is associated with

        Returns:
          pyharp.AttenuatorOptions | str : class object if argument is not empty, otherwise the band name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().bname('band1')
        """
    @typing.overload
    def opacity_files(self) -> list[str]:
        """
        Set or get the list of opacity data files

        Args:
          opacity_files (list): list of opacity data files

        Returns:
          pyharp.AttenuatorOptions | list[str]: class object if argument is not empty, otherwise the list of opacity data files

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
        """
    @typing.overload
    def opacity_files(self, arg0: list[str]) -> AttenuatorOptions:
        """
        Set or get the list of opacity data files

        Args:
          opacity_files (list): list of opacity data files

        Returns:
          pyharp.AttenuatorOptions | list[str]: class object if argument is not empty, otherwise the list of opacity data files

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
        """
    @typing.overload
    def species_ids(self) -> list[int]:
        """
        Set or get the list of dependent species indices

        Args:
          species_ids (list): list of dependent species indices

        Returns:
          pyharp.AttenuatorOptions | list[int]: class object if argument is not empty, otherwise the list of dependent species indices

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().species_ids([1, 2])
        """
    @typing.overload
    def species_ids(self, arg0: list[int]) -> AttenuatorOptions:
        """
        Set or get the list of dependent species indices

        Args:
          species_ids (list): list of dependent species indices

        Returns:
          pyharp.AttenuatorOptions | list[int]: class object if argument is not empty, otherwise the list of dependent species indices

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().species_ids([1, 2])
        """
    @typing.overload
    def type(self) -> str:
        """
        Set or get the type of the opacity source

        Valid options are:
          .. list-table::
            :widths: 15 25
            :header-rows: 1

            * - Key
              - Description
            * - 's8_fuller'
              - S8 absorption data from Fuller et al. (1987)
            * - 'h2so4_simple'
              - H2SO4 absorption data from the simple model
            * - 'rfm-lbl'
              - Line-by-line absorption data computed by RFM
            * - 'rfm-ck'
              - Correlated-k absorption computed from line-by-line data

        Args:
          type (str): type of the opacity source

        Returns:
          pyharp.AttenuatorOptions | str : class object if argument is not empty, otherwise the type

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().type('rfm-lbl')
            >>> print(op)
        """
    @typing.overload
    def type(self, arg0: str) -> AttenuatorOptions:
        """
        Set or get the type of the opacity source

        Valid options are:
          .. list-table::
            :widths: 15 25
            :header-rows: 1

            * - Key
              - Description
            * - 's8_fuller'
              - S8 absorption data from Fuller et al. (1987)
            * - 'h2so4_simple'
              - H2SO4 absorption data from the simple model
            * - 'rfm-lbl'
              - Line-by-line absorption data computed by RFM
            * - 'rfm-ck'
              - Correlated-k absorption computed from line-by-line data

        Args:
          type (str): type of the opacity source

        Returns:
          pyharp.AttenuatorOptions | str : class object if argument is not empty, otherwise the type

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().type('rfm-lbl')
            >>> print(op)
        """
class RadiationBandOptions:
    def __init__(self) -> None:
        """
        Returns:
          pyharp.RadiationBandOptions: class object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().name('band1').outdirs('outdir')
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def disort(self) -> ...:
        """
        Set or get disort options

        Args:
          disort (pydisort.DisortOptions): disort options

        Returns:
          pyharp.RadiationBandOptions | pydisort.DisortOptions: class object if argument is not empty, otherwise the disort options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions, DisortOptions
            >>> op = RadiationBandOptions().disort(DisortOptions().nwave(10))
            >>> print(op)
        """
    @typing.overload
    def disort(self, arg0: ...) -> RadiationBandOptions:
        """
        Set or get disort options

        Args:
          disort (pydisort.DisortOptions): disort options

        Returns:
          pyharp.RadiationBandOptions | pydisort.DisortOptions: class object if argument is not empty, otherwise the disort options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions, DisortOptions
            >>> op = RadiationBandOptions().disort(DisortOptions().nwave(10))
            >>> print(op)
        """
    @typing.overload
    def integration(self) -> str:
        """
        Set or get integration method

        Args:
          integration (str): integration method

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the integration method

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().integration('simpson')
            >>> print(op)
        """
    @typing.overload
    def integration(self, arg0: str) -> RadiationBandOptions:
        """
        Set or get integration method

        Args:
          integration (str): integration method

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the integration method

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().integration('simpson')
            >>> print(op)
        """
    @typing.overload
    def name(self) -> str:
        """
        Set or get radiation band name

        Args:
          name (str): radiation band name

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the band name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().name('band1')
            >>> print(op)
        """
    @typing.overload
    def name(self, arg0: str) -> RadiationBandOptions:
        """
        Set or get radiation band name

        Args:
          name (str): radiation band name

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the band name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().name('band1')
            >>> print(op)
        """
    @typing.overload
    def opacities(self) -> dict[str, AttenuatorOptions]:
        """
        Set or get opacities

        Args:
          opacities (dict[str,pyharp.AttenuatorOptions]): opacities

        Returns:
          pyharp.RadiationBandOptions | dict[str,AttenuatorOptions]: class object if argument is not empty, otherwise the attenuator options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().opacities({'band1': 'opacity1', 'band2': 'opacity2'})
            >>> print(op)
        """
    @typing.overload
    def opacities(self, arg0: dict[str, AttenuatorOptions]) -> RadiationBandOptions:
        """
        Set or get opacities

        Args:
          opacities (dict[str,pyharp.AttenuatorOptions]): opacities

        Returns:
          pyharp.RadiationBandOptions | dict[str,AttenuatorOptions]: class object if argument is not empty, otherwise the attenuator options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().opacities({'band1': 'opacity1', 'band2': 'opacity2'})
            >>> print(op)
        """
    @typing.overload
    def outdirs(self) -> str:
        """
        Set or get outgoing ray directions

        Args:
          outdirs (str): outgoing ray directions

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the outgoing ray directions

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().outdirs('(0, 10), (0, 20)')
            >>> print(op)
        """
    @typing.overload
    def outdirs(self, arg0: str) -> RadiationBandOptions:
        """
        Set or get outgoing ray directions

        Args:
          outdirs (str): outgoing ray directions

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the outgoing ray directions

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().outdirs('(0, 10), (0, 20)')
            >>> print(op)
        """
    def query_waves(self) -> list[float]:
        """
        Query the spectral grids

        Returns:
          list[float]: spectral grids

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().query_waves()
        """
    def query_weights(self) -> list[float]:
        """
        Query the weights

        Returns:
          list[float]: weights

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().query_weights()
        """
    @typing.overload
    def solver_name(self) -> str:
        """
        Set or get solver name

        Args:
          solver_name (str): solver name

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the solver name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().solver_name('disort')
            >>> print(op)
        """
    @typing.overload
    def solver_name(self, arg0: str) -> RadiationBandOptions:
        """
        Set or get solver name

        Args:
          solver_name (str): solver name

        Returns:
          pyharp.RadiationBandOptions | str : class object if argument is not empty, otherwise the solver name

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().solver_name('disort')
            >>> print(op)
        """
    @typing.overload
    def ww(self) -> list[float]:
        """
        Set or get wavelength, wavenumber or weights for a wave grid

        Args:
          ww (list[float]): wavenumbers/wavelengths/weights

        Returns:
          pyharp.RadiationBandOptions | list[float]: class object if argument is not empty, otherwise the wavenumbers/wavelengths/weights

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().ww([1.0, 2.0, 3.0])
            >>> print(op)
        """
    @typing.overload
    def ww(self, arg0: list[float]) -> RadiationBandOptions:
        """
        Set or get wavelength, wavenumber or weights for a wave grid

        Args:
          ww (list[float]): wavenumbers/wavelengths/weights

        Returns:
          pyharp.RadiationBandOptions | list[float]: class object if argument is not empty, otherwise the wavenumbers/wavelengths/weights

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().ww([1.0, 2.0, 3.0])
            >>> print(op)
        """
class RadiationOptions:
    @staticmethod
    def from_yaml(filename: str) -> RadiationOptions:
        """
        Create a :class:`pyharp.RadiationOptions` object from a YAML file

        Args:
          filename (str): YAML file name

        Returns:
          pyharp.RadiationOptions: class object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions.from_yaml('radiation.yaml')
        """
    def __init__(self) -> None:
        """
        Set radiation band options

        Returns:
          pyharp.RadiationOptions: class object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().band_options(['band1', 'band2'])
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def bands(self) -> dict[str, RadiationBandOptions]:
        """
        Set radiation band options

        Args:
          bands (dict[str,pyharp.RadiationBandOptions]): radiation band options

        Returns:
          pyharp.RadiationOptions | dict[str,pyharp.RadiationBandOptions]: class object if argument is not empty, otherwise the radiation band options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().bands({'band1': 'outdir1', 'band2': 'outdir2'})
            >>> print(op)
        """
    @typing.overload
    def bands(self, arg0: dict[str, RadiationBandOptions]) -> RadiationOptions:
        """
        Set radiation band options

        Args:
          bands (dict[str,pyharp.RadiationBandOptions]): radiation band options

        Returns:
          pyharp.RadiationOptions | dict[str,pyharp.RadiationBandOptions]: class object if argument is not empty, otherwise the radiation band options

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().bands({'band1': 'outdir1', 'band2': 'outdir2'})
            >>> print(op)
        """
    @typing.overload
    def outdirs(self) -> str:
        """
        Set outgoing ray directions

        Args:
          outdirs (str): outgoing ray directions

        Returns:
          pyharp.RadiationOptions | str : class object if argument is not empty, otherwise the outgoing ray directions

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().outdirs('(0, 10), (0, 20)')
            >>> print(op)
        """
    @typing.overload
    def outdirs(self, arg0: str) -> RadiationOptions:
        """
        Set outgoing ray directions

        Args:
          outdirs (str): outgoing ray directions

        Returns:
          pyharp.RadiationOptions | str : class object if argument is not empty, otherwise the outgoing ray directions

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().outdirs('(0, 10), (0, 20)')
            >>> print(op)
        """
def bbflux_wavelength(wave: torch.Tensor, temp: float, ncol: int = 1) -> torch.Tensor:
    """
    Calculate blackbody flux using wavelength

    Args:
      wave (torch.Tensor): wavelength [um]
      temp (float): temperature [K]
      ncol (int, optional): number of columns, default to 1

    Returns:
      torch.Tensor: blackbody flux [w/(m^2 um^-1)]

    Examples:
      .. code-block:: python

        >>> from pyharp import bbflux_wavelength
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavelength(wave, temp)
    """
@typing.overload
def bbflux_wavenumber(wave: torch.Tensor, temp: float, ncol: int = 1) -> torch.Tensor:
    """
    Calculate blackbody flux using wavenumber

    Args:
      wave (torch.Tensor): wavenumber [cm^-1]
      temp (float): temperature [K]
      ncol (int, optional) number of columns, default to 1

    Returns:
      torch.Tensor: blackbody flux [w/(m^2 cm^-1)]

    Examples:
      .. code-block:: python

        >>> import torch
        >>> from pyharp import bbflux_wavenumber

        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavenumber(wave, temp)
    """
@typing.overload
def bbflux_wavenumber(wn1: float, wn2: float, temp: torch.Tensor = 1) -> torch.Tensor:
    """
    Calculate blackbody flux using wavenumber

    Args:
      wn1 (double): wavenumber [cm^-1]
      wn2 (double): temperature [K]
      temp (torch.Tensor): number of columns, default to 1

    Returns:
      torch.Tensor: blackbody flux [w/(m^2 cm^-1)]

    Examples:
      .. code-block: python

        >>> import torch
        >>> from pyharp import bbflux_wavenumber
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavenumber(wave, temp)
    """
def calc_dz_hypsometric(pres: torch.Tensor, temp: torch.Tensor, g_ov_R: torch.Tensor) -> torch.Tensor:
    """
    Calculate the height between pressure levels using the hypsometric equation

    .. math::

      dz = \\frac{R}{g} \\cdot T \\cdot \\ln\\left(\\frac{p_1}{p_2}\\right)

    where :math:`R` is the specific gas constant, :math:`g` is the gravity,
    :math:`T` is the temperature, :math:`p_1` and :math:`p_2` are the pressure levels.

    Args:
      pres (torch.Tensor): pressure [pa] at layers
      temp (torch.Tensor): temperature [K] at layers
      g_ov_R (torch.Tensor): gravity over specific gas constant [K/m] at layers

    Returns:
      torch.Tensor: height between pressure levels [m]

    Examples:
      .. code-block:: python

        >>> from pyharp import calc_dz_hypsometric
        >>> pres = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = torch.tensor([300.0, 310.0, 320.0])
        >>> g_ov_R = torch.tensor([1.0, 2.0, 3.0])
        >>> dz = calc_dz_hypsometric(pres, temp, g_ov_R)
    """
def find_resource(filename: str) -> str:
    """
    Find a resource file from the search paths.

    Args:
      filename (str): The name of the resource file.

    Returns:
      str: The full path to the resource file.

    Example:
      .. code-block:: python

        >>> import pyharp

        # find a resource file
        >>> path = pyharp.find_resource("example.txt")
        >>> print(path)  # /path/to/resource/files/example.txt
    """
def interpn(query: list[torch.Tensor], coords: list[torch.Tensor], lookup: torch.Tensor, extrapolate: bool = False) -> torch.Tensor:
    """
    Multidimensional linear interpolation

    Args:
      query_coords (list[torch.Tensor]): Query coordinates
      coords (list[torch.Tensor]): Coordinate arrays, len = ndim, each tensor has shape (nx1,), (nx2,) ...
      lookup (torch.Tensor): Lookup tensor (nx1, nx2, ..., nval)

    Examples:
      .. code-block:: python

        >>> import torch
        >>> from pyharp import interpn
        >>> query = [torch.tensor([0.5]), torch.tensor([0.5])]
        >>> coords = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
        >>> lookup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> interpn(query, coords, lookup)
        tensor(2.5000)
    """
def set_search_paths(path: str) -> list[str]:
    """
    Set the search paths for resource files.

    Args:
      path (str): The search paths

    Return:
      str: The search paths

    Example:
      .. code-block:: python

        >>> import pyharp

        # set the search paths
        >>> pyharp.set_search_paths("/path/to/resource/files")
    """
def shared() -> dict[str, torch.Tensor]:
    """
    `Pyharp` module deposits data -- tensors -- to a shared dictionary, which can be accessed by other modules.
    This function returns a readonly view of the shared data from a key.

    After running the forward method of the :class:`RadiationBand <pyharp.cpp.RadiationBand>`, the shared data with the following keys are available:

      .. list-table::
        :widths: 15 25
        :header-rows: 1

        * - Key
          - Description
        * - "radiation/<band_name>/total_flux"
          - total flux in a band
        * - "radiation/downward_flux"
          - downward flux to surface
        * - "radiation/upward_flux"
          - upward flux to space

    Returns:
      dict[str, torch.Tensor]: Shared readonly data of the pyharp module

    Examples:
      .. code-block:: python

        >>> import pyharp
        >>> import torch

        # ... after calling the forward method

        # get the shared data
        >>> shared = pyharp.shared()

        # get the total flux in a band
        >>> shared["radiation/<band_name>/total_flux"]
    """
__version__: str = '1.3.6'
pyharp =
