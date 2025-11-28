"""
Python bindings for HARP (High-performance Atmospheric Radiation Package)

This module provides Python bindings to the C++ HARP library for
atmospheric radiation calculations.
"""

from typing import overload
import torch

# Index constants for optical properties
kIEX: int  # extinction cross section index
kISS: int  # single scattering albedo index
kIPM: int  # phase moments index

# Index constants for flux direction
kIUP: int  # upward flux index
kIDN: int  # downward flux index

# Module-level functions
def species_names() -> list[str]:
    """
    Retrieves the list of species names.

    Returns:
        list[str]: List of species names
    """
    ...

def species_weights() -> list[float]:
    """
    Retrieves the list of species molecular weights [kg/mol].

    Returns:
        list[float]: List of species molecular weights in kg/mol
    """
    ...

def set_search_paths(path: str) -> str:
    """
    Set the search paths for resource files.

    Args:
        path (str): The search paths

    Return:
        str: The search paths

    Example:
        >>> import pyharp

        # set the search paths
        >>> pyharp.set_search_paths("/path/to/resource/files")
    """
    ...

def get_search_paths() -> str:
    """
    Get the search paths for resource files.

    Return:
        str: The search paths

    Example:
        >>> import pyharp

        # get the search paths
        >>> pyharp.get_search_paths()
    """
    ...

def add_resource_directory(path: str, prepend: bool = True) -> str:
    """
    Add a resource directory to the search paths.

    Args:
        path (str): The resource directory to add.
        prepend (bool): If true, prepend the directory to the search paths. If false, append it.

    Returns:
        str: The updated search paths.

    Example:
        >>> import pyharp

        # add a resource directory
        >>> pyharp.add_resource_directory("/path/to/resource/files")
    """
    ...

def find_resource(filename: str) -> str:
    """
    Find a resource file from the search paths.

    Args:
        filename (str): The name of the resource file.

    Returns:
        str: The full path to the resource file.

    Example:
        >>> import pyharp

        # find a resource file
        >>> path = pyharp.find_resource("example.txt")
        >>> print(path)  # /path/to/resource/files/example.txt
    """
    ...

# Radiation functions
@overload
def bbflux_wavenumber(wave: torch.Tensor, temp: float, ncol: int = 1) -> torch.Tensor:
    """
    Calculate blackbody flux using wavenumber.

    Args:
        wave (torch.Tensor): wavenumber [cm^-1]
        temp (float): temperature [K]
        ncol (int, optional): number of columns, default to 1

    Returns:
        torch.Tensor: blackbody flux [w/(m^2 cm^-1)]

    Examples:
        >>> import torch
        >>> from pyharp import bbflux_wavenumber

        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavenumber(wave, temp)
    """
    ...

@overload
def bbflux_wavenumber(wn1: float, wn2: float, temp: torch.Tensor) -> torch.Tensor:
    """
    Calculate blackbody flux using wavenumber.

    Args:
        wn1 (float): wavenumber [cm^-1]
        wn2 (float): wavenumber [cm^-1]
        temp (torch.Tensor): temperature [K]

    Returns:
        torch.Tensor: blackbody flux [w/(m^2 cm^-1)]

    Examples:
        >>> import torch
        >>> from pyharp import bbflux_wavenumber
        >>> temp = torch.tensor([300.0, 310.0, 320.0])
        >>> flux = bbflux_wavenumber(1.0, 2.0, temp)
    """
    ...

def bbflux_wavelength(wave: torch.Tensor, temp: float, ncol: int = 1) -> torch.Tensor:
    """
    Calculate blackbody flux using wavelength.

    Args:
        wave (torch.Tensor): wavelength [um]
        temp (float): temperature [K]
        ncol (int, optional): number of columns, default to 1

    Returns:
        torch.Tensor: blackbody flux [w/(m^2 um^-1)]

    Examples:
        >>> from pyharp import bbflux_wavelength
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavelength(wave, temp)
    """
    ...

def calc_dz_hypsometric(pres: torch.Tensor, temp: torch.Tensor, g_ov_R: torch.Tensor) -> torch.Tensor:
    """
    Calculate the height between pressure levels using the hypsometric equation.

    .. math::

      dz = \\frac{RT}{g} \\cdot d\\ln p

    where :math:`R` is the specific gas constant, :math:`g` is the gravity,
    :math:`T` is the temperature, :math:`p_1` and :math:`p_2` are the pressure levels.

    Args:
        pres (torch.Tensor): pressure [pa] at layers
        temp (torch.Tensor): temperature [K] at layers
        g_ov_R (torch.Tensor): gravity over specific gas constant [K/m] at layers

    Returns:
        torch.Tensor: height between pressure levels [m]

    Examples:
        >>> from pyharp import calc_dz_hypsometric
        >>> pres = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = torch.tensor([300.0, 310.0, 320.0])
        >>> g_ov_R = torch.tensor([1.0, 2.0, 3.0])
        >>> dz = calc_dz_hypsometric(pres, temp, g_ov_R)
    """
    ...

# Math functions
def interpn(
    query: list[torch.Tensor],
    coords: list[torch.Tensor],
    lookup: torch.Tensor,
    extrapolate: bool = False
) -> torch.Tensor:
    """
    Multidimensional linear interpolation.

    Args:
        query (list[torch.Tensor]): Query coordinates
        coords (list[torch.Tensor]): Coordinate arrays, len = ndim, each tensor has shape (nx1,), (nx2,) ...
        lookup (torch.Tensor): Lookup tensor (nx1, nx2, ..., nval)
        extrapolate (bool): Whether to extrapolate beyond the bounds

    Returns:
        torch.Tensor: Interpolated values

    Examples:
        >>> import torch
        >>> from pyharp import interpn
        >>> query = [torch.tensor([0.5]), torch.tensor([0.5])]
        >>> coords = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
        >>> lookup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> interpn(query, coords, lookup)
        tensor(2.5000)
    """
    ...

# Radiation classes
class RadiationBandOptions:
    """
    Options for radiation band configuration.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().name('band1').outdirs('outdir')
    """

    def __init__(self) -> None:
        """
        Create a new RadiationBandOptions instance.

        Returns:
            RadiationBandOptions: class object
        """
        ...

    def __repr__(self) -> str: ...

    @overload
    def name(self) -> str:
        """Get radiation band name."""
        ...

    @overload
    def name(self, value: str) -> "RadiationBandOptions":
        """
        Set radiation band name.

        Args:
            value (str): radiation band name

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def outdirs(self) -> str:
        """Get outgoing ray directions."""
        ...

    @overload
    def outdirs(self, value: str) -> "RadiationBandOptions":
        """
        Set outgoing ray directions.

        Args:
            value (str): outgoing ray directions

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def solver_name(self) -> str:
        """Get solver name."""
        ...

    @overload
    def solver_name(self, value: str) -> "RadiationBandOptions":
        """
        Set solver name.

        Args:
            value (str): solver name

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def l2l_order(self) -> int:
        """Get layer-to-level interpolation order."""
        ...

    @overload
    def l2l_order(self, value: int) -> "RadiationBandOptions":
        """
        Set layer-to-level interpolation order.

        Args:
            value (int): interpolation order

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def nwave(self) -> int:
        """Get number of spectral waves."""
        ...

    @overload
    def nwave(self, value: int) -> "RadiationBandOptions":
        """
        Set number of spectral waves.

        Args:
            value (int): number of waves

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def ncol(self) -> int:
        """Get number of columns."""
        ...

    @overload
    def ncol(self, value: int) -> "RadiationBandOptions":
        """
        Set number of columns.

        Args:
            value (int): number of columns

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def nlyr(self) -> int:
        """Get number of layers."""
        ...

    @overload
    def nlyr(self, value: int) -> "RadiationBandOptions":
        """
        Set number of layers.

        Args:
            value (int): number of layers

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def nstr(self) -> int:
        """Get number of streams."""
        ...

    @overload
    def nstr(self, value: int) -> "RadiationBandOptions":
        """
        Set number of streams.

        Args:
            value (int): number of streams

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def wavenumber(self) -> list[float]:
        """Get wavenumber grid [cm^-1]."""
        ...

    @overload
    def wavenumber(self, value: list[float]) -> "RadiationBandOptions":
        """
        Set wavenumber grid.

        Args:
            value (list[float]): wavenumber grid [cm^-1]

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def weight(self) -> list[float]:
        """Get spectral weights."""
        ...

    @overload
    def weight(self, value: list[float]) -> "RadiationBandOptions":
        """
        Set spectral weights.

        Args:
            value (list[float]): spectral weights

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def verbose(self) -> bool:
        """Get verbose flag."""
        ...

    @overload
    def verbose(self, value: bool) -> "RadiationBandOptions":
        """
        Set verbose flag.

        Args:
            value (bool): verbose flag

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def disort(self):
        """
        Get disort options.

        Returns:
            pydisort.DisortOptions: disort options
        """
        ...

    @overload
    def disort(self, value) -> "RadiationBandOptions":
        """
        Set disort options.

        Args:
            value (pydisort.DisortOptions): disort options

        Returns:
            RadiationBandOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> from pydisort import DisortOptions
            >>> op = RadiationBandOptions().disort(DisortOptions().nwave(10))
            >>> print(op)
        """
        ...

    @overload
    def opacities(self):
        """
        Get opacities.

        Returns:
            dict: opacities dictionary mapping name to OpacityOptions
        """
        ...

    @overload
    def opacities(self, value: dict) -> "RadiationBandOptions":
        """
        Set opacities.

        Args:
            value (dict): opacities dictionary

        Returns:
            RadiationBandOptions: class object
        """
        ...

class RadiationOptions:
    """
    Options for radiation configuration.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions.from_yaml("config.yaml")
    """

    def __init__(self) -> None:
        """
        Create a new RadiationOptions instance.

        Returns:
            RadiationOptions: class object
        """
        ...

    def __repr__(self) -> str: ...

    @staticmethod
    def from_yaml(filename: str) -> "RadiationOptions":
        """
        Create a RadiationOptions object from a YAML file.

        Args:
            filename (str): YAML file name

        Returns:
            RadiationOptions: class object
        """
        ...

    def ncol(self) -> int:
        """
        Get number of columns from all bands.

        Returns:
            int: number of columns
        """
        ...

    def nlyr(self) -> int:
        """
        Get number of layers from all bands.

        Returns:
            int: number of layers
        """
        ...

    @overload
    def outdirs(self) -> str:
        """Get outgoing ray directions."""
        ...

    @overload
    def outdirs(self, value: str) -> "RadiationOptions":
        """
        Set outgoing ray directions.

        Args:
            value (str): outgoing ray directions

        Returns:
            RadiationOptions: class object
        """
        ...

    @overload
    def bands(self) -> list[RadiationBandOptions]:
        """
        Get radiation band options.

        Returns:
            list[RadiationBandOptions]: list of radiation band options
        """
        ...

    @overload
    def bands(self, value: list[RadiationBandOptions]) -> "RadiationOptions":
        """
        Set radiation band options.

        Args:
            value (list[RadiationBandOptions]): list of radiation band options

        Returns:
            RadiationOptions: class object
        """
        ...

class Radiation:
    """
    Calculate the net radiation flux.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationOptions, Radiation
        >>> op = RadiationOptions.from_yaml("config.yaml")
        >>> rad = Radiation(op)
    """

    options: RadiationOptions
    spectra: torch.Tensor

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RadiationOptions) -> None:
        """
        Create a Radiation instance.

        Args:
            options (RadiationOptions): Radiation options
        """
        ...

    def __repr__(self) -> str: ...

    def module(self, name: str):
        """
        Get a submodule by name.

        Args:
            name (str): name of the submodule (e.g., "band_name.opacity_name")

        Returns:
            The submodule
        """
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """
        Get a buffer by name.

        Args:
            name (str): name of the buffer

        Returns:
            torch.Tensor: the buffer tensor
        """
        ...

    def forward(
        self,
        conc: torch.Tensor,
        dz: torch.Tensor,
        bc: dict[str, torch.Tensor],
        atm: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the net radiation flux.

        Args:
            conc (torch.Tensor): concentration [mol/m^3] (ncol, nlyr, nspecies)
            dz (torch.Tensor): layer thickness [m] (nlyr) or (ncol, nlyr)
            bc (dict[str, torch.Tensor]): boundary conditions
            atm (dict[str, torch.Tensor]): atmospheric parameters (temp, pres, etc.)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - net flux [W/m^2] (ncol, nlyr+1)
                - surface downward flux [W/m^2] (ncol)
                - TOA upward flux [W/m^2] (ncol)
        """
        ...

class RadiationBand:
    """
    Calculate the net radiation flux for a band.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationBandOptions, RadiationBand
        >>> op = RadiationBandOptions().name('band1')
        >>> band = RadiationBand(op)
    """

    options: RadiationBandOptions
    prop: torch.Tensor
    spectrum: torch.Tensor

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: RadiationBandOptions) -> None:
        """
        Create a RadiationBand instance.

        Args:
            options (RadiationBandOptions): Radiation band options
        """
        ...

    def __repr__(self) -> str: ...

    def module(self, name: str):
        """
        Get a submodule by name.

        Args:
            name (str): name of the submodule

        Returns:
            The submodule
        """
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """
        Get a buffer by name.

        Args:
            name (str): name of the buffer

        Returns:
            torch.Tensor: the buffer tensor
        """
        ...

    def forward(
        self,
        conc: torch.Tensor,
        dz: torch.Tensor,
        bc: dict[str, torch.Tensor],
        atm: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the net radiation flux for a band.

        Args:
            conc (torch.Tensor): concentration [mol/m^3] (ncol, nlyr, nspecies)
            dz (torch.Tensor): layer thickness [m] (nlyr) or (ncol, nlyr)
            bc (dict[str, torch.Tensor]): boundary conditions
            atm (dict[str, torch.Tensor]): atmospheric parameters (temp, pres, etc.)

        Returns:
            torch.Tensor: total flux [W/m^2] (ncol, nlyr+1, 2)
        """
        ...

# Integrator
class IntegratorWeight:
    """
    Time integrator weight configuration.

    This class manages integrator weights for multi-stage methods.
    """

    def __init__(self) -> None:
        """Initialize IntegratorWeight with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def wght0(self) -> float:
        """Get weight 0."""
        ...

    @overload
    def wght0(self, value: float) -> "IntegratorWeight":
        """Set weight 0."""
        ...

    @overload
    def wght1(self) -> float:
        """Get weight 1."""
        ...

    @overload
    def wght1(self, value: float) -> "IntegratorWeight":
        """Set weight 1."""
        ...

    @overload
    def wght2(self) -> float:
        """Get weight 2."""
        ...

    @overload
    def wght2(self, value: float) -> "IntegratorWeight":
        """Set weight 2."""
        ...

class IntegratorOptions:
    """
    Time integrator configuration options.

    This class manages time integration parameters.
    """

    def __init__(self) -> None:
        """Initialize IntegratorOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get the integrator type."""
        ...

    @overload
    def type(self, value: str) -> "IntegratorOptions":
        """Set the integrator type."""
        ...

    @overload
    def cfl(self) -> float:
        """Get the CFL number."""
        ...

    @overload
    def cfl(self, value: float) -> "IntegratorOptions":
        """Set the CFL number."""
        ...

class Integrator:
    """
    Time integrator implementation.

    This module handles time integration.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: IntegratorOptions) -> None:
        """
        Construct an Integrator module.

        Args:
            options: Time integrator configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: IntegratorOptions
    stages: int

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def stop(self, steps: int, current_time: float) -> bool:
        """
        Check if integration should stop.

        Args:
            steps: Number of steps taken
            current_time: Current simulation time

        Returns:
            True if should stop, False otherwise
        """
        ...

# Version
__version__: str
