"""
Python bindings for HARP (High-performance Atmospheric Radiation Package)

This module provides Python bindings to the C++ HARP library for
atmospheric radiation calculations.
"""

from typing import Iterator, overload
import torch

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

def shared() -> Iterator[torch.Tensor]:
    """
    Pyharp module deposits data -- tensors -- to a shared dictionary, which can be accessed by other modules.
    This function returns an iterator over the shared data.

    After running the forward method of the :class:`RadiationBand`, the shared data with the following keys are available:

      - "radiation/<band_name>/total_flux": total flux in a band

    Yields:
        torch.Tensor: shared data of the pyharp module

    Examples:
        >>> import pyharp
        >>> import torch

        # ... after calling the forward method

        # loop over the shared data
        >>> for data in pyharp.shared():
        >>>     print(type(data), data.size())  # prints the shared data
    """
    ...

def get_shared(key: str) -> torch.Tensor:
    """
    Get the shared data by key.

    Args:
        key (str): The key of the shared data.

    Returns:
        torch.Tensor: The shared data.

    Example:
        >>> import pyharp
        >>> import torch

        # ... after calling the forward method

        # get the shared data
        >>> data = pyharp.get_shared("radiation/band1/total_flux")
        >>> print(type(data), data.size())  # prints the shared data
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

    def query_waves(self, op_name: str = "") -> list[float]:
        """
        Query the spectral grids.

        Args:
            op_name (str): opacity name

        Returns:
            list[float]: spectral grids
        """
        ...

    def query_weights(self, op_name: str = "") -> list[float]:
        """
        Query the weights.

        Args:
            op_name (str): opacity name

        Returns:
            list[float]: weights
        """
        ...

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
    def ww(self) -> list[float]:
        """Get wavelength, wavenumber or weights for a wave grid."""
        ...

    @overload
    def ww(self, value: list[float]) -> "RadiationBandOptions":
        """
        Set wavelength, wavenumber or weights for a wave grid.

        Args:
            value (list[float]): wavenumbers/wavelengths/weights

        Returns:
            RadiationBandOptions: class object
        """
        ...

    @overload
    def integration(self) -> str:
        """Get integration method."""
        ...

    @overload
    def integration(self, value: str) -> "RadiationBandOptions":
        """
        Set integration method.

        Args:
            value (str): integration method

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
            dict: opacities dictionary
        """
        ...

    @overload
    def opacities(self, value: dict) -> "RadiationBandOptions":
        """
        Set opacities.

        Args:
            value (dict): opacities

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
        >>> op = RadiationOptions().band_options(['band1', 'band2'])
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
    def bands(self):
        """
        Get radiation band options.

        Returns:
            dict: radiation band options dictionary
        """
        ...

    @overload
    def bands(self, value: dict) -> "RadiationOptions":
        """
        Set radiation band options.

        Args:
            value (dict): radiation band options

        Returns:
            RadiationOptions: class object
        """
        ...

class Radiation:
    """
    Calculate the net radiation flux.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().band_options(['band1', 'band2'])
    """

    options: RadiationOptions

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

    def forward(
        self,
        conc: torch.Tensor,
        dz: torch.Tensor,
        bc: dict[str, torch.Tensor],
        kwargs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the net radiation flux.

        Args:
            conc (torch.Tensor): concentration [mol/m^3]
            dz (torch.Tensor): height [m]
            bc (dict[str, torch.Tensor]): boundary conditions
            kwargs (dict[str, torch.Tensor]): additional arguments

        Returns:
            torch.Tensor: net flux [w/m^2]
        """
        ...

class RadiationBand:
    """
    Calculate the net radiation flux for a band.

    Examples:
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().band_options(['band1', 'band2'])
    """

    options: RadiationBandOptions

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

    def forward(
        self,
        conc: torch.Tensor,
        dz: torch.Tensor,
        bc: dict[str, torch.Tensor],
        kwargs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the net radiation flux for a band.

        Args:
            conc (torch.Tensor): concentration [mol/m^3]
            dz (torch.Tensor): height [m]
            bc (dict[str, torch.Tensor]): boundary conditions
            kwargs (dict[str, torch.Tensor]): additional arguments

        Returns:
            torch.Tensor: [W/m^2] (ncol, nlyr+1)
        """
        ...

# Version
__version__: str
