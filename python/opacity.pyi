"""
Opacity module for HARP atmospheric radiation calculations.

This module provides various opacity models for calculating atmospheric opacities.
"""

from typing import overload
import torch

class OpacityOptions:
    """
    Set opacity options.

    Returns:
        pyharp.OpacityOptions: class object

    Examples:
        >>> import torch
        >>> from pyharp.opacity import OpacityOptions
        >>> op = OpacityOptions().type('molecule-line')
    """

    def __init__(self) -> None:
        """Create a new OpacityOptions instance."""
        ...

    def __repr__(self) -> str: ...

    def query_wavenumber(self) -> list[float]:
        """
        Query the wavenumber grid from opacity files.

        Returns:
            list[float]: wavenumber grid [cm^-1]
        """
        ...

    def query_weight(self) -> list[float]:
        """
        Query the spectral weights from opacity files.

        Returns:
            list[float]: spectral weights
        """
        ...

    @overload
    def type(self) -> str:
        """
        Get the type of the opacity source format.

        Returns:
            str: type of the opacity source
        """
        ...

    @overload
    def type(self, value: str) -> "OpacityOptions":
        """
        Set the type of the opacity source format.

        Valid options are: ``jit``, ``molecule-line``, ``molecule-cia``,
        ``fourcolumn``, ``wavetemp``, ``multiband-ck``, ``picaso-ck``,
        ``rayleigh``, ``respq-table``, ``helios``.

        Args:
            value (str): type of the opacity source

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().type('molecule-line')
            >>> print(op)
        """
        ...

    @overload
    def bname(self) -> str:
        """
        Get the name of the band that the opacity is associated with.

        Returns:
            str: band name
        """
        ...

    @overload
    def bname(self, value: str) -> "OpacityOptions":
        """
        Set the name of the band that the opacity is associated with.

        Args:
            value (str): name of the band that the opacity is associated with

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().bname('band1')
        """
        ...

    @overload
    def opacity_files(self) -> list[str]:
        """
        Get the list of opacity data files.

        Returns:
            list[str]: list of opacity data files
        """
        ...

    @overload
    def opacity_files(self, value: list[str]) -> "OpacityOptions":
        """
        Set the list of opacity data files.

        Args:
            value (list[str]): list of opacity data files

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().opacity_files(['file1', 'file2'])
        """
        ...

    @overload
    def species_ids(self) -> list[int]:
        """
        Get the list of dependent species indices.

        Returns:
            list[int]: list of dependent species indices
        """
        ...

    @overload
    def species_ids(self, value: list[int]) -> "OpacityOptions":
        """
        Set the list of dependent species indices.

        Args:
            value (list[int]): list of dependent species indices

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().species_ids([1, 2])
        """
        ...

    @overload
    def jit_kwargs(self) -> list[str]:
        """
        Get the list of kwargs to pass to the JIT module.

        Returns:
            list[str]: list of kwargs
        """
        ...

    @overload
    def jit_kwargs(self, value: list[str]) -> "OpacityOptions":
        """
        Set the list of kwargs to pass to the JIT module.

        Args:
            value (list[str]): list of kwargs to pass to the JIT module

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().jit_kwargs(['temp', 'wavelength'])
            >>> print(op.jit_kwargs())
        """
        ...

    @overload
    def fractions(self) -> list[float]:
        """
        Get fractions of species in cia calculation.

        Returns:
            list[float]: list of species fractions
        """
        ...

    @overload
    def fractions(self, value: list[float]) -> "OpacityOptions":
        """
        Set fractions of species in cia calculation.

        Args:
            value (list[float]): list of species fractions

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().fractions([0.9, 0.1])
        """
        ...

    @overload
    def nmom(self) -> int:
        """
        Get the number of scattering phase-function moments.

        Returns:
            int: number of moments, excluding the implicit zeroth moment
        """
        ...

    @overload
    def nmom(self, value: int) -> "OpacityOptions":
        """
        Set the number of scattering phase-function moments.

        Args:
            value (int): number of moments, excluding the implicit zeroth moment

        Returns:
            OpacityOptions: class object
        """
        ...

    @overload
    def verbose(self) -> bool:
        """
        Get verbose flag.

        Returns:
            bool: verbose flag
        """
        ...

    @overload
    def verbose(self, value: bool) -> "OpacityOptions":
        """
        Set verbose flag.

        Args:
            value (bool): verbose flag

        Returns:
            OpacityOptions: class object
        """
        ...

class JITOpacity:
    """
    JIT opacity model.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import JITOpacity, OpacityOptions
        >>> op = JITOpacity(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a JITOpacity instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using JIT model.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3
            atm (dict[str, torch.Tensor]): atmospheric parameters passed to the JIT model

                The keyword arguments must be provided in the form of a dictionary.
                The keys of the dictionary are the names of the input tensors
                and the values are the corresponding tensors.
                Since the JIT model only accepts positional arguments,
                the keyword arguments are passed according to the order of the keys in the dictionary.

        Returns:
            torch.Tensor: results of the JIT opacity model
        """
        ...

class WaveTemp:
    """
    Wave-Temp opacity data.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import WaveTemp, OpacityOptions
        >>> op = WaveTemp(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a WaveTemp instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using Wave-Temp data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            atm (dict[str, torch.Tensor]): atmospheric parameters.

                Both 'temp' [k] and ('wavenumber' [cm^{-1}] or 'wavelength' [um]) must be provided

        Returns:
            torch.Tensor:
                The shape of the output tensor is (nwave, ncol, nlyr, *),
                where nwave is the number of wavelengths,
                ncol is the number of columns,
                nlyr is the number of layers.
                The last dimension is the optical properties arranged
                in the order of attenuation [1/m], single scattering albedo and scattering phase function.
        """
        ...

class MultiBand:
    """
    Multi-band opacity data.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import MultiBand, OpacityOptions
        >>> op = MultiBand(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a MultiBand instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using multi-band data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            atm (dict[str, torch.Tensor]): atmospheric parameters

                Both 'temp' [k] and 'pres' [pa] must be provided

        Returns:
            torch.Tensor:
                The shape of the output tensor is (nwave, ncol, nlyr, 1),
                where nwave is the number of wavelengths,
                ncol is the number of columns,
                nlyr is the number of layers.
                The last dimension is the optical properties arranged
                in the order of attenuation [1/m], single scattering albedo and scattering phase function.
        """
        ...

class FourColumn:
    """
    Four-column opacity data.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import FourColumn, OpacityOptions
        >>> op = FourColumn(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a FourColumn instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using four-column data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            atm (dict[str, torch.Tensor]): atmospheric parameters

                Either 'wavelength' or 'wavenumber' must be provided
                if 'wavelength' is provided, the unit is um.
                if 'wavenumber' is provided, the unit is cm^{-1}.

        Returns:
            torch.Tensor:
                The shape of the output tensor is (nwave, ncol, nlyr, 2+nmom),
                where nwave is the number of wavelengths,
                ncol is the number of columns,
                nlyr is the number of layers.
                The last dimension is the optical properties arranged
                in the order of attenuation [1/m], single scattering albedo and scattering phase function, where nmom is the number of scattering moments.
        """
        ...

class Rayleigh:
    """
    Gas Rayleigh-scattering opacity evaluated on the active spectral grid.

    Supported species are H2, He, H2O, CH4, N2, and CO2. The returned
    attenuation is calculated from the molar concentrations in ``conc``.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import OpacityOptions, Rayleigh
        >>> op = OpacityOptions().type("rayleigh").species_ids([0]).nmom(4)
        >>> rayleigh = Rayleigh(op)
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """Create a Rayleigh instance from opacity options."""
        ...

    def __repr__(self) -> str: ...

    def module(self, name: str): ...

    def buffer(self, name: str) -> torch.Tensor: ...

    def forward(
        self, conc: torch.Tensor, atm: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate gas Rayleigh optical properties.

        Args:
            conc: Molar concentration with shape ``(ncol, nlyr, nspecies)``
                and units mol/m^3.
            atm: Must contain either one-dimensional ``wavenumber`` in cm^-1
                or ``wavelength`` in um.

        Returns:
            Tensor with shape ``(nwave, ncol, nlyr, 2 + nmom)`` containing
            attenuation [1/m], single-scattering albedo, and Legendre moments
            excluding the zeroth moment.
        """
        ...

class FuWaterIce:
    """
    Fu (1996) solar and Fu et al. (1998) infrared water-ice opacity.

    The configured species supplies ice-water content. When ``re`` is given,
    it takes precedence and is converted internally to Fu generalized
    effective size using ``Dge = 8 re / (3 sqrt(3))``. Otherwise ``Dge`` is
    diagnosed from layer ``temp`` and IWC with the Sun--Rikus/Sun (2001)
    parameterization and limited to the Fu96/Fu98 common size range of
    18.63--129.6 um. The full Fu96 interval tables and Fu98 wavelength-node
    tables cover 0.25--100 um; all optical properties are zero outside that
    range. A scalar ``fu_delta_scale=True`` applies the Fu96 scaling; it
    defaults to false because Toon shortwave already uses delta-Eddington.

    Examples:
        >>> from pyharp.opacity import FuWaterIce, OpacityOptions
        >>> op = (OpacityOptions().type("water-ice-fu96-98")
        ...       .species_ids([0]).nmom(4))
        >>> ice = FuWaterIce(op)
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """Create a FuWaterIce instance from opacity options."""
        ...

    def __repr__(self) -> str: ...

    def module(self, name: str): ...

    def buffer(self, name: str) -> torch.Tensor: ...

    def forward(
        self, conc: torch.Tensor, atm: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate water-ice cloud optical properties.

        Args:
            conc: Molar concentration with shape ``(ncol, nlyr, nspecies)``
                and units mol/m^3.
            atm: Must contain ``wavenumber`` [cm^-1] or ``wavelength`` [um].
                Layer ``re`` [um] takes precedence when present; otherwise
                layer ``temp`` [K] is required. Optional scalar
                ``fu_delta_scale`` enables the Fu96 delta scaling.

        Returns:
            Tensor with shape ``(nwave, ncol, nlyr, 2 + nmom)`` containing
            attenuation [1/m], single-scattering albedo, and HG Legendre
            moments excluding the zeroth moment.
        """
        ...

class MieWaterLiquid:
    """
    Lorenz-Mie opacity for spherical liquid-water droplets.

    The configured species supplies liquid-water content. ``re`` is the
    equivalent monodisperse droplet radius in um. By default the complex
    refractive index is interpolated from Segelstein (1981) over
    0.1--1000 um. The returned Legendre moments use a
    Henyey-Greenstein representation of the exact Mie asymmetry factor.

    Examples:
        >>> from pyharp.opacity import MieWaterLiquid, OpacityOptions
        >>> op = (OpacityOptions().type("water-liquid-mie")
        ...       .species_ids([0]).nmom(4))
        >>> cloud = MieWaterLiquid(op)
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """Create a MieWaterLiquid instance from opacity options."""
        ...

    def __repr__(self) -> str: ...

    def module(self, name: str): ...

    def buffer(self, name: str) -> torch.Tensor: ...

    def forward(
        self, conc: torch.Tensor, atm: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate liquid-water cloud optical properties.

        Args:
            conc: Molar concentration with shape
                ``(ncol, nlyr, nspecies)`` and units mol/m^3.
            atm: Must contain ``wavenumber`` [cm^-1] or ``wavelength`` [um],
                plus layer ``re`` [um]. Optional ``water_density`` is in
                kg/m^3. ``refractive_index_real`` and
                ``refractive_index_imag`` may jointly override the built-in
                liquid-water optical constants.

        Returns:
            Tensor with shape ``(nwave, ncol, nlyr, 2 + nmom)`` containing
            attenuation [1/m], single-scattering albedo, and HG Legendre
            moments excluding the zeroth moment.
        """
        ...

class TemperatureSwitchWaterCloud:
    """
    Temperature-partitioned water-condensate cloud opacity.

    The configured species supplies total condensed water. Fu (1996, 1998)
    ice optics and Lorenz-Mie liquid-water optics are combined following the
    Khairoutdinov and Randall (2003) temperature partition: all liquid at and
    above 273.15 K, all ice at and below 253.15 K, and a linear mixed-phase
    transition between them.

    Examples:
        >>> from pyharp.opacity import TemperatureSwitchWaterCloud, OpacityOptions
        >>> op = (OpacityOptions().type("water-cloud-temperature-switch")
        ...       .species_ids([0]).nmom(4))
        >>> cloud = TemperatureSwitchWaterCloud(op)
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, options: OpacityOptions) -> None: ...

    def __repr__(self) -> str: ...

    def module(self, name: str): ...

    def buffer(self, name: str) -> torch.Tensor: ...

    def forward(
        self, conc: torch.Tensor, atm: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate temperature-partitioned cloud optical properties.

        Args:
            conc: Total water-condensate molar concentration with shape
                ``(ncol, nlyr, nspecies)`` and units mol/m^3.
            atm: Must contain layer ``temp`` [K] and either ``wavenumber``
                [cm^-1] or ``wavelength`` [um]. Liquid droplets use 14 um by
                default; optional ``liquid_re`` [um] overrides it. Optional
                ``ice_re`` [um] takes precedence over the Sun--Rikus/Sun
                diagnostic.

        Returns:
            Tensor with shape ``(nwave, ncol, nlyr, 2 + nmom)`` containing
            attenuation [1/m], single-scattering albedo, and HG Legendre
            moments excluding the zeroth moment.
        """
        ...

class MoleculeLine:
    """
    Molecular line absorption read from a NetCDF dump.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import MoleculeLine, OpacityOptions
        >>> op = MoleculeLine(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a MoleculeLine instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate attenuation using NetCDF line cross sections.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3
            atm (dict[str, torch.Tensor]): atmospheric parameters

                Either 'wavelength' or 'wavenumber' must be provided
                if 'wavelength' is provided, the unit is um.
                if 'wavenumber' is provided, the unit is cm^{-1}.

        Returns:
            torch.Tensor:
                The shape of the output tensor is (nwave, ncol, nlyr, 1),
                where nwave is the number of wavelengths,
                ncol is the number of columns,
                nlyr is the number of layers.
                The last dimension is the attenuation coefficient [1/m].
        """
        ...

class MoleculeCIA:
    """
    Collision-induced absorption read from a NetCDF dump.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import MoleculeCIA, OpacityOptions
        >>> op = MoleculeCIA(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a MoleculeCIA instance.

        Args:
            options (OpacityOptions): Opacity options
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

    def forward(self, conc: torch.Tensor, atm: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate attenuation using NetCDF CIA binary coefficients.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3
            atm (dict[str, torch.Tensor]): atmospheric parameters

                Either 'wavelength' or 'wavenumber' must be provided
                if 'wavelength' is provided, the unit is um.
                if 'wavenumber' is provided, the unit is cm^{-1}.

        Returns:
            torch.Tensor:
                The shape of the output tensor is (nwave, ncol, nlyr, 1),
                where nwave is the number of wavelengths,
                ncol is the number of columns,
                nlyr is the number of layers.
                The last dimension is the attenuation coefficient [1/m].
        """
        ...
