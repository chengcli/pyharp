"""
Opacity module for HARP atmospheric radiation calculations.

This module provides various opacity models for calculating atmospheric opacities.
"""

from typing import overload
import torch

class AttenuatorOptions:
    """
    Set opacity band options.

    Returns:
        pyharp.AttenuatorOptions: class object

    Examples:
        >>> import torch
        >>> from pyharp.opacity import AttenuatorOptions
        >>> op = AttenuatorOptions().band_options(['band1', 'band2'])
    """

    def __init__(self) -> None:
        """Create a new AttenuatorOptions instance."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """
        Get the type of the opacity source format.

        Returns:
            str: type of the opacity source
        """
        ...

    @overload
    def type(self, value: str) -> "AttenuatorOptions":
        """
        Set the type of the opacity source format.

        Valid options are: ``jit``, ``rfm-lbl``, ``rfm-ck``, ``four-column``, ``wavetemp``, ``multiband``.

        Args:
            value (str): type of the opacity source

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().type('rfm-lbl')
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
    def bname(self, value: str) -> "AttenuatorOptions":
        """
        Set the name of the band that the opacity is associated with.

        Args:
            value (str): name of the band that the opacity is associated with

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().bname('band1')
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
    def opacity_files(self, value: list[str]) -> "AttenuatorOptions":
        """
        Set the list of opacity data files.

        Args:
            value (list[str]): list of opacity data files

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
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
    def species_ids(self, value: list[int]) -> "AttenuatorOptions":
        """
        Set the list of dependent species indices.

        Args:
            value (list[int]): list of dependent species indices

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().species_ids([1, 2])
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
    def jit_kwargs(self, value: list[str]) -> "AttenuatorOptions":
        """
        Set the list of kwargs to pass to the JIT module.

        Args:
            value (list[str]): list of kwargs to pass to the JIT module

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().jit_kwargs(['temp', 'wavelength'])
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
    def fractions(self, value: list[float]) -> "AttenuatorOptions":
        """
        Set fractions of species in cia calculation.

        Args:
            value (list[float]): list of species fractions

        Returns:
            AttenuatorOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import AttenuatorOptions
            >>> op = AttenuatorOptions().fractions([0.9, 0.1])
        """
        ...

class JITOpacity:
    """
    JIT opacity model.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import JITOpacity, AttenuatorOptions
        >>> op = JITOpacity(AttenuatorOptions())
    """

    options: AttenuatorOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: AttenuatorOptions) -> None:
        """
        Create a JITOpacity instance.

        Args:
            options (AttenuatorOptions): Attenuator options
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using JIT model.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3
            kwargs (dict[str, torch.Tensor]): keyword arguments passed to the JIT model

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
        >>> from pyharp.opacity import WaveTemp, AttenuatorOptions
        >>> op = WaveTemp(AttenuatorOptions())
    """

    options: AttenuatorOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: AttenuatorOptions) -> None:
        """
        Create a WaveTemp instance.

        Args:
            options (AttenuatorOptions): Attenuator options
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using Wave-Temp data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            kwargs (dict[str, torch.Tensor]): keyword arguments.

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
        >>> from pyharp.opacity import MultiBand, AttenuatorOptions
        >>> op = MultiBand(AttenuatorOptions())
    """

    options: AttenuatorOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: AttenuatorOptions) -> None:
        """
        Create a MultiBand instance.

        Args:
            options (AttenuatorOptions): Attenuator options
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using multi-band data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            kwargs (dict[str, torch.Tensor]): keyword arguments

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
        >>> from pyharp.opacity import FourColumn, AttenuatorOptions
        >>> op = FourColumn(AttenuatorOptions())
    """

    options: AttenuatorOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: AttenuatorOptions) -> None:
        """
        Create a FourColumn instance.

        Args:
            options (AttenuatorOptions): Attenuator options
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using four-column data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3

            kwargs (dict[str, torch.Tensor]): keyword arguments

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

class RFM:
    """
    Line-by-line absorption data computed by RFM.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import RFM, AttenuatorOptions
        >>> op = RFM(AttenuatorOptions())
    """

    options: AttenuatorOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: AttenuatorOptions) -> None:
        """
        Create a RFM instance.

        Args:
            options (AttenuatorOptions): Attenuator options
        """
        ...

    def __repr__(self) -> str: ...

    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate opacity using RFM line-by-line absorption data.

        Args:
            conc (torch.Tensor): concentration of the species in mol/m^3
            kwargs (dict[str, torch.Tensor]): keyword arguments

                Either 'wavelength' or 'wavenumber' must be provided
                if 'wavelength' is provided, the unit is um.
                if 'wavenumber' is provided, the unit is cm^{-1}.

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
