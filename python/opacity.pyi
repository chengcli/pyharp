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
        >>> op = OpacityOptions().type('rfm-lbl')
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

        Valid options are: ``jit``, ``rfm-lbl``, ``rfm-ck``, ``fourcolumn``, ``wavetemp``, ``multiband-ck``, ``helios``.

        Args:
            value (str): type of the opacity source

        Returns:
            OpacityOptions: class object

        Examples:
            >>> import torch
            >>> from pyharp.opacity import OpacityOptions
            >>> op = OpacityOptions().type('rfm-lbl')
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

class RFM:
    """
    Line-by-line absorption data computed by RFM.

    Examples:
        >>> import torch
        >>> from pyharp.opacity import RFM, OpacityOptions
        >>> op = RFM(OpacityOptions())
    """

    options: OpacityOptions

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: OpacityOptions) -> None:
        """
        Create a RFM instance.

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
        Calculate opacity using RFM line-by-line absorption data.

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
                The last dimension is the optical properties arranged
                in the order of attenuation [1/m], single scattering albedo and scattering phase function.
        """
        ...
