from __future__ import annotations
import pyharp
import torch
import torch._C.cpp
import torch._C.cpp.nn
import typing
__all__ = ['H2SO4Simple', 'RFM', 'Radiation', 'RadiationBand', 'S8Fuller']
class H2SO4Simple(torch._C.cpp.nn.Module):
    def __call__(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a new default module.
        """
    @typing.overload
    def __init__(self, options: pyharp.AttenuatorOptions) -> None:
        """
        Construct a H2SO4Simple module
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def buffer(self, arg0: str) -> torch.Tensor:
        ...
    def buffers(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    def children(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def clone(self) -> torch._C.cpp.nn.Module:
        ...
    def cpu(self) -> None:
        ...
    def cuda(self) -> None:
        ...
    def double(self) -> None:
        ...
    def eval(self) -> None:
        ...
    def float(self) -> None:
        ...
    @typing.overload
    def forward(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        H2SO4 absorption data from the simple model

        Args:
          conc (torch.Tensor)
            concentration of the species in mol/cm^3

          kwargs (dict[str, torch.Tensor])
            keyword arguments.
            Either 'wavelength' or 'wavenumber' must be provided
            if 'wavelength' is provided, the unit is nm.
            if 'wavenumber' is provided, the unit is cm^-1.

        Returns:
          torch.Tensor:
            attenuation [1/m], single scattering albedo and scattering phase function.
            The shape of the output tensor is (nwave, ncol, nlyr, 2 + nmom)
            where nwave is the number of wavelengths,
            ncol is the number of columns,
            nlyr is the number of layers,
            2 is for attenuation and scattering coefficients,
            and nmom is the number of moments.

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import H2SO4Simple
            >>> op = H2SO4Simple(AttenuatorOptions())
        """
    def half(self) -> None:
        ...
    def module(self, arg0: str) -> torch._C.cpp.nn.Module:
        ...
    def modules(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def named_buffers(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def named_children(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_modules(self, memo: typing.Any = None, prefix: str = '', remove_duplicate: bool = True) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_parameters(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def parameters(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    @typing.overload
    def to(self, dtype_or_device: typing.Any, non_blocking: bool = False) -> None:
        ...
    @typing.overload
    def to(self, device: typing.Any, dtype: typing.Any, non_blocking: bool = False) -> None:
        ...
    def train(self, mode: bool = True) -> None:
        ...
    def zero_grad(self) -> None:
        ...
    @property
    def _buffers(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def _modules(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    @property
    def _parameters(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def options(self) -> pyharp.AttenuatorOptions:
        ...
    @property
    def training(self) -> bool:
        ...
class RFM(torch._C.cpp.nn.Module):
    def __call__(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a new default module.
        """
    @typing.overload
    def __init__(self, options: pyharp.AttenuatorOptions) -> None:
        """
        Construct a RFM module
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def buffer(self, arg0: str) -> torch.Tensor:
        ...
    def buffers(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    def children(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def clone(self) -> torch._C.cpp.nn.Module:
        ...
    def cpu(self) -> None:
        ...
    def cuda(self) -> None:
        ...
    def double(self) -> None:
        ...
    def eval(self) -> None:
        ...
    def float(self) -> None:
        ...
    @typing.overload
    def forward(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Line-by-line absorption data computed by RFM

        Args:
          conc (torch.Tensor): concentration of the species in mol/cm^3
          kwargs (dict[str, torch.Tensor]): keyword arguments
            Either 'wavelength' or 'wavenumber' must be provided
            if 'wavelength' is provided, the unit is nm.
            if 'wavenumber' is provided, the unit is cm^-1.

        Returns:
          torch.Tensor: attenuation [1/m], single scattering albedo and scattering phase function
            The shape of the output tensor is (nwave, ncol, nlyr, 2 + nmom)
            where nwave is the number of wavelengths,
            ncol is the number of columns,
            nlyr is the number of layers,
            2 is for attenuation and scattering coefficients,
            and nmom is the number of moments.

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RFM
            >>> op = RFM(AttenuatorOptions())
        """
    def half(self) -> None:
        ...
    def module(self, arg0: str) -> torch._C.cpp.nn.Module:
        ...
    def modules(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def named_buffers(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def named_children(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_modules(self, memo: typing.Any = None, prefix: str = '', remove_duplicate: bool = True) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_parameters(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def parameters(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    @typing.overload
    def to(self, dtype_or_device: typing.Any, non_blocking: bool = False) -> None:
        ...
    @typing.overload
    def to(self, device: typing.Any, dtype: typing.Any, non_blocking: bool = False) -> None:
        ...
    def train(self, mode: bool = True) -> None:
        ...
    def zero_grad(self) -> None:
        ...
    @property
    def _buffers(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def _modules(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    @property
    def _parameters(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def options(self) -> pyharp.AttenuatorOptions:
        ...
    @property
    def training(self) -> bool:
        ...
class Radiation(torch._C.cpp.nn.Module):
    def __call__(self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: dict[str, torch.Tensor], arg3: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a new default module.
        """
    @typing.overload
    def __init__(self, options: pyharp.RadiationOptions) -> None:
        """
        Construct a Radiation module
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def buffer(self, arg0: str) -> torch.Tensor:
        ...
    def buffers(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    def children(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def clone(self) -> torch._C.cpp.nn.Module:
        ...
    def cpu(self) -> None:
        ...
    def cuda(self) -> None:
        ...
    def double(self) -> None:
        ...
    def eval(self) -> None:
        ...
    def float(self) -> None:
        ...
    @typing.overload
    def forward(self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: dict[str, torch.Tensor], arg3: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def forward(self, conc: torch.Tensor, dz: torch.Tensor, bc: dict[str, torch.Tensor], kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the net radiation flux

        Args:
          conc (torch.Tensor): concentration [mol/m^3]
          dz (torch.Tensor): height [m]
          bc (dict[str, torch.Tensor]): boundary conditions
          kwargs (dict[str, torch.Tensor]): additional arguments

        Returns:
          torch.Tensor: net flux [w/m^2]

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationOptions
            >>> op = RadiationOptions().band_options(['band1', 'band2'])
        """
    def half(self) -> None:
        ...
    def module(self, arg0: str) -> torch._C.cpp.nn.Module:
        ...
    def modules(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def named_buffers(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def named_children(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_modules(self, memo: typing.Any = None, prefix: str = '', remove_duplicate: bool = True) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_parameters(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def parameters(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    @typing.overload
    def to(self, dtype_or_device: typing.Any, non_blocking: bool = False) -> None:
        ...
    @typing.overload
    def to(self, device: typing.Any, dtype: typing.Any, non_blocking: bool = False) -> None:
        ...
    def train(self, mode: bool = True) -> None:
        ...
    def zero_grad(self) -> None:
        ...
    @property
    def _buffers(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def _modules(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    @property
    def _parameters(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def options(self) -> pyharp.RadiationOptions:
        ...
    @property
    def training(self) -> bool:
        ...
class RadiationBand(torch._C.cpp.nn.Module):
    def __call__(self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: dict[str, torch.Tensor], arg3: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a new default module.
        """
    @typing.overload
    def __init__(self, options: pyharp.RadiationBandOptions) -> None:
        """
        Construct a RadiationBand module
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def buffer(self, arg0: str) -> torch.Tensor:
        ...
    def buffers(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    def children(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def clone(self) -> torch._C.cpp.nn.Module:
        ...
    def cpu(self) -> None:
        ...
    def cuda(self) -> None:
        ...
    def double(self) -> None:
        ...
    def eval(self) -> None:
        ...
    def float(self) -> None:
        ...
    @typing.overload
    def forward(self, arg0: torch.Tensor, arg1: torch.Tensor, arg2: dict[str, torch.Tensor], arg3: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def forward(self, conc: torch.Tensor, dz: torch.Tensor, bc: dict[str, torch.Tensor], kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the net radiation flux for a band

        Args:
          conc (torch.Tensor): concentration [mol/m^3]
          dz (torch.Tensor): height [m]
          bc (dict[str, torch.Tensor]): boundary conditions
          kwargs (dict[str, torch.Tensor]): additional arguments

        Returns:
          torch.Tensor: [W/m^2] (ncol, nlyr+1)

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import RadiationBandOptions
            >>> op = RadiationBandOptions().band_options(['band1', 'band2'])
        """
    def half(self) -> None:
        ...
    def module(self, arg0: str) -> torch._C.cpp.nn.Module:
        ...
    def modules(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def named_buffers(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def named_children(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_modules(self, memo: typing.Any = None, prefix: str = '', remove_duplicate: bool = True) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_parameters(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def parameters(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    @typing.overload
    def to(self, dtype_or_device: typing.Any, non_blocking: bool = False) -> None:
        ...
    @typing.overload
    def to(self, device: typing.Any, dtype: typing.Any, non_blocking: bool = False) -> None:
        ...
    def train(self, mode: bool = True) -> None:
        ...
    def zero_grad(self) -> None:
        ...
    @property
    def _buffers(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def _modules(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    @property
    def _parameters(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def options(self) -> pyharp.RadiationBandOptions:
        ...
    @property
    def training(self) -> bool:
        ...
class S8Fuller(torch._C.cpp.nn.Module):
    def __call__(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a new default module.
        """
    @typing.overload
    def __init__(self, options: pyharp.AttenuatorOptions) -> None:
        """
        Construct a S8Fuller module
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def buffer(self, arg0: str) -> torch.Tensor:
        ...
    def buffers(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    def children(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def clone(self) -> torch._C.cpp.nn.Module:
        ...
    def cpu(self) -> None:
        ...
    def cuda(self) -> None:
        ...
    def double(self) -> None:
        ...
    def eval(self) -> None:
        ...
    def float(self) -> None:
        ...
    @typing.overload
    def forward(self, arg0: torch.Tensor, arg1: dict[str, torch.Tensor]) -> torch.Tensor:
        ...
    @typing.overload
    def forward(self, conc: torch.Tensor, kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        S8 absorption data from Fuller et al. (1987)

        Args:
          conc (torch.Tensor): concentration of the species in mol/cm^3

          kwargs (dict[str, torch.Tensor]): keyword arguments.
            Either 'wavelength' or 'wavenumber' must be provided
            if 'wavelength' is provided, the unit is nm.
            if 'wavenumber' is provided, the unit is cm^-1.

        Returns:
          torch.Tensor:
            attenuation [1/m], single scattering albedo and scattering phase function
            The shape of the output tensor is (nwave, ncol, nlyr, 2 + nmom)
            where nwave is the number of wavelengths,
            ncol is the number of columns,
            nlyr is the number of layers,
            2 is for attenuation and scattering coefficients,
            and nmom is the number of moments.

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import S8Fuller
            >>> op = S8Fuller(AttenuatorOptions())
        """
    def half(self) -> None:
        ...
    def module(self, arg0: str) -> torch._C.cpp.nn.Module:
        ...
    def modules(self) -> list[torch._C.cpp.nn.Module]:
        ...
    def named_buffers(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def named_children(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_modules(self, memo: typing.Any = None, prefix: str = '', remove_duplicate: bool = True) -> torch._C.cpp.OrderedModuleDict:
        ...
    def named_parameters(self, recurse: bool = True) -> torch._C.cpp.OrderedTensorDict:
        ...
    def parameters(self, recurse: bool = True) -> list[torch.Tensor]:
        ...
    @typing.overload
    def to(self, dtype_or_device: typing.Any, non_blocking: bool = False) -> None:
        ...
    @typing.overload
    def to(self, device: typing.Any, dtype: typing.Any, non_blocking: bool = False) -> None:
        ...
    def train(self, mode: bool = True) -> None:
        ...
    def zero_grad(self) -> None:
        ...
    @property
    def _buffers(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def _modules(self) -> torch._C.cpp.OrderedModuleDict:
        ...
    @property
    def _parameters(self) -> torch._C.cpp.OrderedTensorDict:
        ...
    @property
    def options(self) -> pyharp.AttenuatorOptions:
        ...
    @property
    def training(self) -> bool:
        ...
