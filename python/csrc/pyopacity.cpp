// torch
#include <torch/extension.h>

// harp
#include <harp/opacity/attenuator_options.hpp>
#include <harp/opacity/fourcolumn.hpp>
#include <harp/opacity/jit_opacity.hpp>
#include <harp/opacity/multiband.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/wavetemp.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_opacity(py::module &parent) {
  auto m = parent.def_submodule("opacity", "Opacity module");

  auto pyAttenuatorOptions =
      py::class_<harp::AttenuatorOptions>(m, "AttenuatorOptions");

  pyAttenuatorOptions
      .def(py::init<>(), R"doc(
Set opacity band options

Returns:
  pyharp.AttenuatorOptions: class object

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().band_options(['band1', 'band2'])
        )doc")

      .def("__repr__",
           [](const harp::AttenuatorOptions &a) {
             return fmt::format("AttenuatorOptions{}", a);
           })

      .ADD_OPTION(std::string, harp::AttenuatorOptions, type, R"doc(
Set or get the type of the opacity source format

Valid options are, ``jit``, ``rfm-lbl``, ``rfm-ck``, ``four-column``, ``wavetemp``, ``multiband``.
See :ref:`opacity_choices` for more details.

Args:
  type (str): type of the opacity source

Returns:
  AttenuatorOptions | str : class object if argument is not empty, otherwise the type

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().type('rfm-lbl')
    >>> print(op)
        )doc")

      .ADD_OPTION(std::string, harp::AttenuatorOptions, bname, R"doc(
Set or get the name of the band that the opacity is associated with

Args:
  bname (str): name of the band that the opacity is associated with

Returns:
  AttenuatorOptions | str : class object if argument is not empty, otherwise the band name

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().bname('band1')
        )doc")

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions,
                  opacity_files, R"doc(
Set or get the list of opacity data files

Args:
  opacity_files (list): list of opacity data files

Returns:
  AttenuatorOptions | list[str]: class object if argument is not empty, otherwise the list of opacity data files

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
        )doc")

      .ADD_OPTION(std::vector<int>, harp::AttenuatorOptions, species_ids, R"doc(
Set or get the list of dependent species indices

Args:
  species_ids (list[int]): list of dependent species indices

Returns:
  AttenuatorOptions | list[int]: class object if argument is not empty, otherwise the list of dependent species indices

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().species_ids([1, 2])
        )doc")

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions, jit_kwargs,
                  R"doc(
Set or get the list of kwargs to pass to the JIT module

Args:
  jit_kwargs (list[str]): list of kwargs to pass to the JIT module

Returns:
  AttenuatorOptions | list[str]: class object if argument is not empty, otherwise the list of kwargs to pass to the JIT module

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().jit_kwargs(['temp', 'wavelength'])
    >>> print(op.jit_kwargs())
        )doc")

      .ADD_OPTION(std::vector<double>, harp::AttenuatorOptions, fractions,
                  R"doc(
Set or get fractions of species in cia calculatioin

Args:
  fractions (list[float]): list of species fractions

Returns:
  AttenuatorOptions | list[float]: class object if argument is not empty, otherwise the list of species fractions

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import AttenuatorOptions
    >>> op = AttenuatorOptions().fractions([0.9, 0.1])
        )doc");

  ADD_HARP_MODULE(JITOpacity, AttenuatorOptions, R"doc(
JIT opacity model

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
        )doc",
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(WaveTemp, AttenuatorOptions, R"doc(
Wave-Temp opacity data

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

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import WaveTemp, AttenuatorOptions
    >>> op = MultiBand(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(MultiBand, AttenuatorOptions, R"doc(
Multi-band opacity data

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

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import MultiBand, AttenuatorOptions
    >>> op = MultiBand(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(FourColumn, AttenuatorOptions, R"doc(
Four-column opacity data

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

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import FourColumn, AttenuatorOptions
    >>> op = FourColumn(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(RFM, AttenuatorOptions, R"doc(
Line-by-line absorption data computed by RFM

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


Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp.opacity import RFM, AttenuatorOptions
    >>> op = RFM(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));
}
