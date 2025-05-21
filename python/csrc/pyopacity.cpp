// torch
#include <torch/extension.h>

// harp
#include <harp/opacity/attenuator_options.hpp>
#include <harp/opacity/fourcolumn.hpp>
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

Valid options are:
  .. list-table::
    :widths: 15 25
    :header-rows: 1

    * - Key
      - Description
    * - 'rfm-lbl'
      - Line-by-line absorption data computed by RFM
    * - 'rfm-ck'
      - Correlated-k absorption computed from line-by-line data
    * - 'multiband-ck'
      - Multi-band opacity data from saved torch ".pt" file
    * - 'fourcolumn'
      - Four-column opacity data (wavelength [um]/wavenumber [cm^{-1}], cross-section [m^2/kg], ssa, g)

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

  ADD_HARP_MODULE(WaveTemp, AttenuatorOptions, R"doc(
Wave-Temp opacity data

Args:
  conc (torch.Tensor): concentration of the species in mol/cm^3

  kwargs (dict[str, torch.Tensor]): keyword arguments.
    Both 'temp' [k] and ('wavenumber' [cm^{-1}] or 'wavelength' [num]) must be provided

Returns:
  torch.Tensor:
    attenuation [1/m], single scattering albedo and scattering phase function
    The shape of the output tensor is (nwave, ncol, nlyr, 1)
    where nwave is the number of wavelengths,
    ncol is the number of columns,
    nlyr is the number of layers,
    1 is for attenuation coefficients,
    and nmom is the number of scattering moments.

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
  conc (torch.Tensor): concentration of the species in mol/cm^3

  kwargs (dict[str, torch.Tensor]): keyword arguments.
    Both 'temp' [k] and 'pres' [pa] must be provided

Returns:
  torch.Tensor:
    attenuation [1/m], single scattering albedo and scattering phase function
    The shape of the output tensor is (nwave, ncol, nlyr, 1)
    where nwave is the number of wavelengths,
    ncol is the number of columns,
    nlyr is the number of layers,
    1 is for attenuation coefficients,
    and nmom is the number of scattering moments.

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
  conc (torch.Tensor): concentration of the species in mol/cm^3

  kwargs (dict[str, torch.Tensor]): keyword arguments.
    Either 'wavelength' or 'wavenumber' must be provided
    if 'wavelength' is provided, the unit is um.
    if 'wavenumber' is provided, the unit is cm^-1.

Returns:
  torch.Tensor:
    attenuation [1/m], single scattering albedo and scattering phase function
    The shape of the output tensor is (nwave, ncol, nlyr, 2 + nmom)
    where nwave is the number of wavelengths,
    ncol is the number of columns,
    nlyr is the number of layers,
    2 is for attenuation and scattering coefficients,
    and nmom is the number of scattering moments.

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
    >>> from pyharp.opacity import RFM, AttenuatorOptions
    >>> op = RFM(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));
}
