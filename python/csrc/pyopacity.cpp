// torch
#include <torch/extension.h>

// harp
#include <harp/opacity/attenuator_options.hpp>
#include <harp/opacity/h2so4_simple.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/s8_fuller.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_opacity(py::module &m) {
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
    >>> from pyharp import AttenuatorOptions
    >>> op = AttenuatorOptions().band_options(['band1', 'band2'])
        )doc")

      .def("__repr__",
           [](const harp::AttenuatorOptions &a) {
             return fmt::format("AttenuatorOptions{}", a);
           })

      .ADD_OPTION(std::string, harp::AttenuatorOptions, type, R"doc(
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
        )doc")

      .ADD_OPTION(std::string, harp::AttenuatorOptions, bname, R"doc(
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
        )doc")

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions,
                  opacity_files, R"doc(
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
        )doc")

      .ADD_OPTION(std::vector<int>, harp::AttenuatorOptions, species_ids, R"doc(
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
        )doc");

  ADD_HARP_MODULE(S8Fuller, AttenuatorOptions, R"doc(
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
        )doc",
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(H2SO4Simple, AttenuatorOptions, R"doc(
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
    >>> from pyharp import RFM
    >>> op = RFM(AttenuatorOptions())
        )doc",
                  py::arg("conc"), py::arg("kwargs"));
}
