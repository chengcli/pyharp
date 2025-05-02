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
  py::class_<harp::AttenuatorOptions>(m, "AttenuatorOptions", R"(
        Set opacity band options

        Returns:
          AttenuatorOptions object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().band_options(['band1', 'band2'])
        )")
      .def(py::init<>())
      .def("__repr__",
           [](const harp::AttenuatorOptions &a) {
             return fmt::format("AttenuatorOptions{}", a);
           })

      .ADD_OPTION(std::string, harp::AttenuatorOptions, type, R"(
        Set the type of the opacity source

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
          AttenuatorOptions object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().type('rfm-lbl')
            >>> print(op)
        )")

      .ADD_OPTION(std::string, harp::AttenuatorOptions, bname, R"(
        Set or get the name of the band that the opacity is associated with

        Args:
          bname (str): name of the band that the opacity is associated with

        Returns:
          AttenuatorOptions object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().bname('band1')
        )")

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions,
                  opacity_files, R"(
        Set or get the list of opacity data files

        Args:
          opacity_files (list): list of opacity data files

        Returns:
          AttenuatorOptions object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
        )")

      .ADD_OPTION(std::vector<int>, harp::AttenuatorOptions, species_ids, R"(
        Set or get the list of dependent species indices

        Args:
          species_ids (list): list of dependent species indices

        Returns:
          AttenuatorOptions object

        Examples:
          .. code-block:: python

            >>> import torch
            >>> from pyharp import AttenuatorOptions
            >>> op = AttenuatorOptions().species_ids([1, 2])
        )");

  ADD_HARP_MODULE(S8Fuller, AttenuatorOptions);
  ADD_HARP_MODULE(H2SO4Simple, AttenuatorOptions);
  ADD_HARP_MODULE(RFM, AttenuatorOptions);
}
