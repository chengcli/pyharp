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

        Returns
        -------
        AttenuatorOptions object

        Examples
        --------
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
          - 's8_fuller'
          - 'h2so4_simple'
          - 'rfm-lbl'
          - 'rfm-ck'

        Parameters
        ----------
        type : str
            type of the opacity source

        Returns
        -------
        AttenuatorOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import AttenuatorOptions
        >>> op = AttenuatorOptions().type('rfm-lbl')
        >>> print(op)
        )")

      .ADD_OPTION(std::string, harp::AttenuatorOptions, bname, R"(
        Set the name of the band that the opacity is associated with

        Parameters
        ----------
        bname : str
            name of the band that the opacity is associated with

        Returns
        -------
        AttenuatorOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import AttenuatorOptions
        >>> op = AttenuatorOptions().bname('band1')
        )")

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions,
                  opacity_files, R"(
        Set the list of opacity data files

        Parameters
        ----------
        opacity_files : list
            list of opacity data files

        Returns
        -------
        AttenuatorOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import AttenuatorOptions
        >>> op = AttenuatorOptions().opacity_files(['file1', 'file2'])
        )")

      .ADD_OPTION(std::vector<int>, harp::AttenuatorOptions, species_ids, R"(
        Set the list of dependent species indices

        Parameters
        ----------
        species_ids : list
            list of dependent species indices

        Returns
        -------
        AttenuatorOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import AttenuatorOptions
        >>> op = AttenuatorOptions().species_ids([1, 2])
        )");

  ADD_HARP_MODULE(S8Fuller, AttenuatorOptions);
  ADD_HARP_MODULE(H2SO4Simple, AttenuatorOptions);
  ADD_HARP_MODULE(RFM, AttenuatorOptions);
}
