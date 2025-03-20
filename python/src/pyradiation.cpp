// torch
#include <torch/extension.h>

// harp
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_radiation(py::module &m) {
  py::class_<harp::RadiationOptions>(m, "RadiationOptions", R"(
        Set radiation band options

        Returns
        -------
        RadiationOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().band_options(['band1', 'band2'])
        )")
      .def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationOptions &a) {
             return fmt::format("RadiationOptions{}", a);
           })

      .ADD_OPTION(std::string, harp::RadiationOptions, outdirs, R"(
        Set outgoing ray directions

        Parameters
        ----------
        outdirs : str
            outgoing ray directions

        Returns
        -------
        RadiationOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().outdirs('(0, 10), (0, 20)')
        >>> print(op)
        )")

      .ADD_OPTION(harp::RadiationBandDict, harp::RadiationOptions, band_options,
                  R"(
        Set radiation band options

        Parameters
        ----------
        band_options : dict
            radiation band options

        Returns
        -------
        RadiationOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().band_options({'band1': 'outdir1', 'band2': 'outdir2'})
        >>> print(op)
        )");

  ADD_HARP_MODULE(Radiation, RadiationOptions);

  py::class_<harp::RadiationBandOptions>(m, "RadiationBandOptions", R"(
        Set radiation options

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().name('band1').outdirs('outdir')
        )")
      .def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationBandOptions &a) {
             return fmt::format("RadiationBandOptions{}", a);
           })

      .ADD_OPTION(std::string, harp::RadiationBandOptions, name, R"(
        Set radiation band name

        Parameters
        ----------
        name : str
            radiation band name

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().name('band1')
        >>> print(op)
        )")

      .ADD_OPTION(std::string, harp::RadiationBandOptions, outdirs, R"(
        Set outgoing ray directions

        Parameters
        ----------
        outdirs : str
            outgoing ray directions

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().outdirs('(0, 10), (0, 20)')
        >>> print(op)
        )")

      .ADD_OPTION(std::string, harp::RadiationBandOptions, solver_name, R"(
        Set solver name

        Parameters
        ----------
        solver_name : str
            solver name

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().solver_name('disort')
        >>> print(op)
        )")

      .ADD_OPTION(disort::DisortOptions, harp::RadiationBandOptions, disort, R"(
        Set disort options

        Parameters
        ----------
        disort : DisortOptions
            disort options

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions, DisortOptions
        >>> op = RadiationBandOptions().disort(DisortOptions().nwave(10))
        >>> print(op)
        )")

      .ADD_OPTION(std::vector<double>, harp::RadiationBandOptions, ww, R"(
        Set wavelength, wavenumber or weights for a wave grid

        Parameters
        ----------
        ww : List[float]
            wavenumbers

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().ww([1.0, 2.0, 3.0])
        >>> print(op)
        )")

      .ADD_OPTION(std::string, harp::RadiationBandOptions, integration, R"(
        Set integration method

        Parameters
        ----------
        integration : str
            integration method

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().integration('simpson')
        >>> print(op)
        )")

      .ADD_OPTION(harp::AttenuatorDict, harp::RadiationBandOptions, opacities,
                  R"(
        Set opacities

        Parameters
        ----------
        opacities : dict
            opacities

        Returns
        -------
        RadiationBandOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationBandOptions
        >>> op = RadiationBandOptions().opacities({'band1': 'opacity1', 'band2': 'opacity2'})
        >>> print(op)
        )");

  ADD_HARP_MODULE(RadiationBand, RadiationBandOptions);
}
