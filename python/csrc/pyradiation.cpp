// torch
#include <torch/extension.h>

// harp
#include <harp/radiation/bbflux.hpp>
#include <harp/radiation/calc_dz_hypsometric.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_radiation(py::module &m) {
  m.def("bbflux_wavenumber",
        py::overload_cast<torch::Tensor, double, int>(&harp::bbflux_wavenumber),
        R"(
        Calculate blackbody flux using wavenumber

        Parameters
        ----------
        wave : torch.Tensor
            wavenumber [cm^-1]
        temp : float
            temperature [K]
        ncol : int, optional
            number of columns, default to 1

        Returns
        -------
        torch.Tensor
            blackbody flux [w/(m^2 cm^-1)]

        Examples
        --------
        >>> import torch
        >>> from pyharp import bbflux_wavenumber
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavenumber(wave, temp)
        )",
        py::arg("wave"), py::arg("temp"), py::arg("ncol") = 1);

  m.def("bbflux_wavenumber",
        py::overload_cast<double, double, torch::Tensor>(
            &harp::bbflux_wavenumber),
        R"(
        Calculate blackbody flux using wavenumber

        Parameters
        ----------
        wn1 : double
            wavenumber [cm^-1]
        wn2 : double
            temperature [K]
        temp: torch.Tensor
            number of columns, default to 1

        Returns
        -------
        torch.Tensor
            blackbody flux [w/(m^2 cm^-1)]

        Examples
        --------
        >>> import torch
        >>> from pyharp import bbflux_wavenumber
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavenumber(wave, temp)
        )",
        py::arg("wn1"), py::arg("wn2"), py::arg("temp") = 1);

  m.def("bbflux_wavelength", &harp::bbflux_wavelength, R"(
        Calculate blackbody flux using wavelength

        Parameters
        ----------
        wave : torch.Tensor
            wavelength [um]
        temp : float
            temperature [K]
        ncol : int, optional
            number of columns, default to 1

        Returns
        -------
        torch.Tensor
            blackbody flux [w/(m^2 um^-1)]

        Examples
        --------
        >>> from pyharp import bbflux_wavelength
        >>> wave = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = 300.0
        >>> flux = bbflux_wavelength(wave, temp)
        )",
        py::arg("wave"), py::arg("temp"), py::arg("ncol") = 1);

  m.def("calc_dz_hypsometric", &harp::calc_dz_hypsometric, R"(
        Calculate the height between pressure levels using the hypsometric equation

        Parameters
        ----------
        pres : torch.Tensor
            pressure [pa] at layers
        temp : torch.Tensor
            temperature [K] at layers
        g_ov_R : torch.Tensor
            gravity over specific gas constant [K/m] at layers

        Returns
        -------
        torch.Tensor
            height between pressure levels [m]

        Examples
        --------
        >>> from pyharp import calc_dz_hypsometric
        >>> pres = torch.tensor([1.0, 2.0, 3.0])
        >>> temp = torch.tensor([300.0, 310.0, 320.0])
        >>> g_ov_R = torch.tensor([1.0, 2.0, 3.0])
        >>> dz = calc_dz_hypsometric(pres, temp, g_ov_R)
        )",
        py::arg("pres"), py::arg("temp"), py::arg("g_ov_R"));

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

      .def_static("from_yaml", &harp::RadiationOptions::from_yaml, R"(
        Create a `RadiationOptions` object from a YAML file

        Parameters
        ----------
        filename : str
            YAML file name

        Returns
        -------
        RadiationOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions.from_yaml('radiation.yaml')
        )")

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

      .ADD_OPTION(harp::RadiationBandDict, harp::RadiationOptions, bands,
                  R"(
        Set radiation band options

        Parameters
        ----------
        bands : dict
            radiation band options

        Returns
        -------
        RadiationOptions object

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().bands({'band1': 'outdir1', 'band2': 'outdir2'})
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

      .def("query_waves", &harp::RadiationBandOptions::query_waves, R"(
        Query the spectral grids

        Returns
        -------
        List[float]
            spectral grids

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().query_waves()
        )")

      .def("query_weights", &harp::RadiationBandOptions::query_weights, R"(
        Query the weights

        Returns
        -------
        List[float]
            weights

        Examples
        --------
        >>> import torch
        >>> from pyharp import RadiationOptions
        >>> op = RadiationOptions().query_weights()
        )")

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
