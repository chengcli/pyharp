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
        py::arg("wave"), py::arg("temp"), py::arg("ncol") = 1);

  m.def("bbflux_wavenumber",
        py::overload_cast<double, double, torch::Tensor>(
            &harp::bbflux_wavenumber),
        py::arg("wn1"), py::arg("wn2"), py::arg("temp") = 1);

  m.def("bbflux_wavelength", &harp::bbflux_wavelength,
        py::arg("wave"), py::arg("temp"), py::arg("ncol") = 1);

  m.def("calc_dz_hypsometric", &harp::calc_dz_hypsometric,
        py::arg("pres"), py::arg("temp"), py::arg("g_ov_R"));

  auto pyRadiationBandOptions =
      py::class_<harp::RadiationBandOptions>(m, "RadiationBandOptions");

  pyRadiationBandOptions
      .def(py::init<>())

      .def("__repr__",
           [](const harp::RadiationBandOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("RadiationBandOptions(\n{})", ss.str());
           })

      .def("query_waves", &harp::RadiationBandOptions::query_waves,
           py::arg("op_name") = "")

      .def("query_weights", &harp::RadiationBandOptions::query_weights,
           py::arg("op_name") = "")

      .ADD_OPTION(std::string, harp::RadiationBandOptions, name)

      .ADD_OPTION(std::string, harp::RadiationBandOptions, outdirs)

      .ADD_OPTION(std::string, harp::RadiationBandOptions, solver_name)

      .ADD_OPTION(disort::DisortOptions, harp::RadiationBandOptions, disort)

      .ADD_OPTION(std::vector<double>, harp::RadiationBandOptions, ww)

      .ADD_OPTION(std::string, harp::RadiationBandOptions, integration)

      .ADD_OPTION(harp::AttenuatorDict, harp::RadiationBandOptions, opacities);

  auto pyRadiationOptions =
      py::class_<harp::RadiationOptions>(m, "RadiationOptions");

  pyRadiationOptions
      .def(py::init<>())

      .def("__repr__",
           [](const harp::RadiationOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("RadiationOptions(\n{})", ss.str());
           })

      .def_static("from_yaml", &harp::RadiationOptions::from_yaml,
                  py::arg("filename"))

      .ADD_OPTION(std::string, harp::RadiationOptions, outdirs)

      .ADD_OPTION(harp::RadiationBandDict, harp::RadiationOptions, bands);

  ADD_HARP_MODULE(Radiation, RadiationOptions,
                  py::arg("conc"), py::arg("dz"), py::arg("bc"),
                  py::arg("kwargs"));

  ADD_HARP_MODULE(RadiationBand, RadiationBandOptions,
                  py::arg("conc"), py::arg("dz"), py::arg("bc"),
                  py::arg("kwargs"));
}
