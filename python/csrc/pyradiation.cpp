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

  m.def("bbflux_wavelength", &harp::bbflux_wavelength, py::arg("wave"),
        py::arg("temp"), py::arg("ncol") = 1);

  m.def("calc_dz_hypsometric", &harp::calc_dz_hypsometric, py::arg("pres"),
        py::arg("temp"), py::arg("g_ov_R"));

  auto pyRadiationBandOptions =
      py::class_<harp::RadiationBandOptionsImpl, harp::RadiationBandOptions>(
          m, "RadiationBandOptions");

  pyRadiationBandOptions.def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationBandOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("RadiationBandOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, harp::RadiationBandOptionsImpl, name)
      .ADD_OPTION(std::string, harp::RadiationBandOptionsImpl, outdirs)
      .ADD_OPTION(std::string, harp::RadiationBandOptionsImpl, solver_name)
      .ADD_OPTION(disort::DisortOptions, harp::RadiationBandOptionsImpl, disort)
      .ADD_OPTION(std::vector<double>, harp::RadiationBandOptionsImpl,
                  wavenumber)
      .ADD_OPTION(std::vector<double>, harp::RadiationBandOptionsImpl, weight)
      .ADD_OPTION(harp::OpacityDict, harp::RadiationBandOptionsImpl, opacities)
      .ADD_OPTION(int, harp::RadiationBandOptionsImpl, l2l_order)
      .ADD_OPTION(int, harp::RadiationBandOptionsImpl, nwave)
      .ADD_OPTION(int, harp::RadiationBandOptionsImpl, ncol)
      .ADD_OPTION(int, harp::RadiationBandOptionsImpl, nlyr)
      .ADD_OPTION(int, harp::RadiationBandOptionsImpl, nstr)
      .ADD_OPTION(bool, harp::RadiationBandOptionsImpl, verbose);

  auto pyRadiationOptions =
      py::class_<harp::RadiationOptionsImpl, harp::RadiationOptions>(
          m, "RadiationOptions");

  pyRadiationOptions.def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("RadiationOptions(\n{})", ss.str());
           })
      .def_static("from_yaml", &harp::RadiationOptionsImpl::from_yaml,
                  py::arg("filename"))
      .def("ncol", &harp::RadiationOptionsImpl::ncol)
      .def("nlyr", &harp::RadiationOptionsImpl::nlyr)
      .ADD_OPTION(std::string, harp::RadiationOptionsImpl, outdirs)
      .ADD_OPTION(std::vector<harp::RadiationBandOptions>,
                  harp::RadiationOptionsImpl, bands);

  ADD_HARP_MODULE(Radiation, RadiationOptions, py::arg("conc"), py::arg("dz"),
                  py::arg("bc"), py::arg("atm"));
  ADD_HARP_MODULE(RadiationBand, RadiationBandOptions, py::arg("conc"),
                  py::arg("dz"), py::arg("bc"), py::arg("atm"));
}
