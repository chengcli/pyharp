// torch
#include <torch/extension.h>

// fvm
#include <radiation/radiation.hpp>
#include <radiation/radiation_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_radiation(py::module &m) {
  py::class_<harp::RadiationOptions>(m, "RadiationOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationOptions &a) {
             return fmt::format("RadiationOptions{}", a);
           })
      .ADD_OPTION(std::string, harp::RadiationOptions, outdirs)
      .ADD_OPTION(harp::RadiationBandDict, harp::RadiationOptions,
                  band_options);

  ADD_HARP_MODULE(Radiation, RadiationOptions);

  py::class_<harp::RadiationBandOptions>(m, "RadiationBandOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const harp::RadiationBandOptions &a) {
             return fmt::format("RadiationBandOptions{}", a);
           })
      .ADD_OPTION(std::string, harp::RadiationBandOptions, name)
      .ADD_OPTION(std::string, harp::RadiationBandOptions, outdirs)
      .ADD_OPTION(std::string, harp::RadiationBandOptions, solver_name)
      .ADD_OPTION(disort::DisortOptions, harp::RadiationBandOptions, disort)
      .ADD_OPTION(std::vector<double>, harp::RadiationBandOptions, ww)
      .ADD_OPTION(std::string, harp::RadiationBandOptions, integration)
      .ADD_OPTION(harp::AttenuatorDict, harp::RadiationBandOptions, opacities);

  ADD_HARP_MODULE(RadiationBand, RadiationBandOptions);
}
