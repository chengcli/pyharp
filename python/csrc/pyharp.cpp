// torch
#include <torch/extension.h>

// disort
#include <disort/index.h>

// harp
#include <harp/radiation/radiation.hpp>
#include <harp/utils/find_resource.hpp>
#include <harp/utils/mean_molecular_weight.hpp>
#include <harp/utils/write_column_profile.hpp>
#include <harp/utils/write_spectral_profile.hpp>

namespace py = pybind11;

void bind_radiation(py::module& m);
void bind_opacity(py::module& m);
void bind_math(py::module& m);
void bind_constants(py::module& m);
void bind_integrator(py::module&);
void bind_rtsolver(py::module&);

PYBIND11_MODULE(pyharp, m) {
  m.attr("__name__") = "pyharp";

  m.attr("kIEX") = (int)disort::PropertyIndex::IEX;
  m.attr("kISS") = (int)disort::PropertyIndex::ISS;
  m.attr("kIPM") = (int)disort::PropertyIndex::IPM;

  m.attr("kIUP") = (int)disort::DirectionIndex::IUP;
  m.attr("kIDN") = (int)disort::DirectionIndex::IDN;

  bind_opacity(m);
  bind_radiation(m);
  bind_math(m);
  bind_constants(m);
  bind_integrator(m);
  bind_rtsolver(m);

  m.def("species_names",
        []() -> const std::vector<std::string>& { return harp::species_names; })
      .def("species_weights",
           []() -> const std::vector<double>& { return harp::species_weights; })
      .def(
          "set_search_paths",
          [](const std::string path) {
            strcpy(harp::search_paths, path.c_str());
            return harp::deserialize_search_paths(harp::search_paths);
          },
          py::arg("path"))
      .def("get_search_paths",
           []() { return harp::deserialize_search_paths(harp::search_paths); })
      .def(
          "add_resource_directory",
          [](const std::string path, bool prepend) {
            harp::add_resource_directory(path, prepend);
            return harp::deserialize_search_paths(harp::search_paths);
          },
          py::arg("path"), py::arg("prepend") = true)
      .def("find_resource", &harp::find_resource, py::arg("filename"))
      .def("mean_molecular_weight", &harp::mean_molecular_weight,
           py::arg("conc"))
      .def(
          "write_column_profile",
          [](const std::string& filename, torch::Tensor dz, torch::Tensor pres,
             torch::Tensor temp, torch::Tensor conc, torch::Tensor band_flux) {
            harp::write_column_profile(filename, dz, pres, temp, conc,
                                       band_flux);
          },
          py::arg("filename"), py::arg("dz"), py::arg("pres"), py::arg("temp"),
          py::arg("conc"), py::arg("band_flux"))
      .def(
          "write_spectral_profile",
          [](const std::string& filename, const std::vector<double>& wavenumber,
             const std::vector<std::pair<std::string, torch::Tensor>>&
                 transmittance,
             torch::Tensor flux_up_toa, torch::Tensor flux_down_boa) {
            harp::write_spectral_profile(filename, wavenumber, transmittance,
                                         flux_up_toa, flux_down_boa);
          },
          py::arg("filename"), py::arg("wavenumber"), py::arg("transmittance"),
          py::arg("flux_up_toa"), py::arg("flux_down_boa"));
}
