// torch
#include <torch/extension.h>

// fmt
#include <fmt/format.h>

// harp
#include <harp/rtsolver/toon_mckay89.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_rtsolver(py::module &m) {
  auto pyToonMcKay89Options =
      py::class_<harp::ToonMcKay89OptionsImpl, harp::ToonMcKay89Options>(
          m, "ToonMcKay89Options");

  pyToonMcKay89Options.def(py::init<>())
      .def("__repr__",
           [](const harp::ToonMcKay89Options &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ToonMcKay89Options(\n{})", ss.str());
           })
      .ADD_OPTION(std::vector<double>, harp::ToonMcKay89OptionsImpl, wave_lower)
      .ADD_OPTION(std::vector<double>, harp::ToonMcKay89OptionsImpl, wave_upper)
      .ADD_OPTION(bool, harp::ToonMcKay89OptionsImpl, zenith_correction);

  torch::python::bind_module<harp::ToonMcKay89Impl>(m, "ToonMcKay89")
      .def(py::init<>())
      .def(py::init<harp::ToonMcKay89Options>(), py::arg("options"))
      .def_readonly("options", &harp::ToonMcKay89Impl::options)
      .def(
          "forward",
          [](harp::ToonMcKay89Impl &self, torch::Tensor prop, std::string bname,
             torch::optional<torch::Tensor> temf, const py::kwargs &kwargs) {
            // get bc from kwargs
            std::map<std::string, torch::Tensor> bc;
            for (auto item : kwargs) {
              auto key = py::cast<std::string>(item.first);
              auto value = py::cast<torch::Tensor>(item.second);
              bc.emplace(std::move(key), std::move(value));
            }

            for (auto &[key, value] : bc) {
              std::vector<std::string> items = {"fbeam", "albedo", "umu0"};
              // broadcast dimensions to (nwave, ncol)
              if (std::find(items.begin(), items.end(), key) != items.end()) {
                while (value.dim() < 2) {
                  value = value.unsqueeze(0);
                }
              }
            }

            // broadcast dimensions to (nwave, ncol, nlyr, nprop)
            while (prop.dim() < 4) {
              prop = prop.unsqueeze(0);
            }

            return self.forward(prop, &bc, bname, temf);
          },
          py::arg("prop"), py::arg("bname") = "", py::arg("temf") = py::none());
};
