// torch
#include <torch/extension.h>

// harp
#include <harp/radiation/radiation.hpp>
#include <harp/utils/find_resource.hpp>

namespace py = pybind11;

void bind_radiation(py::module &m);
void bind_opacity(py::module &m);
void bind_math(py::module &m);
void bind_constants(py::module &m);

PYBIND11_MODULE(pyharp, m) {
  m.attr("__name__") = "pyharp";

  bind_opacity(m);
  bind_radiation(m);
  bind_math(m);
  bind_constants(m);

  m.def(
      "species_names",
      []() -> const std::vector<std::string> & { return harp::species_names; });

  m.def(
      "species_weights",
      []() -> const std::vector<double> & { return harp::species_weights; });

  m.def(
      "shared",
      []() {
        return py::make_iterator(harp::shared.begin(), harp::shared.end());
      },
      py::keep_alive<0, 1>());

  m.def(
      "get_shared",
      [](const std::string &key) {
        auto it = harp::shared.find(key);
        if (it == harp::shared.end()) {
          throw std::runtime_error("Key not found in shared data");
        }
        return it->second;
      },
      py::arg("key"));

  m.def(
      "set_search_paths",
      [](const std::string path) {
        strcpy(harp::search_paths, path.c_str());
        return harp::deserialize_search_paths(harp::search_paths);
      },
      py::arg("path"));

  m.def(
      "get_search_paths",
      []() { return harp::deserialize_search_paths(harp::search_paths); });

  m.def(
      "add_resource_directory",
      [](const std::string path, bool prepend) {
        harp::add_resource_directory(path, prepend);
        return harp::deserialize_search_paths(harp::search_paths);
      },
      py::arg("path"), py::arg("prepend") = true);

  m.def("find_resource", &harp::find_resource,
        py::arg("filename"));
}
