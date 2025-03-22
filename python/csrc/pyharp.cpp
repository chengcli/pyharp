// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  m.doc() = "Python bindings for harp module";

  bind_radiation(m);
  bind_opacity(m);
  bind_math(m);
  bind_constants(m);

  m.def(
      "shared",
      []() -> const std::unordered_map<std::string, torch::Tensor> & {
        return harp::shared;
      },
      R"(
      Shared readonly data between modules.

      The shared data is a dictionary of tensors, which may contain
      the following keys:
        - "radiation/<band_name>/total_flux": total flux in a band
        - "radiation/downward_flux": downward flux
        - "radiation/upward_flux": upward flux
      )");

  m.def(
      "set_search_paths",
      [](const std::string path) {
        strcpy(harp::search_paths, path.c_str());
        return harp::deserialize_search_paths(harp::search_paths);
      },
      R"(
        Set the search paths for resource files.

        Parameters
        ----------
        path : str
            The search paths.
      )");

  m.def("find_resource", &harp::find_resource, R"(
      Find a resource file.

      Parameters
      ----------
      filename : str
          The name of the resource file.

      Returns
      -------
      str
          The full path to the resource file.
      )");
}
