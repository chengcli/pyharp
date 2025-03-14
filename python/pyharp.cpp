// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// harp
#include <utils/find_resource.hpp>

namespace py = pybind11;

void bind_radiation(py::module &m);
void bind_opacity(py::module &m);

PYBIND11_MODULE(pyharp, m) {
  m.attr("__name__") = "pyharp";
  m.doc() = "Python bindings for harp module";

  bind_radiation(m);
  bind_opacity(m);

  m.def("load_configure", &YAML::LoadFile, R"(
      Load configuration from a YAML file.

      Parameters
      ----------
      arg0 : str
          The path to the YAML file.

      Returns
      -------
      YAML::Node
          The configuration.
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
