// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// yaml-cpp
#include <yaml-cpp/yaml.h>

namespace py = pybind11;

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

  m.def("find_resource", &find_resource, R"(
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
