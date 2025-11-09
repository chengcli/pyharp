// torch
#include <torch/extension.h>

// harp
#include <harp/math/interpolation.hpp>

namespace py = pybind11;

void bind_math(py::module &m) {
  m.def("interpn", &harp::interpn, py::arg("query"), py::arg("coords"),
        py::arg("lookup"), py::arg("extrapolate") = false);
}
