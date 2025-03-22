// torch
#include <torch/extension.h>

// harp
#include <harp/math/interpolation.hpp>

namespace py = pybind11;

void bind_math(py::module &m) {
  m.def("interpn", &harp::interpn, R"(
      Multidimensional linear interpolation

      Parameters
      ----------
      query_coords : list of torch.Tensor
          Query coordinates
      coords : list of torch.Tensor
          Coordinate arrays, len = ndim, each tensor has shape (nx1,), (nx2,) ...
      lookup : torch.Tensor
          Lookup tensor (nx1, nx2, ..., nval)

      Examples
      --------
      >>> from pyharp import interpn
      )",
        py::arg("query"), py::arg("coords"), py::arg("lookup"),
        py::arg("extrapolate") = false);
}
