// torch
#include <torch/extension.h>

// harp
#include <harp/math/interpolation.hpp>

namespace py = pybind11;

void bind_math(py::module &m) {
  m.def("interpn", &harp::interpn, R"doc(
Multidimensional linear interpolation

Args:
  query_coords (list[torch.Tensor]): Query coordinates
  coords (list[torch.Tensor]): Coordinate arrays, len = ndim, each tensor has shape (nx1,), (nx2,) ...
  lookup (torch.Tensor): Lookup tensor (nx1, nx2, ..., nval)

Examples:
  .. code-block:: python

    >>> import torch
    >>> from pyharp import interpn
    >>> query = [torch.tensor([0.5]), torch.tensor([0.5])]
    >>> coords = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
    >>> lookup = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> interpn(query, coords, lookup)
    tensor(2.5000)
      )doc",
        py::arg("query"), py::arg("coords"), py::arg("lookup"),
        py::arg("extrapolate") = false);
}
