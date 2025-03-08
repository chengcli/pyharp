// harp
#include "trapezoid.hpp"

namespace harp {

torch::Tensor trapezoid(torch::Tensor y, torch::Tensor x, int dim) {
  // Check that the input tensor x is 1D
  if (x.dim() != 1) {
    TORCH_CHECK(false, "Input tensor x must be 1D, got ", x.dim(), "D");
  }

  // Check that the input tensors have the same size
  if (x.size(0) != y.size(dim)) {
    TORCH_CHECK(false, "Input tensors must have the same size, got ", x.size(0),
                " and ", y.size(dim));
  }

  int nx = x.size(0);

  // Compute the differences between adjacent elements in the tensor
  auto dx = x.slice(0, 1, nx) - x.slice(0, 0, nx - 1);

  // broadcast to the same shape as y
  auto vec = y.sizes().vec();
  if (dim == -1) dim = vec.size() - 1;

  for (int i = 0; i < vec.size(); ++i)
    if (i != dim) vec[i] = 1;
  vec[dim] = nx - 1;
  dx = dx.view(vec);

  // Compute the integral using trapezoidal rule
  return (0.5 * dx * (y.slice(dim, 1, nx) + y.slice(dim, 0, nx - 1))).sum(dim);
}

}  // namespace harp
