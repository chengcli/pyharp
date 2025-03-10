#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! \brief Integrate a function using the trapezoidal rule
/*!
 * Formula:
 * \f[
 *   \int_{a}^{b} f(x) dx \approx \sum_{i=1}^{N-1} \frac{(x_{i+1}-x_i)}{2}
 *    (f(x_{i+1}) + f(x_i))
 * \f]
 *
 * This function uses the trapezoidal rule to integrate a quantity over
 * a specified dimension.
 * y is the function values at the grid points, x.
 *
 * If unspecified, the last dimension of the input tensors
 * is integrated over. The other dimensions are batched.
 *
 * The grid point tensor x must be 1D.
 *
 * \param y The function values at the grid points
 * \param x The discretized grid points
 * \param dim The dimension to integrate over
 * \return The integrated quantity
 */
torch::Tensor trapezoid(torch::Tensor y, torch::Tensor x, int dim = -1);

}  // namespace harp
