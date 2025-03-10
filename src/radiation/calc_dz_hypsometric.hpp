#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! \brief Calculate the height between pressure levels using the hypsometric
//! equation
/*!
 * Formula:
 * \f[
 *    dz = -\frac{RT}{g} dlnp
 * \f]
 *
 * The last dimension is the vertical dimension. The other dimensions are
 * batched. If the last dimension of pressure is greater than the last dimension
 * of temperature by one, the pressure is interpreted as the pressure at the
 * layer boundaries (level). Otherwise, both the pressure and the temperature
 * are interpreted as at the layer centers.
 *
 * If both pressure and temperature are at the layer centers, the pressure is
 * interpolated to the layer boundaries using a second-order extrapolation
 * method.
 *
 * The nominal direction is from bottom to top, i.e. the first layer is at the
 * bottom.
 *
 * \param pres pressure [pa] at layers
 * \param temp temperature [K] at layers
 * \param g_ov_R gravity over specific gas constant [K/m] at layers
 */
torch::Tensor calc_dz_hypsometric(torch::Tensor pres, torch::Tensor temp,
                                  torch::Tensor g_ov_R);

}  // namespace harp
