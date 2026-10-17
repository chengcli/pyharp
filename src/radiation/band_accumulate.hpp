#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! Add one opacity's optical properties into the band's.
/*!
 * \param prop band properties (nwave, ncol, nlyr, nprop), updated in place:
 *        extinction, extinction-weighted albedo, and albedo- and
 *        extinction-weighted phase moments
 * \param kdata the opacity's (nwave, ncol, nlyr, nprop_k), nprop_k <= nprop
 */
void accumulate_opacity(torch::Tensor& prop, torch::Tensor const& kdata);

//! Turn the band's summed properties into solver input, in place.
/*!
 * Averages the phase moments and the albedo, multiplies the extinction by
 * the layer thickness, and clamps the albedo to at most \p albedo_cap.
 * \param dz layer thickness, (nlyr,) or (ncol, nlyr)
 */
void finish_band_properties(torch::Tensor& prop, torch::Tensor const& dz,
                            double albedo_cap);

}  // namespace harp
