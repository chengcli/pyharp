#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! Can the cloud kernels evaluate these tensors?
/*!
 * The kernels read rows through raw pointers and carry no autograd graph, so
 * they take only float32/float64 CPU or CUDA tensors, all on one device, that
 * nothing is differentiating. The modules keep their tensor expressions for
 * everything else.
 */
bool cloud_kernels_apply(std::initializer_list<torch::Tensor> tensors);

//! Fill \p result (nwave, ncol, nlyr, 2 + nmom) with Fu ice properties.
/*!
 * \param table per-wavelength coefficient rows (nwave, fu::kWidth)
 * \param dge generalized effective size (ncol, nlyr) [um]
 * \param iwc ice water content (ncol, nlyr) [g/m^3]
 * \return uint8 (nwave, ncol, nlyr), 1 where the properties are unphysical
 */
torch::Tensor fu_ice_fill(torch::Tensor& result, torch::Tensor const& table,
                          torch::Tensor const& dge, torch::Tensor const& iwc,
                          bool delta_scale);

//! Fill \p result with the mix of the ice and liquid results, all
//! (nwave, ncol, nlyr, 2 + nmom).
void cloud_switch_fill(torch::Tensor& result, torch::Tensor const& ice,
                       torch::Tensor const& liquid);

}  // namespace harp
