#pragma once

#include <torch/torch.h>

//! \brief Spherical correction by XIZ
/*!
 * xiz 2022 flux scaling so that the heating rate is the same as the
 * plane-parallel scheme volheating scaling: first calculate flux divergence
 * from DISORT using Plane-parallel in a cell then mulitpled by the cell
 * volume divided by dx1f then solve for F using F1*S1-F2*S2 = volheating
 * the top fluxes are the still the same as the plane-paralell values
 *
 * \param flx flux, shape (C, D, W, nlayer)
 * \param x1f cell interface, shape (D, W, nlayer + 1)
 * \param area cell area, shape (D, W, nlayer)
 * \param vol cell volume, shape (D, W, nlayer)
 */
torch::Tensor spherical_flux_correction(torch::Tensor flx, torch::Tensor x1f,
                                        torch::Tensor area, torch::Tensor vol);
