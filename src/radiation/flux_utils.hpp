// torch
#include <torch/torch.h>

// harp
#include <index.h>

#include <math/trapezoid.hpp>

namespace harp {

//! \brief Calculate total flux
/*!
 * Formula 1:
 * \f[
 *    \int_{a}^{b} F(\lambda) d\lambda = \sum_{i=0}^{n-1} \frac{F(\lambda_i) +
 *    F(\lambda_{i+1})}{2} (\lambda_{i+1} - \lambda_i)
 * \f]
 *
 * Formula 2:
 * \f[
 *    \int_{a}^{b} F(\lambda) d\lambda = \sum_{i=0}^{n-1} F(\lambda_i) w_i
 * \f]
 *
 * This function calculates the total flux by integrating the flux over the
 * given wave grid or using specified weights.
 *
 * The first argument is a multi-dimensional spectral flux tensor. Its first
 * dimension is the spectral (wavelength,wavenumber) dimension.
 *
 * The second argument must be a 1D tensor.
 * It can be either the wavelength(number) grid or a 1D weight tensor.
 *
 * The third argument is a string that specifies the type of the second
 * argument. It can be either "wave" or "weight". An error will be raised if the
 * input is not one of these two.
 *
 * \param flux The flux tensor
 * \param wave_or_weight The wavelength or weight tensor
 * \param input The type of the second argument, either "wave" or "weight"
 * \return The integrated total flux
 */
torch::Tensor cal_total_flux(torch::Tensor flux, torch::Tensor wave_or_weight,
                             std::string input) {
  // Check 1D tensor
  TORCH_CHECK(wave_or_weight.dim() == 1, "wave_or_weight must be 1D tensor");

  if (input == "wave") {
    return trapezoid(flux, wave_or_weight, /*dim=*/0);
  } else if (input == "weight") {
    return (flux * wave_or_weight.unsqueeze(1).expand_as(flux)).sum(0);
  } else {
    TORCH_CHECK(false, "input must be either 'wave' or 'weight'");
  }
}

//! \brief Calculate net flux
/*!
 * Formula:
 * \f[
 *   F_{\text{net}} = F_{\text{up}} - F_{\text{down}}
 * \f]
 *
 * The last dimension of the flux tensor must be 2.
 * The first element of the last dimension is the upward flux,
 * and the second element is the downward flux.
 *
 * The net flux is calculated by subtracting the downward flux from the upward
 * flux.
 *
 * \param flux The flux tensor
 * \return The net flux
 */
torch::Tensor cal_net_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IUP) - flux.select(-1, index::IDN);
}

//! \brief Calculate surface flux
/*!
 * Formula:
 * \f[
 *   F_{\text{surface}} = F_{\text{down}}\vert_{z=0}
 * \f]
 *
 * The surface flux is calculated by selecting the downward flux
 * (first element) from the flux tensor.
 *
 * The flux tensor must have at least 2 dimensions
 * and the last dimension must be 2.
 *
 * \param flux The flux tensor
 * \return The surface flux
 */
torch::Tensor cal_surface_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IDN).select(-1, 0);
}

//! \brief Calculate top of atmosphere flux
/*!
 * Formula:
 * \f[
 *   F_{\text{toa}} = F_{\text{up}}\vert_{z=\infty}
 * \f]
 *
 * The top of atmosphere flux is calculated by selecting the upward flux
 * (last element) from the flux tensor.
 *
 * The flux tensor must have at least 2 dimensions
 * and the last dimension must be 2.
 *
 * \param flux The flux tensor
 * \return The top of atmosphere flux
 */
torch::Tensor cal_toa_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IUP).select(-1, -1);
}

}  // namespace harp
