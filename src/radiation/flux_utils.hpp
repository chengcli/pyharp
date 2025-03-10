// torch
#include <torch/torch.h>

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
 * argument. It can be either "wavelength", "wavenumber" or "weight".
 * An error will be raised if the input is not one of these.
 *
 * \param flux The flux tensor
 * \param wave_or_weight The wavelength or weight tensor
 * \param input The type of the second argument, either "wave" or "weight"
 * \return The integrated total flux
 */
torch::Tensor cal_total_flux(torch::Tensor flux, torch::Tensor wave_or_weight,
                             std::string input);

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
torch::Tensor cal_net_flux(torch::Tensor flux);

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
torch::Tensor cal_surface_flux(torch::Tensor flux);

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
torch::Tensor cal_toa_flux(torch::Tensor flux);

//! \brief Spherical correction by XIZ
/*!
 * Formula 1:
 * \f[
 *    H_i = \frac{F_{i+1/2} - F_{i-1/2}}{\Delta z_i} V_i
 * \f]
 *
 * Formula 2:
 * \f[
 *   k_{i+1/2} F_{i+1/2} A_{i+1/2} - k_{i-1/2} F_{i-1/2} A_{i-1/2} = H_i
 * \f]
 *
 * Formula 3:
 * \f[
 *   k_{i-1/2} = \frac{k_{i+1/2} F_{i+1/2} A_{i+1/2} - H_i}{F_{i-1/2} A_{i-1/2}}
 * \f]
 *
 * Formula 4:
 * \f[
 *   k_{n+1/2} = 1.
 * \f]
 *
 * xiz 2022 flux scaling so that the heating rate is the same as the
 * plane-parallel scheme volheating scaling.
 *
 * First calculate flux divergence from DISORT using Plane-parallel in a cell
 * to get the volume heating rate (Formula 1).
 * Assume the heating rate is the same as the plane-parallel scheme (Formula 2).
 * Solve for kappa level by level from top down (Formula 3).
 * The boundary condition assumes that the top flux is the same as the
 * plane-parallel scheme (Formula 4).
 *
 * The input flux tensor is the net flux tensor at each level
 * For all variables, the last dimension is the layer/level dimension.
 * The other dimensions are batched.
 *
 * \param flx flux, shape (..., nlayer + 1, 2)
 * \param dz cell height, shape (..., nlayer)
 * \param area cell area, shape (..., nlayer)
 * \param vol cell volume, shape (..., nlayer)
 * \return the flux scaling factor, shape (..., nlayer)
 */
torch::Tensor spherical_flux_scaling(torch::Tensor flx, torch::Tensor dz,
                                     torch::Tensor area, torch::Tensor vol);

}  // namespace harp
