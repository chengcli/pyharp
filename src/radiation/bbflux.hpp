// torch
#include <torch/torch.h>

namespace harp {

//! \brief calculate blackbody radiance per wavenumber
/*!
 * Formula, with \f$\nu\f$ the wavenumber:
 * \f[
 *    B_\nu = \frac{2 h c^2 \nu^3}{\exp(\frac{h c \nu}{k_B T}) - 1}
 * \f]
 *
 * This is a radiance, per steradian; the hemispheric flux of a blackbody is
 * \f$\pi B_\nu\f$.
 * Input wavenumber is in cm^-1 and must be a 1D tensor.
 * The output is a 2D tensor with shape (nwave, ncol), where nwave is the
 * number of wavenumbers and ncol is the number of columns.
 *
 * \param wavenumber [cm^-1]
 * \param temp [K]
 * \param ncol number of columns
 * \return blackbody radiance [W/(m^2 sr cm^-1)]
 */
torch::Tensor bbflux_wavenumber(torch::Tensor wave, double temp, int ncol = 1);

//! \brief calculate blackbody radiance integrated over a wavenumber band
/*!
 * Calculate the blackbody radiance integrated between two wavenumbers
 * for a tensor of temperatures. This is a radiance, per steradian; the
 * hemispheric flux is pi times it.
 * If wn1 == wn2, return the radiance per wavenumber at that wavenumber
 * [W/(m^2 sr cm^-1)] instead.
 *
 * \param wn1 small wavenumber [cm^-1]
 * \param wn2 large wavenumber [cm^-1]
 * \param temp [K]
 * \return blackbody radiance [W/(m^2 sr)]
 */
torch::Tensor bbflux_wavenumber(double wn1, double wn2, torch::Tensor temp);

//! \brief calculate blackbody radiance integrated over wavenumber bands
/*!
 * As above, element by element; an element with wn1 == wn2 gets the radiance
 * per wavenumber at that wavenumber.
 */
torch::Tensor bbflux_wavenumber(torch::Tensor wn1, torch::Tensor wn2,
                                torch::Tensor temp);

//! \brief calculate blackbody flux using wavelength
/*!
 * Formula:
 * \f[
 *    F_\lambda = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp(\frac{h c}{\lambda k_B
 * T}) - 1}
 * \f]
 *
 * Input wavelength is in um and must be a 1D tensor.
 * The output flux is a 2D tensor with shape (nwave, ncol), where nwave is the
 * number of wavelengths and ncol is the number of columns.
 *
 * \param wavelength [um]
 * \param temp [K]
 * \param ncol number of columns
 * \return blackbody flux [w/(m^2 um^-1)]
 */
torch::Tensor bbflux_wavelength(torch::Tensor wave, double temp, int ncol = 1);

}  // namespace harp
