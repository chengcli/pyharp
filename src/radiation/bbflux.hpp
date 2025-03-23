// torch
#include <torch/torch.h>

namespace harp {

//! \brief calculate blackbody flux using wavenumber
/*!
 * Formula:
 * \f[
 *    F_\nu = \frac{2 h \nu^3}{c^2} \frac{1}{\exp(\frac{h \nu}{k_B T}) - 1}
 * \f]
 *
 * Input wavenumber is in cm^-1 and must be a 1D tensor.
 * The output flux is a 2D tensor with shape (nwave, ncol), where nwave is the
 * number of wavenumbers and ncol is the number of columns.
 *
 * \param wavenumber [cm^-1]
 * \param temp [K]
 * \param ncol number of columns
 * \return blackbody flux [w/(m^2 cm^-1)]
 */
torch::Tensor bbflux_wavenumber(torch::Tensor wave, double temp, int ncol = 1);

//! \brief calculate integrated blackbody flux using wavenumber
/*!
 * Calculate the integrated blackbody flux between two wavenumbers
 * for a tensor of temperatures.
 * If wn1 == wn2, return the flux at that wavenumber.
 *
 * \param wn1 small wavenumber [cm^-1]
 * \param wn2 large wavenumber [cm^-1]
 * \param temp [K]
 * \return blackbody flux [w/(m^2)]
 */
torch::Tensor bbflux_wavenumber(double wn1, double wn2, torch::Tensor temp);

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
