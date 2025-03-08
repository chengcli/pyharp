// torch
#include <torch/torch.h>

// disort
#include <disort/disort.hpp>

namespace harp {

//! \brief create disort options for shortwave grid
/*!
 * This function creates a template disort options for shortwave grid.
 *
 * It turns on the following disort flags:
 *  - lamber
 *  - quiet
 *  - onlyfl
 *  - intensity_correction
 *  - old_intensity_correction
 *
 * More flags can be added after the template is created by modifying the
 * `flags` member of the `DisortOptions` object.
 *
 * Example:
 *
 * ```cpp
 * auto op = disort_options_sw(nwave, ncol, nlyr);
 * op.flags("print-input,print-phase-function");
 * ```
 *
 * \param nwave number of wavenumbers
 * \param ncol number of columns
 * \param nlyr number of layers
 * \param nstr number of streams
 * \return disort options
 */
disort::DisortOptions disort_options_sw(int nwave, int ncol, int nlyr,
                                        int nstr = 8);

//! \brief create disort options for longwave grid
/*!
 * This function creates a template disort options for longwave grid.
 *
 * It turns on the following disort flags:
 *  - lamber
 *  - quiet
 *  - onlyfl
 *  - planck
 *  - intensity_correction
 *  - old_intensity_correction
 *
 * More flags can be added after the template is created by modifying the
 * `flags` member of the `DisortOptions` object.
 *
 * Example:
 *
 * ```cpp
 * auto op = disort_options_sw(nwave, ncol, nlyr);
 * op.flags("print-input,print-phase-function");
 * ```
 *
 * \param wmin minimum wavenumber
 * \param wmax maximum wavenumber
 * \param nwave number of wavenumbers
 * \param ncol number of columns
 * \param nlyr number of layers
 * \return disort options
 */
disort::DisortOptions disort_options_lw(double wmin, double wmax, int nwave,
                                        int ncol, int nlyr, int nstr = 8);

}  // namespace harp
