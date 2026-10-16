#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

//! \brief Add an opacity's optical properties into the band's.
/*!
 * The iterator runs over (nwave, ncol, nlyr) with the band's extinction
 * slice as output and the opacity's as input; each element's row along the
 * contiguous property axis is read and updated in the kernel. \p nprop is
 * the opacity's property count.
 */
using accumulate_opacity_fn = void (*)(at::TensorIterator& iter, int nprop);

//! \brief Average the band's summed properties and form the optical depth.
/*!
 * The iterator runs over (nwave, ncol, nlyr) with the band's extinction
 * slice as output and the layer thickness as input. \p nprop is the band's
 * property count and \p albedo_cap the largest single-scattering albedo the
 * solver is given.
 */
using finish_band_fn = void (*)(at::TensorIterator& iter, int nprop,
                                double albedo_cap);

DECLARE_DISPATCH(accumulate_opacity_fn, call_accumulate_opacity);
DECLARE_DISPATCH(finish_band_fn, call_finish_band);

}  // namespace at::native
