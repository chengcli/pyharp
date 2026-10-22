#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

//! \brief Fu ice optical properties on the (wavelength, column, layer) grid.
/*!
 * The iterator runs over (nwave, ncol, nlyr): outputs are the result's
 * extinction slice, whose contiguous row holds the 2 + nmom properties, and
 * a uint8 flag set where the properties are unphysical; inputs are the
 * wavelength index, Dge and the ice water content. \p table holds one
 * coefficient row per wavelength, laid out as harp::fu describes.
 */
using fu_ice_fill_fn = void (*)(at::TensorIterator& iter,
                                at::Tensor const& table, int nmom,
                                bool delta_scale);

//! \brief Mix ice and liquid optical properties cell by cell.
/*!
 * The iterator runs over (nwave, ncol, nlyr) with the result's, the ice
 * result's and the liquid result's extinction slices; each row along the
 * contiguous property axis holds 2 + nmom values.
 */
using cloud_switch_fill_fn = void (*)(at::TensorIterator& iter, int nmom);

DECLARE_DISPATCH(fu_ice_fill_fn, call_fu_ice_fill);
DECLARE_DISPATCH(cloud_switch_fill_fn, call_cloud_switch_fill);

}  // namespace at::native
