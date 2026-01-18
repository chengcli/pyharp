#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using toon89_fn = void (*)(at::TensorIterator &iter);

//! \brief Toon 1989 longwave solver
/*!
 * Based on Elsie Lee's implementation in Exo-FMS_column_ck, which was
 * based on CHIMERA code by Mike Line.
 * Ported by Xi Zhang to Eigen
 * Ported by Cheng Li to torch
 * Reference: Toon, O.B., 1989, JGR, 94, 16287-16301.
 */
DECLARE_DISPATCH(toon89_fn, call_toon89_lw);

//! \brief Toon 1989 shortwave solver
/*!
 * Based on Elsie Lee's implementation in Exo-FMS_column_ck, which was
 * based on CHIMERA code by Mike Line.
 * Ported by Xi Zhang to Eigen
 * Ported by Cheng Li to torch
 * Reference: Toon, O.B., 1989, JGR, 94,16287-16301.
 */
DECLARE_DISPATCH(toon89_fn, call_toon89_sw);

}  // namespace at::native
