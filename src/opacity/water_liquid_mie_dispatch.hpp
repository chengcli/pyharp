#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

// Mie efficiencies (qext, qsca, g, status) per (wavelength, radius).
using water_liquid_mie_efficiency_fn = void (*)(at::TensorIterator& iter,
                                                int max_order);

// Cell optical properties (extinction, albedo, g) from those efficiencies.
using water_liquid_mie_assemble_fn = void (*)(at::TensorIterator& iter,
                                              double molecular_weight);

DECLARE_DISPATCH(water_liquid_mie_efficiency_fn,
                 call_water_liquid_mie_efficiency);
DECLARE_DISPATCH(water_liquid_mie_assemble_fn, call_water_liquid_mie_assemble);

}  // namespace at::native
