#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using water_liquid_mie_fn = void (*)(at::TensorIterator& iter,
                                     double molecular_weight, int nmom,
                                     int max_order);

DECLARE_DISPATCH(water_liquid_mie_fn, call_water_liquid_mie);

}  // namespace at::native
