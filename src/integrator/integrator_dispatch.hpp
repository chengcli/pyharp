#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using avg3_fn = void (*)(at::TensorIterator &iter, double w1, double w2,
                         double w3);

DECLARE_DISPATCH(avg3_fn, call_average3);

}  // namespace at::native
