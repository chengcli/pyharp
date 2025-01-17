#pragma once

// torch
#include <ATen/TensorIterator.h>

namespace harp {

void call_interpn_cpu(at::TensorIterator& iter);
void call_interpn_cuda(at::TensorIterator& iter);

}  // namespace harp
