// harp
#include "attenuator.hpp"

namespace harp {
torch::Tensor AttenuatorImpl::forward(torch::Tensor var_x) {
  // shape of the original tensor
  auto var_shape = var_x.sizes().vec();
  var_shape.insert(var_shape.begin(), 1);

  // tau + ssa + phase moments starting from 1st order
  var_shape[0] = 2 + options.npmom();
  var_shape[1] = options.nspec();

  return torch::zeros(var_shape, var_x.options());
}
}  // namespace harp
