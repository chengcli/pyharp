// harp
#include "layer2level.hpp"

#include "interp.hpp"

namespace harp {
torch::Tensor layer2level(torch::Tensor var,
                          Layer2LevelOptions const &options) {
  // increase the last dimension by 1
  auto shape = var.sizes().vec();
  shape.back() += 1;
  torch::Tensor out = torch::zeros(shape, var.options());

  // lower boundary
  if (options.blower() == kExtrapolate) {
    out.select(-1, 0) = (3. * var.select(-1, 0) - var.select(-1, 1)) / 2.;
  } else if (options.blower() == kConstant) {
    out.select(-1, 0) = var.select(-1, 0);
  } else {
    throw std::runtime_error("Unsupported boundary condition");
  }

  // interior
  if (options.order() == k4thOrder) {
    Center4Interp interp_cp4;
    interp_cp4->to(var.device());

    out.select(-1, 1) = (var.select(-1, 0) + var.select(-1, 1)) / 2.;
    out.slice(-1, 2, -2) = interp_cp4->forward(var);
    out.slice(-1, -2) = (var.select(-1, -1) + var.select(-1, -2)) / 2.;
  } else if (options.order() == k2ndOrder) {
    out.slice(-1, 1, -1) = (var.slice(-1, 0, -2) + var.slice(-1, 1, -1)) / 2.;
  } else {
    throw std::runtime_error("Unsupported interpolation order");
  }

  // upper boundary
  if (options.bupper() == kExtrapolate) {
    out.select(-1, -1) = (3. * var.select(-1, -1) - var.select(-1, -2)) / 2.;
  } else if (options.bupper() == kConstant) {
    out.select(-1, -1) = var.select(-1, -1);
  } else {
    throw std::runtime_error("Unsupported boundary condition");
  }

  // checks
  if (options.check_positivity()) {
    auto error = torch::nonzero(out < 0);

    if (error.size(0) > 0) {
      std::cout << "Negative values found at cell interface: ";
      std::cout << "indices = " << error << std::endl;
      throw std::runtime_error("layer2level check failed");
    }
  }

  return out;
}
}  // namespace harp
