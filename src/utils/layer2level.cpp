// harp
#include "layer2level.hpp"

#include "interp.hpp"

namespace harp {
torch::Tensor layer2level(torch::Tensor var,
                          Layer2LevelOptions const &options) {
  // increase the last dimension by 1 (lyr -> lvl)
  auto shape = var.sizes().vec();
  shape.back() += 1;
  torch::Tensor out = torch::zeros(shape, var.options());

  int nlyr = var.size(-1);

  if (nlyr == 1) {  // use constant extrapolation
    out.select(-1, 0) = var.select(-1, 0);
  } else {  // lower boundary
    if (options.blower() == kExtrapolate) {
      out.select(-1, 0) = (3. * var.select(-1, 0) - var.select(-1, 1)) / 2.;
    } else if (options.blower() == kConstant) {
      out.select(-1, 0) = var.select(-1, 0);
    } else {
      TORCH_CHECK(false, "Unsupported boundary condition");
    }
  }

  // interior
  if (options.order() == k4thOrder) {
    Center4Interp interp_cp4;
    interp_cp4->to(var.device());

    if (nlyr > 1) {
      out.select(-1, 1) = (var.select(-1, 0) + var.select(-1, 1)) / 2.;
    }

    if (nlyr > 2) {
      out.select(-1, nlyr - 1) =
          (var.select(-1, nlyr - 1) + var.select(-1, nlyr - 2)) / 2.;
    }

    if (nlyr > 3) {
      out.slice(-1, 2, nlyr - 1) = interp_cp4->forward(var.unfold(-1, 4, 1));
    }
  } else if (options.order() == k2ndOrder) {
    if (nlyr > 1) {
      out.slice(-1, 1, nlyr) =
          (var.slice(-1, 0, nlyr - 1) + var.slice(-1, 1, nlyr)) / 2.;
    }
  } else {
    TORCH_CHECK(false, "Unsupported interpolation order");
  }

  if (nlyr == 1) {  // use constant extrapolation
    out.select(-1, nlyr) = var.select(-1, nlyr - 1);
  } else {  // upper boundary
    if (options.bupper() == kExtrapolate) {
      out.select(-1, nlyr) =
          (3. * var.select(-1, nlyr - 1) - var.select(-1, nlyr - 2)) / 2.;
    } else if (options.bupper() == kConstant) {
      out.select(-1, nlyr) = var.select(-1, nlyr - 1);
    } else {
      TORCH_CHECK(false, "Unsupported boundary condition");
    }
  }

  // checks
  if (options.check_positivity()) {
    auto error = torch::nonzero(out < 0);

    if (error.size(0) > 0) {
      std::cout << "Negative values found at cell interface: ";
      std::cout << "indices = " << error << std::endl;
      TORCH_CHECK(false, "layer2level check failed");
    }
  }

  return out;
}
}  // namespace harp
