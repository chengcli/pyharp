// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

namespace harp {
class Center4InterpImpl : public torch::nn::Cloneable<Center4InterpImpl> {
 public:
  //! data
  torch::Tensor cm;

  //! Constructor to initialize the layer
  explicit Center4InterpImpl() { reset(); }
  void reset() override {
    cm = register_buffer(
        "cm", torch::tensor({-1. / 12., 7. / 12., 7. / 12., -1. / 12.},
                            torch::kFloat64));
  }

  torch::Tensor forward(torch::Tensor w) { return matmul(w, cm); }
};
TORCH_MODULE(Center4Interp);
}  // namespace harp
