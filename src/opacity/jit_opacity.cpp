// C/C++
#include <map>
#include <string>

// harp
#include "jit_opacity.hpp"

namespace harp {

JITOpacityImpl::JITOpacityImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.opacity_files().size() > 0,
              "JIT opacities must have more than one file");
  module = torch::jit::load(options.opacity_files()[0]);
  reset();
}

void JITOpacityImpl::reset() {
  // No specific reset actions needed for JITOpacity
}

torch::Tensor JITOpacityImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(conc);
  for (auto key : options.jit_kwargs()) {
    if (kwargs.count(key) > 0) {
      inputs.push_back(kwargs.at(key));
    } else {
      throw std::runtime_error("Missing required argument: " + key);
    }
  }
  return module.forward(inputs).toTensor();
}

}  // namespace harp
