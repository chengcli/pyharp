#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <harp/add_arg.h>

// according to:
// https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html

namespace harp {
struct IntegratorWeight {
  void report(std::ostream& os) const {
    os << "* wght0 = " << wght0() << "\n"
       << "* wght1 = " << wght1() << "\n"
       << "* wght2 = " << wght2() << "\n";
  }
  ADD_ARG(double, wght0) = 0.0;
  ADD_ARG(double, wght1) = 0.0;
  ADD_ARG(double, wght2) = 0.0;
};

struct IntegratorOptionsImpl {
  static std::shared_ptr<IntegratorOptionsImpl> create() {
    return std::make_shared<IntegratorOptionsImpl>();
  }
  static std::shared_ptr<IntegratorOptionsImpl> from_yaml(
      std::string const& filename);

  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* cfl = " << cfl() << "\n"
       << "* tlim = " << tlim() << "\n"
       << "* nlim = " << nlim() << "\n"
       << "* ncycle_out = " << ncycle_out() << "\n"
       << "* max_redo = " << max_redo() << "\n"
       << "* restart = " << restart() << "\n";
  }

  ADD_ARG(std::string, type) = "rk3";
  ADD_ARG(double, cfl) = 0.9;
  ADD_ARG(double, tlim) = 1.e9;
  ADD_ARG(int, nlim) = -1;
  ADD_ARG(int, ncycle_out) = 1;
  ADD_ARG(int, max_redo) = 5;
  ADD_ARG(std::string, restart) = "";
};
using IntegratorOptions = std::shared_ptr<IntegratorOptionsImpl>;

class IntegratorImpl : public torch::nn::Cloneable<IntegratorImpl> {
 public:
  int current_redo = 0;

  //! options with which this `Integrator` was constructed
  IntegratorOptions options;

  //! weights for each stage
  std::vector<IntegratorWeight> stages;

  IntegratorImpl() : options(IntegratorOptionsImpl::create()) {}
  explicit IntegratorImpl(IntegratorOptions const& options);
  void reset() override;

  //! \brief check if the integration should stop
  bool stop(int steps, double current_time);

  //! \brief compute the average of the three input tensors
  torch::Tensor forward(int stage, torch::Tensor u0, torch::Tensor u1,
                        torch::Tensor u2);
};
TORCH_MODULE(Integrator);

}  // namespace harp

#undef ADD_ARG
