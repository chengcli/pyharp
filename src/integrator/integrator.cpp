// yaml
#include <yaml-cpp/yaml.h>

// harp
#include "integrator.hpp"
#include "integrator_dispatch.hpp"

namespace harp {

IntegratorOptions IntegratorOptionsImpl::from_yaml(
    std::string const& filename) {
  auto op = IntegratorOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return op;

  op->type() = config["integration"]["type"].as<std::string>("rk3");
  op->cfl() = config["integration"]["cfl"].as<double>(0.9);
  op->tlim() = config["integration"]["tlim"].as<double>(1.e9);
  op->nlim() = config["integration"]["nlim"].as<int>(-1);
  op->ncycle_out() = config["integration"]["ncycle_out"].as<int>(1);
  op->restart() = config["integration"]["restart"].as<std::string>("");

  return op;
}

IntegratorImpl::IntegratorImpl(IntegratorOptions const& options_)
    : options(options_), current_redo(0) {
  if (options->type() == "rk1" || options->type() == "euler") {
    stages.resize(1);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);
  } else if (options->type() == "rk2") {
    stages.resize(2);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);

    stages[1].wght0(0.5);
    stages[1].wght1(0.5);
    stages[1].wght2(0.5);
  } else if (options->type() == "rk3") {
    stages.resize(3);
    stages[0].wght0(0.0);
    stages[0].wght1(1.0);
    stages[0].wght2(1.0);

    stages[1].wght0(3. / 4.);
    stages[1].wght1(1. / 4.);
    stages[1].wght2(1. / 4.);

    stages[2].wght0(1. / 3.);
    stages[2].wght1(2. / 3.);
    stages[2].wght2(2. / 3.);
  } else if (options->type() == "rk3s4") {
    stages.resize(4);
    stages[0].wght0(0.5);
    stages[0].wght1(0.5);
    stages[0].wght2(0.5);

    stages[1].wght0(0.0);
    stages[1].wght1(1.0);
    stages[1].wght2(0.5);

    stages[2].wght0(2. / 3.);
    stages[2].wght1(1. / 3.);
    stages[2].wght2(1. / 6.);

    stages[3].wght0(0.);
    stages[3].wght1(1.);
    stages[3].wght2(1. / 2.);
  } else {
    throw std::runtime_error("Integrator not implemented: requested type '" + options->type() + "'");
  }

  reset();
}

void IntegratorImpl::reset() {}

bool IntegratorImpl::stop(int steps, double current_time) {
  if (options->nlim() >= 0 && steps >= options->nlim()) {
    return true;  // stop if number of steps exceeds nlim
  }

  if (options->tlim() >= 0 && current_time >= options->tlim()) {
    return true;  // stop if time exceeds tlim
  }

  return false;  // otherwise, continue integration
}

torch::Tensor IntegratorImpl::forward(int s, torch::Tensor u0, torch::Tensor u1,
                                      torch::Tensor u2) {
  if (s < 0 || s >= stages.size()) {
    throw std::runtime_error("Invalid stage");
  }

  auto out = torch::empty_like(u0);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_input(u0)
                  .add_input(u1)
                  .add_input(u2)
                  .build();

  at::native::call_average3(out.device().type(), iter, stages[s].wght0(),
                            stages[s].wght1(), stages[s].wght2());

  return out;
}
}  // namespace harp
