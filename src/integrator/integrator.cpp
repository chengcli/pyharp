// C/C++
#include <map>

// yaml
#include <yaml-cpp/yaml.h>

// harp
#include "integrator.hpp"

namespace harp {

IntegratorOptions IntegratorOptionsImpl::from_yaml(std::string const& filename,
                                                   bool verbose) {
  auto op = IntegratorOptionsImpl::create();
  op->verbose() = verbose;

  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return op;
  return from_yaml(config["integration"], verbose);
}

IntegratorOptions IntegratorOptionsImpl::from_yaml(YAML::Node const& node,
                                                   bool verbose) {
  auto op = IntegratorOptionsImpl::create();

  op->type() = node["type"].as<std::string>("rk3");
  op->cfl() = node["cfl"].as<double>(0.9);
  op->tlim() = node["tlim"].as<double>(1.e9);
  op->nlim() = node["nlim"].as<int>(-1);
  op->ncycle_out() = node["ncycle_out"].as<int>(1);
  op->verbose() = node["verbose"].as<bool>(verbose);
  return op;
}

IntegratorImpl::IntegratorImpl(IntegratorOptions const& options_)
    : options(options_), current_redo(0) {
  // {w0, w1, w2} per stage. 1. - 2. / 3. is exact, so state weights sum to 1;
  // 1./3. + 2./3. is 1 - 2^-54 and drifted a conserved total 5.6e-17 per step
  static std::map<std::string, std::vector<std::vector<double>>> const table = {
      {"rk1", {{0.0, 1.0, 1.0}}},
      {"rk2", {{0.0, 1.0, 1.0}, {0.5, 0.5, 0.5}}},
      {"rk3",
       {{0.0, 1.0, 1.0},
        {3. / 4., 1. / 4., 1. / 4.},
        {1. - 2. / 3., 2. / 3., 2. / 3.}}},
      {"rk3s4",
       {{0.5, 0.5, 0.5},
        {0.0, 1.0, 0.5},
        {2. / 3., 1. - 2. / 3., 1. / 6.},
        {0., 1., 1. / 2.}}}};
  auto it = table.find(options->type() == "euler" ? "rk1" : options->type());
  if (it == table.end()) {
    throw std::runtime_error("Integrator not implemented: requested type '" +
                             options->type() + "'");
  }
  for (auto const& w : it->second)
    stages.push_back(IntegratorWeight().wght0(w[0]).wght1(w[1]).wght2(w[2]));
  reset();
}

void IntegratorImpl::reset() {}

bool IntegratorImpl::stop(int steps, double current_time) {
  return (options->nlim() >= 0 && steps >= options->nlim()) ||
         (options->tlim() >= 0 && current_time >= options->tlim());
}

torch::Tensor IntegratorImpl::forward(int s, torch::Tensor u0, torch::Tensor u1,
                                      torch::Tensor u2,
                                      std::optional<torch::Tensor> out) {
  if (s < 0 || s >= stages.size()) {
    throw std::runtime_error("Invalid stage: s = " + std::to_string(s) +
                             ", valid range = [0, " +
                             std::to_string(stages.size()) + ")");
  }

  auto const& w = stages[s];
  if (!out) return w.wght0() * u0 + w.wght1() * u1 + w.wght2() * u2;
  for (auto const& t : {u0, u1, u2})
    TORCH_CHECK(t.dtype() == out->dtype() && t.device() == out->device(),
                "[Integrator] out, u0, u1 and u2 must share dtype and device");
  at::add_out(*out, w.wght0() * u0, w.wght1() * u1);
  return out->add_(w.wght2() * u2);
}

std::shared_ptr<IntegratorImpl> IntegratorImpl::create(
    IntegratorOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p, "[Integrator] Parent module is null");
  TORCH_CHECK(opts, "[Integrator] Options pointer is null");
  return p->register_module(name, Integrator(opts));
}

}  // namespace harp
