#pragma once

// C/C++
#include <memory>

// torch
#include <torch/nn/module.h>

namespace harp {
class AttenuatorImpl;
class AttenuatorOptions;
//! Choose between [XizH2H2CIA, XizH2HeCIA, Hitran]
std::shared_ptr<AttenuatorImpl> register_module_op(torch::nn::Module *p,
                                                   std::string name,
                                                   AttenuatorOptions const &op);

class RTSolverImpl;
class BeerLambertOptions;
std::shared_ptr<RTSolverImpl> register_module_op(torch::nn::Module *p,
                                                 std::string name,
                                                 BeerLambertOptions const &op);
}  // namespace harp
