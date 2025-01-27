// harp
#include "opacity/attenuator.hpp"
#include "rtsolver/rtsolver.hpp"

namespace harp {
Attenuator register_module_op(torch::nn::Module *p, std::string name,
                              AttenuatorOptions const &op) {
  if (op.type() == "XIZ-H2-H2-CIA") {
    // return p->register_module(name, XizH2H2CIA(op));
  } else if (op.type() == "XIZ-H2-He-CIA") {
    // return p->register_module(name, XizH2HeCIA(op));
  } else if (op.type() == "RFM") {
    return p->register_module(name, AbsorberRFM(op));
  }

  throw std::runtime_error("register_module: unknown type " + op.type());
}

RTSolver register_module_op(torch::nn::Module *p, std::string name,
                            BeerLambertOptions const &op) {
  return p->register_module(name, BeerLambert(op));
}

}  // namespace harp
