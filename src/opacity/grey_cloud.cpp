// harp
#include "grey_cloud.hpp"

#include <harp/constants.h>

#include "scattering_functions.hpp"

namespace harp {

GreyCloudImpl::GreyCloudImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.species_ids().size() >= 1,
              "At least one cloud species is needed");

  for (int i = 0; i < options.species_ids().size(); i++) {
    TORCH_CHECK(options.species_ids()[i] >= 0,
                "Invalid species_id: ", options.species_ids()[i]);
  }
}

torch::Tensor GreyCloud::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  // Check species id in range
  TORCH_CHECK(options.species_ids()[0] < conc.size(-1),
              "Invalid species_id: ", options.species_ids()[0]);
  auto totc = conc.select(-1, options.species_ids()[0]);

  for (int i = 1; i < options.species_ids().size(); i++) {
    TORCH_CHECK(options.species_ids()[i] < conc.size(-1),
                "Invalid species_id: ", options.species_ids()[i]);
    totc += conc.select(-1, options.species_ids()[i]);
  }

  auto result = torch::zeros({ncol, nlyr, 2 + options.nmom()}, conc.options());

  if (totc < 1e-10) {
    return result.unsqueeze(0);
  } else {
    result.select(-1, 0) = options.xsection() * totc * constants::Avogadro;
    result.select(-1, 1) = options.ssa();
    result.narrow(-1, 2, options.nmom()) = double_henyey_greenstein(
        options.nmom(), options.ff(), options.g1(), options.g2());
    return result.unsqueeze(0);
  }
}

}  // namespace harp
