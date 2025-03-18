// torch
#include <torch/torch.h>

// harp
#include <harp/utils/vectorize.hpp>

#include "parse_radiation_direction.hpp"

namespace harp {

torch::Tensor parse_radiation_direction(std::string const &str) {
  float mu = 0.;
  float phi = 0.;

  sscanf(str.data(), "(%f,%f)", &mu, &phi);
  mu = cos(deg2rad(mu));
  phi = deg2rad(phi);

  return torch::tensor({mu, phi}, torch::kFloat32);
}

torch::Tensor parse_radiation_directions(std::string const &str) {
  std::vector<std::string> dstr = Vectorize<std::string>(str.c_str());
  int nray = dstr.size();

  torch::Tensor ray = torch::zeros({nray, 2}, torch::kFloat32);

  for (int i = 0; i < nray; ++i) {
    ray[i] = parse_radiation_direction(dstr[i]);
  }

  return ray;
}
}  // namespace harp
