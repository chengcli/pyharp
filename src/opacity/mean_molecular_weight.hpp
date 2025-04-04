#include <torch/torch.h>

namespace harp {

extern std::vector<double> species_weights;

inline torch::Tensor mean_molecular_weight(torch::Tensor conc) {
  // check if conc has the same number of dimensions as species_weights
  TORCH_CHECK(species_weights.size() == conc.size(-1),
              "The last dimension of 'conc' must be the same as number of "
              "species defined in yaml");
  torch::Tensor ww = torch::tensor(
      species_weights,
      torch::TensorOptions().dtype(conc.dtype()).device(conc.device()));

  // dimension of conc is (ncol, nlyr, nspecies)
  return (conc * ww.unsqueeze(0).squeeze(0)).sum(-1) / conc.sum(-1);
}

}  // namespace harp
