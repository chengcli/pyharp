// harp
#include "atm_to_standard_grid.hpp"

#include <harp/index.h>

namespace harp {
void AtmToStandardGridImpl::reset() {
  xgrid =
      register_buffer("xgrid", torch::zeros({options.ncomp()}, torch::kFloat));
  tgrid =
      register_buffer("tgrid", torch::zeros({options.ntemp()}, torch::kFloat));

  refatm = register_buffer("refatm",
                           torch::zeros({3, options.npres()}, torch::kFloat));
}

torch::Tensor AtmToStandardGridImpl::forward(torch::Tensor var_x, int ix) {
  namespace F = torch::nn::functional;

  auto var_shape = var_x[0].sizes().vec();
  var_shape.push_back(3);
  auto out = torch::zeros(var_shape, var_x.options());

  // rescale log pressure to [-1, 1]
  auto log_refp = refatm[index::IPR].log();
  auto logp = var_x[index::IPR].flatten().log();

  auto log_refp_min = log_refp.min();
  auto log_refp_max = log_refp.max();

  auto logp_scaled =
      2.0 * (logp - log_refp_min) / (log_refp_max - log_refp_min) - 1.0;

  out.select(3, index::IPR) = logp_scaled.view(var_shape);

  // 2d interpolation grid for reference atmosphere
  auto grid = torch::zeros({1, 1, refatm.size(1), 2}, refatm.options());
  grid.select(3, index::IPR) = logp_scaled;

  // interpolated reference temperature at given pressure
  auto op = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder);
  auto reftem = refatm[index::ITM].view({1, 1, 1, -1}).expand({1, 1, 2, -1});
  auto tem = F::grid_sample(reftem, grid, op).view(var_shape);

  // rescale temperature anomaly to [-1, 1]
  auto tema = var_x[index::ITM] - tem;
  out.select(3, index::ITM) =
      2.0 * (tema - tgrid.min()) / (tgrid.max() - tgrid.min()) - 1.0;

  // interpolated reference composition at given pressure
  auto refcom = refatm[index::ICX].view({1, 1, 1, -1}).expand({1, 1, 2, -1});
  auto com = F::grid_sample(refcom, grid, op).view(var_shape);

  // rescale composition scaling to [-1, 1]
  auto comx = var_x[ix] / (com + 1.e-10);  // prevent divide by zero
  out.select(3, index::ICX) =
      2.0 * (comx - xgrid.min()) / (grid.max() - xgrid.min()) - 1.0;

  return out;
}
}  // namespace harp
