// harp
#include "s8_fuller.hpp"

#include <math/interpolation.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

namespace harp {
S8FullerImpl::S8FullerImpl(S8RTOptions const& options_) : options(options_) {
  reset();
}

void S8FullerImpl::reset() {
  auto full_path = find_resource(options.opacity_file());

  // remove comment
  std::string str_file = decomment_file(full_path);

  // read data table
  // read first time to determine dimension
  std::stringstream inp(str_file);
  std::string line;
  std::getline(inp, line);
  int rows = 0, cols = 0;
  char c = ' ';
  if (!line.empty()) {
    rows = 1;
    cols = line[0] == c ? 0 : 1;
    for (int i = 1; i < line.length(); ++i)
      if (line[i - 1] == c && line[i] != c) cols++;
  }
  while (std::getline(inp, line)) ++rows;
  rows--;

  TORCH_CHECK(rows > 0, "Empty file: ", full_path);
  TORCH_CHECK(cols == 3 + S8RTOptions::npmom, "Invalid file: ", full_path);

  kwave = register_buffer("kwave", torch::zeros({rows}, torch::kFloat64));
  kdata =
      register_buffer("kdata", torch::zeros({rows, cols - 1}, torch::kFloat64));

  // read second time
  std::stringstream inp2(str_file);

  // Use an accessor for performance
  auto kwave_accessor = kwave.accessor<double, 1>();
  auto kdata_accessor = kdata.accessor<double, 2>();

  for (int i = 0; i < rows; ++i) {
    inp2 >> kwave_accessor[i];
    for (int j = 1; j < cols; ++j) {
      inp2 >> kdata_accessor[i][j - 1];
    }
  }

  // change wavelength [um] to wavenumber [cm^-1]
  if (options.use_wavenumber()) {
    kwave = 1.e4 / kwave;
  }

  // change extinction x-section [m^2/kg] to [m^2/mol]
  kdata.select(1, 0) *= options.species_mu();
}

torch::Tensor S8FullerImpl::forward(torch::Tensor wave, torch::Tensor conc,
                                    torch::optional<torch::Tensor> pres,
                                    torch::optional<torch::Tensor> temp) {
  int nwve = wave.size(0);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);
  constexpr int nprop = 2 + S8RTOptions::npmom;

  auto out = torch::zeros({nwve, ncol, nlyr, nprop}, conc.options());
  auto dims = torch::tensor(
      {kwave.size(0)},
      torch::TensorOptions().dtype(torch::kInt64).device(conc.device()));

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/3)
                  .add_output(out)
                  .add_owned_const_input(wave.view({-1, 1, 1, 1}))
                  .build();

  if (conc.is_cpu()) {
    call_interpn_cpu<nprop>(iter, kdata, kwave, dims, 1);
  } else if (conc.is_cuda()) {
    // call_interpn_cuda<nprop>(iter, kdata, kwave, dims, 1);
  } else {
    throw std::runtime_error("Unsupported device");
  }

  // attenuation [1/m]
  out.select(3, 0) *= conc.select(2, options.species_id()).unsqueeze(0);

  // attenuation weighted single scattering albedo [1/m]
  out.select(3, 1) *= out.select(3, 0);

  return out;
}

}  // namespace harp
