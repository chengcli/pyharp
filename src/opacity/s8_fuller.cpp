// harp
#include "s8_fuller.hpp"

#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

namespace harp {

S8FullerImpl::S8FullerImpl(S8FullerOptions const& options_)
    : options(options_) {
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

  // read second time
  kdata = register_buffer("kdata", torch::zeros({rows, cols}, torch::kFloat64));
  std::stringstream inp2(str_file);

  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      double value;
      inp2 >> value;
      kdata.index_put_({i, j}, value);
    }

  // change wavelength [um] to wavenumber [cm^-1]
  kdata.select(1, 0) = 1e4 / kdata.select(1, 0);

  // change extinction x-section [m^2/kg] to [m^2/mol]
  kdata.select(1, 1) *= options.species_mu();
}

torch::Tensor S8FullerImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                    torch::Tensor conc) {
  int nwve = kdata.size(0);
  int ncol = temp.size(0);
  int nlyr = temp.size(1);
  int nprop = 2;

  auto result = torch::zeros({nwve, ncol, nlyr, nprop}, temp.options());

  // attenuation [1/m]
  result.select(3, 0) = conc.select(2, options.species_id()).unsqueeze(0) *
                        kdata.select(1, 1).unsqueeze(-1).unsqueeze(-1);

  // single scattering albedo
  result.select(3, 1) =
      result.select(3, 0) * kdata.select(1, 2).unsqueeze(-1).unsqueeze(-1);

  return result;
}

}  // namespace harp
