// torch
#include <harp/math/interpolation.hpp>
#include <harp/utils/fileio.hpp>
#include <harp/utils/find_resource.hpp>

// harp
#include "fourcolumn.hpp"
#include "scattering_functions.hpp"

namespace harp {

extern std::vector<double> species_weights;

FourColumnImpl::FourColumnImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");
  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(options.type().empty() ||
                  (options.type().compare(0, 10, "fourcolumn") == 0),
              "Mismatch type: ", options.type(), " expecting 'fourcolumn'");

  reset();
}

void FourColumnImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

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
  TORCH_CHECK(cols == 4, "Invalid file: ", full_path);

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

  // change extinction x-section [m^2/kg] to [m^2/mol]
  kdata.select(1, 0) *= species_weights[options.species_ids()[0]];
}

torch::Tensor FourColumnImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  torch::Tensor wave;
  if (kwargs.count("wavelength") > 0) {
    wave = kwargs.at("wavelength");
  } else if (kwargs.count("wavenumber") > 0) {
    wave = 1.e4 / kwargs.at("wavenumber");
  } else {
    TORCH_CHECK(false, "wavelength or wavenumber is required in kwargs");
  }

  auto out = torch::zeros({wave.size(0), ncol, nlyr, 2 + options.nmom()},
                          conc.options());
  auto data = interpn({wave}, {kwave}, kdata);

  out.narrow(-1, 0, 2) = data.narrow(1, 0, 2).unsqueeze(1).unsqueeze(1);
  out.narrow(-1, 2, options.nmom()) =
      henyey_greenstein(options.nmom(), data.select(1, 2))
          .unsqueeze(1)
          .unsqueeze(1);

  // Check species id in range
  TORCH_CHECK(options.species_ids()[0] < conc.size(-1),
              "Invalid species_id: ", options.species_ids()[0]);

  // attenuation [1/m]
  out.select(-1, 0) *= conc.select(-1, options.species_ids()[0]).unsqueeze(0);

  return out;
}

}  // namespace harp
