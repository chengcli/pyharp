// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/utils/parse_yaml_input.hpp>
#include <harp/utils/read_dimvar_netcdf.hpp>
#include <harp/utils/read_var_pt.hpp>

#include "opacity_options.hpp"

namespace harp {

extern std::vector<std::string> species_names;

OpacityOptions OpacityOptionsImpl::from_yaml(std::string const& filename,
                                             std::string const& op_name,
                                             std::string bd_name) {
  auto op = std::make_shared<OpacityOptionsImpl>();
  op->bname(bd_name);

  auto config = YAML::LoadFile(filename);
  TORCH_CHECK(config["opacities"][op_name], op_name, " not found in opacities");

  auto my = config["opacities"][op_name];

  TORCH_CHECK(my["type"], "'type' missing in opacity", op_name);
  op->type(my["type"].as<std::string>());

  if (my["data"]) {
    op->opacity_files(my["data"].as<std::vector<std::string>>());
    for (auto& f : op->opacity_files()) {
      replace_pattern_inplace(f, "<band>", bd_name);
    }
  }

  if (my["species"]) {
    for (auto const& sp : my["species"]) {
      auto sp_name = sp.as<std::string>();

      // index sp_name in species
      auto jt = std::find(species_names.begin(), species_names.end(), sp_name);

      TORCH_CHECK(jt != species_names.end(), "species ", sp_name,
                  " not found in species list");
      op->species_ids().push_back(jt - species_names.begin());
    }
  }

  op->jit_kwargs(my["jit_kwargs"].as<std::vector<std::string>>());
  op->nmom(my["nmom"].as<int>(0));

  return op;
}

std::vector<double> OpacityOptionsImpl::query_wavenumber() const {
  if (type().compare(0, 3, "rfm") == 0) {
    if (opacity_files().size() < 1) {
      throw std::runtime_error("no opacity files specified for RFM opacity");
    }
    return read_dimvar_netcdf<double>(opacity_files()[0], "Wavenumber");
  } else if (type().compare(0, 9, "multiband") == 0) {
    if (opacity_files().size() < 1) {
      throw std::runtime_error(
          "no opacity files specified for multiband opacity");
    }
    return read_var_pt<double>(opacity_files()[0], "wavenumber");
  } else {
    throw std::runtime_error(
        "unsupported opacity type for querying wavenumber");
  }
}

std::vector<double> OpacityOptionsImpl::query_weight() const {
  if (type().compare(0, 3, "rfm") == 0) {
    if (opacity_files().size() < 1) {
      throw std::runtime_error("no opacity files specified for RFM opacity");
    }
    return read_dimvar_netcdf<double>(opacity_files()[0], "weights");
  } else if (type().compare(0, 9, "multiband") == 0) {
    if (opacity_files().size() < 1) {
      throw std::runtime_error(
          "no opacity files specified for multiband opacity");
    }
    return read_var_pt<double>(opacity_files()[0], "weights");
  } else {
    throw std::runtime_error("unsupported opacity type for querying weight");
  }
}

}  // namespace harp
