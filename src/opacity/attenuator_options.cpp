// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/utils/parse_yaml_input.hpp>

#include "attenuator_options.hpp"

namespace harp {

extern std::vector<std::string> species_names;

AttenuatorOptions AttenuatorOptionsImpl::from_yaml(std::string const& filename,
                                                   std::string const& op_name,
                                                   std::string bd_name) {
  auto op = std::make_shared<AttenuatorOptionsImpl>();
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

}  // namespace harp
