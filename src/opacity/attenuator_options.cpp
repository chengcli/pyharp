// yaml
#include <yaml-cpp/yaml.h>

// harp
#include <harp/utils/parse_yaml_input.hpp>

#include "attenuator_options.hpp"

namespace harp {

extern std::vector<std::string> species_names;

AttenuatorOptions AttenuatorOptions::from_yaml(YAML::Node const& config,
                                               std::string bd_name) {
  AttenuatorOptions op;
  op.bname(bd_name);

  TORCH_CHECK(config["type"], "'type' missing in opacity config");
  op.type(config["type"].as<std::string>());

  if (config["data"]) {
    op.opacity_files(config["data"].as<std::vector<std::string>>());
    for (auto& f : op.opacity_files()) {
      replace_pattern_inplace(f, "<band>", bd_name);
    }
  }

  if (config["species"]) {
    for (auto const& sp : config["species"]) {
      auto sp_name = sp.as<std::string>();

      // index sp_name in species
      auto jt = std::find(species_names.begin(), species_names.end(), sp_name);

      TORCH_CHECK(jt != species_names.end(), "species ", sp_name,
                  " not found in species list");
      op.species_ids().push_back(jt - species_names.begin());
    }
  }

  if (config["jit_kwargs"]) {
    op.jit_kwargs(config["jit_kwargs"].as<std::vector<std::string>>());
  }

  if (config["fractions"]) {
    op.fractions(config["fractions"].as<std::vector<double>>());
  }

  op.scale(config["scale"].as<double>(1.0));
  op.metallicity(config["metallicity"].as<double>(0.0));
  op.kappa_a(config["kappa_a"].as<double>(0.0));
  op.kappa_b(config["kappa_b"].as<double>(0.0));
  op.kappa_cut(config["kappa_cut"].as<double>(0.0));
  op.diameter(config["diameter"].as<double>(0.0));
  op.xsection(config["xsection"].as<double>(0.0));
  op.ssa(config["ssa"].as<double>(0.0));
  op.ff(config["ff"].as<double>(0.0));
  op.g1(config["g1"].as<double>(0.0));
  op.g2(config["g2"].as<double>(0.0));
  op.nmom(config["nmom"].as<int>(0));

  return op;
}

}  // namespace harp
