#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// C/C++
#include <cctype>
#include <sstream>
#include <string>

// harp
#include <harp/add_arg.h>

namespace harp {

struct ToonMcKay89OptionsImpl {
  ToonMcKay89OptionsImpl() {}
  static std::shared_ptr<ToonMcKay89OptionsImpl> create() {
    return std::make_shared<ToonMcKay89OptionsImpl>();
  }

  std::shared_ptr<ToonMcKay89OptionsImpl> clone() const {
    return std::make_shared<ToonMcKay89OptionsImpl>(*this);
  }
  bool has_flag(std::string flag) const {
    auto normalize = [](std::string value) {
      auto start = value.find_first_not_of(" \t\n\r");
      if (start == std::string::npos) {
        return std::string{};
      }
      auto end = value.find_last_not_of(" \t\n\r");
      value = value.substr(start, end - start + 1);
      for (auto& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
      return value;
    };
    flag = normalize(flag);
    std::stringstream ss(flags());
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (normalize(item) == flag) {
        return true;
      }
    }
    return false;
  }
  bool planck() const { return has_flag("planck"); }
  bool zenith_correction() const { return has_flag("zenith_correction"); }
  bool hard_surface() const { return has_flag("hard_surface"); }
  bool delta_eddington_lw() const { return has_flag("delta_eddington_lw"); }
  void report(std::ostream& os) const {
    os << "* flags = " << flags() << "\n";
    os << "* planck = " << planck() << "\n";
    os << "* zenith_correction = " << zenith_correction() << "\n";
    os << "* top_emission_flag = " << top_emission_flag() << "\n";
    os << "* hard_surface = " << hard_surface() << "\n";
    os << "* delta_eddington_lw = " << delta_eddington_lw() << "\n";

    os << "* wave_lower = ";
    for (auto const& v : wave_lower()) os << v << ", ";
    os << "\n";

    os << "* wave_upper = ";
    for (auto const& v : wave_upper()) os << v << ", ";
    os << "\n";
  }

  //! set lower wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_lower) = {};

  //! set upper wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_upper) = {};

  //! comma-separated solver flags
  ADD_ARG(std::string, flags) = "";

  //! top emission flag
  //!   0 = no incoming radiation at TOA (Toon 1989 default, GCM mode)
  //!   1 = full Planck at TOA (infinite isothermal slab above)
  //!  -1 = auto-compute from first layer: tau_top = dtau[0]*exp(-1),
  //!       Btop = (1-exp(-tau_top/mu)) * B(T_top)  (FMS-style)
  ADD_ARG(int, top_emission_flag) = 0;
};

using ToonMcKay89Options = std::shared_ptr<ToonMcKay89OptionsImpl>;

class ToonMcKay89Impl : public torch::nn::Cloneable<ToonMcKay89Impl> {
 public:
  //! options with which this `ToonMcKay89Impl` was constructed
  ToonMcKay89Options options;

  //! Constructor to initialize the layers
  ToonMcKay89Impl() : options(ToonMcKay89OptionsImpl::create()) {}
  explicit ToonMcKay89Impl(ToonMcKay89Options const& options);
  void reset() override;

  //! Calculate radiative flux
  /*!
   * \param prop optical properties at each level (nwave, ncol, nlyr, nprop)
   * \param bc dictionary of disort boundary conditions
   *        The dimensions of each recognized key are:
   *
   * \param bname name of the radiation band
   * \param temf temperature at each level (ncol, nlvl = nlyr + 1)
   * \return radiative flux or intensity (nwave, ncol, nlvl, nrad)
   */
  torch::Tensor forward(torch::Tensor prop,
                        std::map<std::string, torch::Tensor>* bc,
                        std::string bname = "",
                        torch::optional<torch::Tensor> temf = torch::nullopt);
};
TORCH_MODULE(ToonMcKay89);

}  // namespace harp

#undef ADD_ARG
