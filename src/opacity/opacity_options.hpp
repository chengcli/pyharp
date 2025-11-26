#pragma once

// harp
#include <harp/add_arg.h>

namespace harp {

struct OpacityOptionsImpl {
  static std::shared_ptr<OpacityOptionsImpl> create() {
    return std::make_shared<OpacityOptionsImpl>();
  }

  static std::shared_ptr<OpacityOptionsImpl> from_yaml(
      std::string const& filename, std::string const& op_name,
      std::string bd_name = "");

  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n";
    os << "* bname = " << bname() << "\n";
    os << "* opacity_files = ";
    for (auto const& f : opacity_files()) os << f << ", ";
    os << "\n";
    os << "* species_ids = ";
    for (auto const& s : species_ids()) os << s << ", ";
    os << "\n";
    os << "* jit_kwargs = ";
    for (auto const& kw : jit_kwargs()) os << kw << ", ";
    os << "\n";
    os << "* fractions = ";
    for (auto const& f : fractions()) os << f << ", ";
    os << "* nmom = " << nmom() << "\n";
  }

  std::vector<double> query_wavenumber() const;
  std::vector<double> query_weight() const;

  //! type of the opacity source
  ADD_ARG(std::string, type) = "";

  //! name of the band that the opacity is associated with
  ADD_ARG(std::string, bname) = "";

  //! list of opacity data files
  ADD_ARG(std::vector<std::string>, opacity_files) = {};

  //! list of dependent species indices
  ADD_ARG(std::vector<int>, species_ids) = {};

  //! list of kwargs to pass to the JIT module
  ADD_ARG(std::vector<std::string>, jit_kwargs) = {};

  //! number fraction of species in CIA calculation
  ADD_ARG(std::vector<double>, fractions) = {};

  //! number of scattering moments
  ADD_ARG(int, nmom) = 0;
};
using OpacityOptions = std::shared_ptr<OpacityOptionsImpl>;

}  // namespace harp

#undef ADD_ARG
