#pragma once

namespace harp {

class AbsorberFactory {
 public:
  //! \brief Create an absorber from YAML node
  //!
  //! \param[in] my YAML node containing the current absorber
  //! \param[in] band_name name of the radiation band
  static Absorber create_from(YAML::Node const& my, std::string band_name);

  //! \brief Create absorbers from YAML node
  //!
  //! \param[in] names names of absorbers
  //! \param[in] band_name name of the radiation band
  //! \param[in] rad YAML node containing the radiation input file
  static std::vector<Absorber> create_from(
      std::vector<std::string> const& names, std::string band_name,
      YAML::Node const& rad);

 protected:
  //! \brief Only create an absorber based on its name and class
  //!
  //! \param[in] name name of the absorber
  //! \param[in] type class identifier of the absorber
  static AbsorberPtr create_absorber_partial(std::string name,
                                             std::string type);
};

}  // namespace harp
