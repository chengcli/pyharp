#ifndef SRC_HARP_RADIATION_BAND_HPP_
#define SRC_HARP_RADIATION_BAND_HPP_

// C/C++ headers
#include <map>
#include <memory>
#include <string>
#include <vector>

// athena
#include <athena/athena.hpp>  // Real

// canoe
#include <virtual_groups.hpp>

// exchanger
#include <exchanger/exchanger.hpp>

// harp
#include "spectral_grid.hpp"

namespace YAML {
class Node;
}

class OutputParameters;
class Absorber;
class SpectralGridBase;
struct Direction;

class RadiationBand {
 public:  // public access data
  // implementation of RT Solver
  class RTSolver;
  class RTSolverLambert;
  class RTSolverDisort;

  std::shared_ptr<LinearExchanger<Real, 2>> pexv;

  //! band optical depth
  torch::Tensor btau;

  //! band single scattering albedo
  torch::Tensor bssa;

  //! band phase function moments
  torch::Tensor bpmom;

  //! band upward flux (shallow reference to fluxup)
  torch::Tensor bflxup;

  //! band downward flux (shallow reference to flxdn)
  torch::Tensor bflxdn;

  //! \brief band top-of-the-atmosphere radiance 
  torch::Tensor btoa;

 public:  // constructor and destructor
  RadiationBand(std::string name, YAML::Node const &rad, bool load_opacity);
  virtual ~RadiationBand();

 public:  // member functions
  //! \brief Allocate memory for radiation band
  void Resize(int nc1, int nc2, int nc3, int nstr, MeshBlock const *pmb);

  //! \brief Create radiative transfer solver from YAML node
  //!
  //! \param[in] rad YAML node containing whole radiation configuration
  std::shared_ptr<RTSolver> CreateRTSolverFrom(std::string const &name,
                                               YAML::Node const &rad);

  //! Get number of spectral grids
  size_t GetNumSpecGrids() const { return pgrid_->spec.size(); }

  //! Get number of absorbers
  size_t GetNumAbsorbers() const { return absorbers_.size(); }

  std::vector<std::shared_ptr<Absorber>> const &Absorbers() const {
    return absorbers_;
  }

  //! Get number of phase function moments
  size_t GetNumPhaseMoments() const { return pmom_.GetDim1() - 1; }

  //! Get number of phase function moments
  size_t GetNumLayers() const { return tem_.size() - 2 * NGHOST; }

  //! Get range of wavenumbers
  std::pair<Real, Real> GetRange() const { return wrange_; }

  //! Get an individual absorber
  std::shared_ptr<Absorber> GetAbsorber(int i) { return absorbers_[i]; }

  //! Get an individual absorber by name
  std::shared_ptr<Absorber> GetAbsorberByName(std::string const &name);

  //! Get number of outgoing rays
  size_t GetNumOutgoingRays() { return rayOutput_.size(); }

  //! Get cosine polar angle of the outgoing ray
  Real GetCosinePolarAngle(int n) const { return rayOutput_[n].mu; }

  //! Get azimuthal angle of the outgoing ray
  Real GetAzimuthalAngle(int n) const { return rayOutput_[n].phi; }

 public:  // python bindings
  Real const *_GetToaPtr() const { return toa_.data(); }
  Real const *_GetTauPtr() const { return tau_.data(); }

 public:  // inbound functions
  //! \brief Set spectral properties for an air column
  //!
  //! \param[in] air mole fraction representation of air column
  //! \param[in] pcoord coordinates
  //! \param[in] gH0 grav * H0
  //! \param[in] k horizontal index
  //! \param[in] j horizontal index
  void SetSpectralProperties(AirColumn &air, Real const *x1f, Real gH0 = 0,
                             int k = 0, int j = 0);

  //! \brief Calculate band radiative fluxes
  RadiationBand const *CalBandFlux(MeshBlock const *pmb, int k, int j);

  //! \brief Calculate band radiances
  RadiationBand const *CalBandRadiance(MeshBlock const *pmb, int k, int j);

  //! \brief Set outgoing ray directions
  //!
  //! \param[in] rayOutput outgoing ray directions
  void SetOutgoingRays(std::vector<Direction> const &ray) { rayOutput_ = ray; }

 public:  // ASCIIOutputGroup functions
  void WriteAsciiHeader(OutputParameters const *) const override;
  void WriteAsciiData(OutputParameters const *) const override;

 public:  // StringReprGroup functions
  std::string ToString() const override;

 public:  // Exchanger functions
  //! \brief Pack temperature at cell face into send buffer 0
  void packTemperature();

  //! \brief Unpack temperature at cell face from receive buffer 0
  bool unpackTemperature(void *arg);

  //! \brief Pack data in all spectral grids into send buffer 1
  //!
  //! \param[in] b spectral bin index
  void packSpectralProperties();

  //! \brief Unpack data from receive buffer 1 into spectral grid b
  void unpackSpectralProperties(int b, void *arg);

 protected:
  //! radiative transfer solver
  std::shared_ptr<RTSolver> psolver_;

  //! spectral grid
  std::shared_ptr<SpectralGridBase> pgrid_;

  //! \brief identify wave- number (wavelength, frequency) range
  std::pair<Real, Real> wrange_;

  //! all absorbers
  std::vector<std::shared_ptr<Absorber>> absorbers_;

  //! outgoing rays
  std::vector<Direction> rayOutput_;

  //! \brief spectral bin optical depth
  //!
  //! This is a two-dimensional array. The first dimension is the size of the
  //! spectral bin and the second dimension is the size of the vertical
  //! dimension.
  torch::Tensor tau_;

  //! \brief spectral bin single scattering albedo
  //!
  //! This is a two-dimensional array. The first dimension is the size of the
  //! spectral bin and the second dimension is the size of the vertical
  //! dimension.
  torch::Tensor ssa_;

  //! \brief spectral bin top-of-the-atmosphere radiance
  //!
  //! This is a two-dimensional array. The first dimension is the size of the
  //! spectral bin and the second dimension is the size of the vertical
  //! dimension.
  torch::Tensor toa_;

  //! \brief spectral bin upward flux
  //!
  //! This is a two-dimensional array. The first dimension is the size of the
  //! spectral bin and the second dimension is the size of the vertical
  //! dimension.
  torch::Tensor flxup_;

  //! \brief spectral bin downward flux
  //!
  //! This is a two-dimensional array. The first dimension is the size of the
  //! spectral bin and the second dimension is the size of the vertical
  //! dimension.
  torch::Tensor flxdn_;

  //! \brief spectral bin phase function moments
  //!
  //! This is a three-dimensional array. The first dimension is the size of the
  //! spectral bin. The second dimension is the size of the vertical dimension.
  //! The third dimension is the size of the phase function moments (1 + npmom).
  torch::Tensor pmom_;

  //! temperature at cell center
  std::vector<Real> tem_;

  //! temperature at cell boundary (face)
  std::vector<Real> temf_;
};

using RadiationBandPtr = std::shared_ptr<RadiationBand>;
using RadiationBandContainer = std::vector<RadiationBandPtr>;

class RadiationBandsFactory {
 public:
  static RadiationBandContainer CreateFrom(ParameterInput *pin,
                                           std::string key);
  static RadiationBandContainer CreateFrom(std::string filename);

  static int GetBandId(std::string const &bname) {
    if (band_id_.find(bname) == band_id_.end()) {
      return -1;
    }
    return band_id_.at(bname);
  }

 protected:
  static std::map<std::string, int> band_id_;
  static int next_band_id_;
};

#endif  // SRC_HARP_RADIATION_BAND_HPP_