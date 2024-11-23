// C/C++
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// opacity

// harp
#include <opacity/attenuator.hpp>
#include <opacity/parse_radiation_direction.hpp>

#include "radiation.hpp"
#include "radiation_band.hpp"
// #include "rt_solvers.hpp"

namespace harp {
RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const &options_)
    : options(options_) {
  reset();
}

void RadiationBandImpl::reset() {
  wgt =
      register_buffer("wgt", torch::tensor({options.nspec()}, torch::kFloat32));

  opt = register_buffer("opt", torch::zeros({3 + options.nstr(), options.nc3(),
                                             options.nc2(), options.nc1()},
                                            torch::kFloat32));

  auto str = options.outdirs();
  if (!str.empty()) {
    rayOutput = register_buffer("rayOutput", parse_radiation_directions(str));
  }

  // set attenuators
  for (auto name : options.absorbers()) {
    absorbers[name] = Absorber(options.absorber_options().at(name));
  }

  if (my["opacity"]) {
    if (!my["opacity"].IsSequence()) {
      throw RuntimeError("RadiationBand", "opacity must be a sequence");
    }

    auto names = my["opacity"].as<std::vector<std::string>>();
    absorbers_ = AbsorberFactory::CreateFrom(names, GetName(), rad);

    if (load_opacity) {
      for (auto &ab : absorbers_) {
        ab->LoadOpacity(RadiationBandsFactory::GetBandId(myname));
        // Correlated-k absorbers need to modify the spectral grid
        ab->ModifySpectralGrid(pgrid_->spec);
      }
    }
  }

  // set flags
  if (my["flags"]) {
    if (!my["flags"].IsSequence()) {
      throw RuntimeError("RadiationBand", "flags must be a sequence");
    }

    auto flag_strs = my["flags"].as<std::vector<std::string>>();
    for (auto flag : flag_strs) {
      SetFlag(RadiationHelper::parse_radiation_flags(flag));
    }
  }

  // set rt solver
  // rt_solver = CreateRTSolverFrom(my["rt-solver"].as<std::string>(), rad);
}

void RadiationBand::Resize(int nc1, int nc2, int nc3, int nstr,
                           MeshBlock const *pmb) {
  // allocate memory for spectral properties
  tem_.resize(nc1);
  temf_.resize(nc1 + 1);

  tau_.NewAthenaArray(pgrid_->spec.size(), nc1);
  tau_.ZeroClear();

  ssa_.NewAthenaArray(pgrid_->spec.size(), nc1);
  ssa_.ZeroClear();

  pmom_.NewAthenaArray(pgrid_->spec.size(), nc1, nstr + 1);
  pmom_.ZeroClear();

  // spectral grids properties
  toa_.NewAthenaArray(pgrid_->spec.size(), rayOutput_.size(), nc3, nc2);
  toa_.ZeroClear();

  flxup_.NewAthenaArray(pgrid_->spec.size(), nc3, nc2, nc1);
  flxup_.ZeroClear();

  flxdn_.NewAthenaArray(pgrid_->spec.size(), nc3, nc2, nc1);
  flxdn_.ZeroClear();

  // band properties
  btau.NewAthenaArray(nc3, nc2, nc1);
  bssa.NewAthenaArray(nc3, nc2, nc1);
  bpmom.NewAthenaArray(nstr + 1, nc3, nc2, nc1);

  btoa.NewAthenaArray(rayOutput_.size(), nc3, nc2);
  bflxup.NewAthenaArray(nc3, nc2, nc1 + 1);
  bflxdn.NewAthenaArray(nc3, nc2, nc1 + 1);

  // exchange buffer
  pexv = std::make_shared<LinearExchanger<Real, 2>>(GetName());

  int nlayers = GetNumLayers();
  int npmom = GetNumPhaseMoments();
  pexv->send_buffer[0].resize(temf_.size());
  pexv->send_buffer[1].resize(nlayers * (npmom + 3));

  pexv->Regroup(pmb, X1DIR);
  int nblocks = pexv->GetGroupSize();
  pexv->recv_buffer[0].resize(nblocks * pexv->send_buffer[0].size());
  pexv->recv_buffer[1].resize(nblocks * pexv->send_buffer[1].size());

  if (psolver_ != nullptr) {
    psolver_->Resize(nblocks * (nc1 - 2 * NGHOST), nstr);
  }
}
RadiationBand const *RadiationBand::CalBandFlux(MeshBlock const *pmb, int k,
                                                int j) {
  // reset flux of this column
  for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
    bflxup(k, j, i) = 0.;
    bflxdn(k, j, i) = 0.;
  }

  psolver_->Prepare(pmb, k, j);
  psolver_->CalBandFlux(pmb, k, j);

  return this;
}

RadiationBand const *RadiationBand::CalBandRadiance(MeshBlock const *pmb, int k,
                                                    int j) {
  // reset radiance of this column
  for (int n = 0; n < GetNumOutgoingRays(); ++n) {
    btoa(n, k, j) = 0.;
  }

  psolver_->Prepare(pmb, k, j);
  psolver_->CalBandRadiance(pmb, k, j);

  return this;
}

void RadiationBand::WriteAsciiHeader(OutputParameters const *pout) const {
  if (!TestFlag(RadiationFlags::WriteBinRadiance)) return;

  char fname[80], number[6];
  snprintf(number, sizeof(number), "%05d", pout->file_number);
  snprintf(fname, sizeof(fname), "%s.radiance.%s.txt", GetName().c_str(),
           number);
  FILE *pfile = fopen(fname, "w");

  fprintf(pfile, "# Bin Radiances of Band %s: %.3g - %.3g\n", GetName().c_str(),
          wrange_.first, wrange_.second);
  fprintf(pfile, "# Ray output size: %lu\n", rayOutput_.size());

  fprintf(pfile, "# Polar angles: ");
  for (auto &ray : rayOutput_) {
    fprintf(pfile, "%.3f", rad2deg(acos(ray.mu)));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "# Azimuthal angles: ");
  for (auto &ray : rayOutput_) {
    fprintf(pfile, "%.3f", rad2deg(ray.phi));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "#%12s%12s", "wave1", "wave2");
  for (size_t j = 0; j < rayOutput_.size(); ++j) {
    fprintf(pfile, "%12s%lu", "Radiance", j + 1);
  }
  fprintf(pfile, "%12s\n", "weight");

  fclose(pfile);
}

void RadiationBand::WriteAsciiData(OutputParameters const *pout) const {
  if (!TestFlag(RadiationFlags::WriteBinRadiance)) return;

  char fname[80], number[6];
  snprintf(number, sizeof(number), "%05d", pout->file_number);
  snprintf(fname, sizeof(fname), "%s.radiance.%s.txt", GetName().c_str(),
           number);
  FILE *pfile = fopen(fname, "w");

  for (size_t i = 0; i < pgrid_->spec.size(); ++i) {
    fprintf(pfile, "%13.3g%12.3g", pgrid_->spec[i].wav1, pgrid_->spec[i].wav2);
    for (size_t j = 0; j < rayOutput_.size(); ++j) {
      fprintf(pfile, "%12.3f", toa_(i, j));
    }
    if (TestFlag(RadiationFlags::Normalize) &&
        (wrange_.first != wrange_.second)) {
      fprintf(pfile, "%12.3g\n",
              pgrid_->spec[i].wght / (wrange_.second - wrange_.first));
    } else {
      fprintf(pfile, "%12.3g\n", pgrid_->spec[i].wght);
    }
  }

  fclose(pfile);
}

std::string RadiationBand::ToString() const {
  std::stringstream ss;
  ss << "RadiationBand: " << GetName() << std::endl;
  ss << "Absorbers: ";
  for (auto &ab : absorbers_) {
    ss << ab->GetName() << ", ";
  }
  ss << std::endl << "RT-Solver: " << psolver_->GetName();
  return ss.str();
}

std::shared_ptr<RadiationBand::RTSolver> RadiationBand::CreateRTSolverFrom(
    std::string const &rt_name, YAML::Node const &rad) {
  std::shared_ptr<RTSolver> psolver;

  if (rt_name == "Lambert") {
    psolver = std::make_shared<RTSolverLambert>(this, rad);
#ifdef RT_DISORT
  } else if (rt_name == "Disort") {
    psolver = std::make_shared<RTSolverDisort>(this, rad);
#endif  // RT_DISORT
  } else {
    throw NotFoundError("RadiationBand", rt_name);
  }

  return psolver;
}

}  // namespace harp
