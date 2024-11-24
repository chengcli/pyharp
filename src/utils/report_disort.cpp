// C/C++
#include <sstream>

// harp
#include <rtsolver/cdisort213/cdisort.h>  // disort_state

#include "report_disort.hpp"

void report_disort_atmosphere(disort_state const& ds, std::ostream& os) {
  os << "- Levels = " << ds.nlyr << std::endl;
  os << "- Radiation Streams = " << ds.nstr << std::endl;
  os << "- Phase function moments = " << ds.nmom << std::endl;
}

void report_disort_output(disort_state const& ds, std::ostream& os) {
  os << "- User azimuthal angles = " << ds.nphi << std::endl << "  : ";
  for (int i = 0; i < ds.nphi; ++i) {
    os << ds.phi[i] / M_PI * 180. << ", ";
  }
  os << std::endl;
  os << "- User polar angles = " << ds.numu << std::endl << "  : ";
  for (int i = 0; i < ds.numu; ++i) {
    os << acos(ds.umu[i]) / M_PI * 180. << ", ";
  }
  os << std::endl;
  os << "- User optical depths = " << ds.ntau << std::endl << "  : ";
  for (int i = 0; i < ds.ntau; ++i) {
    os << ds.utau[i] << ", ";
  }
  os << std::endl;
}

void report_disort_bcs(disort_state const& ds, std::ostream& os) {
  os << "- Bottom temperature = " << ds.bc.btemp << std::endl;
  os << "- Albedo = " << ds.bc.albedo << std::endl;
  os << "- Top temperature = " << ds.bc.ttemp << std::endl;
  os << "- Top emissivity = " << ds.bc.temis << std::endl;
  os << "- Bottom isotropic illumination = " << ds.bc.fluor << std::endl;
  os << "- Top isotropic illumination = " << ds.bc.fisot << std::endl;
  os << "- Solar beam = " << ds.bc.fbeam << std::endl;
  os << "- Cosine of solar zenith angle = " << ds.bc.umu0 << std::endl;
  os << "- Solar azimuth angle = " << ds.bc.phi0 << std::endl;
}

void report_disort_flags(disort_state const& ds, std::ostream& os) {
  if (ds.flag.ibcnd) {
    os << "- Spectral boundary condition (ibcnd) = True" << std::endl;
  } else {
    os << "- Spectral boundary condition (ibcnd) = False" << std::endl;
  }

  if (ds.flag.usrtau) {
    os << "- User optical depth (usrtau) = True" << std::endl;
  } else {
    os << "- User optical depth (usrtau) = False" << std::endl;
  }

  if (ds.flag.usrang) {
    os << "- User angles (usrang) = True" << std::endl;
  } else {
    os << "- User angles (usrang) = False" << std::endl;
  }

  if (ds.flag.lamber) {
    os << "- Lambertian surface (lamber) = True" << std::endl;
  } else {
    os << "- Lambertian surface (lamber) = False" << std::endl;
  }

  if (ds.flag.planck) {
    os << "- Planck function (planck) = True" << std::endl;
  } else {
    os << "- Planck function (planck) = False" << std::endl;
  }

  if (ds.flag.spher) {
    os << "- Spherical correction (spher) = True" << std::endl;
  } else {
    os << "- Spherical correction (spher) = False" << std::endl;
  }

  if (ds.flag.onlyfl) {
    os << "- Only calculate fluxes (onlyfl) = True" << std::endl;
  } else {
    os << "- Only calculate fluxes (onlyfl) = False" << std::endl;
  }

  if (ds.flag.intensity_correction) {
    os << "- Intensity correction (intensity_correction) = True" << std::endl;
  } else {
    os << "- Intensity correction (intensity_correction) = False" << std::endl;
  }

  if (ds.flag.old_intensity_correction) {
    os << "- Old intensity correction (old_intensity_correction) = True"
       << std::endl;
  } else {
    os << "- Old intensity correction (old_intensity_correction) = False"
       << std::endl;
  }

  if (ds.flag.general_source) {
    os << "- General source function (general_source) = True" << std::endl;
  } else {
    os << "- General source function (general_source) = False" << std::endl;
  }

  if (ds.flag.output_uum) {
    os << "- Output uum (output_uum) = True" << std::endl;
  } else {
    os << "- Output uum (output_uum) = False" << std::endl;
  }
}

std::string report_disort(disort_state const& ds) {
  std::stringstream ss;

  ss << "Disort is configured with:" << std::endl;
  report_disort_flags(ds, ss);
  ss << "- Accuracy = " << ds.accur << std::endl;

  ss << "Boundary condition:" << std::endl;
  report_disort_bcs(ds, ss);

  ss << "Dimensions:" << std::endl;
  if (ds.dtauc != nullptr) {
    report_disort_atmosphere(ds, ss);
    report_disort_output(ds, ss);
    ss << "Disort is finalized.";
  } else {
    ss << "Disort is not yet finalized.";
  }

  return ss.str();
}
