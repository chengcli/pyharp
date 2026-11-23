#pragma once

namespace harp {
namespace constants {
//! The one home of kBoltz, hPlanck and Rgas, C++ and Python alike (Python reads
//! them from pyharp.constants): the exact SI values (2019 redefinition).

//! Molar gas constant N_A k_B [J/(mol K)], exact in SI
double const Rgas = 8.31446261815324;
double const Rgas_cgs = Rgas * 1.E7;
//! Boltzmann constant [J/K], exact in SI
double const kBoltz = 1.380649E-23;
double const kBoltz_cgs = kBoltz * 1.E7;
double const Lo = 2.68719E25;

//! Avogadro's Number @f$ N_{\mathrm{A}} @f$ [number/mol]
double const Avogadro = 6.02214179E23;

//! Planck constant [J s], exact in SI
double const hPlanck = 6.62607015E-34;
double const hPlanck_cgs = hPlanck * 1.E7;
double const cLight = 2.99792458E8;
double const cLight_cgs = 2.99792458E10;
double const stefanBoltzmann = 5.670374419E-8;

//! Electron Mass @f$ m_e @f$ [kg]
const double ElectronMass = 9.1093837015e-31;
}  // namespace constants
}  // namespace harp
