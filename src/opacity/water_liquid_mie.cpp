// C/C++
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <vector>

// disort
#include <disort/index.h>

// harp
#include "water_liquid_mie.hpp"
#include "water_liquid_mie_data.hpp"

namespace harp {

extern std::vector<double> species_weights;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct MieEfficiency {
  double qext;
  double qsca;
  double g;
};

std::complex<double> lentz_log_derivative(
    std::complex<double> z, int n) {
  // Continued-fraction coefficients:
  // a_j = (-1)^(j+1) * (2*n + 2*j - 1) / z
  auto const a1 = (2.0 * n + 1.0) / z;
  auto const a2 = -(2.0 * n + 3.0) / z;
  // U_2 = P_2 / P_1
  // V_2 = Q_2 / Q_1
  auto u = a2 + 1.0 / a1;
  auto v = a2;
  // delta_2 = F_2 / F_1
  auto ratio = u / v;
  // F_2 = F_1 * delta_2
  auto runratio = a1 * ratio;
  int iterations = 0;
  int j = 3;
  while (std::abs(ratio - 1.0) > 1.0e-12) {
    TORCH_CHECK(
        ++iterations < 100000,
        "Mie logarithmic-derivative continued fraction did not converge");
    double const sign = (j % 2 == 1) ? 1.0 : -1.0;
    auto const aj = sign * (2.0 * n + 2.0 * j - 1.0) / z;
    u = aj + 1.0 / u;
    v = aj + 1.0 / v;
    ratio = u / v;
    runratio *= ratio;
    ++j;
  }
  return (-1.0 * n) / z + runratio;
}

torch::Tensor layer_field(torch::Tensor value, torch::Tensor const& conc,
                          char const* name) {
  value = value.to(torch::kCPU, torch::kFloat64);
  if (value.dim() == 0) {
    return value.expand({conc.size(0), conc.size(1)}).contiguous();
  }
  TORCH_CHECK(value.dim() == 2 && value.size(0) == conc.size(0) &&
                  value.size(1) == conc.size(1),
              "Mie water liquid expects ", name,
              " shape (ncol, nlyr) or a scalar; got ", value.sizes());
  return value.contiguous();
}

torch::Tensor spectral_field(torch::Tensor value, int64_t nwave,
                             char const* name) {
  value = value.to(torch::kCPU, torch::kFloat64);
  if (value.dim() == 0) {
    return value.expand({nwave}).contiguous();
  }
  TORCH_CHECK(value.dim() == 1 && value.size(0) == nwave,
              "Mie water liquid expects ", name,
              " to be scalar or match the 1D spectral grid; got ",
              value.sizes());
  return value.contiguous();
}

std::pair<double, double> water_refractive_index(double wavelength) {
  using namespace mie_water_liquid_data;
  TORCH_CHECK(wavelength >= kSegelsteinWater[0][0] &&
                  wavelength <=
                      kSegelsteinWater[kSegelsteinWaterSize - 1][0],
              "Built-in liquid-water refractive index supports wavelength "
              "0.1--1000 um; got ",
              wavelength,
              " um. Provide refractive_index_real and "
              "refractive_index_imag to override it.");

  auto const* begin = std::begin(kSegelsteinWater);
  auto const* end = std::end(kSegelsteinWater);
  auto const* upper = std::lower_bound(
      begin, end, wavelength,
      [](auto const& row, double value) { return row[0] < value; });
  if (upper == begin) return {upper[0][1], upper[0][2]};
  if (upper == end) {
    auto const& row = kSegelsteinWater[kSegelsteinWaterSize - 1];
    return {row[1], row[2]};
  }
  if ((*upper)[0] == wavelength) return {(*upper)[1], (*upper)[2]};

  auto const* lower = upper - 1;
  double const x0 = std::log((*lower)[0]);
  double const x1 = std::log((*upper)[0]);
  double const f = (std::log(wavelength) - x0) / (x1 - x0);
  double const n = (*lower)[1] + f * ((*upper)[1] - (*lower)[1]);
  double k;
  if ((*lower)[2] > 0.0 && (*upper)[2] > 0.0) {
    k = std::exp(std::log((*lower)[2]) +
                 f * (std::log((*upper)[2]) - std::log((*lower)[2])));
  } else {
    k = (*lower)[2] + f * ((*upper)[2] - (*lower)[2]);
  }
  return {n, k};
}

// Full homogeneous-sphere Lorenz-Mie efficiencies. The recurrence follows
// Bohren and Huffman (1983) and the numerical layout used by miepython:
// m uses the absorbing n-i*k convention and x is the external size parameter.
MieEfficiency mie_efficiency(std::complex<double> m, double x) {
  TORCH_CHECK(std::isfinite(x) && x > 0.0,
              "Mie size parameter must be finite and positive; got ", x);
  TORCH_CHECK(std::isfinite(m.real()) && std::isfinite(m.imag()) &&
                  m.real() > 0.0 && m.imag() <= 0.0,
              "Mie refractive index must have n > 0 and k >= 0");

  // Avoid cancellation in the Riccati-Bessel recurrence in the Rayleigh
  // limit. This branch is asymptotically exact as x -> 0.
  if (x < 1.0e-3) {
    auto const m2 = m * m;
    auto const alpha = (m2 - 1.0) / (m2 + 2.0);
    double const qsca = (8.0 / 3.0) * std::pow(x, 4) * std::norm(alpha);
    double const qabs = std::max(0.0, -4.0 * x * alpha.imag());
    return {qsca + qabs, qsca, 0.0};
  }

  // Wiscombe (1980) criterion for 8 < x < 4200.
  int const nstop =
      std::max(1, static_cast<int>(x + 4.05 * std::cbrt(x) + 2.0));
  auto const z = m * x;
  int const derivative_order = nstop + 1;
  TORCH_CHECK(derivative_order < 2000000,
              "Mie size parameter is too large for an online calculation: ",
              x);

  // D_n(z) = psi'_n(z)/psi_n(z). Wiscombe's criterion chooses the stable
  // recurrence direction; the downward start uses Lentz's continued
  // fraction, matching miepython's validated implementation.
  std::vector<std::complex<double>> d(
      static_cast<std::size_t>(derivative_order + 1));
  double const nreal = m.real();
  double const kappa = std::abs(m.imag());
  bool const use_downward =
      nreal < 1.0 || nreal > 10.0 || kappa > 10.0 ||
      x * kappa >= 3.9 - 10.8 * nreal + 13.78 * nreal * nreal;
  if (use_downward) {
    auto last = lentz_log_derivative(z, derivative_order);
    for (int n = derivative_order; n > 0; --n) {
      auto const nz = (1.0 * n) / z;
      last = nz - 1.0 / (last + nz);
      d[static_cast<std::size_t>(n - 1)] = last;
    }
  } else {
    auto const exponential = std::exp(std::complex<double>(0.0, -2.0) * z);
    d[1] = -1.0 / z +
           (1.0 - exponential) /
               ((1.0 - exponential) / z -
                std::complex<double>(0.0, 1.0) * (1.0 + exponential));
    for (int n = 2; n <= derivative_order; ++n) {
      auto const nz = (1.0 * n) / z;
      d[static_cast<std::size_t>(n)] =
          1.0 / (nz - d[static_cast<std::size_t>(n - 1)]) - nz;
    }
  }

  std::vector<std::complex<double>> a(static_cast<std::size_t>(nstop));
  std::vector<std::complex<double>> b(static_cast<std::size_t>(nstop));
  double psi_nm1 = std::sin(x); // n-1 (n minus 1) Riccati-Bessel function
  double psi_n = psi_nm1 / x - std::cos(x);
  std::complex<double> xi_nm1(psi_nm1, std::cos(x)); //psi_nm1 + i*chi_nm1
  std::complex<double> xi_n(psi_n, std::cos(x) / x + std::sin(x));

  for (int n = 1; n <= nstop; ++n) {
    auto const dn = d[static_cast<std::size_t>(n)];
    auto const nx = n / x;
    auto const da = dn / m + nx;
    auto const db = m * dn + nx;
    a[static_cast<std::size_t>(n - 1)] =
        (da * psi_n - psi_nm1) / (da * xi_n - xi_nm1); // Mie-scattering coefficient a_n
    b[static_cast<std::size_t>(n - 1)] =
        (db * psi_n - psi_nm1) / (db * xi_n - xi_nm1); // Mie-scattering coefficient b_n

    double const psi_np1 = (2.0 * n + 1.0) * psi_n / x - psi_nm1;
    auto const xi_np1 = (2.0 * n + 1.0) * xi_n / x - xi_nm1;
    psi_nm1 = psi_n;
    psi_n = psi_np1;
    xi_nm1 = xi_n;
    xi_n = xi_np1;
  }

  double ext_sum = 0.0;
  double sca_sum = 0.0;
  double g_sum = 0.0; 
  for (int n = 1; n <= nstop; ++n) {
    auto const an = a[static_cast<std::size_t>(n - 1)];
    auto const bn = b[static_cast<std::size_t>(n - 1)];
    double const weight = 2.0 * n + 1.0;
    ext_sum += weight * (an.real() + bn.real()); // Wiscombe 1980, eq.1a
    sca_sum += weight * (std::norm(an) + std::norm(bn)); // Wiscombe 1980, eq.1b
    g_sum += weight / (n * (n + 1.0)) *
             std::real(an * std::conj(bn)); //complex conjugate
    if (n < nstop) {
      auto const an1 = a[static_cast<std::size_t>(n)];
      auto const bn1 = b[static_cast<std::size_t>(n)];
      g_sum += n * (n + 2.0) / (n + 1.0) *
               std::real(an * std::conj(an1) + bn * std::conj(bn1)); // Wiscombe 1980, eq.1c
    }
  }

  double const factor = 2.0 / (x * x);
  double qext = factor * ext_sum;
  double const qsca = factor * sca_sum;
  TORCH_CHECK(std::isfinite(qext) && std::isfinite(qsca) && qext >= 0.0 &&
                  qsca >= 0.0,
              "Non-finite or negative Mie efficiency for x=", x);
  // Roundoff can make a nonabsorbing calculation very slightly violate
  // qext >= qsca. Enforce nonnegative absorption before forming the albedo.
  qext = std::max(qext, qsca);
  double const g = qsca > std::numeric_limits<double>::min()
                       ? std::clamp(4.0 * g_sum / (x * x * qsca), -0.999999,
                                    0.999999)
                       : 0.0;
  return {qext, qsca, g};
}

}  // namespace

MieWaterLiquidImpl::MieWaterLiquidImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->type().empty() ||
                  options->type() == "water-liquid-mie",
              "Mismatch opacity type: ", options->type(),
              " expecting 'water-liquid-mie'");
  TORCH_CHECK(options->species_ids().size() == 1,
              "Mie water liquid requires exactly one H2O(l) species");
  TORCH_CHECK(options->species_ids()[0] >= 0,
              "Invalid Mie water-liquid species_id: ",
              options->species_ids()[0]);
  TORCH_CHECK(options->nmom() >= 1,
              "Mie water liquid requires nmom >= 1; got ", options->nmom());
  reset();
}

torch::Tensor MieWaterLiquidImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(conc.dim() == 3,
              "Mie water liquid expects conc shape (ncol, nlyr, nspecies); "
              "got ",
              conc.sizes());
  auto const species_id = options->species_ids()[0];
  TORCH_CHECK(species_id < conc.size(2),
              "Invalid Mie water-liquid species_id: ", species_id,
              " for conc with ", conc.size(2), " species");
  TORCH_CHECK(species_id < species_weights.size(),
              "Mie water-liquid species_id has no molecular weight: ",
              species_id);

  auto liquid_conc = conc.select(-1, species_id)
                         .to(torch::kCPU, torch::kFloat64)
                         .contiguous();
  TORCH_CHECK(torch::all(torch::isfinite(liquid_conc)).item<bool>() &&
                  torch::all(liquid_conc >= 0.0).item<bool>(),
              "Mie water-liquid concentration must be finite and nonnegative");

  torch::Tensor wavelength;
  if (kwargs.count("wavelength") > 0) {
    wavelength = kwargs.at("wavelength");
  } else if (kwargs.count("wavenumber") > 0) {
    auto wavenumber = kwargs.at("wavenumber");
    TORCH_CHECK(torch::all(torch::isfinite(wavenumber)).item<bool>() &&
                    torch::all(wavenumber > 0.0).item<bool>(),
                "Mie water-liquid wavenumber must be finite and positive");
    wavelength = 1.0e4 / wavenumber;
  } else {
    TORCH_CHECK(
        false,
        "Mie water liquid requires wavenumber [cm^-1] or wavelength [um]");
  }
  TORCH_CHECK(wavelength.dim() == 1,
              "Mie water liquid expects a 1D spectral grid; got ",
              wavelength.sizes());
  wavelength = wavelength.to(torch::kCPU, torch::kFloat64).contiguous();
  TORCH_CHECK(torch::all(torch::isfinite(wavelength)).item<bool>() &&
                  torch::all(wavelength > 0.0).item<bool>(),
              "Mie water-liquid wavelength must be finite and positive");

  TORCH_CHECK(kwargs.count("re") > 0,
              "Mie water liquid requires droplet radius re [um]");
  auto re = layer_field(kwargs.at("re"), conc, "re");
  TORCH_CHECK(torch::all(torch::isfinite(re)).item<bool>() &&
                  torch::all(re > 0.0).item<bool>(),
              "Mie water-liquid re must be finite and positive");

  torch::Tensor density;
  if (kwargs.count("water_density") > 0) {
    density = layer_field(kwargs.at("water_density"), conc, "water_density");
  } else {
    density = torch::full(
        {conc.size(0), conc.size(1)},
        mie_water_liquid_data::kWaterDensity,
        torch::TensorOptions().dtype(torch::kFloat64));
  }
  TORCH_CHECK(torch::all(torch::isfinite(density)).item<bool>() &&
                  torch::all(density > 0.0).item<bool>(),
              "Mie water-liquid density must be finite and positive");

  bool const override_real = kwargs.count("refractive_index_real") > 0;
  bool const override_imag = kwargs.count("refractive_index_imag") > 0;
  TORCH_CHECK(override_real == override_imag,
              "Provide both refractive_index_real and "
              "refractive_index_imag, or neither");
  torch::Tensor ref_real;
  torch::Tensor ref_imag;
  if (override_real) {
    ref_real = spectral_field(kwargs.at("refractive_index_real"),
                              wavelength.size(0), "refractive_index_real");
    ref_imag = spectral_field(kwargs.at("refractive_index_imag"),
                              wavelength.size(0), "refractive_index_imag");
    TORCH_CHECK(torch::all(torch::isfinite(ref_real)).item<bool>() &&
                    torch::all(ref_real > 0.0).item<bool>() &&
                    torch::all(torch::isfinite(ref_imag)).item<bool>() &&
                    torch::all(ref_imag >= 0.0).item<bool>(),
                "Mie refractive indices require finite n > 0 and k >= 0");
  } else {
    ref_real = torch::empty_like(wavelength);
    ref_imag = torch::empty_like(wavelength);
    auto* wave_ptr = wavelength.data_ptr<double>();
    auto* real_ptr = ref_real.data_ptr<double>();
    auto* imag_ptr = ref_imag.data_ptr<double>();
    for (int64_t iw = 0; iw < wavelength.size(0); ++iw) {
      auto const [n, k] = water_refractive_index(wave_ptr[iw]);
      real_ptr[iw] = n;
      imag_ptr[iw] = k;
    }
  }

  auto result_cpu = torch::zeros(
      {wavelength.size(0), conc.size(0), conc.size(1), 2 + options->nmom()},
      torch::TensorOptions().dtype(torch::kFloat64));
  auto* conc_ptr = liquid_conc.data_ptr<double>();
  auto* wave_ptr = wavelength.data_ptr<double>();
  auto* re_ptr = re.data_ptr<double>();
  auto* density_ptr = density.data_ptr<double>();
  auto* real_ptr = ref_real.data_ptr<double>();
  auto* imag_ptr = ref_imag.data_ptr<double>();
  auto result = result_cpu.accessor<double, 4>();
  int64_t const ncol = conc.size(0);
  int64_t const nlyr = conc.size(1);
  double const molecular_weight = species_weights.at(species_id);  // kg/mol

  for (int64_t iw = 0; iw < wavelength.size(0); ++iw) {
    std::complex<double> const m(real_ptr[iw], -imag_ptr[iw]);
    std::unordered_map<double, MieEfficiency> efficiency_cache;
    for (int64_t icol = 0; icol < ncol; ++icol) {
      for (int64_t ilyr = 0; ilyr < nlyr; ++ilyr) {
        auto const layer = icol * nlyr + ilyr;
        double const molar_conc = conc_ptr[layer];
        if (molar_conc == 0.0) continue;
        double const radius_m = re_ptr[layer] * 1.0e-6;
        auto [cached, inserted] = efficiency_cache.try_emplace(
            re_ptr[layer], MieEfficiency{0.0, 0.0, 0.0});
        if (inserted) {
          double const x = 2.0 * kPi * re_ptr[layer] / wave_ptr[iw];
          cached->second = mie_efficiency(m, x);
        }
        auto const& mie = cached->second;
        double const mass_extinction =
            3.0 * mie.qext / (4.0 * density_ptr[layer] * radius_m); // kext = Qext*pi*r^2 / (4/3*rho_w*r^3) m^2/kg
        result[iw][icol][ilyr][disort::IEX] = // m^-1
            molar_conc * molecular_weight * mass_extinction;
        result[iw][icol][ilyr][disort::ISS] =
            mie.qext > 0.0 ? std::clamp(mie.qsca / mie.qext, 0.0, 1.0) : 0.0;
        double moment = mie.g;
        for (int imom = 0; imom < options->nmom(); ++imom) {
          result[iw][icol][ilyr][disort::IPM + imom] = moment;
          moment *= mie.g;
        }
      }
    }
  }

  TORCH_CHECK(torch::all(torch::isfinite(result_cpu)).item<bool>() &&
                  torch::all(result_cpu.select(-1, disort::IEX) >= 0.0)
                      .item<bool>(),
              "Mie water-liquid returned invalid optical properties");
  return result_cpu.to(conc.options());
}

}  // namespace harp
