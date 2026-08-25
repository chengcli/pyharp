// C/C++
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// disort
#include <disort/index.h>

// harp
#include <harp/utils/find_resource.hpp>
#include <harp/utils/read_data_tensor.hpp>

#include "water_liquid_mie.hpp"
#include "water_liquid_mie_dispatch.hpp"
#include "water_liquid_mie_impl.h"

namespace harp {

extern std::vector<double> species_weights;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kWaterDensity = 997.0;  // kg/m^3 at 25 C

torch::Tensor layer_field(torch::Tensor value, torch::Tensor const& conc,
                          char const* name) {
  value = value.to(conc.options());
  if (value.dim() == 0) {
    return value.expand({conc.size(0), conc.size(1)}).contiguous();
  }
  TORCH_CHECK(value.dim() == 2 && value.size(0) == conc.size(0) &&
                  value.size(1) == conc.size(1),
              "Mie water liquid expects ", name,
              " shape (ncol, nlyr) or a scalar; got ", value.sizes());
  return value.contiguous();
}

torch::Tensor spectral_field(torch::Tensor value,
                             torch::Tensor const& wavelength,
                             char const* name) {
  auto const nwave = wavelength.size(0);
  value = value.to(wavelength.options());
  if (value.dim() == 0) {
    return value.expand({nwave}).contiguous();
  }
  TORCH_CHECK(
      value.dim() == 1 && value.size(0) == nwave, "Mie water liquid expects ",
      name, " to be scalar or match the 1D spectral grid; got ", value.sizes());
  return value.contiguous();
}

torch::Tensor water_refractive_indices(torch::Tensor const& wavelength,
                                       torch::Tensor const& segelstein_water) {
  auto table = segelstein_water.to(wavelength.options());
  auto grid = table.select(1, 0).contiguous();
  TORCH_CHECK(torch::all(wavelength >= grid[0]).item<bool>() &&
                  torch::all(wavelength <= grid[-1]).item<bool>(),
              "Built-in liquid-water refractive index supports wavelength "
              "0.1--1000 um. Provide refractive_index_real and "
              "refractive_index_imag to override it.");

  auto upper = torch::searchsorted(grid, wavelength,
                                   /*out_int32=*/false, /*right=*/true);
  upper = torch::clamp(upper, 0, grid.size(0) - 1);
  auto lower = torch::clamp(upper - 1, 0, grid.size(0) - 1);
  auto exact_lower = upper == 0;
  lower = torch::where(exact_lower, upper, lower);

  auto w0 = grid.index_select(0, lower);
  auto w1 = grid.index_select(0, upper);
  auto n0 = table.select(1, 1).index_select(0, lower);
  auto n1 = table.select(1, 1).index_select(0, upper);
  auto k0 = table.select(1, 2).index_select(0, lower);
  auto k1 = table.select(1, 2).index_select(0, upper);
  auto denom = torch::where(w1 == w0, torch::ones_like(w1),
                            torch::log(w1) - torch::log(w0));
  auto frac = torch::where(w1 == w0, torch::zeros_like(w1),
                           (torch::log(wavelength) - torch::log(w0)) / denom);
  auto n = n0 + frac * (n1 - n0);
  auto k = torch::where(
      (k0 > 0.0) & (k1 > 0.0),
      torch::exp(torch::log(k0) + frac * (torch::log(k1) - torch::log(k0))),
      k0 + frac * (k1 - k0));
  return torch::stack({n, k}, 1).contiguous();
}

int mie_max_order(double max_x) {
  int const nstop =
      std::max(1, static_cast<int>(max_x + 4.05 * std::cbrt(max_x) + 2.0));
  int const derivative_order = nstop + 1;
  TORCH_CHECK(
      derivative_order < 2000000,
      "Mie size parameter is too large for an online calculation: ", max_x);
  return derivative_order + 1;
}

}  // namespace

void call_water_liquid_mie_cpu(at::TensorIterator& iter,
                               double molecular_weight, int nmom,
                               int max_order) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_water_liquid_mie_cpu", [&] {
    int const grain_size = iter.numel() / at::get_num_threads();
    using ComplexScalar = Complex<scalar_t>;
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::vector<ComplexScalar> work(
              static_cast<std::size_t>(3 * max_order));
          auto* work_ptr = work.data();
          for (int64_t i = 0; i < n; ++i) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto prop = reinterpret_cast<int64_t*>(data[1] + i * strides[1]);
            auto conc = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wave = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto re = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto density =
                reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            auto real = reinterpret_cast<scalar_t*>(data[6] + i * strides[6]);
            auto imag = reinterpret_cast<scalar_t*>(data[7] + i * strides[7]);
            auto const value = water_liquid_mie_property(
                *prop, *conc, *wave, *re, *density, *real, *imag,
                static_cast<scalar_t>(molecular_weight), nmom, work_ptr,
                max_order);
            *out = value;
          }
        },
        grain_size);
  });
}

}  // namespace harp

namespace at::native {

DEFINE_DISPATCH(call_water_liquid_mie);
REGISTER_ALL_CPU_DISPATCH(call_water_liquid_mie,
                          &harp::call_water_liquid_mie_cpu);

}  // namespace at::native

namespace harp {

MieWaterLiquidImpl::MieWaterLiquidImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->type().empty() || options->type() == "water-liquid-mie",
              "Mismatch opacity type: ", options->type(),
              " expecting 'water-liquid-mie'");
  TORCH_CHECK(options->species_ids().size() == 1,
              "Mie water liquid requires exactly one H2O(l) species");
  TORCH_CHECK(
      options->species_ids()[0] >= 0,
      "Invalid Mie water-liquid species_id: ", options->species_ids()[0]);
  TORCH_CHECK(options->nmom() >= 1, "Mie water liquid requires nmom >= 1; got ",
              options->nmom());
  reset();
}

void MieWaterLiquidImpl::reset() {
  segelstein_water = register_buffer(
      "segelstein_water",
      read_data_tensor(find_resource("opacity/cloud/segelstein_water.txt")));
  TORCH_CHECK(segelstein_water.dim() == 2 && segelstein_water.size(1) == 3,
              "Liquid-water refractive-index table must have columns "
              "wavelength, n, k; got ",
              segelstein_water.sizes());
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
  TORCH_CHECK(
      species_id < species_weights.size(),
      "Mie water-liquid species_id has no molecular weight: ", species_id);

  auto liquid_conc = conc.select(-1, species_id).contiguous();
  TORCH_CHECK(torch::all(torch::isfinite(liquid_conc)).item<bool>() &&
                  torch::all(liquid_conc >= 0.0).item<bool>(),
              "Mie water-liquid concentration must be finite and nonnegative");

  torch::Tensor wavelength;
  if (kwargs.count("wavelength") > 0) {
    wavelength = kwargs.at("wavelength").to(conc.options()).contiguous();
  } else if (kwargs.count("wavenumber") > 0) {
    auto wavenumber = kwargs.at("wavenumber").to(conc.options()).contiguous();
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
  TORCH_CHECK(torch::all(torch::isfinite(wavelength)).item<bool>() &&
                  torch::all(wavelength > 0.0).item<bool>(),
              "Mie water-liquid wavelength must be finite and positive");

  TORCH_CHECK(kwargs.count("re") > 0,
              "Mie water liquid requires droplet radius re [um]");
  auto re = layer_field(kwargs.at("re"), conc, "re").contiguous();
  TORCH_CHECK(torch::all(torch::isfinite(re)).item<bool>() &&
                  torch::all(re > 0.0).item<bool>(),
              "Mie water-liquid re must be finite and positive");

  torch::Tensor density;
  if (kwargs.count("water_density") > 0) {
    density = layer_field(kwargs.at("water_density"), conc, "water_density")
                  .contiguous();
  } else {
    density = torch::full({conc.size(0), conc.size(1)}, kWaterDensity,
                          conc.options());
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
    ref_real = spectral_field(kwargs.at("refractive_index_real"), wavelength,
                              "refractive_index_real");
    ref_imag = spectral_field(kwargs.at("refractive_index_imag"), wavelength,
                              "refractive_index_imag");
    TORCH_CHECK(torch::all(torch::isfinite(ref_real)).item<bool>() &&
                    torch::all(ref_real > 0.0).item<bool>() &&
                    torch::all(torch::isfinite(ref_imag)).item<bool>() &&
                    torch::all(ref_imag >= 0.0).item<bool>(),
                "Mie refractive indices require finite n > 0 and k >= 0");
  } else {
    auto const nk = water_refractive_indices(wavelength, segelstein_water);
    ref_real = nk.select(1, 0).contiguous();
    ref_imag = nk.select(1, 1).contiguous();
  }

  int64_t const ncol = conc.size(0);
  int64_t const nlyr = conc.size(1);
  int64_t const nwave = wavelength.size(0);
  int64_t const nprop = 2 + options->nmom();
  double const molecular_weight = species_weights.at(species_id);  // kg/mol

  auto max_x =
      (2.0 * kPi * re.max() / wavelength.min()).to(torch::kCPU).item<double>();
  int const max_order = mie_max_order(max_x);

  auto result = torch::empty({nwave, ncol, nlyr, nprop}, conc.options());
  auto prop =
      torch::arange(
          nprop,
          torch::TensorOptions().dtype(torch::kLong).device(conc.device()))
          .view({1, 1, 1, nprop})
          .expand({nwave, ncol, nlyr, nprop});
  auto conc_view =
      liquid_conc.view({1, ncol, nlyr, 1}).expand({nwave, ncol, nlyr, nprop});
  auto wave_view =
      wavelength.view({nwave, 1, 1, 1}).expand({nwave, ncol, nlyr, nprop});
  auto re_view = re.view({1, ncol, nlyr, 1}).expand({nwave, ncol, nlyr, nprop});
  auto density_view =
      density.view({1, ncol, nlyr, 1}).expand({nwave, ncol, nlyr, nprop});
  auto real_view =
      ref_real.view({nwave, 1, 1, 1}).expand({nwave, ncol, nlyr, nprop});
  auto imag_view =
      ref_imag.view({nwave, 1, 1, 1}).expand({nwave, ncol, nlyr, nprop});

  auto iter = at::TensorIteratorConfig()
                  .add_output(result)
                  .add_input(prop)
                  .add_input(conc_view)
                  .add_input(wave_view)
                  .add_input(re_view)
                  .add_input(density_view)
                  .add_input(real_view)
                  .add_input(imag_view)
                  .check_all_same_dtype(false)
                  .build();
  at::native::call_water_liquid_mie(iter.device_type(), iter, molecular_weight,
                                    options->nmom(), max_order);

  TORCH_CHECK(
      torch::all(torch::isfinite(result)).item<bool>() &&
          torch::all(result.select(-1, disort::IEX) >= 0.0).item<bool>(),
      "Mie water-liquid returned invalid optical properties");
  return result;
}

}  // namespace harp
