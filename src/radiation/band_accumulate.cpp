// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

// disort
#include <disort/index.h>

// harp
#include "band_accumulate.hpp"
#include "band_accumulate_dispatch.hpp"
#include "band_accumulate_impl.h"

namespace harp {

namespace {

bool tracks_gradient(torch::Tensor const& t) {
  return (torch::GradMode::is_enabled() && t.requires_grad()) ||
         t._fw_grad(/*level=*/0).defined();
}

// The in-place kernels carry no autograd graph and read whole rows through
// raw pointers, so they take only contiguous, same-typed CPU or CUDA tensors
// that nothing is differentiating. Everything else keeps the tensor
// expressions below, which produce the same bits.
bool kernel_applies(torch::Tensor const& prop, torch::Tensor const& other) {
  auto const dtype = prop.scalar_type();
  return (prop.device().is_cpu() || prop.device().is_cuda()) &&
         other.device() == prop.device() &&
         (dtype == torch::kFloat32 || dtype == torch::kFloat64) &&
         other.scalar_type() == dtype && prop.dim() == 4 &&
         prop.is_contiguous() && !tracks_gradient(prop) &&
         !tracks_gradient(other);
}

torch::Tensor divide_where_positive(torch::Tensor numerator,
                                    torch::Tensor denominator) {
  while (denominator.dim() < numerator.dim()) {
    denominator = denominator.unsqueeze(-1);
  }
  auto const positive = denominator > 0.0;
  auto const safe_denominator =
      torch::where(positive, denominator, torch::ones_like(denominator));
  return torch::where(positive, numerator / safe_denominator,
                      torch::zeros_like(numerator));
}

void call_accumulate_opacity_cpu(at::TensorIterator& iter, int nprop) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "accumulate_opacity_cpu", [&] {
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int64_t i = 0; i < n; ++i) {
        auto prop = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto k = reinterpret_cast<scalar_t const*>(data[1] + i * strides[1]);
        accumulate_opacity_row(prop, k, nprop);
      }
    });
  });
}

void call_finish_band_cpu(at::TensorIterator& iter, int nprop,
                          double albedo_cap) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "finish_band_cpu", [&] {
    auto const cap = static_cast<scalar_t>(albedo_cap);
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int64_t i = 0; i < n; ++i) {
        auto prop = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto dz = *reinterpret_cast<scalar_t const*>(data[1] + i * strides[1]);
        finish_band_row(prop, nprop, dz, cap);
      }
    });
  });
}

}  // namespace

void accumulate_opacity(torch::Tensor& prop, torch::Tensor const& kdata) {
  int const nprop = kdata.size(-1);
  TORCH_CHECK(nprop <= prop.size(-1), "opacity returns ", nprop,
              " properties, the band holds ", prop.size(-1));

  if (kernel_applies(prop, kdata) && kdata.dim() == 4 &&
      kdata.sizes().slice(0, 3) == prop.sizes().slice(0, 3)) {
    auto k = kdata.contiguous();
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(prop.select(-1, disort::IEX))
                    .add_owned_input(k.select(-1, disort::IEX))
                    .build();
    at::native::call_accumulate_opacity(iter.device_type(), iter, nprop);
    return;
  }

  // attenuation coefficients
  prop.select(-1, disort::IEX) += kdata.select(-1, disort::IEX);

  // attenuation weighted single scattering albedo
  if (nprop > 1) {
    prop.select(-1, disort::ISS) +=
        kdata.select(-1, disort::ISS) * kdata.select(-1, disort::IEX);
  }

  // attenuation + single scattering albedo weighted phase moments
  if (nprop > 2) {
    prop.narrow(-1, disort::IPM, nprop - 2) +=
        kdata.narrow(-1, disort::IPM, nprop - 2) *
        (kdata.select(-1, disort::ISS) * kdata.select(-1, disort::IEX))
            .unsqueeze(-1);
  }
}

void finish_band_properties(torch::Tensor& prop, torch::Tensor const& dz,
                            double albedo_cap) {
  int const nprop = prop.size(-1);

  if (kernel_applies(prop, dz) && (dz.dim() == 1 || dz.dim() == 2) &&
      dz.size(-1) == prop.size(2) &&
      (dz.dim() == 1 || dz.size(0) == prop.size(1))) {
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .add_owned_output(prop.select(-1, disort::IEX))
                    .add_owned_input(dz.unsqueeze(0))
                    .build();
    at::native::call_finish_band(iter.device_type(), iter, nprop, albedo_cap);
    return;
  }

  // average phase moments
  if (nprop > 2) {
    auto scattering = prop.select(-1, disort::ISS);
    auto moments = prop.narrow(-1, disort::IPM, nprop - 2);
    moments.copy_(divide_where_positive(moments, scattering));
  }

  // average single scattering albedo
  if (nprop > 1) {
    auto extinction = prop.select(-1, disort::IEX);
    auto scattering = prop.select(-1, disort::ISS);
    scattering.copy_(divide_where_positive(scattering, extinction));
  }

  // attenuation coefficients -> optical thickness
  prop.select(-1, disort::IEX) *= dz.unsqueeze(0);

  if (nprop > 1) {
    prop.select(-1, disort::ISS).clamp_max_(albedo_cap);
  }
}

}  // namespace harp

namespace at::native {

DEFINE_DISPATCH(call_accumulate_opacity);
DEFINE_DISPATCH(call_finish_band);
REGISTER_ALL_CPU_DISPATCH(call_accumulate_opacity,
                          &harp::call_accumulate_opacity_cpu);
REGISTER_ALL_CPU_DISPATCH(call_finish_band, &harp::call_finish_band_cpu);

}  // namespace at::native
