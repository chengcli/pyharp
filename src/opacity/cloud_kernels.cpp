// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

// disort
#include <disort/index.h>

// harp
#include "cloud_kernels.hpp"
#include "cloud_kernels_dispatch.hpp"
#include "cloud_kernels_impl.h"

namespace harp {

namespace {

bool tracks_gradient(torch::Tensor const& t) {
  return (torch::GradMode::is_enabled() && t.requires_grad()) ||
         t._fw_grad(/*level=*/0).defined();
}

void call_fu_ice_fill_cpu(at::TensorIterator& iter, at::Tensor const& table,
                          int nmom, bool delta_scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "fu_ice_fill_cpu", [&] {
    auto rows = table.data_ptr<scalar_t>();
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int64_t i = 0; i < n; ++i) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto invalid = reinterpret_cast<uint8_t*>(data[1] + i * strides[1]);
        auto wave = *reinterpret_cast<int64_t const*>(data[2] + i * strides[2]);
        auto d = *reinterpret_cast<scalar_t const*>(data[3] + i * strides[3]);
        auto iwc = *reinterpret_cast<scalar_t const*>(data[4] + i * strides[4]);
        *invalid = fu_ice_row(out, rows + wave * fu::kWidth, d, iwc, nmom,
                              delta_scale);
      }
    });
  });
}

void call_cloud_switch_fill_cpu(at::TensorIterator& iter, int nmom) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cloud_switch_fill_cpu", [&] {
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int64_t i = 0; i < n; ++i) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto ice = reinterpret_cast<scalar_t const*>(data[1] + i * strides[1]);
        auto liquid =
            reinterpret_cast<scalar_t const*>(data[2] + i * strides[2]);
        cloud_switch_row(out, ice, liquid, nmom);
      }
    });
  });
}

}  // namespace

bool cloud_kernels_apply(std::initializer_list<torch::Tensor> tensors) {
  auto const& first = *tensors.begin();
  auto const device = first.device();
  auto const dtype = first.scalar_type();
  if (!device.is_cpu() && !device.is_cuda()) return false;
  if (dtype != torch::kFloat32 && dtype != torch::kFloat64) return false;
  for (auto const& t : tensors) {
    if (t.device() != device || t.scalar_type() != dtype ||
        tracks_gradient(t)) {
      return false;
    }
  }
  return true;
}

torch::Tensor fu_ice_fill(torch::Tensor& result, torch::Tensor const& table,
                          torch::Tensor const& dge, torch::Tensor const& iwc,
                          bool delta_scale) {
  TORCH_CHECK(result.is_contiguous() && result.dim() == 4,
              "fu_ice_fill: result must be contiguous (nwave, ncol, nlyr, "
              "nprop)");
  int64_t const nwave = result.size(0);
  int const nmom = result.size(-1) - 2;
  TORCH_CHECK(table.sizes() == torch::IntArrayRef({nwave, fu::kWidth}),
              "fu_ice_fill: coefficient table shape ", table.sizes());

  auto invalid = torch::empty(result.sizes().slice(0, 3),
                              result.options().dtype(torch::kUInt8));
  auto wave_index = torch::arange(nwave, result.options().dtype(torch::kLong))
                        .view({nwave, 1, 1});
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_owned_output(result.select(-1, disort::IEX))
                  .add_output(invalid)
                  .add_owned_input(wave_index)
                  .add_owned_input(dge.unsqueeze(0))
                  .add_owned_input(iwc.unsqueeze(0))
                  .build();
  at::native::call_fu_ice_fill(iter.device_type(), iter, table.contiguous(),
                               nmom, delta_scale);
  return invalid;
}

void cloud_switch_fill(torch::Tensor& result, torch::Tensor const& ice,
                       torch::Tensor const& liquid) {
  TORCH_CHECK(result.is_contiguous() && result.dim() == 4,
              "cloud_switch_fill: result must be contiguous (nwave, ncol, "
              "nlyr, nprop)");
  TORCH_CHECK(ice.sizes() == result.sizes() && liquid.sizes() == result.sizes(),
              "cloud_switch_fill: ice ", ice.sizes(), " and liquid ",
              liquid.sizes(), " must match the result ", result.sizes());
  int const nmom = result.size(-1) - 2;

  auto ice_c = ice.contiguous();
  auto liquid_c = liquid.contiguous();
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .add_owned_output(result.select(-1, disort::IEX))
                  .add_owned_input(ice_c.select(-1, disort::IEX))
                  .add_owned_input(liquid_c.select(-1, disort::IEX))
                  .build();
  at::native::call_cloud_switch_fill(iter.device_type(), iter, nmom);
}

}  // namespace harp

namespace at::native {

DEFINE_DISPATCH(call_fu_ice_fill);
DEFINE_DISPATCH(call_cloud_switch_fill);
REGISTER_ALL_CPU_DISPATCH(call_fu_ice_fill, &harp::call_fu_ice_fill_cpu);
REGISTER_ALL_CPU_DISPATCH(call_cloud_switch_fill,
                          &harp::call_cloud_switch_fill_cpu);

}  // namespace at::native
