// C/C++
#include <cmath>
#include <limits>
#include <optional>

// external
#include <gtest/gtest.h>

// torch
#include <torch/csrc/autograd/forward_grad.h>

// harp
#include <harp/opacity/scattering_functions.hpp>

// tests
#include "device_testing.hpp"

TEST_P(DeviceTest, HenyeyGreensteinMatchesCumprod) {
  auto g = 1.98 * torch::rand({37, 11, 5}, torch::device(device).dtype(dtype)) -
           0.99;

  for (int nmom = 0; nmom <= 6; ++nmom) {
    auto result = harp::henyey_greenstein(nmom, g);

    auto vec = g.sizes().vec();
    vec.push_back(nmom);
    EXPECT_EQ(result.sizes(), vec) << "nmom = " << nmom;
    if (nmom == 0) continue;

    auto expected = torch::cumprod(g.unsqueeze(-1).expand(vec), -1);
    // Up to g^2 the two are the same product. Beyond that cumprod can round
    // differently: on the CPU it accumulates float32 in double, and a CUDA
    // scan may group a long product differently from repeated
    // multiplication.
    bool bitwise = nmom <= 2 || (dtype == torch::kFloat64 &&
                                 (nmom <= 3 || device.type() == torch::kCPU));
    if (bitwise) {
      EXPECT_TRUE(torch::equal(result, expected)) << "nmom = " << nmom;
    } else {
      EXPECT_TRUE(torch::allclose(result, expected)) << "nmom = " << nmom;
    }
  }
}

TEST_P(DeviceTest, HenyeyGreensteinRejectsBadInput) {
  auto g = torch::full({3}, 0.5, torch::device(device).dtype(dtype));
  EXPECT_THROW(harp::henyey_greenstein(-1, g), c10::Error);

  g[1] = 1.0;
  EXPECT_THROW(harp::henyey_greenstein(2, g), c10::Error);
}

// Zero requested moments still depend on g: the empty result must stay in the
// graph, so that a backward through it gives a defined zero gradient.
TEST_P(DeviceTest, HenyeyGreensteinEmptyMomentsKeepGraph) {
  auto g = torch::tensor({0.25, -0.5}, torch::device(device).dtype(dtype))
               .requires_grad_(true);
  auto result = harp::henyey_greenstein(0, g);

  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 0}));
  EXPECT_EQ(result.scalar_type(), g.scalar_type());
  EXPECT_EQ(result.device(), g.device());
  ASSERT_TRUE(result.requires_grad()) << "nmom = 0 output is detached from g";

  result.backward(torch::empty_like(result));
  ASSERT_TRUE(g.grad().defined());
  EXPECT_TRUE(torch::equal(g.grad(), torch::zeros_like(g)));
}

// The moments are a new tensor: writing to them must neither fail nor reach g.
TEST_P(DeviceTest, HenyeyGreensteinOutputIsOwnedAndWritable) {
  auto values = torch::tensor({0.25, -0.5}, torch::device(device).dtype(dtype));
  for (int nmom : {0, 1, 3}) {
    auto g = values.clone().requires_grad_(true);
    auto result = harp::henyey_greenstein(nmom, g);

    EXPECT_FALSE(result.is_view()) << "nmom = " << nmom;
    EXPECT_FALSE(result.storage().is_alias_of(g.storage()))
        << "nmom = " << nmom;

    torch::NoGradGuard no_grad;
    EXPECT_NO_THROW(result.fill_(0.125)) << "nmom = " << nmom;
    EXPECT_TRUE(torch::equal(g, values)) << "nmom = " << nmom;
  }
}

// Integer and bool inputs pass the range check only when zero; their moments
// keep the input dtype.
TEST_P(DeviceTest, HenyeyGreensteinKeepsIntegerAndBoolDtypes) {
  for (auto int_dtype : {torch::kBool, torch::kInt8, torch::kUInt8,
                         torch::kInt16, torch::kInt32, torch::kInt64}) {
    auto g = torch::zeros({3}, torch::device(device).dtype(int_dtype));
    for (int nmom : {0, 1, 3}) {
      auto result = harp::henyey_greenstein(nmom, g);
      EXPECT_EQ(result.scalar_type(), int_dtype)
          << int_dtype << ", nmom = " << nmom;
      EXPECT_EQ(result.device(), g.device());
      EXPECT_EQ(result.sizes(), (std::vector<int64_t>{3, nmom}));
      EXPECT_TRUE(torch::all(result == 0).item<bool>())
          << int_dtype << ", nmom = " << nmom;
    }
  }
}

namespace {

// Allowances for comparing against float64 powers of the same g.
double rtol(torch::Dtype dtype) {
  return dtype == torch::kFloat64 ? 1.0e-11 : 3.0e-5;
}
double atol(torch::Dtype dtype) {
  return dtype == torch::kFloat64 ? 1.0e-12 : 1.0e-6;
}

// HG's original tolerance, the torch::allclose defaults (rtol 1e-5,
// atol 1e-8) used by HenyeyGreensteinMatchesCumprod since #165: results may
// differ in the last bits. Integer and bool results must match exactly.
bool within_tolerance(torch::Tensor const& a, torch::Tensor const& b) {
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type() ||
      a.device() != b.device()) {
    return false;
  }
  auto x = a.detach().cpu();
  auto y = b.detach().cpu();
  if (!x.is_floating_point()) return torch::equal(x, y);
  return torch::allclose(x, y);
}

}  // namespace

// Moments and their gradients from zero up to the largest g below one, for
// nmom from 0 to 64, against float64 powers of the same representable g.
TEST_P(DeviceTest, HenyeyGreensteinEdgeValuesAndGradients) {
  double const below_one = dtype == torch::kFloat64
                               ? std::nextafter(1.0, 0.0)
                               : double(std::nextafterf(1.0f, 0.0f));
  std::vector<double> values = {0.0,   -0.0,   0.5,     -0.5,      0.99,
                                -0.99, 0.9999, -0.9999, below_one, -below_one};
  auto g64 = torch::tensor(values, torch::kFloat64)
                 .to(dtype)
                 .to(torch::kFloat64);  // the values as dtype represents them
  for (int nmom : {0, 1, 2, 6, 32, 64}) {
    auto g = g64.to(device, dtype).requires_grad_(true);
    auto result = harp::henyey_greenstein(nmom, g);
    ASSERT_EQ(result.sizes(),
              (std::vector<int64_t>{int64_t(values.size()), nmom}));

    auto k = torch::arange(1, nmom + 1, torch::kFloat64);
    auto expected = g64.unsqueeze(-1).pow(k);
    EXPECT_TRUE(torch::allclose(result.to(torch::kCPU, torch::kFloat64),
                                expected, rtol(dtype), atol(dtype)))
        << "nmom = " << nmom;

    // Odd moments change sign with g, even ones do not, exactly.
    auto mirrored = harp::henyey_greenstein(nmom, -g.detach());
    auto parity = torch::pow(-1.0, k).to(device, dtype);
    EXPECT_TRUE(within_tolerance(mirrored, result.detach() * parity))
        << "nmom = " << nmom;

    auto w = torch::linspace(1.0, 2.0, nmom, torch::kFloat64);
    auto grad =
        torch::autograd::grad({(result * w.to(device, dtype)).sum()}, {g})[0];
    // d/dg sum_k w_k g^k = sum_k w_k k g^(k-1); at g = 0 only k = 1 is left.
    auto terms = (w * k) * g64.unsqueeze(-1).pow(k - 1);
    auto expected_grad = terms.sum(-1);
    auto scale = terms.abs().sum(-1);
    auto error = (grad.to(torch::kCPU, torch::kFloat64) - expected_grad).abs();
    EXPECT_TRUE(
        torch::all(error <= atol(dtype) + rtol(dtype) * scale).item<bool>())
        << "nmom = " << nmom << ", max error " << error.max().item<double>();
  }
}

// g = +-1, NaN and +-inf are rejected for every nmom, including 0.
TEST_P(DeviceTest, HenyeyGreensteinRejectsBoundaryAndNonFiniteG) {
  double const inf = std::numeric_limits<double>::infinity();
  for (double bad : {1.0, -1.0, std::nan(""), inf, -inf}) {
    auto g = torch::tensor({0.25, bad}, torch::device(device).dtype(dtype));
    for (int nmom : {0, 3}) {
      EXPECT_THROW(harp::henyey_greenstein(nmom, g), c10::Error)
          << "g = " << bad << ", nmom = " << nmom;
    }
  }
}

// An empty g gives an empty result that still backpropagates.
TEST_P(DeviceTest, HenyeyGreensteinEmptyInput) {
  for (auto shape : {std::vector<int64_t>{0}, std::vector<int64_t>{2, 0}}) {
    for (int nmom : {0, 3}) {
      auto g = torch::zeros(shape, torch::device(device).dtype(dtype))
                   .requires_grad_(true);
      auto result = harp::henyey_greenstein(nmom, g);
      auto expected = shape;
      expected.push_back(nmom);
      EXPECT_EQ(result.sizes(), expected);
      ASSERT_TRUE(result.requires_grad());
      result.backward(torch::empty_like(result));
      ASSERT_TRUE(g.grad().defined());
      EXPECT_EQ(g.grad().sizes(), g.sizes());
    }
  }
}

// Second derivatives through create_graph: d2/dg2 sum_k w_k g^k.
TEST_P(DeviceTest, HenyeyGreensteinSecondDerivative) {
  std::vector<double> values = {-0.9, -0.3, 0.0, 0.2, 0.7};
  auto g64 =
      torch::tensor(values, torch::kFloat64).to(dtype).to(torch::kFloat64);
  for (int nmom : {0, 1, 2, 6}) {
    auto g = g64.to(device, dtype).requires_grad_(true);
    auto w = torch::linspace(1.0, 2.0, nmom, torch::kFloat64);
    auto result = harp::henyey_greenstein(nmom, g);
    auto first = torch::autograd::grad({(result * w.to(device, dtype)).sum()},
                                       {g}, {}, /*retain_graph=*/true,
                                       /*create_graph=*/true)[0];
    if (nmom == 0) {
      // Nothing depends on g: the first derivative is zero and constant.
      EXPECT_TRUE(torch::equal(first, torch::zeros_like(g)));
      continue;
    }
    // A first derivative that does not depend on g has no graph; its
    // derivative is zero.
    auto second = first.requires_grad()
                      ? torch::autograd::grad({first.sum()}, {g})[0]
                      : torch::zeros_like(g);
    auto k = torch::arange(1, nmom + 1, torch::kFloat64);
    auto terms =
        (w * k * (k - 1)) * g64.unsqueeze(-1).pow((k - 2).clamp_min(0));
    terms = torch::where(k >= 2, terms, torch::zeros_like(terms));
    auto error =
        (second.to(torch::kCPU, torch::kFloat64) - terms.sum(-1)).abs();
    EXPECT_TRUE(
        torch::all(error <= atol(dtype) + rtol(dtype) * terms.abs().sum(-1))
            .item<bool>())
        << "nmom = " << nmom << ", max error " << error.max().item<double>();
  }
}

// The two-lobe mixture broadcasts its three inputs and passes gradients to
// all of them, including when no moment is requested.
TEST_P(DeviceTest, DoubleHenyeyGreensteinBroadcastGradients) {
  auto opt = torch::device(device).dtype(dtype);
  auto ff0 = torch::tensor({0.2, 0.5, 0.9, 0.4, 0.6, 0.3}, opt).view({2, 3, 1});
  auto g10 = torch::tensor({0.6, -0.2}, opt).view({2, 1});
  auto g20 = torch::tensor({-0.4, 0.1, 0.8}, opt).view({1, 3});

  for (int nmom : {0, 1, 4}) {
    auto ff = ff0.clone().requires_grad_(true);
    auto g1 = g10.clone().requires_grad_(true);
    auto g2 = g20.clone().requires_grad_(true);
    auto result = harp::double_henyey_greenstein(nmom, ff, g1, g2);
    EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 3, nmom}));
    ASSERT_TRUE(result.requires_grad()) << "nmom = " << nmom;

    auto k = torch::arange(1, nmom + 1, opt);
    auto p1 = g10.unsqueeze(-1).pow(k);  // (2, 1, nmom)
    auto p2 = g20.unsqueeze(-1).pow(k);  // (1, 3, nmom)
    auto expected = ff0 * p1 + (1.0 - ff0) * p2;
    EXPECT_TRUE(
        torch::allclose(result.detach(), expected, rtol(dtype), atol(dtype)))
        << "nmom = " << nmom;

    auto w =
        torch::linspace(1.0, 2.0, std::max(nmom, 1), opt).narrow(0, 0, nmom);
    auto grads = torch::autograd::grad({(result * w).sum()}, {ff, g1, g2});
    for (auto const& gr : grads) ASSERT_TRUE(gr.defined()) << "nmom = " << nmom;

    auto d1 = (w * k * g10.unsqueeze(-1).pow(k - 1));  // d p1 / d g1
    auto d2 = (w * k * g20.unsqueeze(-1).pow(k - 1));
    auto expected_ff = ((p1 - p2) * w).sum(-1, true);
    auto expected_g1 = (ff0 * d1).sum(-1).sum(1, true);
    auto expected_g2 = ((1.0 - ff0) * d2).sum(-1).sum(0, true);
    EXPECT_TRUE(
        torch::allclose(grads[0], expected_ff, rtol(dtype), atol(dtype)));
    EXPECT_TRUE(
        torch::allclose(grads[1], expected_g1, rtol(dtype), atol(dtype)));
    EXPECT_TRUE(
        torch::allclose(grads[2], expected_g2, rtol(dtype), atol(dtype)));
  }
}

namespace {

// Enters a forward-AD level and exits it again when the test ends, also
// when an assertion fails or an exception is thrown.
class DualLevel {
 public:
  DualLevel() : level_(torch::autograd::ForwardADLevel::get_next_idx()) {}
  ~DualLevel() { torch::autograd::ForwardADLevel::release_idx(level_); }
  DualLevel(DualLevel const&) = delete;
  DualLevel& operator=(DualLevel const&) = delete;
  uint64_t level() const { return level_; }

 private:
  uint64_t level_;
};

// The moments as separately rounded products g, g*g, (g*g)*g, ..., stacked.
torch::Tensor product_route(int nmom, torch::Tensor const& g) {
  if (nmom == 0) return g.unsqueeze(-1).narrow(-1, 0, 0).clone();
  std::vector<torch::Tensor> moments = {g};
  for (int k = 1; k < nmom; ++k) moments.push_back(moments.back() * g);
  return torch::stack(moments, -1);
}

// Bitwise comparison, so that -0 differs from +0; used to check that g is
// left untouched.
bool same_bits(torch::Tensor const& a, torch::Tensor const& b) {
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type()) {
    return false;
  }
  auto bits = [](torch::Tensor t) {
    t = t.detach().contiguous().cpu();
    switch (t.scalar_type()) {
      case torch::kFloat64:
        return t.view(torch::kInt64);
      case torch::kFloat32:
        return t.view(torch::kInt32);
      case torch::kFloat16:
      case torch::kBFloat16:
        return t.view(torch::kInt16);
      default:
        return t;
    }
  };
  return torch::equal(bits(a), bits(b));
}

// The 8192 values of the audit HG fixture, linspace(-0.8, 0.8) in float64.
std::vector<double> audit_g() {
  std::vector<double> out(8192);
  double const step = 1.6 / 8191;
  for (int i = 0; i < 8192; ++i) out[i] = i * step - 0.8;
  out.back() = 0.8;
  return out;
}

// Signed zeros, subnormals, the smallest normal and the values next to +-1
// in the given floating dtype.
std::vector<double> special_g(torch::Dtype dtype) {
  bool const f32 = dtype == torch::kFloat32;
  double const tiny = f32 ? double(std::numeric_limits<float>::denorm_min())
                          : std::numeric_limits<double>::denorm_min();
  double const normal = f32 ? double(std::numeric_limits<float>::min())
                            : std::numeric_limits<double>::min();
  double const below_one =
      f32 ? double(std::nextafterf(1.0f, 0.0f)) : std::nextafter(1.0, 0.0);
  return {0.0,       -0.0,       tiny, -tiny, 3 * tiny, normal, -normal,
          below_one, -below_one, 0.5,  -0.5,  0.3,      -0.7};
}

std::vector<torch::Tensor> fixtures(torch::Device device, torch::Dtype dtype) {
  auto opt = torch::TensorOptions().device(device).dtype(dtype);
  auto f64 = torch::kFloat64;
  torch::manual_seed(169);
  return {
      torch::tensor(audit_g(), f64).to(opt),
      torch::tensor(special_g(dtype), f64).to(opt),
      (1.98 * torch::rand({37, 11, 5}, torch::kFloat64) - 0.99).to(opt),
      torch::tensor(0.3, f64).to(opt),
      torch::tensor(-0.0, f64).to(opt),
      torch::zeros({0}, opt),
      torch::zeros({2, 0}, opt),
  };
}

}  // namespace

// Detached and tracked calls agree with g, g*g, (g*g)*g, ... stacked, to
// HG's original tolerance.
TEST_P(DeviceTest, HenyeyGreensteinTrackedAndDetachedAgree) {
  for (auto const& g : fixtures(device, dtype)) {
    for (int nmom : {0, 1, 2, 3, 6, 32, 64}) {
      auto expected = product_route(nmom, g);
      auto detached = harp::henyey_greenstein(nmom, g);
      auto tracked =
          harp::henyey_greenstein(nmom, g.clone().requires_grad_(true));
      EXPECT_EQ(detached.device(), g.device());
      EXPECT_TRUE(within_tolerance(detached, expected))
          << "detached, shape " << g.sizes() << ", nmom = " << nmom;
      EXPECT_TRUE(within_tolerance(tracked, expected))
          << "tracked, shape " << g.sizes() << ", nmom = " << nmom;
    }
  }
}

// The moments of a contiguous input without gradients are owned, writable
// and leave g alone, and are the same products under no_grad, inference
// mode, and grad mode re-enabled inside inference mode.
TEST_P(DeviceTest, HenyeyGreensteinPlainInputOwnedInEveryGradMode) {
  auto g = torch::tensor(audit_g(), torch::kFloat64).to(device, dtype);
  auto const values = g.clone();
  for (int nmom : {1, 3, 6, 32}) {
    auto result = harp::henyey_greenstein(nmom, g);
    EXPECT_TRUE(within_tolerance(result, product_route(nmom, g)))
        << "nmom = " << nmom;
    EXPECT_FALSE(result.is_view());
    EXPECT_FALSE(result.storage().is_alias_of(g.storage()));
    EXPECT_TRUE(result.is_contiguous());
    result.fill_(0.125);
    EXPECT_TRUE(same_bits(g, values)) << "nmom = " << nmom;
  }

  for (int mode = 0; mode < 3; ++mode) {
    std::optional<torch::NoGradGuard> no_grad;
    std::optional<c10::InferenceMode> inference;
    std::optional<torch::AutoGradMode> enable_grad;
    if (mode == 0) no_grad.emplace();
    if (mode >= 1) inference.emplace();
    if (mode == 2) enable_grad.emplace(true);
    auto result = harp::henyey_greenstein(6, g);
    EXPECT_TRUE(within_tolerance(result, product_route(6, g)))
        << "mode " << mode;
  }
}

// Inputs created inside inference mode, used there and after leaving it.
TEST_P(DeviceTest, HenyeyGreensteinInputsCreatedInInferenceMode) {
  torch::Tensor g, inside;
  std::vector<torch::Tensor> fixture;
  {
    c10::InferenceMode inference;
    g = torch::tensor(audit_g(), torch::kFloat64).to(device, dtype);
    ASSERT_TRUE(g.is_inference());
    fixture = fixtures(device, dtype);
    for (int nmom : {0, 1, 3, 6, 32}) {
      auto result = harp::henyey_greenstein(nmom, g);
      EXPECT_TRUE(result.is_inference()) << "nmom = " << nmom;
      EXPECT_FALSE(result.requires_grad()) << "nmom = " << nmom;
      EXPECT_TRUE(within_tolerance(result, product_route(nmom, g)))
          << "nmom = " << nmom;
      if (nmom > 0) {
        EXPECT_FALSE(result.storage().is_alias_of(g.storage()));
      }
    }
    for (auto const& f : fixture) {
      ASSERT_TRUE(f.is_inference());
      for (int nmom : {0, 1, 2, 6, 64}) {
        EXPECT_TRUE(within_tolerance(harp::henyey_greenstein(nmom, f),
                                     product_route(nmom, f)))
            << "shape " << f.sizes() << ", nmom = " << nmom;
      }
    }
    inside = harp::henyey_greenstein(6, g);
  }

  // The same inference tensors, used after inference mode has ended.
  auto outside = harp::henyey_greenstein(6, g);
  EXPECT_TRUE(within_tolerance(outside, inside));
  EXPECT_FALSE(outside.requires_grad());
  for (auto const& f : fixture) {
    EXPECT_TRUE(
        within_tolerance(harp::henyey_greenstein(3, f), product_route(3, f)))
        << "shape " << f.sizes();
  }
}

// Tracked, non-leaf and non-contiguous inputs and other dtypes: the same
// products, the input dtype, and for tracked inputs a graph back to g.
TEST_P(DeviceTest, HenyeyGreensteinTrackedStridedAndOtherDtypeInputs) {
  auto base = torch::tensor(audit_g(), torch::kFloat64).to(device, dtype);

  auto leaf = base.clone().requires_grad_(true);
  auto nonleaf = leaf * 1.0;
  auto strided = base.view({-1, 2}).select(-1, 0);  // stride 2
  ASSERT_FALSE(strided.is_contiguous());
  for (auto const& [name, g] :
       std::vector<std::pair<std::string, torch::Tensor>>{
           {"leaf", leaf}, {"nonleaf", nonleaf}, {"strided", strided}}) {
    auto result = harp::henyey_greenstein(6, g);
    EXPECT_TRUE(within_tolerance(result, product_route(6, g.detach()))) << name;
    EXPECT_EQ(result.requires_grad(), g.requires_grad()) << name;
  }

  // A tracked input under no_grad and inference mode: the same products.
  {
    torch::NoGradGuard no_grad;
    EXPECT_TRUE(within_tolerance(harp::henyey_greenstein(6, leaf),
                                 product_route(6, base)))
        << "tracked under no_grad";
  }
  {
    c10::InferenceMode inference;
    EXPECT_TRUE(within_tolerance(harp::henyey_greenstein(6, leaf),
                                 product_route(6, base)))
        << "tracked under inference mode";
  }

  // The non-leaf input passes its gradient back to the leaf.
  auto out = harp::henyey_greenstein(3, nonleaf);
  auto grad = torch::autograd::grad({out.sum()}, {leaf})[0];
  auto x = base.to(torch::kFloat64);
  auto expected = 1.0 + 2.0 * x + 3.0 * x * x;
  EXPECT_TRUE(torch::allclose(grad.to(torch::kFloat64), expected,
                              dtype == torch::kFloat64 ? 1e-12 : 1e-5, 1e-6));

  // Reduced precision, integer and bool inputs.
  for (auto other : {torch::kFloat16, torch::kBFloat16, torch::kInt32,
                     torch::kInt64, torch::kBool}) {
    auto g = other == torch::kFloat16 || other == torch::kBFloat16
                 ? base.to(other)
                 : torch::zeros({16}, torch::device(device).dtype(other));
    auto result = harp::henyey_greenstein(6, g);
    EXPECT_EQ(result.scalar_type(), other);
    EXPECT_TRUE(within_tolerance(result, product_route(6, g))) << other;
  }
}

// A dual tensor does not require grad but carries a forward tangent: with
// or without no_grad, its primal and JVP are those of the products.
TEST_P(DeviceTest, HenyeyGreensteinForwardModeMatchesTheProducts) {
  auto primal = torch::tensor(audit_g(), torch::kFloat64).to(device, dtype);
  auto tangent = torch::linspace(-1.0, 2.0, primal.numel(), primal.options());
  DualLevel level;
  auto dual = at::_make_dual(primal, tangent, level.level());
  ASSERT_FALSE(dual.requires_grad());
  for (bool no_grad_mode : {false, true}) {
    std::optional<torch::NoGradGuard> no_grad;
    if (no_grad_mode) no_grad.emplace();
    for (int nmom : {1, 3, 6, 32}) {
      auto out = harp::henyey_greenstein(nmom, dual);
      auto expected = product_route(nmom, dual);
      auto [out_primal, out_tangent] = at::_unpack_dual(out, level.level());
      auto [ref_primal, ref_tangent] =
          at::_unpack_dual(expected, level.level());
      ASSERT_TRUE(out_tangent.defined()) << "nmom = " << nmom;
      EXPECT_TRUE(within_tolerance(out_primal, ref_primal))
          << "nmom = " << nmom;
      EXPECT_TRUE(within_tolerance(out_tangent, ref_tangent))
          << "nmom = " << nmom;
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
