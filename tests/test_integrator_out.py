"""forward(..., out=) writes w0 u0 + w1 u1 + w2 u2 into out, bit for bit the allocating form: every scheme and
stage, FP32 and FP64, out distinct from the inputs and out aliasing u1."""

import pytest
import torch

import pyharp


def bits(t):
    return t.view(torch.int64 if t.dtype == torch.float64 else torch.int32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["rk1", "rk2", "rk3", "rk3s4"])
def test_out_equals_the_allocating_form(kind, dtype):
    op = pyharp.IntegratorOptions()
    op.type(kind)
    it = pyharp.Integrator(op)
    gen = torch.Generator().manual_seed(7)
    u0, u1, u2 = (torch.randn(5, 33, 17, generator=gen, dtype=torch.float64).mul(1e3).to(dtype) for _ in range(3))
    for s in range(len(it.stages)):
        ref = it.forward(s, u0, u1, u2)
        out = torch.empty_like(u0)
        assert torch.equal(bits(it.forward(s, u0, u1, u2, out=out)), bits(ref)) and torch.equal(bits(out), bits(ref))
        v = u1.clone()
        ptr = v.data_ptr()
        it.forward(s, u0, v, u2, out=v)
        assert v.data_ptr() == ptr and torch.equal(bits(v), bits(ref)), f"{kind} stage {s}: out aliasing u1"
    bad = torch.empty(0, dtype=dtype)  # wrong shape, empty so torch does not warn: resized per out=, no extra check
    got = it.forward(0, u0, u1, u2, out=bad)
    assert got.shape == bad.shape == u0.shape and torch.equal(bits(got), bits(it.forward(0, u0, u1, u2)))
    other = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(RuntimeError, match="share dtype and device"):
        it.forward(0, u0, u1, u2, out=torch.empty(u0.shape, dtype=other))
