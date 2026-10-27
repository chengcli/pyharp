"""Closed box: the Runge-Kutta stages conserve the total of the state. A stage computes
w0 u0 + w1 u1 + w2 dt L(u1); with a conservative L (zero total) the total changes only if w0 + w1 != 1. The
per-step change is summed exactly (math.fsum over the new and the negated old cells), so it resolves far below
the round-off of the total itself."""

import math
from fractions import Fraction

import pytest
import torch

import pyharp


def integrator(kind):
    op = pyharp.IntegratorOptions()
    op.type(kind)
    return pyharp.Integrator(op)


def per_step_mass_change(kind, exchange_rate, n=200_000, steps=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    u = 1.0 + torch.rand(n, generator=gen, dtype=torch.float64)
    it = integrator(kind)
    exchange = lambda v: exchange_rate * (torch.roll(v, 1) - v)   # periodic, conservative
    changes = []
    for _ in range(steps):
        u0 = u1 = u
        for s in range(len(it.stages)):
            u1 = it.forward(s, u0, u1, exchange(u1))
        total = math.fsum(u.tolist())
        changes.append(math.fsum(u1.tolist() + (-u).tolist()) / total)
        u = u1
    return changes


@pytest.mark.parametrize("kind", ["rk1", "rk2", "rk3"])
def test_closed_box_at_rest_keeps_its_mass(kind):
    # nothing moves: only the state weights act on the total (rk3 before the fix: up to 3.7e-17 per step)
    worst = max(abs(c) for c in per_step_mass_change(kind, 0.0))
    assert worst <= 2e-18, f"{kind}: per-step relative mass change up to {worst:.2e}"


@pytest.mark.parametrize("kind", ["rk1", "rk2", "rk3"])
def test_closed_box_with_exchange_keeps_its_mass(kind):
    # a conservative exchange between cells: what remains is the round-off of the exchange term itself, a few
    # 1e-18 for any weights that sum to 1 (rk3 before the fix: mean -5.8e-17 per step)
    changes = per_step_mass_change(kind, 1e-3)
    mean = sum(changes) / len(changes)
    assert abs(mean) <= 5e-18, f"{kind}: mean per-step relative mass change {mean:+.2e}"


@pytest.mark.parametrize("kind", ["rk1", "rk2", "rk3"])
def test_state_weights_sum_to_exactly_one(kind):
    for s, w in enumerate(integrator(kind).stages):
        assert Fraction(w.wght0()) + Fraction(w.wght1()) == 1, f"{kind} stage {s}: {w.wght0()!r} + {w.wght1()!r}"
