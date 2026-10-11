"""Tests for the canonical WOA implementation (woa.py).

Reference: Mirjalili & Lewis (2016), "The Whale Optimization Algorithm".
"""

import numpy as np

import woa


def test_a_decreases_from_2_to_0():
    t = 50
    first = woa.decreasing_a(0, t)
    last = woa.decreasing_a(t - 1, t)
    assert first == 2.0 - 2.0 / t
    assert last == 0.0
    values = [woa.decreasing_a(i, t) for i in range(t)]
    assert all(b < a for a, b in zip(values, values[1:])), "a must decrease"


def test_a_stays_in_range():
    for i in range(100):
        a = woa.decreasing_a(i, 100)
        assert 0.0 <= a <= 2.0


def test_coefficients_in_expected_range():
    rng = np.random.default_rng(0)
    for _ in range(500):
        A, C = woa.coefficients(a=1.0, rng=rng)
        assert -1.0 <= A <= 1.0  # A in [-a, a]
        assert 0.0 <= C <= 2.0   # C = 2*r, r in [0, 1]


def test_clip_to_bounds():
    bounds = [(0.01, 0.1), (10.0, 200.0), (0.0001, 0.1), (0.05, 0.5), (0.1, 0.5)]
    out = woa.clip_to_bounds(np.array([-5.0, 500.0, 0.5, 0.3, 0.05]), bounds)
    assert out[0] == 0.01   # clamped to lower
    assert out[1] == 200.0  # clamped to upper
    assert out[3] == 0.3    # untouched inside range


def test_lambda_upper_bound_respected():
    """Regression test: lamda must be able to reach 0.5, not 0.1.

    (The legacy code clamped lamda with UPPER_BOUND_EPSILON = 0.1.)
    """
    bounds = [(0.05, 0.5)]

    def fitness(x):
        return -x[0]  # optimum pushes towards the upper bound

    best, _fit, _hist = woa.optimize(
        fitness, bounds, n_agents=10, n_iterations=30, seed=1
    )
    assert best[0] > 0.1, "lamda range must extend beyond the old 0.1 clamp"
    assert best[0] <= 0.5


def test_optimize_sphere_function():
    """WOA should (near-)minimise a simple convex function."""

    def sphere(x):
        return float(np.sum(x ** 2))

    best, best_fitness, history = woa.optimize(
        sphere,
        bounds=[(-5.0, 5.0)] * 3,
        n_agents=20,
        n_iterations=60,
        seed=42,
    )
    assert best_fitness < 1e-2
    assert all(-5.0 <= v <= 5.0 for v in best)
    assert history[-1] <= history[0], "best fitness must not worsen"


def test_optimize_is_reproducible_with_seed():
    def sphere(x):
        return float(np.sum(x ** 2))

    kwargs = dict(bounds=[(-5.0, 5.0)] * 2, n_agents=10, n_iterations=10, seed=7)
    _, f1, _ = woa.optimize(sphere, **kwargs)
    _, f2, _ = woa.optimize(sphere, **kwargs)
    assert f1 == f2
