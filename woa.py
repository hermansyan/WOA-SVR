# -*- coding: utf-8 -*-
"""Whale Optimization Algorithm (WOA).

Canonical implementation following Mirjalili & Lewis (2016),
"The Whale Optimization Algorithm", Advances in Engineering Software 95:51-67.

Mathematical model
------------------
Encircling prey (Eq. 2.1-2.2)::

    D = |C * X*(t) - X(t)|
    X(t+1) = X*(t) - A * D

Spiral bubble-net (Eq. 2.5)::

    X(t+1) = D' * exp(b*l) * cos(2*pi*l) + X*(t),  D' = |X*(t) - X(t)|

Search for prey / exploration (Eq. 2.7-2.8)::

    D = |C * X_rand - X|
    X(t+1) = X_rand - A * D

Coefficients (Eq. 2.3-2.4)::

    A = 2*a*r - a        with a decreasing linearly from 2 to 0
    C = 2*r              with r ~ U(0, 1)

Update rule (Eq. 2.6): with p ~ U(0, 1),
  - p < 0.5 and |A| < 1  -> encircle the best solution (exploitation)
  - p < 0.5 and |A| >= 1 -> move towards a random agent (exploration)
  - p >= 0.5             -> spiral update around the best solution
"""

import math

import numpy as np


def decreasing_a(iteration, max_iterations):
    """Control parameter ``a``: decreases linearly from 2 to 0.

    This is the canonical schedule from the WOA paper. It drives the
    exploration/exploitation balance: |A| >= 1 favours exploration,
    |A| < 1 favours exploitation.
    """
    return 2.0 - (iteration + 1) / max_iterations * 2.0


def coefficients(a, rng):
    """Draw the WOA coefficient vectors A and C for one agent."""
    r = rng.random()
    A = 2.0 * a * r - a
    C = 2.0 * r
    return A, C


def clip_to_bounds(position, bounds):
    """Amend a position that left the search space (paper pseudo-code)."""
    clipped = np.array(
        [min(max(x, lo), hi) for x, (lo, hi) in zip(position, bounds)],
        dtype=float,
    )
    return clipped


def optimize(
    fitness_fn,
    bounds,
    n_agents=30,
    n_iterations=50,
    spiral_b=2.0,
    seed=None,
    verbose=False,
):
    """Minimize ``fitness_fn`` with the Whale Optimization Algorithm.

    Parameters
    ----------
    fitness_fn : callable
        ``fitness_fn(x) -> float``; lower is better. ``x`` is a 1-D
        numpy array of length ``len(bounds)``.
    bounds : sequence of (low, high)
        Search-space bounds per dimension.
    n_agents : int
        Whale population size.
    n_iterations : int
        Maximum number of iterations.
    spiral_b : float
        Shape constant of the logarithmic spiral (paper: ``b``).
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    (best_position, best_fitness, history)
        Best solution found, its fitness, and the best-fitness history.
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    # Initialise population uniformly inside the bounds: shape (dim, n_agents)
    population = np.array(
        [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_agents)]
    ).T

    fitness = np.array([fitness_fn(population[:, i]) for i in range(n_agents)])
    best_idx = int(np.argmin(fitness))
    best_position = population[:, best_idx].copy()
    best_fitness = float(fitness[best_idx])
    history = [best_fitness]

    for iteration in range(n_iterations):
        a = decreasing_a(iteration, n_iterations)
        for i in range(n_agents):
            current = population[:, i]
            A, C = coefficients(a, rng)
            p = rng.random()

            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey (Eq. 2.1-2.2)
                    D = abs(C * best_position - current)
                    new_position = best_position - A * D
                else:
                    # Search for prey: random agent (Eq. 2.7-2.8)
                    j = rng.integers(0, n_agents)
                    random_agent = population[:, j]
                    D = abs(C * random_agent - current)
                    new_position = random_agent - A * D
            else:
                # Spiral bubble-net (Eq. 2.5)
                D_prime = abs(best_position - current)
                l = rng.uniform(-1.0, 1.0)
                new_position = (
                    D_prime * math.exp(spiral_b * l) * math.cos(2 * math.pi * l)
                    + best_position
                )

            population[:, i] = clip_to_bounds(new_position, bounds)

        fitness = np.array([fitness_fn(population[:, i]) for i in range(n_agents)])
        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < best_fitness:
            best_fitness = float(fitness[best_idx])
            best_position = population[:, best_idx].copy()
        history.append(best_fitness)

        if verbose:
            print(f"iter {iteration + 1}/{n_iterations}  best fitness: {best_fitness:.6f}")

    return best_position, best_fitness, history
