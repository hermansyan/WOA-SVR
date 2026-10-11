# -*- coding: utf-8 -*-
"""Demo: optimise a linear regression (y = m*x + c) with WOA.

Fitness is the mean squared error on ``data/regression.xlsx``.
Run:  python main.py [--agents 40] [--iterations 20] [--seed 42]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import woa


def load_data(path="data/regression.xlsx"):
    raw = pd.read_excel(path)
    inputs = np.array([(x, 1) for x in raw["X"]]).reshape(-1, 2)
    outputs = raw["Y"].to_numpy()
    return raw, inputs, outputs


def mse_fitness_factory(inputs, outputs):
    def fitness(solution):
        predicted = np.dot(inputs, solution)
        return float(np.mean((outputs - predicted) ** 2))

    return fitness


def main(agents=40, iterations=20, seed=42):
    raw, inputs, outputs = load_data()
    fitness = mse_fitness_factory(inputs, outputs)

    best, best_fitness, _history = woa.optimize(
        fitness,
        bounds=[(0.0, 100.0), (0.0, 100.0)],  # (m, c)
        n_agents=agents,
        n_iterations=iterations,
        seed=seed,
        verbose=True,
    )
    print(f"best m = {best[0]:.4f}, c = {best[1]:.4f}, MSE = {best_fitness:.6f}")

    prediction = np.dot(inputs, best)
    plt.scatter(
        raw.loc[raw["Tipe"] == 0]["X"],
        raw.loc[raw["Tipe"] == 0]["Y"],
        s=100, c="blue", alpha=0.65, label="Tipe A",
    )
    plt.scatter(
        raw.loc[raw["Tipe"] == 1]["X"],
        raw.loc[raw["Tipe"] == 1]["Y"],
        s=100, c="red", alpha=0.65, label="Tipe B",
    )
    plt.plot(
        raw["X"], prediction, color="#000000", lw=3,
        label=f"WOA Result, m = {best[0]:.2f}, c = {best[1]:.2f}",
    )
    plt.title("Metode WOA", size=20)
    plt.xlabel("X", size=14)
    plt.ylabel("Y", size=14)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOA linear-regression demo")
    parser.add_argument("--agents", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(agents=args.agents, iterations=args.iterations, seed=args.seed)
