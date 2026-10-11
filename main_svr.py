# -*- coding: utf-8 -*-
"""WOA-SVR: optimise SVR hyperparameters with the Whale Optimization Algorithm.

Each whale is a 5-D vector ``[cLR, C, epsilon, lamda, sigma]``. Fitness is the
training MSE of the SVR trained with those hyperparameters (see svr.py).

Run:  python main_svr.py [--agents 10] [--iterations 10] [--seed 42]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import SVR_Model
import svr
import woa

# Search-space bounds for the five SVR hyperparameters.
BOUNDS = [
    (0.01, 0.1),      # cLR      (learning rate)
    (10.0, 200.0),    # C_value  (regularisation)
    (0.0001, 0.1),    # epsilon  (insensitive tube)
    (0.05, 0.5),      # lamda    (Hessian regularisation)
    (0.1, 0.5),       # sigma    (RBF kernel width)
]
PARAM_NAMES = ["cLR", "C_value", "epsilon", "lamda", "sigma"]


def fitness_factory(data_path="data/dataCovid.csv", proportion=0.7):
    """Build a fitness function closing over the dataset.

    Trains an SVR for each candidate hyperparameter vector and returns the
    training MSE. Note: optimising on training error risks overfitting; the
    held-out test MSE is reported separately for reference.
    """

    def fitness(solution):
        model = SVR_Model.SVR_Model()
        for name, value in zip(PARAM_NAMES, solution):
            setattr(model, name, float(value))
        trained = svr.exec_svr_woa(
            model=model, data_path=data_path, proportion=proportion
        )
        return trained.errors

    return fitness


def main(agents=10, iterations=10, seed=42,
         data_path="data/dataCovid.csv", proportion=0.7):
    fitness = fitness_factory(data_path=data_path, proportion=proportion)

    best, best_fitness, _history = woa.optimize(
        fitness,
        bounds=BOUNDS,
        n_agents=agents,
        n_iterations=iterations,
        seed=seed,
        verbose=True,
    )
    print("Best hyperparameters:")
    for name, value in zip(PARAM_NAMES, best):
        print(f"  {name:8s} = {value:.6f}")
    print(f"Best training MSE: {best_fitness:.6f}")

    # Final evaluation with the best hyperparameters (incl. honest test MSE)
    model = SVR_Model.SVR_Model()
    for name, value in zip(PARAM_NAMES, best):
        setattr(model, name, float(value))
    trained = svr.exec_svr_woa(
        model=model, data_path=data_path, proportion=proportion,
        is_print_log=False,
    )
    print(f"Held-out test MSE: {trained.test_mse:.6f}")

    plt.plot(
        range(len(trained.training_data)), trained.training_data,
        color="red", lw=1, label="Data latih (aktual)",
    )
    plt.plot(
        range(len(trained.training_prediction)), trained.training_prediction,
        color="blue", lw=1, label="Prediksi (training)",
    )
    plt.title("SVR-WOA", size=20)
    plt.xlabel("Hari ke-", size=14)
    plt.ylabel("Jumlah Kasus", size=14)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOA-SVR hyperparameter search")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="data/dataCovid.csv")
    args = parser.parse_args()
    main(agents=args.agents, iterations=args.iterations, seed=args.seed,
         data_path=args.data)
