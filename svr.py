# -*- coding: utf-8 -*-
"""Support Vector Regression (SVR) engine with RBF kernel.

Hand-rolled SVR (no scikit-learn): min-max normalisation, RBF kernel from
pairwise squared distances, Hessian matrix, and sequential learning of the
dual variables alpha / alpha*.
"""

import csv

import numpy as np

import SVR_Model


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def read_csv(file_name):
    """Read a ``;``-delimited CSV of ints into a 2-D list."""
    array_2d = []
    with open(file_name, "r") as csvfile:
        for row in csv.reader(csvfile, delimiter=";"):
            array_2d.append([int(i) for i in row])
    return array_2d


def split_data(data, proportion):
    """Positional split into training / testing parts."""
    cut = int(np.floor(proportion * len(data)))
    return data[:cut], data[cut:]


def get_max(data):
    return max(max(row) for row in data)


def get_min(data):
    return min(min(row) for row in data)


def normalization(data, proportion):
    """Min-max normalisation to [0, 1].

    The min/max statistics are computed from the *training* portion only,
    so no information from the test portion leaks into the training data.
    Returns ``(train, test, min_value, max_value)``.
    """
    n_train = int(np.floor(proportion * len(data)))
    train = data[:n_train]
    max_value = float(get_max(train))
    min_value = float(get_min(train))
    span = (max_value - min_value) or 1.0
    scaled = np.array(
        [[(value - min_value) / span for value in row] for row in data],
        dtype=float,
    )
    return scaled[:n_train], scaled[n_train:], min_value, max_value


def denormalize(values, min_value, max_value):
    """Map normalised values back to the original scale."""
    values = np.asarray(values, dtype=float)
    return values * (max_value - min_value) + min_value


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def calc_dist(row1, row2):
    """Squared Euclidean distance over the feature columns (excludes target)."""
    return float(sum((a - b) ** 2 for a, b in zip(row1[:-1], row2[:-1])))


def get_dist(data):
    """Full pairwise squared-distance matrix."""
    n = len(data)
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            distance[i][j] = calc_dist(data[i], data[j])
    return distance


def rbf_scalar(dist2, sigma):
    """RBF kernel from a squared distance value."""
    return float(np.exp(-(dist2 / (2 * sigma ** 2))))


def rbf_features(a, b, sigma):
    """RBF kernel between two feature vectors."""
    dist2 = sum((x - y) ** 2 for x, y in zip(a, b))
    return rbf_scalar(dist2, sigma)


def get_kernel_rbf(data_dist, sigma):
    """RBF kernel matrix from a squared-distance matrix."""
    kernel = np.zeros_like(data_dist)
    for i in range(len(data_dist)):
        for j in range(len(data_dist[i])):
            kernel[i][j] = rbf_scalar(data_dist[i][j], sigma)
    return kernel


def get_hessian(kernel_data, lamda):
    """Hessian matrix: kernel plus the squared regularisation term."""
    return np.array(
        [
            [kernel_data[i][j] + lamda ** 2 for j in range(len(kernel_data[i]))]
            for i in range(len(kernel_data))
        ]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_MSE(prediction, actual):
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual - prediction) ** 2))


def calc_MAE(prediction, actual):
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(actual - prediction)))


# ---------------------------------------------------------------------------
# Training (sequential learning of the dual variables)
# ---------------------------------------------------------------------------
def exec_svr_woa(
    model=None,
    data_path="data/dataCovid.csv",
    proportion=0.7,
    iter_max=100,
    is_print_log=False,
):
    """Train the SVR model with the given hyperparameters.

    Parameters
    ----------
    model : SVR_Model.SVR_Model
        Holds ``cLR``, ``C_value``, ``epsilon``, ``lamda`` and ``sigma``.
        Updated in place with training results.
    data_path : str
        Path to the ``;``-delimited dataset (features..., target).
    proportion : float
        Training fraction for the positional split.
    iter_max : int
        Maximum sequential-learning iterations.
    is_print_log : bool
        Verbose per-iteration logging.

    Returns the updated model. ``model.errors`` is the best *training* MSE
    reached; ``model.test_mse`` reports the held-out MSE for reference.
    """
    if model is None:
        model = SVR_Model.SVR_Model()

    data_all = read_csv(data_path)
    data_training, data_testing, min_value, max_value = normalization(
        data_all, proportion
    )

    x_training = np.array(data_training)[:, :-1].tolist()
    y_training = np.array(data_training)[:, -1].tolist()
    x_testing = np.array(data_testing)[:, :-1].tolist()
    y_testing = np.array(data_testing)[:, -1].tolist()

    distance = get_dist(data_training)
    kernel = get_kernel_rbf(distance, model.sigma)
    hessian_matrix = get_hessian(kernel, model.lamda)

    # --- Sequential learning -------------------------------------------
    # Step 1: initialise alpha, alpha* and helpers with 0
    n = len(data_training)
    alpha = [0.0] * n
    alpha_star = [0.0] * n
    e_value = [0.0] * n
    y_pred = [0.0] * n
    gamma = model.cLR / np.max(hessian_matrix)

    x = 0
    min_mse = float("inf")
    iterate = True
    while iterate:
        if is_print_log:
            print(x)
        # Step 2.1: compute E_i
        for i in range(n):
            total = sum(
                (a_s - a) * h
                for h, a_s, a in zip(hessian_matrix[i], alpha_star, alpha)
            )
            e_value[i] = y_training[i] - total
        # Step 2.2: compute delta alpha / delta alpha*
        delta_alpha_star = [
            min(max(gamma * (e - model.epsilon), -a_s), model.C_value - a_s)
            for e, a_s in zip(e_value, alpha_star)
        ]
        delta_alpha = [
            min(max(gamma * (-e - model.epsilon), -a), model.C_value - a)
            for e, a in zip(e_value, alpha)
        ]
        # Step 2.3: update alpha / alpha*
        alpha = [a + b for a, b in zip(alpha, delta_alpha)]
        alpha_star = [a + b for a, b in zip(alpha_star, delta_alpha_star)]

        for i in range(n):
            y_pred[i] = sum(
                (a_s - a) * h
                for h, a_s, a in zip(hessian_matrix[i], alpha_star, alpha)
            )

        converged = max(delta_alpha_star) < model.epsilon and max(
            delta_alpha
        ) < model.epsilon
        mse = calc_MSE(y_pred, y_training)
        if mse < min_mse:
            min_mse = mse
        if converged or x >= iter_max:
            iterate = False
        x += 1

    if is_print_log:
        print("ITERASI = " + str(x))

    # --- Results ---------------------------------------------------------
    dual_coef = [a_s - a for a_s, a in zip(alpha_star, alpha)]

    model.last_iteration = x
    model.errors = min_mse
    model.alpha_function = dual_coef
    model.training_data = np.array(data_all[: len(data_training)])[:, -1]
    model.training_prediction = denormalize(y_pred, min_value, max_value)

    # Held-out evaluation (honest test error, not used for optimisation)
    test_pred_norm = [
        sum(coef * rbf_features(xt, xr, model.sigma) for coef, xr in zip(dual_coef, x_training))
        for xt in x_testing
    ]
    model.test_prediction = denormalize(test_pred_norm, min_value, max_value)
    model.test_mse = calc_MSE(
        model.test_prediction, np.array(data_all[len(data_training):])[:, -1]
    )
    # Backwards-compatible alias (historical name; these are training preds)
    model.validation_result = model.training_prediction

    return model


if __name__ == "__main__":
    default_model = SVR_Model.SVR_Model()
    trained = exec_svr_woa(model=default_model, is_print_log=True)
    print("training MSE:", trained.errors)
    print("test MSE:", trained.test_mse)
