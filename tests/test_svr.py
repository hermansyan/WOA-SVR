"""Tests for the SVR engine (svr.py)."""

import numpy as np
import pytest

import SVR_Model
import svr


def test_rbf_scalar():
    assert svr.rbf_scalar(0.0, 0.3) == pytest.approx(1.0)
    assert 0.0 < svr.rbf_scalar(4.0, 0.3) < 1.0
    # larger distance -> smaller kernel value
    assert svr.rbf_scalar(4.0, 0.3) < svr.rbf_scalar(1.0, 0.3)


def test_kernel_matrix_properties():
    dist = svr.get_dist([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    kernel = svr.get_kernel_rbf(dist, sigma=0.5)
    assert np.allclose(kernel, kernel.T), "kernel must be symmetric"
    assert np.allclose(np.diag(kernel), 1.0), "diagonal must be 1"


def test_calc_dist_excludes_target_column():
    # only the feature columns (all but last) enter the distance
    assert svr.calc_dist([1.0, 2.0, 999.0], [1.0, 2.0, -999.0]) == pytest.approx(0.0)
    assert svr.calc_dist([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == pytest.approx(25.0)


def test_hessian_adds_lambda_squared():
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])
    hessian = svr.get_hessian(kernel, lamda=0.1)
    assert np.allclose(hessian, kernel + 0.01)


def test_split_data_proportion():
    data = [[i] for i in range(10)]
    train, test = svr.split_data(data, 0.7)
    assert len(train) == 7 and len(test) == 3


def test_normalization_uses_train_stats_only():
    """No data leakage: extremes in the test split must not affect scaling."""
    data = [[float(i), float(i)] for i in range(10)] + [[1000.0, 1000.0]]
    train, test, lo, hi = svr.normalization(data, 0.9)
    # 11 rows * 0.9 -> 9 train rows (0..8); the 1000.0 outlier is in test
    assert lo == 0.0 and hi == 8.0, "stats must come from the train split"
    assert max(max(row) for row in train) == pytest.approx(1.0)
    assert test[0][0] > 1.0, "test extremes stay outside [0, 1]"


def test_metrics():
    assert svr.calc_MSE([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert svr.calc_MSE([0.0], [2.0]) == pytest.approx(4.0)
    assert svr.calc_MAE([0.0], [2.0]) == pytest.approx(2.0)


def _write_csv(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(";".join(str(int(v)) for v in row) + "\n")


def test_svr_trains_on_synthetic_data(tmp_path):
    """Smoke test: full training run on a tiny linear dataset."""
    csv_path = tmp_path / "tiny.csv"
    _write_csv(csv_path, [[i, i + 1, 2 * i + 1] for i in range(1, 15)])
    model = SVR_Model.SVR_Model(cLR=0.05, C_value=100, epsilon=0.001,
                                lamda=0.1, sigma=0.5)
    trained = svr.exec_svr_woa(model=model, data_path=str(csv_path),
                               proportion=0.7, iter_max=20)
    assert np.isfinite(trained.errors)
    assert len(trained.training_prediction) == len(trained.training_data)
    assert len(trained.test_prediction) > 0
    assert np.isfinite(trained.test_mse)


def test_svr_model_defaults():
    m = SVR_Model.SVR_Model()
    assert (m.cLR, m.C_value, m.epsilon, m.lamda, m.sigma) == (
        0.05, 100, 0.0005, 0.1, 0.3)
    assert list(m.solutions) == [0.05, 100, 0.0005, 0.1, 0.3]
