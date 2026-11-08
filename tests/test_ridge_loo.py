from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from qksearch.backend import NumpyBackend
from qksearch.dataset import AngleScaler, sine_dataset
from qksearch.iqp import build_heuristic
from qksearch.metric import RidgeLooMse
from qksearch.search import GreedySearch, search


def _brute_force_loo_mse(K: np.ndarray, y: np.ndarray, alpha: float) -> float:
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        krr = KernelRidge(alpha=alpha, kernel="precomputed")
        krr.fit(K[np.ix_(keep, keep)], y[keep])
        errors.append(y[i] - krr.predict(K[i:i + 1, keep])[0])
    return float(np.mean(np.square(errors)))


def _problem(seed: int = 0, n: int = 25):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=n)
    return rbf_kernel(X, gamma=2.0), y


@pytest.mark.parametrize("alpha", [1e-3, 1e-1, 1.0])
def test_closed_form_matches_refitting(alpha):
    K, y = _problem()
    metric = RidgeLooMse(alphas=(alpha,), normalize=False)
    expected = _brute_force_loo_mse(K, y, alpha)
    assert metric.loo_mse(K, y)[0] == pytest.approx(expected, rel=1e-8)
    assert metric(K, y) == pytest.approx(-expected, rel=1e-8)


def test_normalized_score_is_loo_r2_minus_one():
    K, y = _problem(seed=1)
    raw = RidgeLooMse(normalize=False)(K, y)
    assert RidgeLooMse()(K, y) == pytest.approx(raw / np.var(y))


def test_grid_keeps_the_best_alpha():
    K, y = _problem(seed=2)
    alphas = (1e-4, 1e-2, 1.0, 100.0)
    metric = RidgeLooMse(alphas=alphas, normalize=False)
    per_alpha = [RidgeLooMse(alphas=(a,), normalize=False)(K, y) for a in alphas]
    assert metric(K, y) == pytest.approx(max(per_alpha))
    assert metric.best_alpha(K, y) == alphas[int(np.argmax(per_alpha))]


def test_batch_matches_single_calls_across_chunks():
    grams = np.stack([_problem(seed=s)[0] for s in range(7)])
    _, y = _problem()
    metric = RidgeLooMse(alphas=(1e-3, 1e-1), chunk_size=3)
    expected = np.array([metric(K, y) for K in grams])
    np.testing.assert_allclose(metric.batch(grams, y), expected, rtol=1e-10)


def test_uninformative_kernel_scores_below_informative_one():
    K, y = _problem(seed=3)
    metric = RidgeLooMse()
    assert metric(K, y) > metric(np.eye(len(y)), y)


def test_constant_target_does_not_divide_by_zero():
    K, _ = _problem()
    assert np.isfinite(RidgeLooMse()(K, np.ones(K.shape[0])))


@pytest.mark.parametrize("alphas", [(), (0.0,), (-1.0, 1.0)])
def test_rejects_invalid_alphas(alphas):
    with pytest.raises(ValueError):
        RidgeLooMse(alphas=alphas)


def test_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        RidgeLooMse(chunk_size=0)


def test_drives_a_circuit_search():
    raw = sine_dataset(n_samples=20, n_features=2, noise=0.05, seed=0)
    data = AngleScaler.fit(raw.X).apply(raw)
    metric = RidgeLooMse()

    def build():
        return build_heuristic(data, n_qubits=2, n_data_gates=3,
                               regularization="none",
                               backend=NumpyBackend(2), metric=metric)

    heuristic = build()
    result = search((), heuristic, GreedySearch())
    assert result.best_state is not None
    gram = heuristic.evaluator.gram(result.best_state)
    assert result.best_score == pytest.approx(metric(gram, data.y))

    fresh = build()
    states = [result.best_state, result.best_state[:1], ()]
    np.testing.assert_allclose(
        fresh.evaluator.kta_many(states),
        [metric(fresh.evaluator.gram(s), data.y) for s in states])
