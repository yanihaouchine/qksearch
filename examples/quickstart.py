"""Search a 2-qubit IQP feature map for a noisy sine and evaluate it.

Run with ``python examples/quickstart.py`` (a few seconds on a laptop).
"""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from qksearch import (
    AngleScaler,
    BestFirstSearch,
    GreedySearch,
    NumpyBackend,
    RegressionDataset,
    RidgeLooMse,
    TargetAlignment,
    build_heuristic,
    describe,
    iqp_circuit,
    search,
    sine_dataset,
)

N_QUBITS = 2
N_GATES = 6
N_TRAIN = 40


def split(seed: int = 0) -> tuple[RegressionDataset, RegressionDataset]:
    raw = sine_dataset(n_samples=120, n_features=N_QUBITS, seed=seed)
    data = AngleScaler.fit(raw.X[:N_TRAIN], scale=np.pi / 4).apply(raw)
    return (RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN]),
            RegressionDataset(data.X[N_TRAIN:], data.y[N_TRAIN:]))


def test_r2(state, train: RegressionDataset, test: RegressionDataset) -> float:
    backend = NumpyBackend(N_QUBITS)
    circuit = iqp_circuit(state, N_QUBITS)
    model = KernelRidge(alpha=1e-3, kernel="precomputed")
    model.fit(backend.gram(circuit, train.X), train.y)
    return r2_score(test.y, model.predict(backend.cross_gram(circuit, test.X, train.X)))


def main() -> None:
    train, test = split()
    runs = {
        "greedy / KTA": (GreedySearch(), TargetAlignment()),
        "best-first / KTA": (BestFirstSearch(), TargetAlignment()),
        "best-first / LOO MSE": (BestFirstSearch(), RidgeLooMse()),
    }
    for name, (strategy, metric) in runs.items():
        heuristic = build_heuristic(train, n_qubits=N_QUBITS, n_data_gates=N_GATES,
                                    backend=NumpyBackend(N_QUBITS), metric=metric)
        result = search((), heuristic, strategy, budget=200)
        print(f"{name:22s} score {result.best_score:+.3f}  "
              f"test R² {test_r2(result.best_state, train, test):+.3f}  "
              f"({result.n_evaluated} circuits scored)")
        print(f"{'':22s} {describe(result.best_state)}")


if __name__ == "__main__":
    main()
