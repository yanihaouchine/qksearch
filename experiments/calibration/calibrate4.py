from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import RIDGE_ALPHA
from qksearch.dataset import AngleScaler, RegressionDataset, sine_dataset
from qksearch.dataset.mof import mof_dataset
from qksearch.iqp import build_heuristic, iqp_circuit
from qksearch.search import GreedySearch, MctsSearch, search

N_TRAIN, N_TEST, N_GATES = 40, 80, 6
SEEDS = (0, 1, 2)
CONFIGS = [("none", 0.0), ("dla", 0.05), ("dla", 0.10), ("dla", 0.20), ("dla", 0.50),
           ("variance", 0.02), ("variance", 0.05), ("variance", 0.20)]


def split(raw):
    d = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
    return (RegressionDataset(d.X[:N_TRAIN], d.y[:N_TRAIN]),
            RegressionDataset(d.X[N_TRAIN:N_TRAIN + N_TEST], d.y[N_TRAIN:N_TRAIN + N_TEST]))


def r2_of(state, n_qubits, train, test):
    b = NumpyBackend(n_qubits)
    full = iqp_circuit(state, n_qubits)
    K_tr, K_te = b.gram(full, train.X), b.cross_gram(full, test.X, train.X)
    krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(K_tr, train.y)
    return float(r2_score(test.y, krr.predict(K_te)))


for name, raw_fn, nq in (
        ("sine", lambda s: sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=2,
                                        noise=0.05, seed=s), 2),
        ("mof", lambda s: mof_dataset(n_samples=N_TRAIN + N_TEST, seed=s), 4)):
    print(f"\n{'='*72}\n{name} — search over {N_GATES} gates, test R²\n{'='*72}")
    print(f"  {'regularization':22s} {'greedy':>18s} {'MCTS(400)':>18s}")
    for reg, w in CONFIGS:
        out = {}
        for strat_name, strat in (("greedy", GreedySearch()),
                                  ("mcts", MctsSearch(n_iterations=400, seed=0))):
            scores = []
            for s in SEEDS:
                train, test = split(raw_fn(s))
                h = build_heuristic(train, n_qubits=nq, n_data_gates=N_GATES,
                                    regularization=reg, weight=w,
                                    backend=NumpyBackend(nq))
                res = search((), h, strat)
                scores.append(r2_of(res.best_state, nq, train, test))
            out[strat_name] = scores
        g, m = np.array(out["greedy"]), np.array(out["mcts"])
        print(f"  {reg + ' w=' + format(w, '.2f'):22s} "
              f"{g.mean():+8.3f} ±{g.std():5.3f} {m.mean():+8.3f} ±{m.std():5.3f}   "
              f"g={np.round(g, 3)} m={np.round(m, 3)}")
