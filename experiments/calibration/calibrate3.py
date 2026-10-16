from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import RIDGE_ALPHA
from qksearch.dataset import AngleScaler, RegressionDataset, sine_dataset
from qksearch.dataset.mof import mof_dataset
from qksearch.dla import DlaCalculator
from qksearch.gates import IDENTITY_NAME, GateSpec, ProductGateSet
from qksearch.iqp import IqpKernelHeuristic, describe, iqp_circuit
from qksearch.metric import TargetAlignment

N_TRAIN, N_TEST = 40, 80
SEEDS = (0, 1, 2)


def gate_set(n_qubits, n_features):
    return ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=n_qubits, n_features=n_features, with_identity=True)


def enumerate_states(gs, depth):
    states = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def table(raw, n_qubits, depth):
    data = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
    train = RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN])
    test = RegressionDataset(data.X[N_TRAIN:N_TRAIN + N_TEST],
                             data.y[N_TRAIN:N_TRAIN + N_TEST])
    gs = gate_set(n_qubits, train.n_features)
    backend = NumpyBackend(n_qubits)
    h = IqpKernelHeuristic(gate_set=gs, backend=backend, data=train,
                           metric=TargetAlignment(), max_depth=depth,
                           n_qubits=n_qubits, regularizer=None)
    dla = DlaCalculator()
    iu = np.triu_indices(train.n_samples, k=1)
    dim_max = 4 ** gs.n_active - 1

    kta, var, dim, r2, labels = [], [], [], [], []
    for state in enumerate_states(gs, depth):
        bare = tuple(op for op in state if op[0] != IDENTITY_NAME)
        full = iqp_circuit(bare, n_qubits)
        K_tr, K_te = backend.gram(full, train.X), backend.cross_gram(full, test.X, train.X)
        krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(K_tr, train.y)
        kta.append(h.kta(state)); var.append(float(K_tr[iu].var()))
        dim.append(dla.dimension(full) / dim_max)
        r2.append(float(r2_score(test.y, krr.predict(K_te))))
        labels.append(describe(bare) if bare else "∅")
    return (np.array(kta), np.array(var), np.array(dim), np.array(r2), labels)


def scan(name, tables, score, ws, lbl):
    print(f"\n### {name} — {lbl} sweep")
    print(f"  {lbl:>6} {'ρ(score,R²)':>12} {'R²(top-1)':>10} {'R² top-3':>9} "
          f"{'R² top-5':>9}   (means over {len(tables)} draws)")
    base = None
    for w in ws:
        rhos, t1, t3, t5 = [], [], [], []
        for kta, var, dim, r2, _ in tables:
            s = score(w, kta, var, dim)
            o = np.argsort(-s)
            rhos.append(spearmanr(s, r2).statistic)
            t1.append(r2[o[0]]); t3.append(r2[o[:3]].mean()); t5.append(r2[o[:5]].mean())
        row = (np.mean(rhos), np.mean(t1), np.mean(t3), np.mean(t5))
        if base is None:
            base = row
        flag = "  ←" if all(a >= b - 1e-9 for a, b in zip(row, base)) else ""
        print(f"  {w:6.3f} {row[0]:+12.4f} {row[1]:+10.3f} {row[2]:+9.3f} "
              f"{row[3]:+9.3f}{flag}")


for name, raw_fn, nq, depth in (
        ("sine (2 qubits, depth 4)",
         lambda s: sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=2,
                                noise=0.05, seed=s), 2, 4),
        ("mof (4 qubits, depth 3)",
         lambda s: mof_dataset(n_samples=N_TRAIN + N_TEST, seed=s), 4, 3)):
    tables = [table(raw_fn(s), nq, depth) for s in SEEDS]
    print(f"\n{'='*76}\n{name}\n{'='*76}")
    for i, s in enumerate(SEEDS):
        kta, var, dim, r2, labels = tables[i]
        print(f"  seed {s}: ρ(KTA,R²)={spearmanr(kta, r2).statistic:+.3f}  "
              f"ρ(Var,R²)={spearmanr(var, r2).statistic:+.3f}  "
              f"ρ(DLA,R²)={spearmanr(dim, r2).statistic:+.3f}  "
              f"R²(argmax KTA)={r2[np.argmax(kta)]:+.3f}  R²max={r2.max():+.3f}")
    scan(name, tables, lambda w, k, v, d: k - w * d,
         np.array([0, .025, .05, .075, .1, .125, .15, .2, .25, .3, .4, .5, .75, 1.0]),
         "lambda")
    scan(name, tables, lambda w, k, v, d: k + 4 * w * v,
         np.array([0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, 1.0]), "c")
