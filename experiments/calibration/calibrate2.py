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


def split(raw):
    data = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
    return (RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN]),
            RegressionDataset(data.X[N_TRAIN:N_TRAIN + N_TEST],
                              data.y[N_TRAIN:N_TRAIN + N_TEST]))


def table(raw, n_qubits, depth):
    train, test = split(raw)
    gs = gate_set(n_qubits, train.n_features)
    backend = NumpyBackend(n_qubits)
    heuristic = IqpKernelHeuristic(gate_set=gs, backend=backend, data=train,
                                   metric=TargetAlignment(), max_depth=depth,
                                   n_qubits=n_qubits, regularizer=None)
    dla = DlaCalculator()
    iu = np.triu_indices(train.n_samples, k=1)
    dim_max = 4 ** gs.n_active - 1

    rows = []
    for state in enumerate_states(gs, depth):
        bare = tuple(op for op in state if op[0] != IDENTITY_NAME)
        full = iqp_circuit(bare, n_qubits)
        K_tr = backend.gram(full, train.X)
        K_te = backend.cross_gram(full, test.X, train.X)
        krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(K_tr, train.y)
        rows.append(dict(label=describe(bare) if bare else "∅",
                         kta=heuristic.kta(state),
                         var=float(K_tr[iu].var()),
                         dla=dla.dimension(full),
                         r2=float(r2_score(test.y, krr.predict(K_te)))))
    rows.sort(key=lambda r: r["kta"], reverse=True)
    return rows, dim_max


def report(name, rows, dim_max):
    kta = np.array([r["kta"] for r in rows])
    var = np.array([r["var"] for r in rows])
    dim = np.array([r["dla"] for r in rows]) / dim_max
    r2 = np.array([r["r2"] for r in rows])
    order = np.argsort(-r2)
    rank_r2 = np.empty(len(rows), int)
    rank_r2[order] = np.arange(1, len(rows) + 1)

    print(f"\n{'='*78}\n{name} — {len(rows)} circuits, dim_max = {dim_max}\n{'='*78}")
    print(f"KTA   ∈ [{kta.min():+.4f}, {kta.max():+.4f}]   "
          f"gap #1−#2 = {kta[0]-kta[1]:.4f}   amplitude = {kta.max()-kta.min():.4f}")
    print(f"Var   ∈ [{var.min():.4f}, {var.max():.4f}]   "
          f"term 4·Var ∈ [{4*var.min():.4f}, {4*var.max():.4f}] (for c=1)")
    print(f"DLA/dim_max ∈ [{dim.min():.4f}, {dim.max():.4f}]   "
          f"observed dims: {sorted(set(r['dla'] for r in rows))}")
    for lbl, v in (("KTA", kta), ("Var", var), ("DLA", dim)):
        rho, p = spearmanr(v, r2)
        print(f"  ρ({lbl:3s}, R²) = {rho:+.3f}  (p={p:.1g})")
    best = int(np.argmax(r2))
    print(f"  best R²: {r2[best]:+.3f}  (KTA rank {best+1})   "
          f"R² of the KTA champion: {r2[0]:+.3f}")

    def sweep(score, lbl, hi):
        print(f"\n  --- {lbl} sweep ∈ [0, {hi}] ---")
        print(f"  {lbl:>7} {'sel.':>5} {'R²(sel)':>8} {'R²rank':>7} {'ρ(score,R²)':>12}  circuit")
        prev = None
        for w in np.round(np.arange(0, hi + 1e-9, 0.005), 4):
            s = score(w)
            j = int(np.argmax(s))
            if j != prev:
                rho = spearmanr(s, r2).statistic
                print(f"  {w:7.3f} {j+1:5d} {r2[j]:+8.3f} {rank_r2[j]:7d} "
                      f"{rho:+12.3f}  {rows[j]['label'][:44]}")
                prev = j
        print(f"  {'w':>7} {'ρ(score,R²)':>12} {'mean R² top-5':>15} {'mean R² top-10':>16}")
        for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
            if w > hi:
                continue
            s = score(w)
            top = np.argsort(-s)
            print(f"  {w:7.2f} {spearmanr(s, r2).statistic:+12.3f} "
                  f"{r2[top[:5]].mean():+15.3f} {r2[top[:10]].mean():+16.3f}")

    sweep(lambda w: kta - w * dim, "lambda", 1.0)
    sweep(lambda w: kta + 4.0 * w * var, "c", 1.0)
    return kta, var, dim, r2


rows_s, dm_s = table(sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=2,
                                  noise=0.05, seed=0), 2, 4)
report("sine (2 qubits, depth 4)", rows_s, dm_s)

rows_m, dm_m = table(mof_dataset(n_samples=N_TRAIN + N_TEST, seed=0), 4, 3)
report("mof (4 qubits, depth 3)", rows_m, dm_m)
