from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import RIDGE_ALPHA
from qksearch.dataset import AngleScaler, RegressionDataset, sine_dataset
from qksearch.dla import DlaCalculator
from qksearch.enumerate_circuits import (MAX_DEPTH, N_FEATURES, N_QUBITS, N_TEST,
                                N_TRAIN, NOISE_REG, SEED, DIM_MAX,
                                enumerate_states, gate_set, strip_identity)
from qksearch.iqp import IqpKernelHeuristic, describe, iqp_circuit
from qksearch.metric import TargetAlignment

raw = sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=N_FEATURES,
                   noise=NOISE_REG, seed=SEED)
data = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
train = RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN])
test = RegressionDataset(data.X[N_TRAIN:], data.y[N_TRAIN:])

gs = gate_set()
backend = NumpyBackend(N_QUBITS)
heuristic = IqpKernelHeuristic(gate_set=gs, backend=backend, data=train,
                               metric=TargetAlignment(), max_depth=MAX_DEPTH,
                               n_qubits=N_QUBITS, regularizer=None)
dla = DlaCalculator()
iu = np.triu_indices(train.n_samples, k=1)

rows = []
for state in enumerate_states(gs):
    bare = strip_identity(state)
    full = iqp_circuit(bare, N_QUBITS)
    K_tr = backend.gram(full, train.X)
    K_te = backend.cross_gram(full, test.X, train.X)
    krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(K_tr, train.y)
    rows.append(dict(
        label=describe(bare) if bare else "∅ (identity)",
        n=len(bare),
        kta=heuristic.kta(state),
        var=float(K_tr[iu].var()),
        dla=dla.dimension(full),
        r2=float(r2_score(test.y, krr.predict(K_te))),
    ))

rows.sort(key=lambda r: r["kta"], reverse=True)
kta = np.array([r["kta"] for r in rows])
var = np.array([r["var"] for r in rows])
dim = np.array([r["dla"] for r in rows]) / DIM_MAX
r2 = np.array([r["r2"] for r in rows])

print(f"{'rg':>3}  {'circuit':52s} {'KTA':>8} {'Var':>7} {'DLA':>4} {'R2':>8}")
for i, r in enumerate(rows, 1):
    print(f"{i:3d}  {r['label']:52s} {r['kta']:+8.4f} {r['var']:7.4f} "
          f"{r['dla']:2d}/{DIM_MAX} {r['r2']:+8.3f}")

print("\n--- Spearman correlations with test R² ---")
for name, v in (("KTA", kta), ("Var(K)", var), ("dim(DLA)", dim)):
    rho, p = spearmanr(v, r2)
    print(f"  {name:9s} ρ = {rho:+.3f}   (p = {p:.2g})")
rho, p = spearmanr(var, kta); print(f"  Var vs KTA   ρ = {rho:+.3f} (p={p:.2g})")
rho, p = spearmanr(dim, kta); print(f"  DLA vs KTA   ρ = {rho:+.3f} (p={p:.2g})")

best_r2 = int(np.argmax(r2))
print(f"\nbest absolute R²: #{best_r2+1} {rows[best_r2]['label']} "
      f"R²={r2[best_r2]:+.3f} KTA={kta[best_r2]:+.4f} "
      f"Var={var[best_r2]:.4f} DLA={rows[best_r2]['dla']}")


def sweep(score_fn, name, ws):
    print(f"\n--- {name} sweep: selected circuit (argmax of the score) ---")
    prev = None
    for w in ws:
        j = int(np.argmax(score_fn(w)))
        if j != prev:
            print(f"  {name}={w:6.3f} → #{j+1:2d} {rows[j]['label'][:46]:46s} "
                  f"KTA={kta[j]:+.4f} Var={var[j]:.4f} DLA={rows[j]['dla']:2d} "
                  f"R²={r2[j]:+.3f}")
            prev = j


ws = np.round(np.arange(0.0, 2.001, 0.005), 4)
sweep(lambda w: kta - w * dim, "lambda", ws)
sweep(lambda w: kta + 4.0 * w * var, "c", ws)

print("\n--- exact thresholds against the KTA champion ---")
lam = [( (kta[0] - kta[j]) / (dim[0] - dim[j]), j) for j in range(1, len(rows))
       if dim[j] < dim[0]]
lam_star = min(lam, key=lambda t: t[0])
print(f"  λ* = {lam_star[0]:.4f} → from there on, #{lam_star[1]+1} "
      f"{rows[lam_star[1]]['label'][:40]} takes the lead (R²={r2[lam_star[1]]:+.3f})")
cs = [(((kta[0] - kta[j]) / (4.0 * (var[j] - var[0]))), j)
      for j in range(1, len(rows)) if var[j] > var[0]]
c_star = min(cs, key=lambda t: t[0])
print(f"  c* = {c_star[0]:.4f} → from there on, #{c_star[1]+1} "
      f"{rows[c_star[1]]['label'][:40]} takes the lead (R²={r2[c_star[1]]:+.3f})")

print("\n--- amplitude of the terms over the 35 circuits (for w = 1) ---")
print(f"  KTA          : [{kta.min():+.4f}, {kta.max():+.4f}]  spread {kta.ptp():.4f}"
      f"   #1−#2 = {kta[0]-kta[1]:.4f}")
print(f"  DLA penalty  : [{-dim.max():+.4f}, {-dim.min():+.4f}]  spread {dim.ptp():.4f}")
print(f"  Var bonus    : [{4*var.min():+.4f}, {4*var.max():+.4f}]  spread {4*var.ptp():.4f}")
