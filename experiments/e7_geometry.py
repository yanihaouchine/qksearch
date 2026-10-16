from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e7_geometry.csv"
N_QUBITS, DEPTH = 4, 6
TRAIN_SIZES = (25, 50, 100, 150, 200)
SEED = 0


def run() -> None:
    from sklearn.metrics.pairwise import rbf_kernel

    from qksearch.backend import NumpyBackend
    from qksearch.compare_kernels import tune_rbf_gamma
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.gates import GateSpec, ProductGateSet
    from qksearch.geometry import geometric_difference
    from qksearch.iqp import describe, iqp_circuit

    t0 = time.time()
    gs = ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=N_QUBITS, n_features=N_QUBITS, with_identity=True)
    states = [()]
    for _ in range(DEPTH):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    space = common.read_csv("e4_space.csv")
    assert len(states) == len(space)
    i_opt = int(space["kta"].idxmax())
    layer = strip_identity(states[i_opt])
    circuit = iqp_circuit(layer, N_QUBITS)
    print(f"fixed circuit: {describe(layer)} (KTA {space['kta'].max():+.4f})")

    backend = NumpyBackend(N_QUBITS)
    rows: list[dict] = []
    for n_train in TRAIN_SIZES:
        train, _ = common.mof_split(SEED, n_train=n_train)
        gamma = tune_rbf_gamma(train)
        K_q = backend.gram(circuit, train.X)
        K_rbf = rbf_kernel(train.X, gamma=gamma)
        g = float(geometric_difference(K_q, K_rbf))
        rows.append({"n_train": n_train, "geom": g,
                     "sqrt_n": float(np.sqrt(n_train)), "rbf_gamma": gamma,
                     "circuit": describe(layer)})
        print(f"N={n_train:4d}  g={g:.3f}  √N={np.sqrt(n_train):.2f}")
    common.write_csv(CSV, rows)
    common.write_meta("e7_geometry", time.time() - t0,
                      train_sizes=list(TRAIN_SIZES), seed=SEED,
                      circuit=describe(layer))


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(df["n_train"], df["sqrt_n"], color=common.BASELINE, lw=1.4, ls="--",
            label=r"$\sqrt{N}$ (max. possible)")
    ax.plot(df["n_train"], df["geom"], color=common.BLUE, lw=1.6, marker="o",
            ms=4, label=r"$g(K_{\mathrm{IQP}}, K_{\mathrm{RBF}})$")
    ax.axhline(1.0, color=common.INK2, lw=0.8, ls=":")
    ax.text(df["n_train"].iloc[-1], 1.0, " classically\n reproducible",
            fontsize=6.5, color=common.INK2, va="center", ha="left")
    ax.set_xlabel("training-set size $N$ (MOF)")
    ax.set_ylabel("geometric difference $g$")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.5)
    common.save_fig(fig, "fig_e7_geometry")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
