from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common
from experiments.sampling import random_iqp_circuit

CSV = "e1_concentration.csv"
QUBITS = list(range(2, 13))
DEPTH_FACTORS = (1, 2, 4)
N_CIRCUITS = 30
N_SAMPLES = 40
MASTER_SEED = 0


def run() -> None:
    from qksearch.backend import NumpyBackend

    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED)
    rows: list[dict] = []
    for n in QUBITS:
        backend = NumpyBackend(n)
        X = rng.uniform(-np.pi / 2, np.pi / 2, size=(N_SAMPLES, n))
        iu = np.triu_indices(N_SAMPLES, k=1)
        for factor in DEPTH_FACTORS:
            for k in range(N_CIRCUITS):
                circuit = random_iqp_circuit(n, factor * n, rng)
                K = backend.gram(circuit, X)
                off = K[iu]
                rows.append({
                    "n_qubits": n, "depth_factor": factor,
                    "n_gates": factor * n, "circuit_idx": k,
                    "var_offdiag": float(off.var()),
                    "mean_offdiag": float(off.mean()),
                })
        print(f"n={n} done ({time.time() - t0:.1f}s)")
    common.write_csv(CSV, rows)
    common.write_meta("e1_concentration", time.time() - t0,
                      qubits=QUBITS, depth_factors=list(DEPTH_FACTORS),
                      n_circuits=N_CIRCUITS, n_samples=N_SAMPLES,
                      master_seed=MASTER_SEED)


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    colors = {1: common.BASELINE, 2: common.BLUE, 4: common.RED}
    for factor, sub in df.groupby("depth_factor"):
        med = sub.groupby("n_qubits")["var_offdiag"].median()
        q1 = sub.groupby("n_qubits")["var_offdiag"].quantile(0.25)
        q3 = sub.groupby("n_qubits")["var_offdiag"].quantile(0.75)
        c = colors[factor]
        ax.plot(med.index, med.values, color=c, lw=1.6, marker="o", ms=3,
                label=f"$L = {factor}n$ gates")
        ax.fill_between(med.index, q1.values, q3.values, color=c, alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}\,[K_{ij}]$ (off-diagonal)")
    ax.set_xticks(df["n_qubits"].unique())
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(loc="lower left")
    common.save_fig(fig, "fig_e1_concentration")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
