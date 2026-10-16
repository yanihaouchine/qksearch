from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import BASELINE, BLUE, FIG_DIR, INK, INK2, MUTED, RED
from qksearch.dla import DlaCalculator
from qksearch.enumerate_circuits import (
    DIM_MAX,
    N_QUBITS,
    enumerate_states,
    gate_set,
    sine_split,
    strip_identity,
)
from qksearch.iqp import IqpKernelHeuristic
from qksearch.metric import TargetAlignment

MAX_DEPTH = 6
FIG_PATH = os.path.join(FIG_DIR, f"fig_trends_depth{MAX_DEPTH}.png")


def collect(depth: int = MAX_DEPTH) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    train, _ = sine_split()
    gs = gate_set()
    heuristic = IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=N_QUBITS,
        regularizer=None)
    dla = DlaCalculator()

    ktas: dict[int, list[float]] = defaultdict(list)
    dims: dict[int, list[int]] = defaultdict(list)
    for state in enumerate_states(gs, depth=depth):
        n_gates = len(strip_identity(state))
        ktas[n_gates].append(heuristic.kta(state))
        dims[n_gates].append(dla.dimension(heuristic.circuit(state)))
    return {k: (np.asarray(ktas[k]), np.asarray(dims[k], dtype=float))
            for k in sorted(ktas)}


FS_TITLE, FS_LABEL, FS_ANNOT, FS_TICK, FS_NOTE = 14, 12.5, 10.5, 11, 9


def _panel(ax, ks, values, color, label, ylabel, fmt) -> None:
    means = np.array([v.mean() for v in values])
    for k, v in zip(ks, values):
        ax.scatter(np.full(v.size, k), v, s=16, color=BASELINE,
                   edgecolors="none", zorder=2)
    ax.plot(ks, means, color=color, lw=2.0, zorder=3)
    ax.scatter(ks, means, s=34, color=color, edgecolors="#ffffff",
               linewidths=1.2, zorder=4)
    for k, m in zip(ks, means):
        ax.annotate(format(m, fmt), (k, m), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=FS_ANNOT, color=INK2)
    ax.set_ylabel(ylabel, color=INK2, fontsize=FS_LABEL)
    ax.set_title(label, loc="left", color=INK, fontsize=FS_TITLE, pad=5)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(x=0.06, y=0.20)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(groups: dict[int, tuple[np.ndarray, np.ndarray]], path: str) -> None:
    ks = list(groups)
    fig, (ax_kta, ax_dla) = plt.subplots(
        2, 1, figsize=(5.4, 3.9), sharex=True,
        gridspec_kw={"hspace": 0.38, "left": 0.13, "right": 0.98,
                     "top": 0.93, "bottom": 0.20})

    _panel(ax_kta, ks, [groups[k][0] for k in ks], BLUE,
           "Mean KTA", "KTA", "+.3f")
    _panel(ax_dla, ks, [groups[k][1] for k in ks], RED,
           f"Mean dim(DLA)  —  maximum: {DIM_MAX}",
           "dim(DLA)", ".2f")

    ax_dla.set_xticks(ks)
    ax_dla.set_xlabel("number of gates in the diagonal block",
                      color=INK2, fontsize=FS_LABEL, labelpad=6)
    total = sum(groups[k][0].size for k in ks)
    fig.text(0.13, 0.02, f"grey dots: every individual circuit ({total} in total)",
             color=MUTED, fontsize=FS_NOTE)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    groups = collect()
    print(f"=== IQP circuits, {N_QUBITS} qubits, gates {{rz, rzz, id}}, "
          f"depth <= {MAX_DEPTH} ===")
    print(f"{'layers':>7} {'n':>5} {'mean KTA':>9} {'std KTA':>9} "
          f"{'mean DLA':>9} {'std DLA':>9}")
    for k, (kta, dim) in groups.items():
        print(f"{k:7d} {kta.size:5d} {kta.mean():+9.4f} {kta.std():9.4f} "
              f"{dim.mean():9.2f} {dim.std():9.2f}")

    figure(groups, FIG_PATH)
    print(f"\nfigure → {FIG_PATH}")


if __name__ == "__main__":
    main()
