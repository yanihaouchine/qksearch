from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import (
    BASELINE,
    BLUE,
    FIG_DIR,
    INK,
    INK2,
    MUTED,
    RED,
    evaluate_regression,
)
from qksearch.dataset import RegressionDataset
from qksearch.enumerate_circuits import (
    MAX_DEPTH,
    N_FEATURES,
    N_PAIRS_EXPR,
    N_QUBITS,
    SEED,
    enumerate_states,
    gate_set,
    sine_split,
)
from qksearch.expressibility import expressibility
from qksearch.gates import ProductGateSet, State
from qksearch.iqp import DEFAULT_WEIGHTS, REGULARIZERS, IqpKernelHeuristic, describe
from qksearch.metric import TargetAlignment
from qksearch.search import MctsSearch, search

DEPTH = MAX_DEPTH
N_RUNS = 10
MCTS_ITERATIONS = 200
N_PAIRS = N_PAIRS_EXPR

OBJECTIVES = ("none", "dla", "variance")
LABELS = {"none": "KTA", "dla": "KTA − λ·DLA", "variance": "KTA + c·Var"}
COLORS = {"none": INK, "dla": RED, "variance": BLUE}

WEIGHTS = dict(DEFAULT_WEIGHTS)

FIG_PATH = os.path.join(FIG_DIR, "fig_stats_mcts.png")


@dataclass(frozen=True)
class Sample:
    state: State
    kta: float
    expr: float
    r2: float


@dataclass(frozen=True)
class Stats:
    n: int
    kta_mean: float
    kta_std: float
    expr_mean: float
    expr_std: float
    r2_mean: float
    r2_std: float

    @classmethod
    def of(cls, samples: Sequence[Sample]) -> "Stats":
        kta = np.asarray([s.kta for s in samples], dtype=float)
        expr = np.asarray([s.expr for s in samples], dtype=float)
        r2 = np.asarray([s.r2 for s in samples], dtype=float)
        return cls(len(samples),
                   float(kta.mean()), float(kta.std()),
                   float(expr.mean()), float(expr.std()),
                   float(r2.mean()), float(r2.std()))


def build_heuristic(objective: str, train: RegressionDataset, gs: ProductGateSet,
                    backend: NumpyBackend, depth: int = DEPTH) -> IqpKernelHeuristic:
    weight = WEIGHTS[objective]
    regularizer = (REGULARIZERS[objective](gs.n_active, weight)
                   if weight > 0 else None)
    return IqpKernelHeuristic(
        gate_set=gs, backend=backend, data=train, metric=TargetAlignment(),
        max_depth=depth, n_qubits=N_QUBITS, regularizer=regularizer)


def measure(heuristic: IqpKernelHeuristic, backend: NumpyBackend, state: State,
            train: RegressionDataset, test: RegressionDataset) -> Sample:
    circuit = heuristic.circuit(state)
    gram = heuristic.evaluator.gram(state)
    K_test = backend.cross_gram(circuit, test.X, train.X)
    return Sample(
        state=state,
        kta=heuristic.kta(state),
        expr=expressibility(backend, circuit, N_FEATURES,
                            n_pairs=N_PAIRS, seed=SEED),
        r2=evaluate_regression(describe(state), gram, K_test, train, test)["r2"],
    )


def enumerate_stats(depth: int = DEPTH) -> tuple[Stats, list[Sample]]:
    train, test = sine_split()
    gs = gate_set()
    backend = NumpyBackend(N_QUBITS)
    heuristic = build_heuristic("none", train, gs, backend, depth)
    samples = [measure(heuristic, backend, state, train, test)
               for state in enumerate_states(gs, depth=depth)]
    return Stats.of(samples), samples


def mcts_stats(depth: int = DEPTH,
               n_runs: int = N_RUNS) -> dict[str, tuple[Stats, list[Sample]]]:
    train, test = sine_split()
    gs = gate_set()
    backend = NumpyBackend(N_QUBITS)

    results: dict[str, tuple[Stats, list[Sample]]] = {}
    for objective in OBJECTIVES:
        heuristic = build_heuristic(objective, train, gs, backend, depth)
        samples: list[Sample] = []
        for seed in range(n_runs):
            strategy = MctsSearch(n_iterations=MCTS_ITERATIONS, seed=seed,
                                  record_complete_only=True)
            best = search((), heuristic, strategy).best_state
            if best is None:
                print(f"  [{objective}] seed {seed}: no complete circuit, run skipped")
                continue
            samples.append(measure(heuristic, backend, best, train, test))
        results[objective] = (Stats.of(samples), samples)
    return results


def print_table(enum: Stats, mcts: dict[str, tuple[Stats, list[Sample]]]) -> None:
    header = (f"{'objective':>14} {'n':>4} {'mean KTA':>9} {'± std':>8} "
              f"{'mean expr':>9} {'± std':>8} {'mean R²':>9} {'± std':>8}")
    print(header)
    print("-" * len(header))

    def row(name: str, s: Stats) -> None:
        print(f"{name:>14} {s.n:4d} {s.kta_mean:+9.4f} {s.kta_std:8.4f} "
              f"{s.expr_mean:9.4f} {s.expr_std:8.4f} "
              f"{s.r2_mean:+9.4f} {s.r2_std:8.4f}")

    row("enumeration", enum)
    for objective in OBJECTIVES:
        row(LABELS[objective], mcts[objective][0])


FS_LABEL, FS_ANNOT, FS_TICK, FS_NOTE = 12, 11, 10.5, 9


def _panel(ax, enum_values: np.ndarray, means: Sequence[float],
           stds: Sequence[float], ylabel: str, fmt: str) -> None:
    ax.scatter(np.zeros(enum_values.size), enum_values, s=16, color=BASELINE,
               edgecolors="none", zorder=2)
    ax.scatter([0], [enum_values.mean()], s=42, color=MUTED,
               edgecolors="#ffffff", linewidths=1.2, zorder=4)
    ax.annotate(format(enum_values.mean(), fmt), (0, enum_values.mean()),
                textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=FS_ANNOT, color=INK2)

    for i, objective in enumerate(OBJECTIVES, start=1):
        color = COLORS[objective]
        ax.errorbar([i], [means[i - 1]], yerr=[stds[i - 1]], fmt="none",
                    ecolor=color, elinewidth=1.6, capsize=5, zorder=3)
        ax.scatter([i], [means[i - 1]], s=42, color=color,
                   edgecolors="#ffffff", linewidths=1.2, zorder=4)
        ax.annotate(format(means[i - 1], fmt), (i, means[i - 1]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=FS_ANNOT, color=INK2)

    ax.set_ylabel(ylabel, color=INK2, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.margins(y=0.26)
    ax.set_xlim(-0.5, len(OBJECTIVES) + 0.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(enum_samples: Sequence[Sample],
           mcts: dict[str, tuple[Stats, list[Sample]]], path: str) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(5.6, 5.6), sharex=True,
        gridspec_kw={"hspace": 0.24, "left": 0.14, "right": 0.98,
                     "top": 0.955, "bottom": 0.145})

    panels = [
        ("kta", "KTA", "+.3f"),
        ("expr", "D_KL", ".3f"),
        ("r2", "R²", "+.3f"),
    ]
    for ax, (attr, ylabel, fmt) in zip(axes, panels):
        values = np.asarray([getattr(s, attr) for s in enum_samples], dtype=float)
        means = [getattr(mcts[o][0], f"{attr}_mean") for o in OBJECTIVES]
        stds = [getattr(mcts[o][0], f"{attr}_std") for o in OBJECTIVES]
        _panel(ax, values, means, stds, ylabel, fmt)

    axes[-1].set_xticks(range(len(OBJECTIVES) + 1))
    axes[-1].set_xticklabels(["enumeration"] + [LABELS[o] for o in OBJECTIVES],
                             color=INK2, fontsize=FS_LABEL)
    fig.text(0.14, 0.042,
             f"grey dots: the {len(enum_samples)} circuits at depth {DEPTH}",
             color=MUTED, fontsize=FS_NOTE)
    fig.text(0.14, 0.008,
             f"bars: ± std over {N_RUNS} MCTS runs ({MCTS_ITERATIONS} it.) "
             f"—  D_KL: lower = more expressive",
             color=MUTED, fontsize=FS_NOTE)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(f"=== IQP circuits, {N_QUBITS} qubits, gates {{rz, rzz, id}}, "
          f"depth {DEPTH} ===")
    print(f"weights: λ(dla)={WEIGHTS['dla']:g}, c(variance)={WEIGHTS['variance']:g}\n")

    enum, enum_samples = enumerate_stats()
    mcts = mcts_stats()
    print_table(enum, mcts)

    figure(enum_samples, mcts, FIG_PATH)
    print(f"\nfigure → {FIG_PATH}")


if __name__ == "__main__":
    main()
