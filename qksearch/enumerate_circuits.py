from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import rbf_kernel

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import (
    BASELINE,
    BLUE,
    FIG_DIR,
    GRID,
    INK,
    INK2,
    MUTED,
    RED,
    RIDGE_ALPHA,
    evaluate_regression,
    tune_rbf_gamma,
)
from qksearch.dataset import AngleScaler, RegressionDataset, sine_dataset
from qksearch.dla import DlaCalculator
from qksearch.expressibility import expressibility
from qksearch.gates import ANGLE_ENCODING, IDENTITY_NAME, GateSpec, ProductGateSet, State
from qksearch.geometry import geometric_difference
from qksearch.iqp import IqpKernelHeuristic, describe
from qksearch.metric import TargetAlignment

N_QUBITS = 2
N_FEATURES = 2
MAX_DEPTH = 4
N_TRAIN, N_TEST = 40, 80
NOISE_REG, SEED = 0.05, 0
N_PAIRS_EXPR = 20000

DIM_MAX = 4 ** N_QUBITS - 1
FIG_PATH = os.path.join(FIG_DIR, "fig_enumeration_depth4.png")


def gate_set() -> ProductGateSet:
    return ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=N_QUBITS, n_features=N_FEATURES, with_identity=True)


def enumerate_states(gs: ProductGateSet, depth: int = MAX_DEPTH) -> list[State]:
    states: list[State] = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def strip_identity(state: State) -> State:
    return tuple(op for op in state if op[0] != IDENTITY_NAME)


def sine_split() -> tuple[RegressionDataset, RegressionDataset]:
    raw = sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=N_FEATURES,
                       noise=NOISE_REG, seed=SEED)
    data = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
    return (RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN]),
            RegressionDataset(data.X[N_TRAIN:], data.y[N_TRAIN:]))


@dataclass(frozen=True)
class Row:
    state: State
    kta: float
    variance: float
    dla: int
    r2: float
    geom: float
    expr: float


def evaluate(train: RegressionDataset, test: RegressionDataset,
             gamma: float) -> list[Row]:
    gs = gate_set()
    states = enumerate_states(gs)
    assert len(states) == 35, f"expected 35 circuits, got {len(states)}"

    backend = NumpyBackend(N_QUBITS)
    heuristic = IqpKernelHeuristic(
        gate_set=gs, backend=backend, data=train,
        metric=TargetAlignment(), max_depth=MAX_DEPTH, n_qubits=N_QUBITS,
        regularizer=None)
    dla = DlaCalculator()
    iu = np.triu_indices(train.n_samples, k=1)
    K_rbf = rbf_kernel(train.X, gamma=gamma)

    rows = []
    for state in states:
        circuit = heuristic.circuit(state)
        gram = heuristic.evaluator.gram(state)
        K_test = backend.cross_gram(circuit, test.X, train.X)
        rows.append(Row(
            state=strip_identity(state),
            kta=heuristic.kta(state),
            variance=float(gram[iu].var()),
            dla=dla.dimension(circuit),
            r2=evaluate_regression(describe(state), gram, K_test, train, test)["r2"],
            geom=geometric_difference(gram, K_rbf),
            expr=expressibility(backend, circuit, N_FEATURES,
                                n_pairs=N_PAIRS_EXPR, seed=SEED),
        ))
    rows.sort(key=lambda r: r.kta, reverse=True)
    return rows


BOX_W, BOX_H = 0.74, 0.30
SUB = ("₀", "₁", "₂", "₃")
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(i: int) -> str:
    return str(i).translate(_SUBSCRIPT)


def wire_positions(n_qubits: int) -> dict[int, float]:
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_qubits == 1:
        return {0: 0.5}
    half = min(0.22 * (n_qubits - 1), 0.32)
    return {q: float(y) for q, y in
            enumerate(np.linspace(0.5 + half, 0.5 - half, n_qubits))}


def box_height(wires: dict[int, float]) -> float:
    ys = sorted(wires.values())
    if len(ys) < 2:
        return BOX_H
    return min(BOX_H, 0.75 * min(b - a for a, b in zip(ys, ys[1:])))


N_COLS = 1 + MAX_DEPTH + 1
Y_WIRE = wire_positions(N_QUBITS)


def gate_label(name: str, f: tuple[int, ...] | None) -> str:
    if f is None:
        return name.upper()
    if ANGLE_ENCODING.get(name, "").startswith("havlicek"):
        angle = "".join(f"(π−x{_sub(i)})" for i in f)
        if ANGLE_ENCODING[name].endswith("scaled"):
            angle += "/2π"
    else:
        angle = "".join(f"x{_sub(i)}" for i in f)
    return f"{name.upper()} {angle}"


def _box(ax, x: float, y0: float, y1: float, label: str, color: str,
         box_h: float = BOX_H, fontsize: float = 8.5) -> None:
    ax.add_patch(plt.Rectangle((x - BOX_W / 2, y0 - box_h / 2),
                               BOX_W, (y1 - y0) + box_h,
                               facecolor=color, edgecolor="none", zorder=2))
    ax.text(x, (y0 + y1) / 2, label, fontsize=fontsize, color="#ffffff",
            ha="center", va="center", zorder=3)


def draw_circuit(ax, layer: State, *, wires: dict[int, float] | None = None,
                 n_cols: int = N_COLS, fontsize: float = 8.5) -> None:
    wires = Y_WIRE if wires is None else wires
    box_h = box_height(wires)
    ax.set_xlim(-0.60, n_cols)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    for q, y in wires.items():
        ax.plot([-0.10, n_cols - 0.05], [y, y], color=INK2, lw=1.0, zorder=1)
        ax.text(-0.35, y, f"q{_sub(q)}", fontsize=fontsize + 0.5, color=INK2,
                ha="center", va="center")

    for q, y in wires.items():
        _box(ax, 0.5, y, y, "H", MUTED, box_h, fontsize)
        _box(ax, n_cols - 0.5, y, y, "H", MUTED, box_h, fontsize)

    for col, (name, qubits, f) in enumerate(layer, start=1):
        ys = [wires[q] for q in qubits]
        if len(ys) > 1:
            ax.plot([col + 0.5, col + 0.5], [min(ys), max(ys)],
                    color=BLUE, lw=1.4, zorder=1)
        _box(ax, col + 0.5, min(ys), max(ys), gate_label(name, f), BLUE,
             box_h, fontsize)


def fig_table(rows: list[Row], path: str, *, n_qubits: int = N_QUBITS,
              max_depth: int = MAX_DEPTH, dim_max: int = DIM_MAX,
              row_height: float = 0.92, fontsize: float = 8.5,
              separator: float = 0.15,
              header: str = "circuit  —  H⊗ⁿ · diagonal layer · H⊗ⁿ") -> None:
    wires = wire_positions(n_qubits)
    n_cols = 1 + max_depth + 1
    kta_max = max(r.kta for r in rows) or 1.0
    var_max = max(r.variance for r in rows) or 1.0
    r2_max = max(max(r.r2 for r in rows), 1e-12)
    geom_max = max(max(r.geom for r in rows), 1e-12)
    log_expr = {id(r): np.log10(max(r.expr, 1e-6)) for r in rows}
    lo, hi = min(log_expr.values()), max(log_expr.values())
    expr_bar = {k: (hi - v) / (hi - lo) if hi > lo else 1.0
                for k, v in log_expr.items()}

    n = len(rows)
    fig = plt.figure(figsize=(14.2, 0.30 + row_height * n))
    grid = fig.add_gridspec(n + 1, 2, width_ratios=[2.7, 2.4],
                            left=0.042, right=0.985, top=0.995, bottom=0.005,
                            hspace=0.0, wspace=0.02)

    head_l = fig.add_subplot(grid[0, 0])
    head_r = fig.add_subplot(grid[0, 1])
    for ax in (head_l, head_r):
        ax.axis("off")
    head_l.text(0.01, 0.35, header,
                transform=head_l.transAxes, fontsize=11, color=INK, weight="bold")
    for x, label in ((0.02, "Var(K)"), (0.185, "dim(DLA)"), (0.35, "KTA"),
                     (0.515, "R²"), (0.68, "g(Q,RBF)"), (0.845, "Expr↓")):
        head_r.text(x, 0.35, label, transform=head_r.transAxes, fontsize=11,
                    color=INK, weight="bold")

    for i, row in enumerate(rows, start=1):
        left = fig.add_subplot(grid[i, 0])
        draw_circuit(left, row.state, wires=wires, n_cols=n_cols,
                     fontsize=fontsize)
        left.text(-0.01, 0.5, f"{i:2d}", transform=left.transAxes, fontsize=10,
                  color=MUTED, ha="right", va="center", family="monospace")

        right = fig.add_subplot(grid[i, 1])
        right.set_xlim(0, 1)
        right.set_ylim(0, 1)
        right.axis("off")
        right.add_patch(plt.Rectangle((0.02, 0.14), 0.135 * row.variance / var_max,
                                      0.13, color=BASELINE, lw=0))
        right.add_patch(plt.Rectangle((0.35, 0.14), 0.135 * max(row.kta, 0.0) / kta_max,
                                      0.13, color=BLUE, lw=0))
        right.add_patch(plt.Rectangle((0.515, 0.14), 0.135 * max(row.r2, 0.0) / r2_max,
                                      0.13, color=BLUE, lw=0))
        right.add_patch(plt.Rectangle((0.68, 0.14), 0.135 * row.geom / geom_max,
                                      0.13, color=RED, lw=0))
        right.add_patch(plt.Rectangle((0.845, 0.14), 0.135 * expr_bar[id(row)],
                                      0.13, color=BASELINE, lw=0))
        for x, text in ((0.02, f"{row.variance:.4f}"),
                        (0.185, f"{row.dla}/{dim_max}"),
                        (0.35, f"{row.kta:+.4f}"),
                        (0.515, f"{row.r2:+.3f}"),
                        (0.68, f"{row.geom:.3f}"),
                        (0.845, f"{row.expr:.3f}")):
            right.text(x, 0.42, text, fontsize=11, color=INK, family="monospace")

        if i < len(rows):
            box_l, box_r = left.get_position(), right.get_position()
            y = box_l.y0 - separator * box_l.height
            fig.add_artist(Line2D([box_l.x0, box_r.x1], [y, y],
                                  color=GRID, lw=0.8, transform=fig.transFigure))

    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_table(rows: list[Row], *, dim_max: int = DIM_MAX) -> None:
    print(f"{'rank':>4}  {'circuit':38s} {'gates':>6} {'Var(K)':>9} "
          f"{'dim(DLA)':>9} {'KTA':>9} {'R²':>8} {'g(Q,RBF)':>9} {'Expr↓':>8}")
    for i, row in enumerate(rows, start=1):
        label = describe(row.state) if row.state else "∅ (identity)"
        print(f"{i:4d}  {label:38s} {len(row.state):6d} {row.variance:9.4f} "
              f"{row.dla:>{8 - len(str(dim_max))}d}/{dim_max} {row.kta:+9.4f} {row.r2:+8.3f} "
              f"{row.geom:9.3f} {row.expr:8.3f}")


def main() -> None:
    train, test = sine_split()
    gamma = tune_rbf_gamma(train)
    print(f"=== IQP circuits, {N_QUBITS} qubits, gates {{rz, rzz, id}}, "
          f"depth ≤ {MAX_DEPTH} ===")
    print(f"sine dataset: {train.n_samples} training points, "
          f"{test.n_samples} test points, {train.n_features} features, "
          f"noise {NOISE_REG}, seed {SEED}")
    print(f"classical reference: RBF kernel, γ = {gamma:g} (tuned by KTA)\n")

    rows = evaluate(train, test, gamma)
    print_table(rows)

    by_depth = np.bincount([len(r.state) for r in rows], minlength=MAX_DEPTH + 1)
    print(f"\n{len(rows)} circuits — breakdown by gate count: "
          + ", ".join(f"{k}→{c}" for k, c in enumerate(by_depth)))
    print("Var(K): variance of the strict upper triangle of the Gram matrix; the "
          "`GramVarianceBonus` bonus is 4·weight times larger.")
    print(f"R²: KernelRidge (α = {RIDGE_ALPHA:g}) fitted on the train set, evaluated "
          f"on the {test.n_samples} test points.")
    print("g(Q,RBF): geometric difference of Huang et al. 2021, bounded by "
          f"√N = {np.sqrt(train.n_samples):.2f}. g ≈ 1: the RBF reproduces the "
          "quantum kernel; g → √N: a quantum separation is possible (necessary, "
          "not sufficient).")
    print(f"Expr↓: expressivity of Sim et al. 2019, D_KL between the fidelity "
          f"distribution ({N_PAIRS_EXPR} pairs drawn from [−π/2, π/2]^{N_FEATURES}) "
          "and the Haar one. Smaller = more expressive; the bar is inverted so that "
          "\"long bar = good\" holds everywhere.")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_table(rows, FIG_PATH)
    print(f"\nfigure → {FIG_PATH}")


if __name__ == "__main__":
    main()
