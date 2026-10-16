from __future__ import annotations

import os
from dataclasses import dataclass
from functools import reduce
from itertools import groupby
from operator import mul
from typing import Callable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, rbf_kernel
from sklearn.svm import SVC

from qksearch.backend import NumpyBackend, QiskitBackend
from qksearch.dataset import (
    AngleScaler,
    BaseDataset,
    Dataset,
    RegressionDataset,
    mof_dataset,
    moons_dataset,
    sine_dataset,
    sine_target,
)
from qksearch.gates import ANGLE_ENCODING, State
from qksearch.gates import RANDOM_ORDER, OrderSeed
from qksearch.iqp import (angle_label, build_heuristic, default_metric, describe,
                 iqp_circuit)
from qksearch.metric import CenteredAlignment, Metric, TargetAlignment
from qksearch.search import MctsSearch ,search
N_QUBITS = 2
N_DATA_GATES = 8
N_TRAIN, N_TEST = 40, 80
NOISE, SEED = 0.15, 0
NOISE_REG = 0.05
RIDGE_ALPHA = 1e-3
GRID_RES = 60
N_QUBITS_MOF = 4
N_TRAIN_MOF = 100
DLA_WEIGHT = 0.05
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"      
NEUTRAL = "#f0efec"                    
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]                
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("div_br", [BLUE, NEUTRAL, RED])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 170,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 9,
})


def find_quantum_kernel(train: BaseDataset, *, n_qubits: int = N_QUBITS,
                        n_data_gates: int = N_DATA_GATES, backend=None,
                        regularization: str = "dla",
                        weight: float = 0.2,
                        order_seed: OrderSeed = RANDOM_ORDER) -> tuple[State, dict]:
    heuristic = build_heuristic(train, n_qubits=n_qubits, n_data_gates=n_data_gates,
                                regularization=regularization, weight=weight,
                                backend=backend, order_seed=order_seed)
    result = search((), heuristic, MctsSearch())
    assert result.best_state is not None
    circuit = iqp_circuit(result.best_state, n_qubits)
    stats = {"kta": heuristic.kta(result.best_state), "score": result.best_score,
             "n_evaluated": result.n_evaluated, "backend": heuristic.backend,
             "order_seed": heuristic.gate_set.order_seed}
    return circuit, stats


def tune_rbf_gamma(train: BaseDataset, metric: Optional[Metric] = None) -> float:
    metric = metric if metric is not None else default_metric(train)
    gammas = np.logspace(-2, 2, 41)
    scores = [metric(rbf_kernel(train.X, gamma=g), train.y) for g in gammas]
    return float(gammas[int(np.argmax(scores))])


def ideal_gram(y: np.ndarray) -> np.ndarray:
   
    target = np.outer(y, y)
    return np.maximum(target, 0.0) / target.max()


def evaluate(name: str, K_train: np.ndarray, K_test: np.ndarray,
             train: Dataset, test: Dataset) -> dict:
   
    svm = SVC(kernel="precomputed").fit(K_train, train.y)
    return {
        "name": name,
        "kta": default_metric(train)(K_train, train.y),
        "accuracy": float((svm.predict(K_test) == test.y).mean()),
        "model": svm,
    }


def evaluate_regression(name: str, K_train: np.ndarray, K_test: np.ndarray,
                        train: RegressionDataset, test: RegressionDataset) -> dict:
  
    krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(K_train, train.y)
    return {
        "name": name,
        "kta": default_metric(train)(K_train, train.y),
        "r2": float(r2_score(test.y, krr.predict(K_test))),
        "model": krr,
    }

def class_masks(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return y < 0, y > 0


def scatter_classes(ax, X, y, *, filled: bool = True, size: float = 48) -> None:
   
    for mask, color in zip(class_masks(y), (BLUE, RED)):
        if filled:
            ax.scatter(X[mask, 0], X[mask, 1], s=size, c=color,
                       edgecolors=SURFACE, linewidths=1.4, zorder=3)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=size, facecolors="none",
                       edgecolors=color, linewidths=1.6, zorder=3)


def split_legend(ax) -> None:
    handles = [
        Line2D([], [], ls="", marker="o", ms=8, mfc=BLUE, mec=SURFACE, label="class −1"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=RED, mec=SURFACE, label="class +1"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=MUTED, mec=SURFACE, label="train"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=MUTED, label="test"),
    ]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.1,
              borderaxespad=0.0, labelcolor=INK2)


def fig_dataset(train: Dataset, test: Dataset, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    scatter_classes(ax, train.X, train.y, filled=True)
    scatter_classes(ax, test.X, test.y, filled=False)
    split_legend(ax)
    ax.set_title("“Two moons” dataset — not linearly separable", pad=12)
    ax.set_xlabel("x₁ (angle, rad)")
    ax.set_ylabel("x₂ (angle, rad)")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def fig_gram(train: BaseDataset, K_q: np.ndarray, K_rbf: np.ndarray,
             path: str, *, split: bool = True, suptitle: str = "") -> None:

    order = np.argsort(train.y, kind="stable")
    ideal = ideal_gram(train.y)
    align = CenteredAlignment()
    panels = [
        ("Ideal target", ideal, f"KTA = {align(ideal, train.y):.3f} (reference)"),
        ("IQP kernel (searched)", K_q, f"KTA = {align(K_q, train.y):.3f}"),
        ("RBF kernel (tuned γ)", K_rbf, f"KTA = {align(K_rbf, train.y):.3f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.1), constrained_layout=True)
    for ax, (title, K, note) in zip(axes, panels):
        im = ax.imshow(K[np.ix_(order, order)], cmap=CMAP_SEQ, vmin=0, vmax=1,
                       interpolation="nearest")
        if split:
            pos = int((train.y < 0).sum()) - 0.5
            ax.axhline(pos, color=SURFACE, lw=2)
            ax.axvline(pos, color=SURFACE, lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(note, color=INK2, fontsize=10, labelpad=8)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, aspect=28, pad=0.015)
    cbar.ax.tick_params(labelsize=9, color=MUTED)
    cbar.outline.set_visible(False)
    fig.suptitle(suptitle or ("Gram matrices (train, samples sorted by class) — "
                              "a good kernel reproduces the blocks of the target"),
                 fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def decision_grid(lim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lin = np.linspace(*lim, GRID_RES)
    gx, gy = np.meshgrid(lin, lin)
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def fig_boundaries(train: Dataset, test: Dataset, results: dict[str, dict],
                   grids: dict[str, np.ndarray], lim: tuple[float, float],
                   path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    for ax, (name, Z) in zip(axes, grids.items()):
        vmax = np.abs(Z).max()
        gx, gy, _ = decision_grid(lim)
        ax.contourf(gx, gy, Z, levels=14, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
                    alpha=0.30, antialiased=True)
        ax.contour(gx, gy, Z, levels=[0.0], colors=[INK2], linewidths=1.2)
        scatter_classes(ax, train.X, train.y, filled=True, size=40)
        scatter_classes(ax, test.X, test.y, filled=False, size=40)
        acc = results[name]["accuracy"]
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f"test accuracy = {acc:.1%}", color=INK2, fontsize=10, labelpad=8)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(*lim), ax.set_ylim(*lim)
    split_legend(axes[0])
    fig.suptitle("SVM decision boundaries (line: boundary; wash: signed margin)",
                 fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def rounded_bar(ax, x: float, height: float, width: float, color: str) -> None:
    r = 0.012
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1,
        fc=color, ec="none", zorder=3))
    ax.add_patch(Rectangle((x - width / 2, 0), width, min(r, height),
                           fc=color, ec="none", zorder=3))


@dataclass(frozen=True)
class ScorePanel:
   

    title: str
    key: str
    fmt: Callable[[float], str]
    ref: float
    ref_label: str
    ref_ha: str = "right"


def fig_scores(results: dict[str, dict], panels: Sequence[ScorePanel], path: str,
               suptitle: str) -> None:
   
    names = list(results)
    colors = {n: BLUE if "IQP" in n else BASELINE for n in names}

    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 4.4), constrained_layout=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        for i, n in enumerate(names):
            v = results[n][panel.key]
            rounded_bar(ax, i, max(v, 0.0), 0.42, colors[n])
            ax.text(i, max(v, 0.0) + 0.03, panel.fmt(v),
                    ha="center", va="bottom", color=INK2, fontsize=10)
        ax.axhline(panel.ref, color=MUTED, lw=1, ls=(0, (4, 3)))
        x = len(names) - 0.45 if panel.ref_ha == "right" else -0.55
       
        ax.text(x, max(panel.ref + 0.02, 0.12), panel.ref_label,
                ha=panel.ref_ha, color=MUTED, fontsize=9)
        ax.set_title(panel.title, fontsize=11, pad=10)
        ax.set_xticks(range(len(names)), names, color=INK2, fontsize=10)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", length=0)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


CIRCUIT_STYLE = {
    "backgroundcolor": SURFACE,
    "linecolor": INK2,
    "textcolor": INK,
    "gatetextcolor": "#ffffff",
    "subfontsize": 9,
    "displaycolor": {
        "h": [MUTED, "#ffffff"],
        "rz": [BLUE, "#ffffff"],
        "rzz": [BLUE, "#ffffff"],
        "rzzh": [BLUE, "#ffffff"],
        "rzzhn": [BLUE, "#ffffff"],
    },
}


def draw_circuit(state: State, n_qubits: int, n_features: int, path: str,
                 title: Optional[str] = None) -> None:
    from qiskit.circuit import Parameter, QuantumCircuit

    drawer = QiskitBackend(n_qubits=n_qubits)
    params = [Parameter(f"x{f}") for f in range(n_features)]
    encoded: dict[str, Parameter] = {}
    qc = QuantumCircuit(n_qubits)
    for op, run in groupby(state):
        name, qubits, f = op
        k = sum(1 for _ in run)
        if f is None:
            for _ in range(k):
                drawer._apply(qc, name, qubits, None)
            continue
        if name in ANGLE_ENCODING:
            label = angle_label(name, f)
            angle = encoded.setdefault(label, Parameter(label))
        else:
            angle = reduce(mul, (params[i] for i in f))
        drawer._apply(qc, name, qubits, k * angle)

    fig = qc.draw(output="mpl", style=CIRCUIT_STYLE, fold=-1)
    if title is not None:
        fig.suptitle(title, fontsize=11, color=INK)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig_circuit(state: State, n_qubits: int, n_features: int,
                kta: float, path: str) -> None:
    draw_circuit(state, n_qubits, n_features, path,
                 title=f"Selected IQP circuit   (train KTA = {kta:+.3f})")


def scatter_targets(ax, X: np.ndarray, y: np.ndarray, vmax: float, *,
                    filled: bool = True, size: float = 48):
  
    return ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
        s=size if filled else size * 0.62, marker="o" if filled else "D",
        edgecolors=INK2, linewidths=0.8, zorder=3 if filled else 2)


def target_legend(ax) -> None:
    handles = [
        Line2D([], [], ls="", marker="o", ms=8, mfc=MUTED, mec=INK2, label="train"),
        Line2D([], [], ls="", marker="D", ms=6, mfc=MUTED, mec=INK2, label="test"),
    ]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.1,
              borderaxespad=0.0, labelcolor=INK2)


def fig_dataset_regression(train: RegressionDataset, test: RegressionDataset,
                           path: str) -> None:
    vmax = float(np.abs(np.concatenate([train.y, test.y])).max())
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    scatter_targets(ax, train.X, train.y, vmax, filled=True)
    im = scatter_targets(ax, test.X, test.y, vmax, filled=False)
    target_legend(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=28, pad=0.02)
    cbar.set_label("target y (standardized)", color=INK2, fontsize=10)
    cbar.ax.tick_params(labelsize=9, color=MUTED)
    cbar.outline.set_visible(False)
    ax.set_title("“Product of sines” dataset — continuous, non-additive target", pad=12)
    ax.set_xlabel("x₁ (angle, rad)")
    ax.set_ylabel("x₂ (angle, rad)")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


MOF_LABELS = ("BET surface area (m²/g)", "pore volume (m³/g)",
              "pressure (bar)", "temperature (K)")


def fig_dataset_mof(raw: RegressionDataset, n_train: int, path: str) -> None:
    vmax = float(np.abs(raw.y).max())
    fig, axes = plt.subplots(1, raw.n_features, figsize=(13.2, 3.9),
                             constrained_layout=True)
    for j, (ax, label) in enumerate(zip(axes, MOF_LABELS)):
        for X, y, size, marker, z in ((raw.X[:n_train], raw.y[:n_train], 34, "o", 3),
                                      (raw.X[n_train:], raw.y[n_train:], 20, "D", 2)):
            ax.scatter(X[:, j], y, c=y, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
                       s=size, marker=marker, edgecolors=INK2, linewidths=0.7,
                       zorder=z)
        ax.set_xlabel(label, color=INK2, fontsize=10)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("uptake (standardized)", color=INK2, fontsize=10)
    target_legend(axes[0])
    fig.suptitle("MOF dataset — hydrogen uptake, 4 physical features "
                 "against the target", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_surfaces(train: RegressionDataset, fields: dict[str, np.ndarray],
                 notes: dict[str, str], lim: tuple[float, float], path: str) -> None:
   
    gx, gy, _ = decision_grid(lim)
    vmax = max(float(np.abs(Z).max()) for Z in fields.values())
    levels = np.linspace(-vmax, vmax, 17)
    fig, axes = plt.subplots(1, len(fields), figsize=(12.6, 4.5), constrained_layout=True)
    for ax, (name, Z) in zip(axes, fields.items()):
        im = ax.contourf(gx, gy, Z, levels=levels, cmap=CMAP_DIV, antialiased=True)
        ax.contour(gx, gy, Z, levels=levels[1::2], colors=[INK2], linewidths=0.45,
                   alpha=0.35)
        ax.scatter(train.X[:, 0], train.X[:, 1], c=train.y, cmap=CMAP_DIV,
                   vmin=-vmax, vmax=vmax, s=26, edgecolors=SURFACE,
                   linewidths=1.0, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(notes[name], color=INK2, fontsize=10, labelpad=8)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(*lim), ax.set_ylim(*lim)
    cbar = fig.colorbar(im, ax=axes, shrink=0.86, aspect=28, pad=0.015)
    cbar.ax.tick_params(labelsize=9, color=MUTED)
    cbar.outline.set_visible(False)
    fig.suptitle("Regression surfaces (evenly spaced contours, dots: "
                 "training samples, shared color scale)", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_predictions(test: RegressionDataset, preds: dict[str, np.ndarray],
                    r2: dict[str, float], path: str) -> None:
   
    lo = min(float(test.y.min()), *(float(p.min()) for p in preds.values())) - 0.2
    hi = max(float(test.y.max()), *(float(p.max()) for p in preds.values())) + 0.2
    fig, axes = plt.subplots(1, len(preds), figsize=(9.6, 4.8), constrained_layout=True)
    for ax, (name, pred) in zip(np.atleast_1d(axes), preds.items()):
        color = BLUE if "IQP" in name else BASELINE
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.scatter(test.y, pred, s=44, c=color, edgecolors=SURFACE, linewidths=1.2,
                   zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f"true target          R² = {r2[name]:.3f}",
                      color=INK2, fontsize=10, labelpad=8)
        ax.set_ylabel("prediction", color=INK2, fontsize=10)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Test-set predictions (dashed: perfect prediction) — "
                 "a cloud spread along the diagonal, not two clumps", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def run_classification() -> None:
   
    print("\n=== Classification — “two moons” dataset ===")
    raw = moons_dataset(n_samples=N_TRAIN + N_TEST, noise=NOISE, seed=SEED)
    scaler = AngleScaler.fit(raw.X[:N_TRAIN])
    data = scaler.apply(raw)
    train = Dataset(data.X[:N_TRAIN], data.y[:N_TRAIN])
    test = Dataset(data.X[N_TRAIN:], data.y[N_TRAIN:])

    state, stats = find_quantum_kernel(train)
    backend = stats["backend"]
    print(f"Selected IQP circuit ({stats['n_evaluated']} circuits evaluated, "
          f"canonical order seed {stats['order_seed']}):")
    print(f"  {describe(state)}")
    K_q_train = backend.gram(state, train.X)
    K_q_test = backend.cross_gram(state, test.X, train.X)


    gamma = tune_rbf_gamma(train)
    print(f"RBF: γ = {gamma:.3g} (max KTA on the train set)")
    K_rbf_train = rbf_kernel(train.X, gamma=gamma)
    K_rbf_test = rbf_kernel(test.X, train.X, gamma=gamma)
    K_cos_train = cosine_similarity(train.X)
    K_cos_test = cosine_similarity(test.X, train.X)

    results = {
        "Quantum IQP": evaluate("Quantum IQP", K_q_train, K_q_test, train, test),
        "RBF": evaluate("RBF", K_rbf_train, K_rbf_test, train, test),
        "Cosine": evaluate("Cosine", K_cos_train, K_cos_test, train, test),
    }
    for r in results.values():
        print(f"{r['name']:14s} KTA(train) = {r['kta']:+.4f}   "
              f"accuracy(test) = {r['accuracy']:.1%}")


    pad = 0.25
    lim = (float(data.X.min()) - pad, float(data.X.max()) + pad)
    _, _, pts = decision_grid(lim)
    grids = {
        "SVM + quantum IQP kernel": results["Quantum IQP"]["model"]
            .decision_function(backend.cross_gram(state, pts, train.X))
            .reshape(GRID_RES, GRID_RES),
        "SVM + RBF kernel": results["RBF"]["model"]
            .decision_function(rbf_kernel(pts, train.X, gamma=gamma))
            .reshape(GRID_RES, GRID_RES),
    }

    ideal_kta = TargetAlignment()(ideal_gram(train.y), train.y)
    fig_dataset(train, test, os.path.join(FIG_DIR, "fig1_dataset.png"))
    fig_gram(train, K_q_train, K_rbf_train,
             os.path.join(FIG_DIR, "fig2_gram.png"))
    fig_boundaries(train, test, {"SVM + quantum IQP kernel": results["Quantum IQP"],
                                 "SVM + RBF kernel": results["RBF"]},
                   grids, lim, os.path.join(FIG_DIR, "fig3_boundaries.png"))
    fig_scores(results, [
        ScorePanel("Kernel–target alignment (train)", "kta", lambda v: f"{v:.3f}",
                   ideal_kta, "ceiling (kernel in [0,1])", "right"),
        ScorePanel("Classification accuracy (test)", "accuracy", lambda v: f"{v:.1%}",
                   0.5, "chance", "left"),
    ], os.path.join(FIG_DIR, "fig4_scores.png"),
        "Our IQP kernel (blue) against the classical kernels")
    fig_circuit(state, N_QUBITS, train.n_features,
                results["Quantum IQP"]["kta"],
                os.path.join(FIG_DIR, "fig5_circuit.png"))


def run_regression() -> None:
   
    print("\n=== Regression — “product of sines” dataset ===")
    raw = sine_dataset(n_samples=N_TRAIN + N_TEST, n_features=2,
                       noise=NOISE_REG, seed=SEED)
    scaler = AngleScaler.fit(raw.X[:N_TRAIN])
    data = scaler.apply(raw)
    train = RegressionDataset(data.X[:N_TRAIN], data.y[:N_TRAIN])
    test = RegressionDataset(data.X[N_TRAIN:], data.y[N_TRAIN:])

    state, stats = find_quantum_kernel(train)
    backend = stats["backend"]
    print(f"Selected IQP circuit ({stats['n_evaluated']} circuits evaluated, "
          f"canonical order seed {stats['order_seed']}):")
    print(f"  {describe(state)}")
    K_q_train = backend.gram(state, train.X)
    K_q_test = backend.cross_gram(state, test.X, train.X)

    gamma = tune_rbf_gamma(train)
    print(f"RBF: γ = {gamma:.3g} (max KTA on the train set)")
    K_rbf_train = rbf_kernel(train.X, gamma=gamma)
    K_rbf_test = rbf_kernel(test.X, train.X, gamma=gamma)
    K_lin_train = linear_kernel(train.X)
    K_lin_test = linear_kernel(test.X, train.X)

    results = {
        "Quantum IQP": evaluate_regression("Quantum IQP", K_q_train, K_q_test,
                                           train, test),
        "RBF": evaluate_regression("RBF", K_rbf_train, K_rbf_test, train, test),
        "Linear": evaluate_regression("Linear", K_lin_train, K_lin_test, train, test),
    }
    for r in results.values():
        print(f"{r['name']:14s} KTA(train) = {r['kta']:+.4f}   "
              f"R²(test) = {r['r2']:+.3f}")

    pad = 0.25
    lim = (float(data.X.min()) - pad, float(data.X.max()) + pad)
    _, _, pts = decision_grid(lim)
   
    truth = sine_target(scaler.inverse_transform(pts))
    fields = {
        "Ground truth  y = sin(x₁)·sin(x₂)": ((truth - truth.mean()) / truth.std())
            .reshape(GRID_RES, GRID_RES),
        "Ridge + quantum IQP kernel": results["Quantum IQP"]["model"]
            .predict(backend.cross_gram(state, pts, train.X))
            .reshape(GRID_RES, GRID_RES),
        "Ridge + RBF kernel": results["RBF"]["model"]
            .predict(rbf_kernel(pts, train.X, gamma=gamma))
            .reshape(GRID_RES, GRID_RES),
    }
    notes = {
        "Ground truth  y = sin(x₁)·sin(x₂)": "noiseless target",
        "Ridge + quantum IQP kernel": f"R² test = {results['Quantum IQP']['r2']:.3f}",
        "Ridge + RBF kernel": f"R² test = {results['RBF']['r2']:.3f}",
    }

    ideal_kta = TargetAlignment()(ideal_gram(train.y), train.y)
    fig_dataset_regression(train, test, os.path.join(FIG_DIR, "fig6_dataset_regression.png"))
    fig_gram(train, K_q_train, K_rbf_train,
             os.path.join(FIG_DIR, "fig7_gram_regression.png"),
             split=False,
             suptitle="Gram matrices (train, samples sorted by increasing target) — "
                      "a good kernel reproduces the gradient of the target")
    fig_surfaces(train, fields, notes, lim,
                 os.path.join(FIG_DIR, "fig8_surfaces_regression.png"))
    fig_predictions(
        test,
        {"Ridge + quantum IQP kernel": results["Quantum IQP"]["model"].predict(K_q_test),
         "Ridge + RBF kernel": results["RBF"]["model"].predict(K_rbf_test)},
        {"Ridge + quantum IQP kernel": results["Quantum IQP"]["r2"],
         "Ridge + RBF kernel": results["RBF"]["r2"]},
        os.path.join(FIG_DIR, "fig9_predictions_regression.png"))
    fig_scores(results, [
        ScorePanel("Kernel–target alignment (train)", "kta", lambda v: f"{v:.3f}",
                   ideal_kta, "ceiling (kernel in [0,1])", "right"),
        ScorePanel("Regression R² (test)", "r2", lambda v: f"{v:.3f}",
                   0.0, "predicting the mean", "right"),
    ], os.path.join(FIG_DIR, "fig10_scores_regression.png"),
        "Regression: our IQP kernel (blue) against the classical kernels")
    fig_circuit(state, N_QUBITS, train.n_features,
                results["Quantum IQP"]["kta"],
                os.path.join(FIG_DIR, "fig11_circuit_regression.png"))


def run_mof() -> None:
    print("\n=== Regression — MOF, hydrogen uptake (real measurements) ===")
    raw = mof_dataset(seed=SEED)
    scaler = AngleScaler.fit(raw.X[:N_TRAIN_MOF])
    data = scaler.apply(raw)
    train = RegressionDataset(data.X[:N_TRAIN_MOF], data.y[:N_TRAIN_MOF])
    test = RegressionDataset(data.X[N_TRAIN_MOF:], data.y[N_TRAIN_MOF:])
    print(f"{raw.n_samples} measurements, {raw.n_features} features — "
          f"{train.n_samples} train / {test.n_samples} test")

    state, stats = find_quantum_kernel(
        train, n_qubits=N_QUBITS_MOF, backend=NumpyBackend(N_QUBITS_MOF),
        regularization="dla", weight=DLA_WEIGHT)
    backend = stats["backend"]
    print(f"Selected IQP circuit ({stats['n_evaluated']} circuits evaluated, "
          f"canonical order seed {stats['order_seed']}):")
    print(f"  {describe(state)}")
    K_q_train = backend.gram(state, train.X)
    K_q_test = backend.cross_gram(state, test.X, train.X)

    gamma = tune_rbf_gamma(train)
    print(f"RBF: γ = {gamma:.3g} (max KTA on the train set)")
    K_rbf_train = rbf_kernel(train.X, gamma=gamma)
    K_rbf_test = rbf_kernel(test.X, train.X, gamma=gamma)
    K_lin_train = linear_kernel(train.X)
    K_lin_test = linear_kernel(test.X, train.X)

    results = {
        "Quantum IQP": evaluate_regression("Quantum IQP", K_q_train, K_q_test,
                                           train, test),
        "RBF": evaluate_regression("RBF", K_rbf_train, K_rbf_test, train, test),
        "Linear": evaluate_regression("Linear", K_lin_train, K_lin_test, train, test),
    }
    for r in results.values():
        print(f"{r['name']:14s} KTA(train) = {r['kta']:+.4f}   "
              f"R²(test) = {r['r2']:+.3f}")

    ideal_kta = TargetAlignment()(ideal_gram(train.y), train.y)
    fig_dataset_mof(raw, N_TRAIN_MOF, os.path.join(FIG_DIR, "fig12_dataset_mof.png"))
    fig_gram(train, K_q_train, K_rbf_train,
             os.path.join(FIG_DIR, "fig13_gram_mof.png"),
             split=False,
             suptitle="MOF — Gram matrices (train, samples sorted by increasing "
                      "target)")
    fig_predictions(
        test,
        {"Ridge + quantum IQP kernel": results["Quantum IQP"]["model"].predict(K_q_test),
         "Ridge + RBF kernel": results["RBF"]["model"].predict(K_rbf_test)},
        {"Ridge + quantum IQP kernel": results["Quantum IQP"]["r2"],
         "Ridge + RBF kernel": results["RBF"]["r2"]},
        os.path.join(FIG_DIR, "fig14_predictions_mof.png"))
    fig_scores(results, [
        ScorePanel("Kernel–target alignment (train)", "kta", lambda v: f"{v:.3f}",
                   ideal_kta, "ceiling (kernel in [0,1])", "right"),
        ScorePanel("Regression R² (test)", "r2", lambda v: f"{v:.3f}",
                   0.0, "predicting the mean", "right"),
    ], os.path.join(FIG_DIR, "fig15_scores_mof.png"),
        "MOF: our IQP kernel (blue) against the classical kernels")
    fig_circuit(state, N_QUBITS_MOF, train.n_features,
                results["Quantum IQP"]["kta"],
                os.path.join(FIG_DIR, "fig16_circuit_mof.png"))


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    print("canonical gate order: drawn at each launch "
          "(the seed of each search is printed with its result)")
    run_classification()
    run_regression()
    run_mof()
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
