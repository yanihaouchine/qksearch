from __future__ import annotations

import argparse
import os
import time

import numpy as np

import qksearch.compare_kernels as ck
from experiments import common
from experiments.e5_benchmark import _r2, _rbf_gamma_cv
from qksearch.dataset import RegressionDataset
from qksearch.dla import DlaCalculator
from qksearch.iqp import iqp_circuit

FIG_DIR = ck.FIG_DIR
DEPTH = 6
LABELS = {0: "x0 · BET", 1: "x1 · volume", 2: "x2 · pressure", 3: "x3 · temperature"}


def drop(data: RegressionDataset, idx: int) -> RegressionDataset:
    keep = [i for i in range(data.X.shape[1]) if i != idx]
    return RegressionDataset(data.X[:, keep], data.y)


def correlations():
    import pandas as pd
    from scipy import stats

    from qksearch.dataset.mof import CSV_PATH, FEATURES, TARGET

    raw = pd.read_csv(CSV_PATH)
    y = raw[TARGET].to_numpy(float)
    rows = []
    for i, f in enumerate(FEATURES):
        v = raw[f].to_numpy(float)
        rows.append({"i": i, "pearson": abs(np.corrcoef(v, y)[0, 1]),
                     "spearman": abs(stats.spearmanr(v, y).statistic)})
    x0x1 = float(np.corrcoef(raw[FEATURES[0]], raw[FEATURES[1]])[0, 1])
    return rows, x0x1, raw[FEATURES[0]].to_numpy(float), \
        raw[FEATURES[1]].to_numpy(float), y


def multiple_corr(idx: tuple[int, ...]) -> float:
    import pandas as pd

    from qksearch.dataset.mof import CSV_PATH, FEATURES, TARGET

    raw = pd.read_csv(CSV_PATH)
    X = raw[[FEATURES[i] for i in idx]].to_numpy(float)
    y = raw[TARGET].to_numpy(float)
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    r2 = 1.0 - float((residual ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return float(np.sqrt(max(r2, 0.0)))


def ablation_r2(seeds: range) -> dict[int, list[float]]:
    from sklearn.metrics.pairwise import rbf_kernel

    out: dict[int, list[float]] = {i: [] for i in range(4)}
    for seed in seeds:
        train, test = common.mof_split(seed)
        for i in range(4):
            tr, te = drop(train, i), drop(test, i)
            gamma = _rbf_gamma_cv(tr)
            out[i].append(_r2(rbf_kernel(tr.X, gamma=gamma),
                              rbf_kernel(te.X, tr.X, gamma=gamma), tr, te))
    return out


def _reduced_space(n: int = 3):
    from qksearch.gates import GateSpec, ProductGateSet

    gs = ProductGateSet([GateSpec("rz", 1, uses_feature=True),
                         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
                        n_qubits=n, n_features=n, with_identity=True)
    states = [()]
    for _ in range(DEPTH):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return gs, states


def ablation_ktas(seeds: range):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    gs, states = _reduced_space()
    dla = DlaCalculator()
    dims = np.array([dla.dimension(iqp_circuit(s, 3)) for s in states], dtype=float)

    ktas: dict[int, list[np.ndarray]] = {i: [] for i in range(4)}
    for seed in seeds:
        train, _ = common.mof_split(seed)
        for i in range(4):
            heuristic = IqpKernelHeuristic(
                gate_set=gs, backend=NumpyBackend(3), data=drop(train, i),
                metric=TargetAlignment(), max_depth=DEPTH, n_qubits=3)
            ktas[i].append(np.asarray(heuristic.evaluator.kta_many(states)))
    return ktas, dims / (4 ** 3 - 1)


def full_ktas(seeds: range):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    gs, states = _reduced_space(4)
    dla = DlaCalculator()
    dims = np.array([dla.dimension(iqp_circuit(s, 4)) for s in states], dtype=float)

    out = []
    for seed in seeds:
        train, _ = common.mof_split(seed)
        heuristic = IqpKernelHeuristic(
            gate_set=gs, backend=NumpyBackend(4), data=train,
            metric=TargetAlignment(), max_depth=DEPTH, n_qubits=4)
        out.append(np.asarray(heuristic.evaluator.kta_many(states)))
    return out, dims / (4 ** 4 - 1)


def heuristic_scores(ktas, norm_dims, weight: float) -> dict[int, list[float]]:
    return {i: [float(np.max(k - weight * norm_dims)) for k in ktas[i]]
            for i in ktas}


def _paired(a: list[float], b: list[float]) -> tuple[float, float, int]:
    from scipy import stats

    d = np.asarray(a) - np.asarray(b)
    p = float(stats.ttest_rel(a, b).pvalue)
    return float(d.mean()), p, int((d > 0).sum())


def figure_heuristic(ktas, norm_dims, full, weight, seeds, fig_dir):
    import matplotlib.pyplot as plt

    n = len(seeds)
    os.makedirs(fig_dir, exist_ok=True)
    keep = {0: ck.RED, 1: ck.BLUE, 2: ck.BASELINE, 3: ck.BASELINE}
    heur = heuristic_scores(ktas, norm_dims, weight)

    means = {i: float(np.mean(v)) for i, v in heur.items()}
    if full is not None:
        full_scores, full_norm_dims = full
        means[None] = float(np.mean(
            [np.max(k - weight * full_norm_dims) for k in full_scores]))
    order = sorted(means, key=lambda i: means[i])

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    pos = np.arange(len(order))
    vals = [means[i] for i in order]
    for p_, v, i in zip(pos, vals, order):
        if i is None:
            ax.barh(p_, v, height=0.62, color=ck.BASELINE, hatch="///",
                    edgecolor=ck.SURFACE, linewidth=0, zorder=2)
        else:
            ax.barh(p_, v, height=0.62, color=keep[i], zorder=2)
        ax.text(v + max(vals) * 0.02, p_, f"{v:+.3f}", va="center", ha="left",
                fontsize=9, color=ck.INK2)
    ax.set_yticks(pos, ["nothing dropped" if i is None else LABELS[i] for i in order])
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel(f"mean score over {n} splits, λ = {weight:g}", color=ck.INK2)
    ax.set_ylabel("dropped feature", color=ck.INK2)
    if full is not None:
        ax.set_title("hatched: the ceiling, the full space that contains "
                     "the four reduced ones", loc="left", fontsize=8.5,
                     color=ck.INK2, pad=8)
    ax.xaxis.grid(True, zorder=1)
    ax.set_axisbelow(True)

    fig.suptitle("The search criterion: KTA − λ·DLA, exact optimum",
                 x=0.015, ha="left", fontsize=12, color=ck.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(fig_dir, "fig_ablation_heuristic.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  figure → {path}")


def figures(heur, r2, corr_rows, x0x1, x0, x1, y, weight, seeds, fig_dir):
    import matplotlib.pyplot as plt

    n = len(seeds)
    os.makedirs(fig_dir, exist_ok=True)
    keep = {0: ck.RED, 1: ck.BLUE, 2: ck.BASELINE, 3: ck.BASELINE}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

    def ranking(ax, values, title, xlabel, fmt):
        order = sorted(range(4), key=lambda i: values[i])
        pos = np.arange(4)
        vals = [values[i] for i in order]
        ax.barh(pos, vals, height=0.62, color=[keep[i] for i in order], zorder=2)
        ax.set_yticks(pos, [LABELS[i] for i in order])
        for p, v in zip(pos, vals):
            ax.text(v + max(vals) * 0.015, p, fmt(v), va="center", ha="left",
                    fontsize=9, color=ck.INK2)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_title(title, loc="left", color=ck.INK)
        ax.set_xlabel(xlabel, color=ck.INK2)
        ax.xaxis.grid(True, zorder=1)
        ax.set_axisbelow(True)

    ranking(axes[0, 0], {i: float(np.mean(v)) for i, v in heur.items()},
            "The search criterion  (KTA − λ·DLA, exact optimum)",
            f"mean score over {n} splits, λ = {weight:g}", lambda v: f"{v:+.3f}")
    ranking(axes[0, 1], {i: float(np.median(v)) for i, v in r2.items()},
            "The performance  (RBF, γ by 5-fold CV)",
            f"median test R² over {n} splits", lambda v: f"{v:+.3f}")
    axes[0, 0].set_ylabel("dropped feature", color=ck.INK2)

    def duel(ax, a, b, title, xlabel):
        d = np.asarray(a) - np.asarray(b)
        pos = np.arange(n)
        ax.barh(pos, d, height=0.62, zorder=2,
                color=[ck.BLUE if v > 0 else ck.RED for v in d])
        ax.axvline(0, color=ck.MUTED, lw=1.0, zorder=3)
        ax.set_yticks(pos, [f"split {s}" for s in seeds], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=ck.INK)
        ax.set_xlabel(xlabel, color=ck.INK2)
        ax.xaxis.grid(True, zorder=1)
        ax.set_axisbelow(True)
        lim = float(np.abs(d).max()) * 1.45
        ax.set_xlim(-lim, lim)
        won = int((d > 0).sum())
        right = float(d.sum()) > 0
        ax.text(0.015 if right else 0.985, 0.985,
                f"dropping x1 wins {won}/{n}",
                ha="left" if right else "right", va="top",
                transform=ax.transAxes, fontsize=9, color=ck.INK2)
        j = int(np.argmax(np.abs(d)))
        if abs(d[j]) > 3 * np.median(np.abs(d)):
            ax.annotate(f"split {seeds[j]}: {d[j]:+.2f}\ncarries the whole gap",
                        xy=(d[j], j), xytext=(0.62 if d[j] < 0 else 0.38, 0.42),
                        textcoords="axes fraction", fontsize=8, color=ck.INK2,
                        ha="left" if d[j] < 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color=ck.MUTED, lw=0.8))
        return d

    duel(axes[1, 0], heur[1], heur[0], "Split-by-split duel — search criterion",
         "score(without x1) − score(without x0)")
    duel(axes[1, 1], r2[1], r2[0], "Split-by-split duel — performance",
         "R²(without x1) − R²(without x0)")

    fig.suptitle("Drop x0 or x1? The two criteria disagree",
                 fontsize=13, color=ck.INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(fig_dir, "fig_x0_vs_x1.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  figure → {path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    from matplotlib.patches import Patch

    pos = np.arange(4)
    w = 0.36
    colors = [keep[i] for i in range(4)]
    ax1.bar(pos - w / 2, [r["pearson"] for r in corr_rows], w, color=colors,
            zorder=2)
    ax1.bar(pos + w / 2, [r["spearman"] for r in corr_rows], w, color=colors,
            hatch="///", edgecolor=ck.SURFACE, linewidth=0, zorder=2)
    ax1.set_xticks(pos, [LABELS[i].replace(" · ", "\n") for i in range(4)],
                   fontsize=8)
    ax1.set_ylim(0, max(max(r["pearson"], r["spearman"])
                        for r in corr_rows) * 1.32)
    ax1.set_ylabel("|correlation| with the uptake", color=ck.INK2)
    ax1.set_title("x0 and x1 tie on Pearson,\nnot on rank",
                  loc="left", color=ck.INK)
    ax1.legend(handles=[Patch(facecolor=ck.MUTED, label="Pearson"),
                        Patch(facecolor=ck.MUTED, hatch="///",
                              edgecolor=ck.SURFACE, label="Spearman (rank)")],
               loc="upper center", ncol=2, fontsize=8)
    ax1.yaxis.grid(True, zorder=1)
    ax1.set_axisbelow(True)

    sc = ax2.scatter(x0, x1, c=y, cmap=ck.CMAP_SEQ, s=28,
                     edgecolors=ck.BASELINE, linewidths=0.5, zorder=2)
    ax2.set_xlabel("x0 · BET (m²/g)", color=ck.INK2)
    ax2.set_ylabel("x1 · pore volume (m³/g)", color=ck.INK2)
    ax2.set_title(f"… because they are collinear: r = {x0x1:.3f}",
                  loc="left", color=ck.INK)
    ax2.grid(True, zorder=1)
    ax2.set_axisbelow(True)
    fig.colorbar(sc, ax=ax2, label="uptake (wt %)")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_x0_vs_x1_data.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  figure → {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10,
                   help="number of MOF splits (default: 10)")
    p.add_argument("--lambda", dest="weight", type=float,
                   default=common.PRODUCTION_WEIGHTS["dla"],
                   help="weight of the DLA penalty (default: 0.05)")
    p.add_argument("--no-fig", action="store_true", help="no PNG")
    p.add_argument("--ceiling", action="store_true",
                   help="add the 'nothing dropped' case: the optimum over the full "
                        "space, an upper bound on the four ablations (+4 min)")
    p.add_argument("--figs", choices=("all", "heuristic"), default="all",
                   help="'heuristic': only the search criterion, "
                        "without the RBF ablation (default: all)")
    p.add_argument("--fig-dir", default=FIG_DIR)
    args = p.parse_args()
    seeds = range(args.seeds)

    t0 = time.time()
    corr_rows, x0x1, x0, x1, y = correlations()
    print("=== the data")
    print(f"  r(x0, x1) = {x0x1:+.3f}  (collinearity)")
    for r in corr_rows:
        print(f"  {LABELS[r['i']]:>18}: |Pearson| = {r['pearson']:.3f}   "
              f"|Spearman| = {r['spearman']:.3f}")
    from qksearch.dataset.mof import TARGET

    r_x0 = multiple_corr((0, 2, 3))
    r_x1 = multiple_corr((1, 2, 3))
    print(f"  multiple correlation with x4 ({TARGET}): "
          f"(x0,x2,x3) R = {r_x0:.4f}   (x1,x2,x3) R = {r_x1:.4f}   "
          f"gap = {r_x1 - r_x0:+.4f}")

    ktas, norm_dims = ablation_ktas(seeds)
    full = full_ktas(seeds) if args.ceiling else None
    heur = heuristic_scores(ktas, norm_dims, args.weight)
    r2 = ablation_r2(seeds) if args.figs == "all" else None
    print(f"\n=== ablation over {args.seeds} splits ({time.time() - t0:.1f}s)")
    print(f"{'dropped':>18} | {'heuristic score':>18}"
          + (f" | {'R² test (RBF-CV)':>18}" if r2 else ""))
    for i in range(4):
        print(f"{LABELS[i]:>18} | {np.mean(heur[i]):>+18.4f}"
              + (f" | {np.median(r2[i]):>+18.4f}" if r2 else ""))
    if full is not None:
        none = [float(np.max(k - args.weight * full[1])) for k in full[0]]
        print(f"{'(none)':>18} | {np.mean(none):>+18.4f}{'':>21}"
              "← ceiling: the full space contains the reduced ones")

    print("\n=== the duel, paired split by split")
    duels = [("search criterion", heur)]
    if r2:
        duels.append(("RBF performance", r2))
    for name, data in duels:
        mean, pval, won = _paired(data[1], data[0])
        verdict = "drop x1" if mean > 0 else "drop x0"
        print(f"  {name:>24}: Δ(without x1 − without x0) = {mean:+.4f}   "
              f"p = {pval:.3f}   → {verdict} ({won}/{args.seeds} splits)")

    lo = min(np.max(k1 - lam * norm_dims) - np.max(k0 - lam * norm_dims)
             for lam in np.linspace(0.0, 1.0, 41)
             for k1, k0 in zip(ktas[1], ktas[0]))
    print(f"  robustness to the weight: over λ ∈ [0, 1], the worst per-split gap "
          f"stays at {lo:+.4f}")

    if not args.no_fig:
        print()
        figure_heuristic(ktas, norm_dims, full, args.weight, list(seeds),
                         args.fig_dir)
        if r2:
            figures(heur, r2, corr_rows, x0x1, x0, x1, y, args.weight,
                    list(seeds), args.fig_dir)
    print(f"\n  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
