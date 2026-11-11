from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e5_benchmark.csv"
SEEDS = list(range(10))
N_DATA_GATES = 6
MCTS_ITERATIONS = 1000
GAMMAS = np.logspace(-2, 2, 41)
OBJECTIVES = {
    "quantum-kta": ("none", 0.0),
    "quantum-dla": ("dla", common.PRODUCTION_WEIGHTS["dla"]),
    "quantum-var": ("variance", common.PRODUCTION_WEIGHTS["variance"]),
}
DATASETS = {"sine": 2, "mof": 4}


def _r2(K_train, K_test, train, test) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    krr = KernelRidge(alpha=common.RIDGE_ALPHA,
                      kernel="precomputed").fit(K_train, train.y)
    return float(r2_score(test.y, krr.predict(K_test)))


def _rbf_gamma_cv(train) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.model_selection import KFold

    folds = list(KFold(n_splits=5, shuffle=True, random_state=0)
                 .split(train.X))
    best_gamma, best_score = GAMMAS[0], -np.inf
    for gamma in GAMMAS:
        K = rbf_kernel(train.X, gamma=gamma)
        scores = []
        for tr, va in folds:
            krr = KernelRidge(alpha=common.RIDGE_ALPHA, kernel="precomputed")
            krr.fit(K[np.ix_(tr, tr)], train.y[tr])
            scores.append(r2_score(train.y[va], krr.predict(K[np.ix_(va, tr)])))
        score = float(np.mean(scores))
        if score > best_score:
            best_gamma, best_score = float(gamma), score
    return best_gamma


def run() -> None:
    from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

    from qksearch.backend import NumpyBackend
    from qksearch.compare_kernels import tune_rbf_gamma
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.iqp import build_heuristic, describe, iqp_circuit
    from qksearch.metric import TargetAlignment
    from qksearch.search import MctsSearch, search

    t0 = time.time()
    kta_metric = TargetAlignment()
    rows: list[dict] = []
    for dataset, n in DATASETS.items():
        for seed in SEEDS:
            train, test = (common.sine_split(seed) if dataset == "sine"
                           else common.mof_split(seed))

            for method, (regularization, weight) in OBJECTIVES.items():
                heuristic = build_heuristic(
                    train, n_qubits=n, n_data_gates=N_DATA_GATES,
                    regularization=regularization, weight=weight,
                    backend=NumpyBackend(n))
                res = search((), heuristic,
                             MctsSearch(n_iterations=MCTS_ITERATIONS, seed=seed))
                circuit = iqp_circuit(res.best_state, n)
                K_train = heuristic.backend.gram(circuit, train.X)
                K_test = heuristic.backend.cross_gram(circuit, test.X, train.X)
                rows.append({
                    "dataset": dataset, "seed": seed, "method": method,
                    "kta_train": float(kta_metric(K_train, train.y)),
                    "r2_test": _r2(K_train, K_test, train, test),
                    "n_evaluated": res.n_evaluated,
                    "detail": describe(strip_identity(res.best_state)) or "empty",
                })

            gamma_kta = tune_rbf_gamma(train)
            gamma_cv = _rbf_gamma_cv(train)
            classical = {
                "rbf-kta": (rbf_kernel(train.X, gamma=gamma_kta),
                            rbf_kernel(test.X, train.X, gamma=gamma_kta),
                            f"gamma={gamma_kta:g}"),
                "rbf-cv": (rbf_kernel(train.X, gamma=gamma_cv),
                           rbf_kernel(test.X, train.X, gamma=gamma_cv),
                           f"gamma={gamma_cv:g}"),
                "linear": (linear_kernel(train.X),
                           linear_kernel(test.X, train.X), ""),
            }
            for method, (K_train, K_test, detail) in classical.items():
                rows.append({
                    "dataset": dataset, "seed": seed, "method": method,
                    "kta_train": float(kta_metric(K_train, train.y)),
                    "r2_test": _r2(K_train, K_test, train, test),
                    "n_evaluated": 0, "detail": detail,
                })
            print(f"{dataset} seed {seed} done ({time.time() - t0:.1f}s)")

    common.write_csv(CSV, rows)
    common.write_meta("e5_benchmark", time.time() - t0, seeds=SEEDS,
                      n_data_gates=N_DATA_GATES, mcts_iterations=MCTS_ITERATIONS,
                      objectives={k: list(v) for k, v in OBJECTIVES.items()})


ORDER = ["quantum-kta", "quantum-dla", "quantum-var", "rbf-kta", "rbf-cv", "linear"]
LABELS = {
    "quantum-kta": "IQP (KTA)",
    "quantum-dla": "IQP (KTA$-\\lambda\\,$DLA)",
    "quantum-var": "IQP (KTA$+c\\,$Var)",
    "rbf-kta": "RBF ($\\gamma$ by KTA)",
    "rbf-cv": "RBF ($\\gamma$ by 5-fold CV)",
    "linear": "linear",
}


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=False)
    titles = {"sine": "synthetic sine (40 train / 80 test)",
              "mof": "MOF H$_2$ uptake (100 train / 194 test)"}
    ymin = -0.55
    for ax, dataset in zip(axes, ("sine", "mof")):
        sub = df[df["dataset"] == dataset]
        rng = np.random.default_rng(0)
        for i, method in enumerate(ORDER):
            vals = sub[sub["method"] == method]["r2_test"].to_numpy()
            color = common.BLUE if method.startswith("quantum") else common.MUTED
            med = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.errorbar([i], [med], yerr=[[med - q1], [q3 - med]], color=color,
                        marker="o", ms=5, capsize=3, lw=1.4, zorder=3)
            shown = np.clip(vals, ymin + 0.02, None)
            jit = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(i + jit, shown, s=7, color=color, alpha=0.35, lw=0)
            for v in vals[vals < ymin]:
                ax.annotate(f"{v:.1f}", (i, ymin + 0.03), fontsize=6,
                            color=color, ha="center", va="bottom")
        ax.set_ylim(ymin, 1.05)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=35,
                           ha="right", fontsize=7)
        ax.set_title(titles[dataset], fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].set_ylabel("test $R^2$")
    fig.tight_layout()
    common.save_fig(fig, "fig_e5_benchmark")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
