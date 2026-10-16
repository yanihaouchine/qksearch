from __future__ import annotations

import sys
import time


from experiments import common

CSV = "e3_expressivity.csv"
N_PAIRS = 20000
SEED = 0
SPACES = {"sine": {"n_qubits": 2, "depth": 4},
          "mof": {"n_qubits": 4, "depth": 4}}


def run() -> None:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import rbf_kernel

    from qksearch.backend import NumpyBackend
    from qksearch.compare_kernels import tune_rbf_gamma
    from qksearch.dla import DlaCalculator
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.expressibility import expressibility
    from qksearch.gates import GateSpec, ProductGateSet
    from qksearch.geometry import geometric_difference
    from qksearch.iqp import IqpKernelHeuristic, describe
    from qksearch.kernel import GramVarianceBonus
    from qksearch.metric import TargetAlignment

    t0 = time.time()
    dla = DlaCalculator()
    rows: list[dict] = []
    for space, cfg in SPACES.items():
        n, depth = cfg["n_qubits"], cfg["depth"]
        train, test = (common.sine_split(SEED) if space == "sine"
                       else common.mof_split(SEED))
        gamma = tune_rbf_gamma(train)
        K_rbf = rbf_kernel(train.X, gamma=gamma)

        gs = ProductGateSet(
            [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
            n_qubits=n, n_features=n, with_identity=True)
        states = [()]
        for _ in range(depth):
            states = [s + (op,) for s in states for op in gs.placements(s)]
        assert len(states) == common.space_size(n, depth)

        backend = NumpyBackend(n)
        heuristic = IqpKernelHeuristic(
            gate_set=gs, backend=backend, data=train,
            metric=TargetAlignment(), max_depth=depth, n_qubits=n,
            regularizer=GramVarianceBonus(weight=1.0))
        ktas = heuristic.evaluator.kta_many(states)

        for i, (s, kta) in enumerate(zip(states, ktas)):
            circuit = heuristic.circuit(s)
            gram = heuristic.evaluator.gram(s)
            K_test = backend.cross_gram(circuit, test.X, train.X)
            krr = KernelRidge(alpha=common.RIDGE_ALPHA,
                              kernel="precomputed").fit(gram, train.y)
            rows.append({
                "space": space, "circuit_id": i,
                "describe": describe(strip_identity(s)) or "empty",
                "n_gates": len(strip_identity(s)),
                "kta": float(kta),
                "var4": heuristic.evaluator.feature(s),
                "dla": dla.dimension(circuit),
                "expr": float(expressibility(backend, circuit, n,
                                             n_pairs=N_PAIRS, seed=SEED)),
                "r2": float(r2_score(test.y, krr.predict(K_test))),
                "geom": float(geometric_difference(gram, K_rbf)),
                "rbf_gamma": gamma,
            })
            if (i + 1) % 200 == 0:
                print(f"{space} {i + 1}/{len(states)} ({time.time() - t0:.1f}s)")
        print(f"{space} fait ({time.time() - t0:.1f}s)")

    sine = [r for r in rows if r["space"] == "sine"]
    best = max(sine, key=lambda r: r["kta"])
    worst = min(sine, key=lambda r: r["kta"])
    assert abs(best["kta"] - 0.1975) < 1e-3, best
    assert abs(best["r2"] - 0.981) < 2e-3, best
    assert abs(worst["kta"] - 0.0048) < 1e-3, worst
    assert abs(worst["r2"] - (-0.059)) < 2e-3, worst

    common.write_csv(CSV, rows)
    common.write_meta("e3_expressivity", time.time() - t0,
                      spaces=SPACES, n_pairs=N_PAIRS, seed=SEED)


def plot() -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    common.report_style()
    df = common.read_csv(CSV)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5))
    titles = {"sine": "synthetic sine — 35 circuits, 2 qubits",
              "mof": "MOF H$_2$ uptake — 1001 circuits, 4 qubits"}
    ymin = {"sine": -0.45, "mof": -1.0}
    for ax, space in zip(axes, ("sine", "mof")):
        sub = df[df["space"] == space]
        lo = ymin[space]
        shown = sub[sub["r2"] >= lo]
        clipped = len(sub) - len(shown)
        sc = ax.scatter(shown["expr"], shown["r2"], c=shown["kta"],
                        cmap="viridis", s=13, lw=0, alpha=0.85)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_ylim(lo, 1.08)
        rho_e = spearmanr(sub["expr"], sub["r2"]).statistic
        rho_k = spearmanr(sub["kta"], sub["r2"]).statistic
        ax.set_title(titles[space], fontsize=8)
        ax.text(0.03, 0.04,
                rf"$\rho$(expr., $R^2$) = {-rho_e:+.2f}"
                "\n"
                rf"$\rho$(KTA, $R^2$) = {rho_k:+.2f}",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        if clipped:
            ax.text(0.03, 0.97, f"{clipped} circuits below {lo:g} not shown",
                    transform=ax.transAxes, fontsize=6.5, ha="left",
                    va="top", color=common.INK2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=22)
        cbar.set_label("KTA (train)", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.5)
    axes[0].set_ylabel("test $R^2$")
    fig.supxlabel(r"expressibility $D_{KL}$ to Haar"
                  r"  [$\longrightarrow$ more expressive]", fontsize=8.5)
    fig.tight_layout()
    common.save_fig(fig, "fig_e3_expressivity")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
