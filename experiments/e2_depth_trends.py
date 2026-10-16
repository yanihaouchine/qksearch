from __future__ import annotations

import sys
import time


from experiments import common

CSV = "e2_depth_trends.csv"
CONFIGS = {
    "2q": {"n_qubits": 2, "depth": 6, "seeds": list(range(10))},
    "4q": {"n_qubits": 4, "depth": 5, "seeds": list(range(5))},
}


def enumerate_states(gs, depth):
    states = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def run() -> None:
    from qksearch.backend import NumpyBackend
    from qksearch.dla import DlaCalculator
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.gates import GateSpec, ProductGateSet
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.kernel import GramVarianceBonus
    from qksearch.metric import TargetAlignment

    t0 = time.time()
    rows: list[dict] = []
    dla = DlaCalculator()
    for config, cfg in CONFIGS.items():
        n, depth = cfg["n_qubits"], cfg["depth"]
        gs = ProductGateSet(
            [GateSpec("rz", 1, uses_feature=True),
             GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
            n_qubits=n, n_features=n, with_identity=True)
        states = enumerate_states(gs, depth)
        assert len(states) == common.space_size(n, depth), \
            (len(states), common.space_size(n, depth))

        static = {}
        for s in states:
            static[s] = {"n_gates": len(strip_identity(s))}

        for seed in cfg["seeds"]:
            train, _ = common.sine_split(seed, n_features=n)
            heuristic = IqpKernelHeuristic(
                gate_set=gs, backend=NumpyBackend(n), data=train,
                metric=TargetAlignment(), max_depth=depth, n_qubits=n,
                regularizer=GramVarianceBonus(weight=1.0))
            ktas = heuristic.evaluator.kta_many(states)
            for s, kta in zip(states, ktas):
                if "dla" not in static[s]:
                    static[s]["dla"] = dla.dimension(heuristic.circuit(s))
                rows.append({
                    "config": config, "seed": seed,
                    "n_gates": static[s]["n_gates"],
                    "kta": float(kta),
                    "dla": static[s]["dla"],
                    "var4": heuristic.evaluator.feature(s),
                })
            print(f"{config} seed {seed} fait ({time.time() - t0:.1f}s)")
    common.write_csv(CSV, rows)
    common.write_meta("e2_depth_trends", time.time() - t0,
                      configs={k: v for k, v in CONFIGS.items()})


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.4))
    colors = {"2q": common.BLUE, "4q": common.RED}
    labels = {"2q": "2 qubits (84 circuits)", "4q": "4 qubits (3003 circuits)"}

    for config, sub in df.groupby("config"):
        c = colors[config]
        per_seed = sub.groupby(["seed", "n_gates"])["kta"].mean().unstack()
        mean, std = per_seed.mean(axis=0), per_seed.std(axis=0)
        ax1.plot(mean.index, mean.values, color=c, lw=1.6, marker="o", ms=3,
                 label=labels[config])
        ax1.fill_between(mean.index, (mean - std).values, (mean + std).values,
                         color=c, alpha=0.18, lw=0)
        best = sub.groupby(["seed", "n_gates"])["kta"].max().unstack()
        ax1.plot(best.mean(axis=0).index, best.mean(axis=0).values, color=c,
                 lw=1.1, ls="--", alpha=0.7)

        dims = sub[sub["seed"] == sub["seed"].min()]
        dmean = dims.groupby("n_gates")["dla"].mean()
        ax2.plot(dmean.index, dmean.values, color=c, lw=1.6, marker="o", ms=3,
                 label=labels[config])

    ax1.set_xlabel("number of data-encoding gates")
    ax1.set_ylabel("kernel–target alignment")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.text(0.97, 0.05, "dashed: best circuit\nsolid: space average",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=7,
             color=common.INK2)
    ax2.set_xlabel("number of data-encoding gates")
    ax2.set_ylabel("mean DLA dimension")
    ax2.legend(loc="upper left", fontsize=7)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.6)
    fig.tight_layout()
    common.save_fig(fig, "fig_e2_depth_trends")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
