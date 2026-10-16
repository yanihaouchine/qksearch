from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e4_search.csv"
CSV_SPACE = "e4_space.csv"
N_QUBITS, DEPTH = 4, 6
BUDGETS = (10, 20, 50, 100, 200, 500, 1000, 2000)
SEEDS = list(range(10))


def _heuristic(train, n, depth):
    from qksearch.backend import NumpyBackend
    from qksearch.gates import GateSpec, ProductGateSet
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.metric import TargetAlignment

    gs = ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=n, n_features=n, with_identity=True)
    return IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(n), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=n,
        regularizer=None), gs


def run() -> None:
    from qksearch.enumerate_circuits import strip_identity
    from qksearch.iqp import describe
    from qksearch.search import GreedySearch, MctsSearch, search

    t0 = time.time()
    train, _ = common.mof_split(0)
    heuristic, gs = _heuristic(train, N_QUBITS, DEPTH)

    states = [()]
    for _ in range(DEPTH):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    assert len(states) == common.space_size(N_QUBITS, DEPTH) == 8008
    ktas = heuristic.evaluator.kta_many(states)
    i_opt = int(np.argmax(ktas))
    opt = float(ktas[i_opt])
    print(f"optimum exhaustif : KTA {opt:+.4f} "
          f"({describe(strip_identity(states[i_opt]))}) "
          f"en {time.time() - t0:.1f}s")
    common.write_csv(CSV_SPACE, [
        {"circuit_id": i, "n_gates": len(strip_identity(s)), "kta": float(k)}
        for i, (s, k) in enumerate(zip(states, ktas))])

    rows: list[dict] = []
    for budget in BUDGETS:
        for seed in SEEDS:
            res = search((), heuristic, MctsSearch(n_iterations=budget, seed=seed))
            best = float(heuristic.kta(res.best_state))
            rows.append({
                "space": "mof8008", "strategy": "mcts", "n_iterations": budget,
                "seed": seed, "n_evaluated": res.n_evaluated,
                "best_kta": best, "frac_of_opt": best / opt,
            })
        done = [r for r in rows if r["n_iterations"] == budget]
        print(f"MCTS {budget:5d} it : frac moyenne "
              f"{np.mean([r['frac_of_opt'] for r in done]):.3f} "
              f"({time.time() - t0:.1f}s)")

    res = search((), heuristic, GreedySearch())
    best = float(heuristic.kta(res.best_state))
    rows.append({"space": "mof8008", "strategy": "greedy", "n_iterations": 0,
                 "seed": 0, "n_evaluated": res.n_evaluated,
                 "best_kta": best, "frac_of_opt": best / opt})
    print(f"greedy: frac {best / opt:.3f} in {res.n_evaluated} evaluations")

    top = max(r["frac_of_opt"] for r in rows
              if r["strategy"] == "mcts" and r["n_iterations"] == 2000)
    assert top > 0.999, f"MCTS 2000 n'atteint pas l'optimum : {top}"

    common.write_csv(CSV, rows)
    common.write_meta("e4_search", time.time() - t0, n_qubits=N_QUBITS,
                      depth=DEPTH, budgets=list(BUDGETS), seeds=SEEDS,
                      space=8008, optimum_kta=opt)


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    space = common.read_csv(CSV_SPACE)

    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    mcts = df[df["strategy"] == "mcts"]
    g = mcts.groupby("n_iterations").agg(
        evals=("n_evaluated", "mean"),
        frac=("frac_of_opt", "mean"),
        std=("frac_of_opt", "std"))
    ax.axhline(1.0, color=common.INK2, lw=0.9, ls="--")
    ax.errorbar(g["evals"], g["frac"], yerr=g["std"], color=common.BLUE,
                lw=1.6, marker="o", ms=3.5, capsize=2, label="MCTS (10 seeds)")
    greedy = df[df["strategy"] == "greedy"].iloc[0]
    ax.plot([greedy["n_evaluated"]], [greedy["frac_of_opt"]], marker="s",
            ms=5, color=common.RED, ls="none", label="greedy")
    ax.axvline(len(space), color=common.BASELINE, lw=1.2)
    ax.text(0.965, 0.45, f"exhaustive:\n{len(space)} circuits",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=common.INK2)
    ax.set_xscale("log")
    ax.set_xlabel("kernel evaluations")
    ax.set_ylabel("fraction of exhaustive optimum")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.5)
    common.save_fig(fig, "fig_e4_search")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
