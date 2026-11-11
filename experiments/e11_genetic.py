from __future__ import annotations

import sys
import time

import numpy as np

from experiments import common

CSV = "e11_genetic.csv"
CSV_BEST = "e11_best.csv"
SEEDS = list(range(10))
N_QUBITS = 4
N_FEATURES = 4
MAX_GATES = 6
POP_SIZE = 100
N_HARVESTED = 20
N_GENERATIONS = 40
PATIENCE = 12

MODES: dict[str, dict] = {
    "valid_r1": dict(n_layers=1, phases=(), max_weight=MAX_GATES, max_support=None),
    "prod_r2": dict(n_layers=2, phases=(), max_weight=None, max_support=MAX_GATES),
    "prod_r2_phase": dict(n_layers=2, phases=("t", "cs"),
                          max_weight=None, max_support=MAX_GATES),
}


def _heuristic(train, phases=()):
    from qksearch.backend import NumpyBackend
    from qksearch.genome import layer_gate_set
    from qksearch.iqp import IqpKernelHeuristic
    from qksearch.kernel import GramVarianceBonus
    from qksearch.metric import TargetAlignment

    gs = layer_gate_set(N_QUBITS, N_FEATURES, phases)
    return IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
        metric=TargetAlignment(), max_depth=MAX_GATES, n_qubits=N_QUBITS,
        regularizer=GramVarianceBonus(common.PRODUCTION_WEIGHTS["variance"]))


def _harvested(spec) -> list:
    import csv

    from experiments.population_circuits import gate_set
    from qksearch.genome import encode

    path = common.result_path("population_mof.csv")
    catalog = gate_set().catalog
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("origin", "search") != "search":
                continue
            state = tuple(catalog[int(i)] for i in row["genome_ranks"].split(","))
            state = tuple(op for op in state if op[0] != "id")
            try:
                out.append(encode(state, spec))
            except ValueError:
                continue
    return out[:N_HARVESTED]


def _initial_population(spec, seed: int) -> list:
    from qksearch.search.genetic import seed_population

    rng = np.random.default_rng(seed)
    return seed_population(spec, POP_SIZE, rng, seeds=_harvested(spec),
                           max_support=MAX_GATES, max_weight=MAX_GATES)


def _r2_of(heuristic, state, train, test) -> float:
    from experiments.e5_benchmark import _r2

    circuit = heuristic.circuit(state)
    return _r2(heuristic.evaluator.gram(state),
               heuristic.backend.cross_gram(circuit, test.X, train.X),
               train, test)


def _exhaustive_optimum(heuristic, spec) -> float:
    from qksearch.genome import decode, enumerate_genomes

    genomes = list(enumerate_genomes(spec, max_weight=MAX_GATES))
    assert len(genomes) == common.space_size(N_QUBITS, MAX_GATES), len(genomes)
    return float(heuristic.scores([decode(g, spec) for g in genomes]).max())


def run(modes: tuple[str, ...] = ()) -> None:
    from qksearch.genome import (
        decode,
        iqp_genome_spec,
        n_phase_gates,
        support,
        to_int,
        weight,
    )
    from qksearch.iqp import describe
    from qksearch.search import search
    from qksearch.search.genetic import GeneticSearch

    t0 = time.time()
    common.ensure_dirs()
    curves: list[dict] = []
    best_rows: list[dict] = []
    optima: dict[str, float] = {}

    selected = {m: MODES[m] for m in (modes or MODES)}
    for mode, cfg in selected.items():
        spec = iqp_genome_spec(N_QUBITS, N_FEATURES, n_layers=cfg["n_layers"],
                               phases=cfg["phases"])
        print(f"\n=== {mode} — {spec.describe()} ===")
        for seed in SEEDS:
            train, test = common.mof_split(seed)
            heuristic = _heuristic(train, cfg["phases"])
            pop = _initial_population(spec, seed)
            ga = GeneticSearch(spec, population=pop,
                               n_generations=N_GENERATIONS,
                               max_support=cfg["max_support"] or len(spec.genes),
                               max_weight=cfg["max_weight"],
                               patience=PATIENCE, seed=seed)
            result = search((), heuristic, ga)
            g = ga.best_genome
            state = decode(g, spec)

            for stats in ga.history:
                curves.append(dict(
                    mode=mode, seed=seed, generation=stats.generation,
                    best_fitness=stats.best, mean_fitness=stats.mean,
                    diversity=stats.diversity, n_evaluated=stats.n_evaluated))

            opt = (_exhaustive_optimum(heuristic, spec) if cfg["max_weight"]
                   else float("nan"))
            optima[f"{mode}:{seed}"] = opt
            best_rows.append(dict(
                mode=mode, protocol="ga", seed=seed,
                fitness=result.best_score,
                kta_train=heuristic.kta(state),
                r2_test=_r2_of(heuristic, state, train, test),
                weight=weight(g, spec), support=support(g),
                n_phase_gates=n_phase_gates(g, spec),
                n_layers=cfg["n_layers"], n_evaluated=result.n_evaluated,
                frac_of_opt=result.best_score / opt if opt == opt else float("nan"),
                optimum=opt, genome_int=to_int(g, spec),
                detail=describe(state)))
            flag = "" if opt != opt else (
                " OPT" if result.best_score >= opt - 1e-9
                else f"  {100 * result.best_score / opt:.2f}% of opt")
            print(f"  seed {seed}: fitness {result.best_score:+.6f}  "
                  f"KTA {heuristic.kta(state):+.4f}  R² {best_rows[-1]['r2_test']:+.3f}  "
                  f"{support(g)} gates ({n_phase_gates(g, spec)} phase)  "
                  f"{result.n_evaluated} evals{flag}")

    common.write_csv(CSV, curves)
    common.write_csv(CSV_BEST, best_rows)

    import pandas as pd

    df = pd.DataFrame(best_rows)
    agg = df.groupby("mode").agg(
        fitness=("fitness", "median"), kta=("kta_train", "median"),
        r2=("r2_test", "median"), gates=("support", "median"),
        phases=("n_phase_gates", "median"), evals=("n_evaluated", "median"))
    print("\n" + agg.to_string(float_format=lambda v: f"{v:+.4f}"))

    valid = df[df["mode"] == "valid_r1"]
    hits = len(valid)
    if len(valid):
        hits = int((valid["frac_of_opt"] >= 1 - 1e-9).sum())
        print(f"\nvalid_r1: exact optimum reached {hits}/{len(valid)} seeds, "
              f"median {valid['n_evaluated'].median():.0f} evaluations out of "
              f"{common.space_size(N_QUBITS, MAX_GATES)} circuits")
        assert hits >= 8, f"the GA misses the exact optimum too often: {hits}/{len(valid)}"
    else:
        print("\n! valid_r1 missing: no validation against the exact optimum")

    phase = df[df["mode"] == "prod_r2_phase"]
    kept = int((phase["n_phase_gates"] > 0).sum())
    if len(phase):
        print(f"prod_r2_phase: t/cs kept in {kept}/{len(phase)} best circuits")

    common.write_meta(
        "e11_genetic", time.time() - t0, modes=list(selected),
        population_size=POP_SIZE, n_harvested=N_HARVESTED,
        n_generations=N_GENERATIONS, patience=PATIENCE, seeds=SEEDS,
        max_gates=MAX_GATES,
        variance_weight=common.PRODUCTION_WEIGHTS["variance"],
        space_valid=common.space_size(N_QUBITS, MAX_GATES),
        optimum_valid={k: v for k, v in optima.items()
                       if k.startswith("valid_r1") and v == v},
        exact_hits=hits, phase_kept=kept)


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    best = common.read_csv(CSV_BEST)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.4))
    labels = {"valid_r1": "1 layer\nrz/rzz", "prod_r2": "2 layers\nrz/rzz",
              "prod_r2_phase": "2 layers\n+ t/cs"}

    ax = axes[0]
    opt = best[best["mode"] == "valid_r1"].set_index("seed")["optimum"]
    sub = df[df["mode"] == "valid_r1"].copy()
    sub["frac"] = sub["best_fitness"] / sub["seed"].map(opt)
    curve = sub.groupby("generation")["frac"].agg(["mean", "min"])
    ax.plot(curve.index, curve["mean"], color=common.BLUE, lw=1.6,
            label="mean over 10 seeds")
    ax.plot(curve.index, curve["min"], color=common.BLUE, lw=0.9, ls="--",
            label="worst seed")
    ax.axhline(1.0, color=common.INK2, lw=0.9, ls=":")
    ax.text(curve.index.max(), 1.0, "exact optimum ", fontsize=6.5,
            color=common.INK2, ha="right", va="bottom")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness / exact optimum")
    ax.set_title(f"validation: {common.space_size(N_QUBITS, MAX_GATES)} circuits, "
                 f"median {best[best['mode'] == 'valid_r1']['n_evaluated'].median():.0f}"
                 " evaluations", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")

    ax = axes[1]
    modes = [m for m in MODES if (best["mode"] == m).any()]
    x = np.arange(len(modes))
    for k, (col, label, color) in enumerate(
            [("kta_train", "KTA (train, maximized)", common.BLUE),
             ("r2_test", r"$R^2$ (test, observed)", common.RED)]):
        vals = [best.loc[best["mode"] == m, col].median() for m in modes]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in modes], fontsize=6.5)
    ax.axhline(0, color=common.INK2, lw=0.6)
    ax.set_ylabel("median over 10 seeds")
    ax.set_title("a bigger space buys KTA and loses $R^2$", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.5)

    fig.tight_layout()
    common.save_fig(fig, "fig_e11_genetic")


def _cli_modes() -> tuple[str, ...]:
    for i, a in enumerate(sys.argv):
        if a == "--modes" and i + 1 < len(sys.argv):
            names = tuple(sys.argv[i + 1].split(","))
        elif a.startswith("--modes="):
            names = tuple(a.split("=", 1)[1].split(","))
        else:
            continue
        unknown = [n for n in names if n not in MODES]
        if unknown:
            raise SystemExit(f"unknown mode: {', '.join(unknown)} "
                             f"(expected: {', '.join(MODES)})")
        return names
    return ()


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run(_cli_modes())
    plot()
