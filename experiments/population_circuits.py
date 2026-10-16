from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from qksearch.backend import NumpyBackend
from qksearch.compare_kernels import FIG_DIR, RIDGE_ALPHA, evaluate_regression, tune_rbf_gamma
from qksearch.dataset import RegressionDataset
from qksearch.dla import DlaCalculator
from qksearch.enumerate_circuits import Row, fig_table, print_table, strip_identity
from experiments.common import mof_split, space_size, write_csv
from qksearch.expressibility import expressibility
from qksearch.gates import IDENTITY_OP, GateSpec, ProductGateSet, State
from qksearch.genome import decode, iqp_genome_spec, random_genome
from qksearch.geometry import geometric_difference
from qksearch.iqp import IqpKernelHeuristic, describe
from qksearch.metric import TargetAlignment
from qksearch.search import Individual, PopulationSearch, multiset_distance, search

N_QUBITS = 4
N_FEATURES = 4
MAX_DEPTH = 6
N_TRAIN = 100
SEED = 0

POP_SIZE = 20
N_DESCENTS = 100
MIN_DISTANCE = 2
N_RANDOM = 0
FIG_ROWS = 20

N_PAIRS_EXPR = 20000

DIM_MAX = 4 ** N_QUBITS - 1
FIG_PATH = os.path.join(FIG_DIR, "fig_population_mof.png")
CSV_NAME = "population_mof.csv"


def gate_set() -> ProductGateSet:
    return ProductGateSet(
        [GateSpec("rz", 1, uses_feature=True),
         GateSpec("rzz", 2, uses_feature=True, symmetric=True)],
        n_qubits=N_QUBITS, n_features=N_FEATURES, with_identity=True)


def build_heuristic(train: RegressionDataset, gs: ProductGateSet,
                    depth: int = MAX_DEPTH) -> IqpKernelHeuristic:
    return IqpKernelHeuristic(
        gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
        metric=TargetAlignment(), max_depth=depth, n_qubits=N_QUBITS,
        regularizer=None)


def pad_genome(heuristic: IqpKernelHeuristic, state: State) -> State:
    while len(state) < heuristic.max_depth:
        state = heuristic.extend(state, IDENTITY_OP)
    return state


def random_individuals(heuristic: IqpKernelHeuristic, n: int, *,
                       seed: int) -> list[Individual[State]]:
    spec = iqp_genome_spec(N_QUBITS, N_FEATURES, n_layers=1)
    rng = np.random.default_rng(seed)
    out: list[Individual[State]] = []
    seen: set[State] = set()
    while len(out) < n:
        state = decode(random_genome(spec, rng, max_support=MAX_DEPTH,
                                     max_weight=MAX_DEPTH), spec)
        if state in seen:
            continue
        seen.add(state)
        out.append(Individual(pad_genome(heuristic, state), heuristic.kta(state)))
    return out


def sample_population(heuristic: IqpKernelHeuristic, *, size: int,
                      n_descents: int, min_distance: int, seed: int):
    strategy = PopulationSearch(n_descents=n_descents, lookahead=1, seed=seed,
                                pad=lambda s: pad_genome(heuristic, s))
    result = search((), heuristic, strategy)
    pop = strategy.population(size, key=strip_identity,
                              distance=multiset_distance,
                              min_distance=min_distance)
    return pop, strategy, result


def evaluate(pop: list[Individual[State]], heuristic: IqpKernelHeuristic,
             train: RegressionDataset, test: RegressionDataset,
             gamma: float) -> list[tuple[Individual[State], Row]]:
    backend = heuristic.backend
    dla = DlaCalculator()
    iu = np.triu_indices(train.n_samples, k=1)
    K_rbf = rbf_kernel(train.X, gamma=gamma)

    pairs: list[tuple[Individual[State], Row]] = []
    for ind in pop:
        circuit = heuristic.circuit(ind.state)
        gram = heuristic.evaluator.gram(ind.state)
        K_test = backend.cross_gram(circuit, test.X, train.X)
        pairs.append((ind, Row(
            state=strip_identity(ind.state),
            kta=heuristic.kta(ind.state),
            variance=float(gram[iu].var()),
            dla=dla.dimension(circuit),
            r2=evaluate_regression(describe(ind.state), gram, K_test,
                                   train, test)["r2"],
            geom=geometric_difference(gram, K_rbf),
            expr=expressibility(backend, circuit, N_FEATURES,
                                n_pairs=N_PAIRS_EXPR, seed=SEED),
        )))
    pairs.sort(key=lambda pair: pair[1].kta, reverse=True)
    return pairs


def serialize(state: State) -> str:
    return ";".join(f"{name}({','.join(map(str, qubits))})" if qubits else name
                    for name, qubits, _ in state)


def genome_ranks(state: State, gs: ProductGateSet) -> str:
    return ",".join(str(gs.catalog.index(op)) for op in state)


def write_population(pairs, gs: ProductGateSet, *, seed: int, n_descents: int,
                     min_distance: int, origins: dict) -> str:
    rows = []
    for i, (ind, row) in enumerate(pairs, start=1):
        rows.append({
            "rank": i,
            "origin": origins.get(ind.state, "search"),
            "genome": serialize(ind.state),
            "genome_ranks": genome_ranks(ind.state, gs),
            "circuit": describe(row.state) if row.state else "identity",
            "n_gates": len(row.state),
            "kta": row.kta,
            "variance": row.variance,
            "dla": row.dla,
            "r2": row.r2,
            "geom": row.geom,
            "expr": row.expr,
            "seed": seed,
            "n_descents": n_descents,
            "min_distance": min_distance,
        })
    return write_csv(CSV_NAME, rows)


def min_pairwise_distance(pop: list[Individual[State]]) -> int:
    keys = [strip_identity(ind.state) for ind in pop]
    if len(keys) < 2:
        return 0
    return min(multiset_distance(a, b)
               for i, a in enumerate(keys) for b in keys[i + 1:])


def enumerate_states(gs: ProductGateSet, depth: int = MAX_DEPTH) -> list[State]:
    states: list[State] = [()]
    for _ in range(depth):
        states = [s + (op,) for s in states for op in gs.placements(s)]
    return states


def check_exhaustive(pairs, heuristic: IqpKernelHeuristic,
                     gs: ProductGateSet) -> None:
    states = enumerate_states(gs)
    assert len(states) == space_size(N_QUBITS, MAX_DEPTH), len(states)
    ktas = heuristic.evaluator.kta_many(states)
    opt = float(ktas.max())

    known = set(states)
    unknown = [ind for ind, _ in pairs if ind.state not in known]
    assert not unknown, f"{len(unknown)} genomes outside the canonical space"

    order = np.argsort(-ktas)
    ranks = {states[int(j)]: r for r, j in enumerate(order, start=1)}
    best = pairs[0][1].kta
    print(f"\nexhaustive optimum: KTA {opt:+.4f} "
          f"({describe(strip_identity(states[int(order[0])]))})")
    print(f"best individual   : {best / opt:.4f} of the optimum; "
          f"population ranks in the exhaustive order: "
          + ", ".join(str(ranks[ind.state]) for ind, _ in pairs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED,
                   help="seed of the descents and of the MOF split")
    p.add_argument("--size", type=int, default=POP_SIZE,
                   help="number of individuals to keep")
    p.add_argument("--descents", type=int, default=N_DESCENTS,
                   help="number of stochastic descents to harvest")
    p.add_argument("--min-distance", type=int, default=MIN_DISTANCE,
                   help="minimum gate-multiset distance between two individuals")
    p.add_argument("--random", type=int, default=N_RANDOM, dest="n_random",
                   help="circuits tires au hasard ajoutes a la population")
    p.add_argument("--fig-rows", type=int, default=FIG_ROWS,
                   help="lignes tracees dans la figure (le CSV les garde toutes)")
    p.add_argument("--exhaustive", action="store_true",
                   help="also sweep the 8008 circuits to locate the population")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train, test = mof_split(args.seed, n_train=N_TRAIN)
    gs = gate_set()
    heuristic = build_heuristic(train, gs)
    gamma = tune_rbf_gamma(train)

    print(f"=== IQP circuit population, MOF, {N_QUBITS} qubits, "
          f"gates {{rz, rzz, id}}, depth = {MAX_DEPTH} ===")
    print(f"MOF dataset: {train.n_samples} training points, "
          f"{test.n_samples} test points, {train.n_features} features, "
          f"split seed {args.seed}")
    print(f"classical reference: RBF kernel, γ = {gamma:g} (tuned by KTA)")

    pop, strategy, result = sample_population(
        heuristic, size=args.size, n_descents=args.descents,
        min_distance=args.min_distance, seed=args.seed)
    harvested = {ind.state for ind in pop}
    drawn = [ind for ind in random_individuals(heuristic, args.n_random,
                                               seed=args.seed)
             if ind.state not in harvested]
    origins = {ind.state: "search" for ind in pop}
    origins.update({ind.state: "random" for ind in drawn})
    pop = pop + drawn

    print(f"search: {args.descents} stochastic descents, seed {args.seed} — "
          f"{len(strategy.pool)} full-depth circuits harvested out of "
          f"{space_size(N_QUBITS, MAX_DEPTH)} "
          f"(expanded={result.n_expanded}, evaluated={result.n_evaluated})")
    print(f"population: {len(pop)} individuals "
          f"({len(pop) - len(drawn)} harvested + {len(drawn)} drawn at random), "
          f"min_distance={args.min_distance} on the harvested half, "
          f"{strategy.n_relaxed} added by relaxing it\n")

    pairs = evaluate(pop, heuristic, train, test, gamma)
    print_table([row for _, row in pairs[:args.fig_rows]], dim_max=DIM_MAX)

    by_depth = np.bincount([len(row.state) for _, row in pairs],
                           minlength=MAX_DEPTH + 1)
    print(f"\n{len(pairs)} individuals — breakdown by gate count: "
          + ", ".join(f"{k}→{c}" for k, c in enumerate(by_depth)))
    print("genome: canonical state of length exactly "
          f"{MAX_DEPTH} over the {len(gs.catalog)}-op catalog, `id` as padding; "
          "the CSV carries it both as gate names and as catalog indices, ready "
          "for the crossover/mutation of a genetic algorithm.")
    print("Var(K): variance of the strict upper triangle of the Gram matrix.")
    print(f"R²: KernelRidge (α = {RIDGE_ALPHA:g}) fitted on the train set, "
          f"evaluated on the {test.n_samples} test points.")
    print("g(Q,RBF): geometric difference of Huang et al. 2021, bounded by "
          f"√N = {np.sqrt(train.n_samples):.2f}.")
    print(f"Expr↓: expressivity of Sim et al. 2019, D_KL between the fidelity "
          f"distribution ({N_PAIRS_EXPR} pairs drawn from [−π/2, π/2]^{N_FEATURES}) "
          "and the Haar one. Smaller = more expressive; the bar is inverted so "
          "that \"long bar = good\" holds everywhere.")

    write_population(pairs, gs, seed=args.seed, n_descents=args.descents,
                     min_distance=args.min_distance, origins=origins)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_table([row for _, row in pairs[:args.fig_rows]], FIG_PATH, n_qubits=N_QUBITS,
              max_depth=MAX_DEPTH, dim_max=DIM_MAX, row_height=1.35,
              fontsize=7.5, separator=0.09,
              header="circuit  —  H⊗⁴ · diagonal layer · H⊗⁴")
    print(f"figure → {FIG_PATH}")

    if args.exhaustive:
        check_exhaustive(pairs, heuristic, gs)


if __name__ == "__main__":
    main()
