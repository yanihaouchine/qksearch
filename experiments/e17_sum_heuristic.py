from __future__ import annotations

import io
import math
import sys
import time

import numpy as np

from experiments import common

SEEDS = list(range(10))
N_QUBITS = 4
MAX_DEPTH = 40
ENTANGLERS = ("rzzh", "rzzhn")
EVAL_BUDGET = 8000
CIRCUIT_SEED = 0
SCATTER_SEED = 0
CHUNK_SIZE = 512
N_WORKERS = 2

WEIGHTS = {"kta": 1.0, "mse": 1.0, "var": 1.0, "dla": 1.0}
TERMS = {
    "kta": ("KTA", common.BLUE),
    "mse": ("MSE", common.RED),
    "var": ("Var", "#e8a33d"),
    "dla": ("DLA", "#5a9e6f"),
}
OBJECTIVES = {
    "kta": ("KTA", common.BLUE),
    "sum": ("KTA + MSE + DLA + Var", common.RED),
}
METHODS = {
    "tree": "best-first",
    "random": "uniform",
}

CSV = "e17_sum_heuristic.csv"
SPACE = "e17_sum_space.npz"
FIGURE = "fig_e17_sum_heuristic"


def _gate_set():
    from qksearch.gates import iqp_gate_set

    return iqp_gate_set(N_QUBITS, N_QUBITS, extra_entanglers=ENTANGLERS)


def space_size(m: int, depth: int) -> int:
    return math.comb(m + depth, depth)


def _heuristic_class():
    from qksearch.iqp import IqpKernelHeuristic

    class Tracked(IqpKernelHeuristic):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.visited: list = []

        def score(self, state):
            self.visited.append(state)
            return super().score(state)

        def scores(self, states):
            self.visited.extend(states)
            return super().scores(states)

    return Tracked


def _strategy():
    from qksearch.search import BestFirstSearch

    class EvalBudgetBestFirst(BestFirstSearch):
        def _budget_left(self, budget):
            return budget is None or self._n_evaluated < budget

    return EvalBudgetBestFirst()


def build_heuristic(objective: str, train, dla):
    from qksearch.backend import NumpyBackend
    from qksearch.kernel import DlaPenalty, GramVarianceBonus
    from qksearch.metric import GramTerm, RidgeLooMse, TargetAlignment, WeightedSum

    if objective == "kta":
        metric, regularizer = TargetAlignment(), None
    else:
        metric = WeightedSum({
            "kta": (TargetAlignment(), WEIGHTS["kta"]),
            "mse": (RidgeLooMse(alphas=(common.RIDGE_ALPHA,)), WEIGHTS["mse"]),
            "var": (GramTerm(GramVarianceBonus(1.0)), WEIGHTS["var"]),
        })
        regularizer = DlaPenalty(dla, N_QUBITS, WEIGHTS["dla"])
    return _heuristic_class()(_gate_set(), NumpyBackend(N_QUBITS), train, metric,
                              MAX_DEPTH, N_QUBITS, regularizer=regularizer)


def sample_states(rng: np.random.Generator, catalog, n: int, depth: int):
    m = len(catalog)
    states = []
    for _ in range(n):
        bars = np.sort(rng.choice(m + depth, size=depth, replace=False))
        gates = bars - np.arange(depth)
        states.append(tuple(catalog[g] for g in gates if g < m))
    return states


def measure(states, train, test, dla) -> dict[str, np.ndarray]:
    from qksearch.backend import NumpyBackend
    from qksearch.dataset import RegressionDataset
    from qksearch.iqp import hadamard_layer
    from qksearch.kernel import CircuitEvaluator, DlaPenalty, GramVarianceBonus
    from qksearch.metric import RidgeLooMse, TargetAlignment, ZeroMetric

    joint = RegressionDataset(np.vstack([train.X, test.X]),
                              np.concatenate([train.y, test.y]))
    h = hadamard_layer(N_QUBITS)
    evaluator = CircuitEvaluator(NumpyBackend(N_QUBITS), joint, ZeroMetric(),
                                 prefix=h, suffix=h)
    alignment = TargetAlignment()
    variance = GramVarianceBonus(1.0)
    loo = RidgeLooMse(alphas=(common.RIDGE_ALPHA,))
    penalty = DlaPenalty(dla, N_QUBITS, 1.0)
    n = train.n_samples
    ridge = common.RIDGE_ALPHA * np.eye(n)
    total = float(((test.y - test.y.mean()) ** 2).mean())

    out = {name: np.empty(len(states)) for name in (*TERMS, "r2_test")}
    for start in range(0, len(states), CHUNK_SIZE):
        block = slice(start, start + CHUNK_SIZE)
        psis = np.stack([evaluator.final_states(s) for s in states[block]])
        overlaps = psis @ psis[:, :n].conj().swapaxes(1, 2)
        grams = overlaps.real ** 2 + overlaps.imag ** 2
        K = grams[:, :n]
        idx = np.arange(n)
        K[:, idx, idx] = 1.0
        out["kta"][block] = alignment.batch(K, train.y)
        out["mse"][block] = loo.batch(K, train.y)
        out["var"][block] = variance.batch(K)
        coef = np.linalg.solve(K + ridge, train.y[:, None])
        pred = (grams[:, n:] @ coef)[..., 0]
        out["r2_test"][block] = 1.0 - ((pred - test.y) ** 2).mean(axis=1) / total
    out["dla"] = np.array([penalty(h + s) for s in states])
    out["sum"] = sum(WEIGHTS[t] * out[t] for t in TERMS)
    return out


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _arm_key(seed: int, objective: str, method: str, column: str) -> str:
    return f"s{seed}_{objective}_{method}_{column}"


def run_seed(seed: int, budget: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    from threadpoolctl import threadpool_limits

    from qksearch.dla import DlaCalculator
    from qksearch.iqp import describe
    from qksearch.search import search

    threadpool_limits(1)
    catalog = _gate_set().catalog
    dla = DlaCalculator()
    train, test = common.mof_split(seed)
    arms = {}
    for objective in OBJECTIVES:
        heuristic = build_heuristic(objective, train, dla)
        result = search((), heuristic, _strategy(), budget=budget)
        arms[(objective, "tree")] = (result.best_state, result.best_score,
                                     list(dict.fromkeys(heuristic.visited)),
                                     result.n_evaluated)
        rng = np.random.default_rng(seed)
        states = sample_states(rng, catalog, result.n_evaluated, MAX_DEPTH)
        scores = heuristic.scores(states)
        arms[(objective, "random")] = (states[int(np.argmax(scores))],
                                       float(scores.max()),
                                       list(dict.fromkeys(states)), len(states))

    union = list(dict.fromkeys(s for _, _, visited, _ in arms.values() for s in visited))
    values = measure(union, train, test, dla)
    index = {s: i for i, s in enumerate(union)}

    rows: list[dict] = []
    space: dict[str, np.ndarray] = {}
    for (objective, method), (best, best_score, visited, n_eval) in arms.items():
        sel = np.array([index[s] for s in visited])
        arm = {name: v[sel] for name, v in values.items()}
        i = index[best]
        if not np.isclose(values[objective][i], best_score, atol=1e-6):
            raise RuntimeError(f"score mismatch {objective}/{method}: "
                               f"{values[objective][i]} vs {best_score}")
        row = {
            "seed": seed, "objective": objective, "method": method,
            "n_evaluated": n_eval, "n_distinct": len(visited),
            "score": float(values[objective][i]),
            "kta": float(values["kta"][i]),
            "loo_r2": float(1.0 + values["mse"][i]),
            "var": float(values["var"][i]),
            "dla_dim": int(round(-values["dla"][i] * (4 ** N_QUBITS - 1))),
            "r2_test": float(values["r2_test"][i]),
            "r2_best_seen": float(arm["r2_test"].max()),
            "spearman_r2": _spearman(arm[objective], arm["r2_test"]),
            "n_gates": len(best),
            "detail": describe(best) or "empty",
        }
        for term in TERMS:
            row[f"rho_{term}"] = _spearman(arm[term], arm["r2_test"])
        rows.append(row)
        for name in (*TERMS, "r2_test"):
            space[_arm_key(seed, objective, method, name)] = arm[name].astype(np.float32)
    return rows, space


def run(budget: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from experiments.e17_sum_heuristic import run_seed as worker

    catalog = _gate_set().catalog
    size = space_size(len(catalog), MAX_DEPTH)
    print(f"{len(catalog)} gates, depth <= {MAX_DEPTH}, {size:.2e} circuits", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    space: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(worker, seed, budget) for seed in SEEDS]
        for seed, future in zip(SEEDS, futures):
            seed_rows, seed_space = future.result()
            rows += seed_rows
            space.update(seed_space)
            print(f"seed {seed} ({time.time() - t0:.0f}s)", flush=True)

    common.write_csv(CSV, rows)
    np.savez_compressed(f"{common.RESULTS_DIR}/{SPACE}", **space)

    import pandas as pd

    df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    print(f"\n{'':>18} {'R2':>7} {'rho':>7} {'gates':>6} {'DLA':>5} {'evals':>6}")
    for (objective, method), sub in df.groupby(["objective", "method"], sort=False):
        key = f"{objective}/{method}"
        summary[key] = {col: float(sub[col].median())
                        for col in ("r2_test", "spearman_r2", "n_gates", "dla_dim",
                                    "n_evaluated", "r2_best_seen")}
        summary[key]["n_positive"] = int((sub["r2_test"] > 0).sum())
        s = summary[key]
        print(f"{key:>18} {s['r2_test']:>+7.3f} {s['spearman_r2']:>+7.3f} "
              f"{s['n_gates']:>6.1f} {s['dla_dim']:>5.0f} {s['n_evaluated']:>6.0f}")

    common.write_meta("e17_sum_heuristic", time.time() - t0, seeds=SEEDS,
                      n_qubits=N_QUBITS, max_depth=MAX_DEPTH, budget=budget,
                      n_gates_catalog=len(catalog), space_size=f"{size:.3e}",
                      weights=WEIGHTS, ridge_alpha=common.RIDGE_ALPHA,
                      summary=summary)


PREFIX = {"rz": "x", "rzz": "x", "rzzh": "h", "rzzhn": "hn"}


def _circuit_image(detail: str) -> np.ndarray:
    from collections import Counter

    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from experiments.e10_rzz import _parse_detail
    from qksearch.backend import QiskitBackend

    drawer = QiskitBackend(n_qubits=N_QUBITS)
    style = {
        "backgroundcolor": "#ffffff",
        "linecolor": common.INK2,
        "textcolor": common.INK,
        "gatetextcolor": "#ffffff",
        "displaycolor": {
            "h": [common.MUTED, "#ffffff"],
            "rz": [common.BLUE, "#ffffff"],
            "rzz": [common.RED, "#ffffff"],
        },
    }
    hadamards = tuple(("h", (q,), None) for q in range(N_QUBITS))
    layer = () if detail == "empty" else _parse_detail(str(detail))
    counts = Counter(layer)
    qc = QuantumCircuit(N_QUBITS)
    for name, qubits, f in hadamards:
        drawer._apply(qc, name, qubits, None)
    for (name, qubits, f), k in counts.items():
        tag = PREFIX[name] + "".join(str(i) for i in f)
        drawer._apply(qc, name, qubits, Parameter(tag if k == 1 else f"{k}·{tag}"))
    qc.barrier()
    for name, qubits, f in hadamards:
        drawer._apply(qc, name, qubits, None)
    drawing = qc.draw(output="mpl", style=style, fold=-1, plot_barriers=False)
    buffer = io.BytesIO()
    drawing.savefig(buffer, format="png", dpi=common.FIG_DPI, bbox_inches="tight")
    plt.close(drawing)
    buffer.seek(0)
    return plt.imread(buffer)


def _load_space() -> dict[str, np.ndarray]:
    with np.load(f"{common.RESULTS_DIR}/{SPACE}") as data:
        return {key: data[key].astype(float) for key in data.files}


def _objective_values(space, objective: str, method: str) -> np.ndarray:
    if objective == "kta":
        return space[_arm_key(SCATTER_SEED, objective, method, "kta")]
    return sum(WEIGHTS[t] * space[_arm_key(SCATTER_SEED, objective, method, t)]
               for t in TERMS)


def plot():
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    space = _load_space()

    fig = plt.figure(figsize=(9.6, 11.0))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.9, 1.0, 1.1, 0.9],
                            hspace=0.38, wspace=0.25)
    axes = {}
    for col, objective in enumerate(OBJECTIVES):
        label, colour = OBJECTIVES[objective]
        sub = df[df["objective"] == objective]
        tree = sub[sub["method"] == "tree"].set_index("seed")

        ax = fig.add_subplot(grid[0, col])
        row = tree.loc[CIRCUIT_SEED]
        ax.imshow(_circuit_image(row["detail"]))
        ax.set_axis_off()
        ax.set_title(label, color=colour, fontsize=11, fontweight="bold")
        ax.text(0.0, -0.06, f"$R^2$ = {row['r2_test']:.2f}   {int(row['n_gates'])} gates"
                f"   dim DLA = {int(row['dla_dim'])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color=common.INK2)

        ax = axes.setdefault((1, col), fig.add_subplot(grid[1, col]))
        rng = np.random.default_rng(0)
        series = [(METHODS["tree"], sub[sub["method"] == "tree"]["r2_test"], colour),
                  (METHODS["random"], sub[sub["method"] == "random"]["r2_test"],
                   common.BASELINE),
                  ("best seen", tree["r2_best_seen"], common.MUTED)]
        for i, (name, values, c) in enumerate(series):
            values = values.to_numpy(dtype=float)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)),
                       np.maximum(values, -1.0), s=16, color=c, alpha=0.75, lw=0)
            med = float(np.median(values))
            ax.plot([i - 0.24, i + 0.24], [med, med], color=common.INK, lw=1.6,
                    solid_capstyle="butt", zorder=3)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([name for name, _, _ in series])
        ax.set_ylabel("test $R^2$")
        ax.axhline(0.0, color=common.BASELINE, lw=0.8)
        ax.grid(True, axis="y", alpha=0.5)

        ax = axes.setdefault((2, col), fig.add_subplot(grid[2, col]))
        for method, c, size in (("random", common.BASELINE, 3), ("tree", colour, 4)):
            x = _objective_values(space, objective, method)
            y = space[_arm_key(SCATTER_SEED, objective, method, "r2_test")]
            ax.scatter(x, np.maximum(y, -1.0), s=size, color=c, alpha=0.45, lw=0,
                       rasterized=True, label=METHODS[method])
        rho = _spearman(_objective_values(space, objective, "tree"),
                        space[_arm_key(SCATTER_SEED, objective, "tree", "r2_test")])
        ax.scatter(float(row["score"]), max(float(row["r2_test"]), -1.0), s=60,
                   color=colour, edgecolor=common.INK, lw=0.9, zorder=4)
        ax.text(0.03, 0.97, rf"$\rho$ = {rho:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=common.INK2)
        ax.set_xlabel(f"{label} (train)")
        ax.set_ylabel("test $R^2$")
        ax.legend(loc="lower right", fontsize=7, markerscale=3)
        ax.grid(True, alpha=0.5)

        ax = axes.setdefault((3, col), fig.add_subplot(grid[3, col]))
        names = [*TERMS, None]
        for i, term in enumerate(names):
            column = "spearman_r2" if term is None else f"rho_{term}"
            values = tree[column].to_numpy(dtype=float)
            c = colour if term is None else TERMS[term][1]
            ax.bar(i, float(np.median(values)), color=c, width=0.65, alpha=0.85)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=8,
                       color=common.INK2, lw=0, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([TERMS[t][0] if t else "objective" for t in names])
        ax.axhline(0.0, color=common.BASELINE, lw=0.8)
        ax.set_ylabel(r"Spearman $\rho$ with test $R^2$")
        ax.grid(True, axis="y", alpha=0.5)

    for r in (1, 2, 3):
        lo = min(axes[(r, c)].get_ylim()[0] for c in range(2))
        hi = max(axes[(r, c)].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[(r, c)].set_ylim(lo, hi)

    common.save_fig(fig, FIGURE)
    return fig


def _argument(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run(_argument("--budget", EVAL_BUDGET))
    common.show(plot())
