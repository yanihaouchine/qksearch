from __future__ import annotations

import io
import sys
import time

import numpy as np

from experiments import common

SEEDS = list(range(10))
N_QUBITS = 4
MAX_DEPTH = 20
BUDGET = 400
CHUNK_SIZE = 256
FILTER_WEIGHT = 100.0
CIRCUIT_METHOD = "l2pf"
CIRCUIT_SEED = 6
ALPHAS = tuple(np.logspace(-4, 1, 16))
RFF_FEATURES = 512

OBJECTIVES = {
    "kta": ("KTA", common.BLUE),
    "mse": ("MSE LOO", common.RED),
}
METHODS = {
    "l1": ("1 layer", 1, False, False),
    "l1p": ("1 layer\n+ phases", 1, True, False),
    "l2": ("2 layers", 2, False, False),
    "l2p": ("2 layers\n+ phases", 2, True, False),
    "l2pf": ("2 layers\n+ phases\n+ filter", 2, True, True),
}
CATEGORIES = {
    "product": ("product", common.BASELINE),
    "iqp": ("IQP", "#86b6ef"),
    "matchgate": ("matchgate", "#104281"),
    "rff": ("RFF", "#5a9e6f"),
    "none": ("no certificate", "#e8a33d"),
}

CSV = "e18_simulability.csv"
FIGURE = "fig_e18_simulability"
CIRCUIT_FIGURE = "fig_e18_uncertified_circuit"


def category(rules: frozenset[str]) -> str:
    for rule in ("matchgate", "iqp", "product"):
        if rule in rules:
            return rule
    return "none"


def _gate_set(method: str):
    from qksearch.gates import ENTANGLER_SPECS, GateSpec, LayeredGateSet, ProductGateSet

    _, layers, phases, _ = METHODS[method]
    specs = [GateSpec("rz", 1, uses_feature=True), ENTANGLER_SPECS["rzz"]]
    if phases:
        specs += [GateSpec("t", 1, uses_feature=False),
                  GateSpec("cs", 2, uses_feature=False, symmetric=True)]
    return LayeredGateSet(ProductGateSet(specs, N_QUBITS, N_QUBITS), layers)


def _metric(objective: str):
    from qksearch.metric import RidgeLooMse, TargetAlignment

    if objective == "kta":
        return TargetAlignment()
    return RidgeLooMse(alphas=ALPHAS)


def build_heuristic(objective: str, method: str, train):
    from experiments.e17_sum_heuristic import _heuristic_class
    from qksearch.backend import NumpyBackend
    from qksearch.simulability import SimulabilityPenalty

    regularizer = SimulabilityPenalty(N_QUBITS, FILTER_WEIGHT) if METHODS[method][3] else None
    return _heuristic_class()(_gate_set(method), NumpyBackend(N_QUBITS), train,
                              _metric(objective), MAX_DEPTH, N_QUBITS,
                              regularizer=regularizer)


def _evaluator(data, cache: int = 4096):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import hadamard_layer
    from qksearch.kernel import CircuitEvaluator
    from qksearch.metric import ZeroMetric

    h = hadamard_layer(N_QUBITS)
    return CircuitEvaluator(NumpyBackend(N_QUBITS), data, ZeroMetric(), prefix=h, suffix=h,
                            psi_cache_size=cache)


def magic(states, train) -> np.ndarray:
    from qksearch.simulability import stabilizer_renyi_entropy

    evaluator = _evaluator(train)
    return np.array([stabilizer_renyi_entropy(evaluator.final_states(s)).mean()
                     for s in states])


def _ridge(grams: np.ndarray, train, test) -> tuple[np.ndarray, ...]:
    from qksearch.metric import RidgeLooMse

    n = train.n_samples
    K = grams[:, :n]
    K[:, np.arange(n), np.arange(n)] = 1.0
    errors = RidgeLooMse(alphas=ALPHAS).loo_mse(K, train.y)
    best = errors.argmin(axis=1)
    alpha = np.asarray(ALPHAS)[best]
    coef = np.linalg.solve(K + alpha[:, None, None] * np.eye(n), train.y[:, None])
    pred = (grams[:, n:] @ coef)[..., 0]
    total = float(((test.y - test.y.mean()) ** 2).mean())
    r2 = 1.0 - ((pred - test.y) ** 2).mean(axis=1) / total
    return -errors.min(axis=1) / train.y.var(), alpha, r2


def measure(states, train, test) -> dict[str, np.ndarray]:
    from qksearch.dataset import RegressionDataset
    from qksearch.iqp import hadamard_layer
    from qksearch.metric import TargetAlignment
    from qksearch.simulability import certify, rff_gram

    joint = RegressionDataset(np.vstack([train.X, test.X]),
                              np.concatenate([train.y, test.y]))
    h = hadamard_layer(N_QUBITS)
    evaluator = _evaluator(joint, cache=512)
    alignment = TargetAlignment()
    n = train.n_samples

    out = {name: np.empty(len(states)) for name in ("kta", "mse", "alpha", "r2_test")}
    for start in range(0, len(states), CHUNK_SIZE):
        block = slice(start, start + CHUNK_SIZE)
        psis = np.stack([evaluator.final_states(s) for s in states[block]])
        overlaps = psis @ psis[:, :n].conj().swapaxes(1, 2)
        grams = overlaps.real ** 2 + overlaps.imag ** 2
        out["kta"][block] = alignment.batch(grams[:, :n], train.y)
        out["mse"][block], out["alpha"][block], out["r2_test"][block] = \
            _ridge(grams, train, test)
    out["category"] = np.array([category(certify(h + s + h, N_QUBITS)) for s in states])
    out["r2_rff"] = np.full(len(states), np.nan)
    pending = np.flatnonzero(out["category"] == "none")
    for start in range(0, len(pending), CHUNK_SIZE):
        block = pending[start:start + CHUNK_SIZE]
        rff = np.stack([rff_gram(states[i], joint.X, train.X, RFF_FEATURES) for i in block])
        out["r2_rff"][block] = _ridge(rff, train, test)[2]
    dequantized = out["r2_rff"] >= out["r2_test"]
    out["category"][dequantized] = "rff"
    return out


def _arms(train, budget: int) -> dict:
    from qksearch.search import BestFirstSearch, search

    arms = {}
    for objective in OBJECTIVES:
        for method in METHODS:
            heuristic = build_heuristic(objective, method, train)
            result = search((), heuristic, BestFirstSearch(), budget=budget)
            arms[(objective, method)] = (result.best_state,
                                         list(dict.fromkeys(heuristic.visited)),
                                         result.n_evaluated)
    return arms


def run(budget: int = BUDGET) -> None:
    from qksearch.gates import layer_count
    from qksearch.iqp import describe

    t0 = time.time()
    rows: list[dict] = []
    for seed in SEEDS:
        train, test = common.mof_split(seed)
        arms = _arms(train, budget)
        union = list(dict.fromkeys(s for _, visited, _ in arms.values() for s in visited))
        values = measure(union, train, test)
        index = {s: i for i, s in enumerate(union)}
        magics = magic([best for best, _, _ in arms.values()], train)
        for ((objective, method), (best, visited, n_eval)), m2 in zip(arms.items(), magics):
            i = index[best]
            seen = values["category"][[index[s] for s in visited]]
            row = {
                "seed": seed, "objective": objective, "method": method,
                "n_evaluated": n_eval, "n_distinct": len(visited),
                "certificate": values["category"][i],
                "kta": float(values["kta"][i]),
                "loo_r2": float(1.0 + values["mse"][i]),
                "alpha": float(values["alpha"][i]),
                "r2_test": float(values["r2_test"][i]),
                "r2_rff": float(values["r2_rff"][i]),
                "magic": float(m2),
                "n_gates": sum(op[0] != "hl" for op in best),
                "n_layers": layer_count(best),
                "detail": describe(best) or "empty",
            }
            for name in CATEGORIES:
                row[f"visited_{name}"] = float(np.mean(seen == name))
            rows.append(row)
        print(f"seed {seed}: {len(union)} circuits ({time.time() - t0:.0f}s)")

    common.write_csv(CSV, rows)

    import pandas as pd

    df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    print(f"\n{'':>10} {'cert%':>6} {'seen%':>6} {'R2':>7} {'RFF':>7} {'magic':>6} "
          f"{'gates':>6} {'layers':>6}")
    for (objective, method), sub in df.groupby(["objective", "method"], sort=False):
        key = f"{objective}/{method}"
        summary[key] = {
            "certified": float((sub["certificate"] != "none").mean()),
            "visited_certified": float(1.0 - sub["visited_none"].mean()),
            "r2_test": float(sub["r2_test"].median()),
            "r2_rff": float(sub["r2_rff"].median()),
            "magic": float(sub["magic"].median()),
            "n_gates": float(sub["n_gates"].median()),
            "n_layers": float(sub["n_layers"].median()),
        }
        s = summary[key]
        print(f"{key:>10} {100 * s['certified']:>6.0f} {100 * s['visited_certified']:>6.0f} "
              f"{s['r2_test']:>+7.3f} {s['r2_rff']:>+7.3f} {s['magic']:>6.2f} "
              f"{s['n_gates']:>6.1f} "
              f"{s['n_layers']:>6.1f}")

    common.write_meta("e18_simulability", time.time() - t0, seeds=SEEDS,
                      n_qubits=N_QUBITS, max_depth=MAX_DEPTH, budget=budget,
                      filter_weight=FILTER_WEIGHT, alphas=list(ALPHAS),
                      rff_features=RFF_FEATURES,
                      summary=summary)


def _stacked(ax, shares: dict[str, list[float]]) -> None:
    bottom = np.zeros(len(METHODS))
    for name, (label, colour) in CATEGORIES.items():
        values = np.array(shares[name])
        ax.bar(range(len(METHODS)), values, bottom=bottom, color=colour, width=0.65,
               label=label)
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([])


def _strip(ax, df, column: str, colour: str) -> None:
    rng = np.random.default_rng(0)
    for i, method in enumerate(METHODS):
        values = df[df["method"] == method][column].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), np.maximum(values, -1.0),
                   s=16, color=colour, alpha=0.75, lw=0)
        med = float(np.median(values))
        ax.plot([i - 0.24, i + 0.24], [max(med, -1.0)] * 2, color=common.INK, lw=1.6,
                solid_capstyle="butt", zorder=3)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([])
    ax.grid(True, axis="y", alpha=0.5)


def plot():
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)

    fig, axes = plt.subplots(5, 2, figsize=(9.0, 12.0), sharey="row",
                             gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    df["rff_gap"] = df["r2_rff"] - df["r2_test"]
    for col, (objective, (label, colour)) in enumerate(OBJECTIVES.items()):
        sub = df[df["objective"] == objective]
        selected = {name: [float((sub[sub["method"] == m]["certificate"] == name).mean())
                           for m in METHODS] for name in CATEGORIES}
        visited = {name: [float(sub[sub["method"] == m][f"visited_{name}"].mean())
                          for m in METHODS] for name in CATEGORIES}
        _stacked(axes[0, col], selected)
        _stacked(axes[1, col], visited)
        _strip(axes[2, col], sub, "r2_test", colour)
        axes[2, col].axhline(0.0, color=common.BASELINE, lw=0.8)
        _strip(axes[3, col], sub, "rff_gap", colour)
        axes[3, col].axhline(0.0, color=common.BASELINE, lw=0.8)
        _strip(axes[4, col], sub, "magic", colour)
        axes[4, col].set_xticklabels([m[0] for m in METHODS.values()], fontsize=7.5)
        axes[0, col].set_title(label, color=colour, fontsize=11, fontweight="bold")

    rows = ("selected", "visited", "test $R^2$", "RFF $-$ quantum", "magic $M_2$")
    for row, name in enumerate(rows):
        axes[row, 0].set_ylabel(name)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                      frameon=False)

    common.save_fig(fig, FIGURE)
    return fig


def _circuit_image(detail: str) -> np.ndarray:
    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from experiments.e10_rzz import _angle, _parse_detail
    from qksearch.backend import QiskitBackend
    from qksearch.gates import HADAMARD_LAYER_NAME

    drawer = QiskitBackend(n_qubits=N_QUBITS)
    params = [Parameter(f"x{f}") for f in range(N_QUBITS)]
    style = {
        "backgroundcolor": "#ffffff",
        "linecolor": common.INK2,
        "textcolor": common.INK,
        "gatetextcolor": "#ffffff",
        "displaycolor": {
            "h": [common.MUTED, "#ffffff"],
            "rz": [common.BLUE, "#ffffff"],
            "rzz": [common.RED, "#ffffff"],
            "p": ["#5a9e6f", "#ffffff"],
            "cp": ["#5a9e6f", "#ffffff"],
        },
    }
    qubits = range(N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    qc.h(qubits)
    for name, on, f in () if detail == "empty" else _parse_detail(str(detail)):
        if name == HADAMARD_LAYER_NAME:
            qc.barrier()
            qc.h(qubits)
            qc.barrier()
        else:
            drawer._apply(qc, name, on, _angle(name, f, params))
    qc.h(qubits)
    drawing = qc.draw(output="mpl", style=style, fold=-1, plot_barriers=True)
    buffer = io.BytesIO()
    drawing.savefig(buffer, format="png", dpi=common.FIG_DPI, bbox_inches="tight")
    plt.close(drawing)
    buffer.seek(0)
    return plt.imread(buffer)


def plot_circuit(seed: int):
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    rows = df[(df["method"] == CIRCUIT_METHOD) & (df["seed"] == seed)].set_index("objective")
    fig, axes = plt.subplots(len(OBJECTIVES), 1, figsize=(11.0, 5.2),
                             gridspec_kw={"hspace": 0.35})
    for ax, (objective, (label, colour)) in zip(axes, OBJECTIVES.items()):
        row = rows.loc[objective]
        print(f"{label} [{row['certificate']}]: {row['detail']}")
        ax.imshow(_circuit_image(row["detail"]))
        ax.set_axis_off()
        ax.set_title(f"{label}   $R^2$ = {row['r2_test']:.2f}", color=colour,
                     fontsize=10, fontweight="bold", loc="left")
    common.save_fig(fig, CIRCUIT_FIGURE)
    return fig


def _argument(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run(_argument("--budget", BUDGET))
    figures = (plot(), plot_circuit(_argument("--seed", CIRCUIT_SEED)))
    for figure in figures:
        common.show(figure)
