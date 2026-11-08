from __future__ import annotations

import io
import sys
import time

import numpy as np

from experiments import common

CSV = "e16_mse_heuristic.csv"
SPACE_CSV = "e16_mse_space.csv"
SEEDS = list(range(10))
N_QUBITS = 4
N_DATA_GATES = 6
CIRCUIT_SEED = 0
SCATTER_SEED = 0

OBJECTIVES = {
    "kta": ("none", 0.0, "KTA", common.BLUE),
    "kta-dla": ("dla", common.PRODUCTION_WEIGHTS["dla"], "KTA + DLA", "#86b6ef"),
    "kta-var": ("variance", common.PRODUCTION_WEIGHTS["variance"],
                "KTA + variance", "#104281"),
    "mse": ("mse", 0.0, "MSE LOO", common.RED),
}
ORACLE = ("oracle", "oracle", common.MUTED)


def _gate_set():
    from qksearch.gates import ENTANGLER_SPECS, GateSpec, ProductGateSet

    return ProductGateSet([GateSpec("rz", 1, uses_feature=True),
                           ENTANGLER_SPECS["rzz"]],
                          n_qubits=N_QUBITS, n_features=N_QUBITS,
                          with_identity=True)


def _heuristic(objective, gate_set, train):
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import REGULARIZERS, IqpKernelHeuristic
    from qksearch.metric import RidgeLooMse, TargetAlignment

    regularization, weight, _, _ = OBJECTIVES[objective]
    if regularization == "mse":
        metric, regularizer = RidgeLooMse(alphas=(common.RIDGE_ALPHA,)), None
    else:
        metric = TargetAlignment()
        regularizer = (REGULARIZERS[regularization](N_QUBITS, weight)
                       if weight > 0 else None)
    return IqpKernelHeuristic(
        gate_set=gate_set, backend=NumpyBackend(N_QUBITS), data=train,
        metric=metric, max_depth=N_DATA_GATES, n_qubits=N_QUBITS,
        regularizer=regularizer)


def _test_scores(gate_set, states, train, test, chunk_size: int = 256):
    from qksearch.backend import NumpyBackend
    from qksearch.dataset import RegressionDataset
    from qksearch.iqp import hadamard_layer
    from qksearch.kernel import CircuitEvaluator
    from qksearch.metric import ZeroMetric

    joint = RegressionDataset(np.vstack([train.X, test.X]),
                              np.concatenate([train.y, test.y]))
    h = hadamard_layer(N_QUBITS)
    evaluator = CircuitEvaluator(NumpyBackend(N_QUBITS), joint, ZeroMetric(),
                                 prefix=h, suffix=h)
    n = train.n_samples
    ridge = common.RIDGE_ALPHA * np.eye(n)
    residual_total = float(((test.y - test.y.mean()) ** 2).sum())
    mse = np.empty(len(states))
    for start in range(0, len(states), chunk_size):
        grams = np.stack([evaluator.gram(s)
                          for s in states[start:start + chunk_size]])
        coef = np.linalg.solve(grams[:, :n, :n] + ridge, train.y[:, None])
        pred = (grams[:, n:, :n] @ coef)[..., 0]
        mse[start:start + chunk_size] = ((pred - test.y) ** 2).mean(axis=1)
    return 1.0 - mse * test.n_samples / residual_total, mse


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def run() -> None:
    from qksearch.enumerate_circuits import enumerate_states, strip_identity
    from qksearch.iqp import describe

    t0 = time.time()
    gate_set = _gate_set()
    states = enumerate_states(gate_set, N_DATA_GATES)
    stripped = [strip_identity(s) for s in states]
    n_gates = np.array([len(s) for s in stripped])
    n_entanglers = np.array([sum(len(op[1]) == 2 for op in s) for s in stripped])

    rows: list[dict] = []
    space: list[dict] = []
    for seed in SEEDS:
        train, test = common.mof_split(seed)
        heuristics = {name: _heuristic(name, gate_set, train)
                      for name in OBJECTIVES}
        scores = {name: h.scores(states) for name, h in heuristics.items()}
        kta = heuristics["kta"].evaluator.kta_many(states)
        r2_test, mse_test = _test_scores(gate_set, states, train, test)
        loo_r2 = 1.0 + scores["mse"]

        picks = {name: int(np.argmax(values)) for name, values in scores.items()}
        picks[ORACLE[0]] = int(np.argmax(r2_test))
        for name, i in picks.items():
            values = scores.get(name, r2_test)
            rows.append({
                "seed": seed, "objective": name,
                "score": float(values[i]),
                "kta_train": float(kta[i]),
                "loo_r2_train": float(loo_r2[i]),
                "r2_test": float(r2_test[i]),
                "mse_test": float(mse_test[i]),
                "r2_percentile": float((r2_test <= r2_test[i]).mean() * 100),
                "spearman_r2": (_spearman(values, r2_test)
                                if name in OBJECTIVES else 1.0),
                "n_gates": int(n_gates[i]),
                "n_entanglers": int(n_entanglers[i]),
                "detail": describe(stripped[i]) or "empty",
            })
        for i in range(len(states)):
            space.append({"seed": seed, "index": i,
                          **{name: float(scores[name][i]) for name in OBJECTIVES},
                          "r2_test": float(r2_test[i])})
        print(f"seed {seed} ({time.time() - t0:.0f}s)")

    common.write_csv(CSV, rows)
    common.write_csv(SPACE_CSV, space)

    import pandas as pd

    df = pd.DataFrame(rows)
    summary: dict[str, dict] = {}
    print(f"\n{'':>10} {'R2':>8} {'pct':>6} {'rho':>7} {'gates':>6}")
    for name in (*OBJECTIVES, ORACLE[0]):
        sub = df[df["objective"] == name]
        summary[name] = {
            "r2_test": float(sub["r2_test"].median()),
            "r2_percentile": float(sub["r2_percentile"].median()),
            "spearman_r2": float(sub["spearman_r2"].median()),
            "n_gates": float(sub["n_gates"].median()),
        }
        print(f"{name:>10} {summary[name]['r2_test']:>+8.3f} "
              f"{summary[name]['r2_percentile']:>6.1f} "
              f"{summary[name]['spearman_r2']:>+7.3f} "
              f"{summary[name]['n_gates']:>6.1f}")

    common.write_meta("e16_mse_heuristic", time.time() - t0, seeds=SEEDS,
                      n_qubits=N_QUBITS, n_data_gates=N_DATA_GATES,
                      n_circuits=len(states), ridge_alpha=common.RIDGE_ALPHA,
                      weights=common.PRODUCTION_WEIGHTS, summary=summary)


def _label(name: str) -> str:
    return OBJECTIVES[name][2] if name in OBJECTIVES else ORACLE[1]


def _colour(name: str) -> str:
    return OBJECTIVES[name][3] if name in OBJECTIVES else ORACLE[2]


def _strip(ax, df, names, column, floor=None) -> None:
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        vals = df.loc[df["objective"] == name, column].to_numpy(dtype=float)
        xs = i + rng.uniform(-0.12, 0.12, len(vals))
        shown = vals if floor is None else np.maximum(vals, floor)
        ax.scatter(xs, shown, s=12, color=_colour(name), alpha=0.6, lw=0)
        med = float(np.median(vals))
        ax.plot([i - 0.22, i + 0.22], [med, med], color=common.INK, lw=1.6,
                zorder=3, solid_capstyle="butt")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_label(n).replace(" + ", "\n+ ") for n in names],
                       fontsize=7)
    ax.grid(True, axis="y", alpha=0.5)


def _scatter(ax, space, picks, column, xlabel, shift=0.0, floor=None) -> None:
    sub = space[space["seed"] == SCATTER_SEED]
    xs = (sub[column] + shift).clip(lower=floor)
    ax.scatter(xs, sub["r2_test"].clip(lower=-1.0), s=3,
               color=common.BASELINE, alpha=0.5, lw=0, rasterized=True)
    for name in OBJECTIVES:
        i = int(picks.loc[picks["objective"] == name, "index"].iloc[0])
        point = sub[sub["index"] == i].iloc[0]
        x = point[column] + shift
        ax.scatter(x if floor is None else max(x, floor),
                   max(point["r2_test"], -1.0), s=42,
                   color=_colour(name), edgecolor="white", lw=0.8, zorder=4,
                   label=_label(name))
    rho = _spearman(sub[column], sub["r2_test"])
    ax.text(0.03, 0.97, rf"$\rho$ = {rho:+.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=common.INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test $R^2$")
    ax.set_ylim(bottom=-1.0)
    ax.grid(True, alpha=0.5)


def _picked_indices(space):
    import pandas as pd

    sub = space[space["seed"] == SCATTER_SEED]
    return pd.DataFrame([{"objective": name,
                          "index": int(sub.loc[sub[name].idxmax(), "index"])}
                         for name in OBJECTIVES])


def plot() -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(CSV)
    space = common.read_csv(SPACE_CSV)
    names = [*OBJECTIVES, ORACLE[0]]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))

    ax = axes[0, 0]
    _strip(ax, df, names, "r2_test", floor=-1.0)
    ax.set_ylabel("test $R^2$")

    ax = axes[0, 1]
    _strip(ax, df, list(OBJECTIVES), "spearman_r2")
    ax.axhline(0.0, color=common.BASELINE, lw=0.8)
    ax.set_ylabel(r"Spearman $\rho$")

    picks = _picked_indices(space)
    _scatter(axes[1, 0], space, picks, "kta", "KTA (train)")
    _scatter(axes[1, 1], space, picks, "mse", "LOO $R^2$ (train)",
             shift=1.0, floor=-1.0)
    axes[1, 1].set_xlim(-1.0, 1.0)
    axes[1, 1].plot([-1.0, 1.0], [-1.0, 1.0], color=common.BASELINE, lw=0.8,
                    ls="--", zorder=1)
    axes[1, 1].legend(loc="lower right", fontsize=6.5)

    fig.tight_layout()
    common.save_fig(fig, "fig_e16_mse_heuristic", formats=("pdf",))
    plt.close(fig)

    plot_circuits(df)


def _circuit_image(detail: str) -> np.ndarray:
    import matplotlib.pyplot as plt
    from qiskit.circuit import Parameter, QuantumCircuit

    from experiments.e10_rzz import _angle, _parse_detail
    from qksearch.backend import QiskitBackend

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
        },
    }
    hadamards = tuple(("h", (q,), None) for q in range(N_QUBITS))
    layer = () if detail == "empty" else _parse_detail(str(detail))
    qc = QuantumCircuit(N_QUBITS)
    for name, qubits, f in hadamards + layer + hadamards:
        drawer._apply(qc, name, qubits, _angle(name, f, params))
    drawing = qc.draw(output="mpl", style=style, fold=-1)
    buffer = io.BytesIO()
    drawing.savefig(buffer, format="png", dpi=common.FIG_DPI,
                    bbox_inches="tight")
    plt.close(drawing)
    buffer.seek(0)
    return plt.imread(buffer)


def plot_circuits(df=None) -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    if df is None:
        df = common.read_csv(CSV)
    chosen = df[df["seed"] == CIRCUIT_SEED].set_index("objective")
    groups: dict[str, list[str]] = {}
    for name in (*OBJECTIVES, ORACLE[0]):
        if name in chosen.index:
            groups.setdefault(chosen.loc[name, "detail"], []).append(name)
    rows = [(names[0], ", ".join(_label(n).replace("\n", " ") for n in names),
             chosen.loc[names[0]]) for names in groups.values()]

    images = [_circuit_image(row["detail"]) for _, _, row in rows]
    width = max(image.shape[1] for image in images)
    images = [np.pad(image, ((0, 0), (0, width - image.shape[1]), (0, 0)),
                     constant_values=1.0) for image in images]
    heights = [image.shape[0] / width for image in images]
    fig, axes = plt.subplots(
        len(images), 1, figsize=(7.2, 7.2 * sum(heights) + 0.5 * len(images)),
        height_ratios=heights, constrained_layout=True)
    for ax, image, (name, label, row) in zip(np.atleast_1d(axes), images,
                                             rows):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"{label}   $R^2$ = {row['r2_test']:.2f}",
                     fontsize=8, loc="left", color=_colour(name))
    common.save_fig(fig, "fig_e16_mse_heuristic", formats=("png",))
    plt.close(fig)


FACE_TO_FACE = ("kta", "mse")


def _kernels(detail: str, train, test):
    from experiments.e10_rzz import _parse_detail
    from qksearch.backend import NumpyBackend
    from qksearch.iqp import iqp_circuit

    layer = () if detail == "empty" else _parse_detail(str(detail))
    circuit = iqp_circuit(layer, N_QUBITS)
    backend = NumpyBackend(N_QUBITS)
    return (backend.gram(circuit, train.X),
            backend.cross_gram(circuit, test.X, train.X))


def _predict(K_train, K_test, y_train) -> np.ndarray:
    ridge = common.RIDGE_ALPHA * np.eye(len(y_train))
    return K_test @ np.linalg.solve(K_train + ridge, y_train)


def plot_face_to_face(df=None, seed: int = CIRCUIT_SEED):
    import matplotlib.pyplot as plt

    from qksearch.compare_kernels import CMAP_SEQ

    common.report_style()
    if df is None:
        df = common.read_csv(CSV)
    train, test = common.mof_split(seed)
    order = np.argsort(train.y)
    oracle = df[df["objective"] == ORACLE[0]].set_index("seed")["r2_test"]

    fig = plt.figure(figsize=(9.0, 10.5))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.8, 1.25, 1.25, 1.0],
                            hspace=0.3, wspace=0.28)
    lo = min(train.y.min(), test.y.min())
    hi = max(train.y.max(), test.y.max())
    pad = 0.05 * (hi - lo)
    image = None

    for col, name in enumerate(FACE_TO_FACE):
        colour = _colour(name)
        rows = df[df["objective"] == name].set_index("seed").sort_index()
        row = rows.loc[seed]

        ax = fig.add_subplot(grid[0, col])
        ax.imshow(_circuit_image(row["detail"]))
        ax.set_axis_off()
        ax.set_title(_label(name), color=colour, fontsize=11,
                     fontweight="bold")

        K_train, K_test = _kernels(row["detail"], train, test)
        ax = fig.add_subplot(grid[1, col])
        image = ax.imshow(K_train[np.ix_(order, order)], cmap=CMAP_SEQ,
                          vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        pred = _predict(K_train, K_test, train.y)
        ax = fig.add_subplot(grid[2, col])
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=common.BASELINE, lw=0.8, ls="--")
        ax.scatter(test.y, np.clip(pred, lo - pad, hi + pad), s=10,
                   color=colour, alpha=0.6, lw=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.text(0.04, 0.96, f"$R^2$ = {row['r2_test']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        ax.grid(True, alpha=0.5)

        ax = fig.add_subplot(grid[3, col])
        seeds = rows.index.to_numpy()
        values = rows["r2_test"].to_numpy(dtype=float)
        ax.bar(seeds, np.maximum(values, -1.0), color=colour, width=0.7)
        ax.scatter(seeds, oracle.loc[seeds], marker="_", s=120,
                   color=common.INK2, lw=1.5, zorder=3,
                   label="oracle")
        ax.axhline(float(np.median(values)), color=common.INK, lw=1.0,
                   ls=":", label="median")
        ax.axhline(0.0, color=common.BASELINE, lw=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks(seeds)
        ax.set_xlabel("seed")
        ax.set_ylabel("test $R^2$")
        ax.legend(loc="lower left", fontsize=6.5)
        ax.grid(True, axis="y", alpha=0.5)

    fig.colorbar(image, ax=fig.axes[1::4] + fig.axes[5::4],
                 fraction=0.025, pad=0.02)
    common.save_fig(fig, "fig_e16_side_by_side")
    return fig


def _seed_argument() -> int:
    if "--seed" in sys.argv:
        return int(sys.argv[sys.argv.index("--seed") + 1])
    return CIRCUIT_SEED


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
    common.show(plot_face_to_face(seed=_seed_argument()))
