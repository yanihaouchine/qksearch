from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

from experiments import common

SEEDS = list(range(10))
N_QUBITS = 4
DEFAULT_DEPTH = 10
CIRCUIT_SEED = 0
SCATTER_SEED = 0
CHUNK_SIZE = 512
SCATTER_POINTS = 20000

OBJECTIVES = {
    "kta": ("KTA", common.BLUE),
    "kta-dla": ("KTA + DLA", "#86b6ef"),
    "kta-var": ("KTA + variance", "#104281"),
    "mse": ("MSE LOO", common.RED),
}
ORACLE = ("oracle", "oracle", common.MUTED)


def _csv(depth: int) -> str:
    return f"e16_mse_heuristic_d{depth}.csv"


def _space_path(depth: int) -> str:
    return os.path.join(common.RESULTS_DIR, f"e16_mse_space_d{depth}.npz")


def _figure(name: str, depth: int) -> str:
    return f"{name}_d{depth}"


def _gate_set():
    from qksearch.gates import ENTANGLER_SPECS, GateSpec, ProductGateSet

    return ProductGateSet([GateSpec("rz", 1, uses_feature=True),
                           ENTANGLER_SPECS["rzz"]],
                          n_qubits=N_QUBITS, n_features=N_QUBITS,
                          with_identity=True)


def _dla_penalties(states) -> np.ndarray:
    from qksearch.dla import DlaCalculator
    from qksearch.iqp import iqp_circuit
    from qksearch.kernel import DlaPenalty

    penalty = DlaPenalty(DlaCalculator(), N_QUBITS,
                         common.PRODUCTION_WEIGHTS["dla"])
    return np.array([penalty(iqp_circuit(s, N_QUBITS)) for s in states])


def _evaluate(states, train, test, dla) -> dict[str, np.ndarray]:
    from qksearch.backend import NumpyBackend
    from qksearch.dataset import RegressionDataset
    from qksearch.iqp import hadamard_layer
    from qksearch.kernel import CircuitEvaluator, GramVarianceBonus
    from qksearch.metric import RidgeLooMse, TargetAlignment, ZeroMetric

    joint = RegressionDataset(np.vstack([train.X, test.X]),
                              np.concatenate([train.y, test.y]))
    h = hadamard_layer(N_QUBITS)
    evaluator = CircuitEvaluator(NumpyBackend(N_QUBITS), joint, ZeroMetric(),
                                 prefix=h, suffix=h)
    alignment = TargetAlignment()
    variance = GramVarianceBonus(common.PRODUCTION_WEIGHTS["variance"])
    loo = RidgeLooMse(alphas=(common.RIDGE_ALPHA,))
    n = train.n_samples
    ridge = common.RIDGE_ALPHA * np.eye(n)
    total = float(((test.y - test.y.mean()) ** 2).mean())

    out = {name: np.empty(len(states))
           for name in ("kta", "kta-var", "mse", "r2_test")}
    for start in range(0, len(states), CHUNK_SIZE):
        block = slice(start, start + CHUNK_SIZE)
        psis = np.stack([evaluator.final_states(s) for s in states[block]])
        overlaps = psis @ psis[:, :n].conj().swapaxes(1, 2)
        grams = overlaps.real ** 2 + overlaps.imag ** 2
        K = grams[:, :n]
        kta = alignment.batch(K, train.y)
        out["kta"][block] = kta
        out["kta-var"][block] = kta + variance.batch(K)
        out["mse"][block] = loo.batch(K, train.y)
        coef = np.linalg.solve(K + ridge, train.y[:, None])
        pred = (grams[:, n:] @ coef)[..., 0]
        out["r2_test"][block] = 1.0 - ((pred - test.y) ** 2).mean(axis=1) / total
    out["kta-dla"] = out["kta"] + dla
    return out


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def run(depth: int) -> None:
    from qksearch.enumerate_circuits import enumerate_states, strip_identity
    from qksearch.iqp import describe

    t0 = time.time()
    states = enumerate_states(_gate_set(), depth)
    stripped = [strip_identity(s) for s in states]
    n_gates = np.array([len(s) for s in stripped])
    n_entanglers = np.array([sum(len(op[1]) == 2 for op in s) for s in stripped])
    dla = _dla_penalties(states)
    print(f"{len(states)} circuits, depth {depth} ({time.time() - t0:.0f}s)")

    rows: list[dict] = []
    space = {name: np.empty((len(SEEDS), len(states)), dtype=np.float32)
             for name in (*OBJECTIVES, "r2_test")}
    for k, seed in enumerate(SEEDS):
        train, test = common.mof_split(seed)
        scores = _evaluate(states, train, test, dla)
        r2_test = scores["r2_test"]
        picks = {name: int(np.argmax(scores[name])) for name in OBJECTIVES}
        picks[ORACLE[0]] = int(np.argmax(r2_test))
        for name, i in picks.items():
            rows.append({
                "seed": seed, "objective": name,
                "kta_train": float(scores["kta"][i]),
                "loo_r2_train": float(1.0 + scores["mse"][i]),
                "r2_test": float(r2_test[i]),
                "r2_percentile": float((r2_test <= r2_test[i]).mean() * 100),
                "spearman_r2": (_spearman(scores[name], r2_test)
                                if name in OBJECTIVES else 1.0),
                "n_gates": int(n_gates[i]),
                "n_entanglers": int(n_entanglers[i]),
                "detail": describe(stripped[i]) or "empty",
            })
        for name in space:
            space[name][k] = scores[name]
        print(f"seed {seed} ({time.time() - t0:.0f}s)")

    common.write_csv(_csv(depth), rows)
    np.savez_compressed(_space_path(depth), seeds=np.array(SEEDS),
                        **{name.replace("-", "_"): v for name, v in space.items()})
    print(f"results → {_space_path(depth)}")

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

    common.write_meta(f"e16_mse_heuristic_d{depth}", time.time() - t0,
                      seeds=SEEDS, n_qubits=N_QUBITS, n_data_gates=depth,
                      n_circuits=len(states), ridge_alpha=common.RIDGE_ALPHA,
                      weights=common.PRODUCTION_WEIGHTS, summary=summary)


def _label(name: str) -> str:
    return OBJECTIVES[name][0] if name in OBJECTIVES else ORACLE[1]


def _colour(name: str) -> str:
    return OBJECTIVES[name][1] if name in OBJECTIVES else ORACLE[2]


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


def _load_space(depth: int) -> dict[str, np.ndarray]:
    with np.load(common.result_path(os.path.basename(_space_path(depth)))) as data:
        k = int(np.flatnonzero(data["seeds"] == SCATTER_SEED)[0])
        return {name: data[name.replace("-", "_")][k].astype(float)
                for name in (*OBJECTIVES, "r2_test")}


def _scatter(ax, space, df, column, xlabel, shift=0.0, floor=None) -> None:
    x_all = space[column] + shift
    y_all = space["r2_test"]
    shown = np.random.default_rng(0).permutation(len(x_all))[:SCATTER_POINTS]
    xs = x_all[shown] if floor is None else np.maximum(x_all[shown], floor)
    ax.scatter(xs, np.maximum(y_all[shown], -1.0), s=3,
               color=common.BASELINE, alpha=0.5, lw=0, rasterized=True)
    picked = df[df["seed"] == SCATTER_SEED].set_index("objective")
    source = "kta_train" if column != "mse" else "loo_r2_train"
    for name in OBJECTIVES:
        x = float(picked.loc[name, source])
        ax.scatter(x if floor is None else max(x, floor),
                   max(float(picked.loc[name, "r2_test"]), -1.0), s=42,
                   color=_colour(name), edgecolor="white", lw=0.8, zorder=4,
                   label=_label(name))
    ax.text(0.03, 0.97, rf"$\rho$ = {_spearman(x_all, y_all):+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=common.INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test $R^2$")
    ax.set_ylim(bottom=-1.0)
    ax.grid(True, alpha=0.5)


def plot(depth: int) -> None:
    import matplotlib.pyplot as plt

    common.report_style()
    df = common.read_csv(_csv(depth))
    space = _load_space(depth)
    names = [*OBJECTIVES, ORACLE[0]]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))

    ax = axes[0, 0]
    _strip(ax, df, names, "r2_test", floor=-1.0)
    ax.set_ylabel("test $R^2$")

    ax = axes[0, 1]
    _strip(ax, df, list(OBJECTIVES), "spearman_r2")
    ax.axhline(0.0, color=common.BASELINE, lw=0.8)
    ax.set_ylabel(r"Spearman $\rho$")

    _scatter(axes[1, 0], space, df, "kta", "KTA (train)")
    _scatter(axes[1, 1], space, df, "mse", "LOO $R^2$ (train)",
             shift=1.0, floor=-1.0)
    axes[1, 1].set_xlim(-1.0, 1.0)
    axes[1, 1].plot([-1.0, 1.0], [-1.0, 1.0], color=common.BASELINE, lw=0.8,
                    ls="--", zorder=1)
    axes[1, 1].legend(loc="lower right", fontsize=6.5)

    fig.tight_layout()
    common.save_fig(fig, _figure("fig_e16_mse_heuristic", depth),
                    formats=("pdf",))
    plt.close(fig)

    plot_circuits(df, depth)


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


def plot_circuits(df, depth: int) -> None:
    import matplotlib.pyplot as plt

    common.report_style()
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
    common.save_fig(fig, _figure("fig_e16_mse_heuristic", depth),
                    formats=("png",))
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


def plot_face_to_face(depth: int, seed: int = CIRCUIT_SEED):
    import matplotlib.pyplot as plt

    from qksearch.compare_kernels import CMAP_SEQ

    common.report_style()
    df = common.read_csv(_csv(depth))
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
    common.save_fig(fig, _figure("fig_e16_side_by_side", depth))
    return fig


def _argument(flag: str, default: int) -> int:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


if __name__ == "__main__":
    depth = _argument("--depth", DEFAULT_DEPTH)
    if "--plot-only" not in sys.argv:
        run(depth)
    plot(depth)
    common.show(plot_face_to_face(depth, seed=_argument("--seed", CIRCUIT_SEED)))
