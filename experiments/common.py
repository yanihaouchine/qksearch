from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time

import matplotlib as mpl
import numpy as np

from qksearch.compare_kernels import (
    BASELINE,
    BLUE,
    GRID,
    INK,
    INK2,
    MUTED,
    RED,
)
from qksearch.dataset import (
    AngleScaler,
    RegressionDataset,
    mof_dataset,
    sine_dataset,
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "results")
REPORT_FIG_DIR = os.path.join(REPO_DIR, "report", "figures")

BALANCE_WEIGHTS = {"dla": 1.0, "variance": 0.5}
PRODUCTION_WEIGHTS = {"dla": 0.05, "variance": 0.02}

RIDGE_ALPHA = 1e-3


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORT_FIG_DIR, exist_ok=True)


def report_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.format": "pdf",
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 120,
    })


FIG_FORMATS = tuple(f.strip().lower()
                    for f in os.environ.get("FIG_FORMATS", "pdf,png").split(",")
                    if f.strip())
FIG_DPI = int(os.environ.get("FIG_DPI", "300"))


def save_fig(fig, name: str, formats: tuple[str, ...] = ()) -> str:
    ensure_dirs()
    paths = []
    for ext in (formats or FIG_FORMATS):
        path = os.path.join(REPORT_FIG_DIR, f"{name}.{ext}")
        extra = {} if ext == "pdf" else {"dpi": FIG_DPI}
        fig.savefig(path, bbox_inches="tight", **extra)
        print(f"figure → {path}")
        paths.append(path)
    return paths[0]


def show(fig) -> None:
    import matplotlib.pyplot as plt

    interactive = mpl.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
    if "--no-show" in sys.argv or not interactive:
        plt.close(fig)
        return
    plt.show()


def write_csv(name: str, rows: list[dict]) -> str:
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"results → {path} ({len(rows)} rows)")
    return path


PRODUCERS = {"population_mof.csv": "population_circuits"}


def _producer(name: str) -> str | None:
    if name in PRODUCERS:
        return PRODUCERS[name]
    match = re.match(r"e(\d+)_", name)
    if match is None:
        return None
    prefix = f"e{match.group(1)}_"
    for entry in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if entry.startswith(prefix) and entry.endswith(".py"):
            return entry[:-3]
    return None


def _run_producer(module_name: str, name: str) -> None:
    import importlib
    import inspect

    module = importlib.import_module(f"experiments.{module_name}")
    entry = getattr(module, "run", None) or module.main
    kwargs = {}
    depth = re.search(r"_d(\d+)\.", name)
    if depth and "depth" in inspect.signature(entry).parameters:
        kwargs["depth"] = int(depth.group(1))
    argv = sys.argv
    sys.argv = [module_name, "--no-show"]
    try:
        entry(**kwargs)
    finally:
        sys.argv = argv


def result_path(name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return path
    producer = _producer(name)
    if producer is None:
        raise FileNotFoundError(f"{path} is missing and no experiment produces it")
    print(f"{name} is missing: running experiments.{producer} first")
    _run_producer(producer, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"experiments.{producer} did not write {path}")
    return path


def read_csv(name: str, **kwargs):
    import pandas as pd

    return pd.read_csv(result_path(name), **kwargs)


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def write_meta(experiment: str, wall_time: float, **params) -> None:
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, "meta.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta[experiment] = {
        "git_hash": git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_s": round(wall_time, 2),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "params": params,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def catalog_size(n_qubits: int) -> int:
    return n_qubits + math.comb(n_qubits, 2) + 1


def space_size(n_qubits: int, depth: int) -> int:
    m = catalog_size(n_qubits)
    return math.comb(m + depth - 1, depth)


REFERENCE_SCALE = float(math.pi / 4)
BOUND_SIGMAS = 2.0


def sine_split(seed: int, n_features: int = 2, n_train: int = 40,
               n_test: int = 80, noise: float = 0.05,
               scale: float = REFERENCE_SCALE
               ) -> tuple[RegressionDataset, RegressionDataset]:
    raw = sine_dataset(n_samples=n_train + n_test, n_features=n_features,
                       noise=noise, seed=seed)
    data = AngleScaler.fit(raw.X[:n_train], scale=scale,
                           bound=BOUND_SIGMAS * scale).apply(raw)
    return (RegressionDataset(data.X[:n_train], data.y[:n_train]),
            RegressionDataset(data.X[n_train:], data.y[n_train:]))


def mof_split(seed: int, n_train: int = 100,
              scale: float = REFERENCE_SCALE
              ) -> tuple[RegressionDataset, RegressionDataset]:
    raw = mof_dataset(seed=seed)
    data = AngleScaler.fit(raw.X[:n_train], scale=scale,
                           bound=BOUND_SIGMAS * scale).apply(raw)
    return (RegressionDataset(data.X[:n_train], data.y[:n_train]),
            RegressionDataset(data.X[n_train:], data.y[n_train:]))


def cv_folds(n_samples: int, n_splits: int = 5, seed: int = 0):
    from sklearn.model_selection import KFold

    return list(KFold(n_splits=n_splits, shuffle=True,
                      random_state=seed).split(np.arange(n_samples)))


SINE_ANCHOR = {"best_kta": 0.1975, "best_r2": 0.981,
               "worst_kta": 0.0048, "worst_r2": -0.059}


def ridge_test_r2(backend, circuit, train: RegressionDataset, test: RegressionDataset,
            gram: np.ndarray | None = None) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    if gram is None:
        gram = backend.gram(circuit, train.X)
    krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed").fit(gram, train.y)
    return float(r2_score(test.y, krr.predict(backend.cross_gram(circuit, test.X, train.X))))


def check_sine_anchor(ktas, r2s=None) -> None:
    ktas = np.asarray(ktas, dtype=float)
    best, worst = int(np.argmax(ktas)), int(np.argmin(ktas))
    assert abs(ktas[best] - SINE_ANCHOR["best_kta"]) < 1e-3, ktas[best]
    assert abs(ktas[worst] - SINE_ANCHOR["worst_kta"]) < 1e-3, ktas[worst]
    if r2s is not None:
        r2s = np.asarray(r2s, dtype=float)
        assert abs(r2s[best] - SINE_ANCHOR["best_r2"]) < 2e-3, r2s[best]
        assert abs(r2s[worst] - SINE_ANCHOR["worst_r2"]) < 2e-3, r2s[worst]


def cv_r2(K: np.ndarray, y: np.ndarray, folds=None, seed: int = 0) -> float:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import r2_score

    if folds is None:
        folds = cv_folds(len(y), seed=seed)
    scores = []
    for tr, va in folds:
        krr = KernelRidge(alpha=RIDGE_ALPHA, kernel="precomputed")
        krr.fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(r2_score(y[va], krr.predict(K[np.ix_(va, tr)])))
    return float(np.mean(scores))


__all__ = [
    "BASELINE", "BLUE", "GRID", "INK", "INK2", "MUTED", "RED",
    "BALANCE_WEIGHTS", "PRODUCTION_WEIGHTS", "RIDGE_ALPHA",
    "RESULTS_DIR", "REPORT_FIG_DIR", "REPO_DIR",
    "ensure_dirs", "report_style", "save_fig", "show", "write_csv", "read_csv",
    "result_path",
    "FIG_FORMATS", "FIG_DPI",
    "write_meta", "git_hash", "catalog_size", "space_size",
    "sine_split", "mof_split", "REFERENCE_SCALE", "BOUND_SIGMAS",
    "cv_folds", "cv_r2", "SINE_ANCHOR", "ridge_test_r2", "check_sine_anchor",
]
