"""Hydrogen uptake of metal-organic frameworks (MOF): 294 measurements.

Features: BET surface area (m^2/g), pore volume (m^3/g), pressure (bar) and
temperature (K, 30 to 300); target: hydrogen uptake (wt %).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import RegressionDataset

FEATURES = ("bet_m2_g", "volume_m3_g", "pressure_bar", "temperature_k")
TARGET = "adsorption_pct"

CSV_PATH = Path(__file__).with_name("mof_dataset.csv")


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        header = [name.strip() for name in f.readline().rstrip("\n").split(",")]
    missing = [name for name in (*FEATURES, TARGET) if name not in header]
    if missing:
        raise ValueError(f"columns missing from {path.name}: {missing}")
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    X = table[:, [header.index(name) for name in FEATURES]]
    y = table[:, header.index(TARGET)]
    return X, y


def mof_dataset(
    n_samples: int | None = None,
    seed: int = 0,
    path: Path | str = CSV_PATH,
) -> RegressionDataset:
    """Shuffled MOF dataset with standardized targets.

    ``n_samples`` draws a random subset (default: all rows, in random order).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    X, y = _read_csv(path)

    rng = np.random.default_rng(seed)
    if n_samples is None:
        order = rng.permutation(len(X))
    else:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if n_samples > len(X):
            raise ValueError(
                f"n_samples={n_samples} exceeds the {len(X)} available points"
            )
        order = rng.choice(len(X), size=n_samples, replace=False)
    return RegressionDataset(X[order], y[order]).standardized()
