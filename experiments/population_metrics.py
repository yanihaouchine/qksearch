from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from experiments.common import RESULTS_DIR, result_path
from qksearch.compare_kernels import BASELINE, BLUE, FIG_DIR, GRID, INK, INK2, MUTED

CSV_PATH = os.path.join(RESULTS_DIR, "population_mof.csv")
FIG_PATH = os.path.join(FIG_DIR, "fig_population_metrics.png")


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    color: str
    fmt: str
    discrete: bool = False
    selected_on: bool = False


SPECS = (
    MetricSpec("kta", "KTA", BLUE, "{:+.4f}", selected_on=True),
    MetricSpec("expr", "Expr ↓", BASELINE, "{:.3f}"),
    MetricSpec("n_gates", "gate count", MUTED, "{:.0f}", discrete=True),
)

N_COLS = 3


def load(path: str) -> pd.DataFrame:
    if os.path.dirname(os.path.abspath(path)) == os.path.abspath(RESULTS_DIR):
        path = result_path(os.path.basename(path))
    df = pd.read_csv(path)
    missing = [s.column for s in SPECS if s.column not in df.columns]
    if missing:
        raise ValueError(f"columns missing from {os.path.basename(path)}: {missing}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"{'metric':12s} {'n':>4} {'min':>10} {'median':>10} {'mean':>10} "
          f"{'max':>10} {'std':>10}   selection")
    for spec in SPECS:
        v = df[spec.column].to_numpy(dtype=float)
        cells = " ".join(f"{spec.fmt.format(x):>10s}" for x in
                         (v.min(), float(np.median(v)), v.mean(), v.max()))
        std = f"{v.std(ddof=1):10.4f}" if len(v) > 1 else f"{'—':>10s}"
        print(f"{spec.label:12s} {len(v):4d} {cells} {std}   "
              + ("ranked on it" if spec.selected_on else "free"))


def _panel(ax, values: np.ndarray, spec: MetricSpec, bins: int) -> None:
    if spec.discrete:
        levels = np.arange(int(values.min()), int(values.max()) + 1)
        counts = np.array([(values == level).sum() for level in levels], dtype=float)
        ax.bar(levels, counts, width=0.7, color=spec.color, lw=0)
        ax.set_xticks(levels)
    else:
        span = float(np.ptp(values))
        lo, hi = ((values.min() - 0.5, values.max() + 0.5) if span == 0.0
                  else (values.min(), values.max()))
        counts, _, _ = ax.hist(values, bins=bins, range=(lo, hi),
                               color=spec.color, lw=0)
        ax.plot(values, np.full(len(values), 0.04), marker="|", ms=7,
                ls="none", color=INK2, alpha=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_ylim(0.0, max(float(np.max(counts)), 1.0) * 1.42)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    median = float(np.median(values))
    ax.axvline(median, ymax=0.72, color=INK, lw=1.1, ls="--", zorder=3)
    ax.set_title(spec.label, loc="left", fontsize=10, color=INK)
    ax.text(0.97, 0.94, f"med {spec.fmt.format(median)}\n"
                        f"[{spec.fmt.format(values.min())}, "
                        f"{spec.fmt.format(values.max())}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color=INK2, family="monospace")
    ax.set_ylabel("individuals", fontsize=8, color=INK2)
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_distributions(df: pd.DataFrame, path: str, *, bins: int) -> None:
    n_rows = -(-len(SPECS) // N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(10.0, 3.1 * n_rows))
    flat = axes.ravel()

    for ax, spec in zip(flat, SPECS):
        _panel(ax, df[spec.column].to_numpy(dtype=float), spec, bins)
    for ax in flat[len(SPECS):]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH, help="population CSV to read")
    p.add_argument("--bins", type=int, default=7,
                   help="histogram bins for the continuous metrics")
    p.add_argument("--out", default=FIG_PATH, help="figure to write")
    p.add_argument("--no-show", action="store_true",
                   help="do not open a window (the figure is still written)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load(args.csv)

    origin = ""
    for column, tag in (("seed", "seed"), ("n_descents", "descents"),
                        ("min_distance", "min_distance")):
        if column in df.columns:
            origin += f", {tag} {df[column].iloc[0]}"
    print(f"=== metric distributions over {len(df)} individuals "
          f"({os.path.basename(args.csv)}) ===")
    print(origin.lstrip(", ") or "no run parameters recorded in the CSV")
    print()
    summarize(df)
    if "origin" in df.columns and (df["origin"] != "search").any():
        counts = df["origin"].value_counts()
        print(f"\npopulation: {counts.get('search', 0)} harvested by search + "
              f"{counts.get('random', 0)} drawn at random. 'ranked on it' only "
              "holds for the harvested half; the random draws widen every spread "
              "and are what the lower tail of the KTA column is made of.")
    else:
        print("\n'ranked on it': the search selected the population on this "
              "metric, so its spread reflects the selection, not the circuit "
              "space.")

    fig_distributions(df, args.out, bins=args.bins)
    print(f"\nfigure → {args.out}")


if __name__ == "__main__":
    main()
