from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR as RESULTS
from qksearch.compare_kernels import BASELINE, BLUE, FIG_DIR, GRID, INK, INK2, MUTED, RED, SURFACE

MODES = ("valid_r1", "prod_r2", "prod_r2_phase")

TEXT = {
    "fr": {
        "modes": ("1 couche\nrz, rzz", "2 couches\nrz, rzz", "2 couches\n+ portes T, CS"),
        "f1_title": "L'algorithme génétique retrouve l'optimum exact",
        "f1_sub": ("Espace de {space} circuits énumérable : on connaît la réponse.\n"
                   "L'algorithme n'en évalue que {evals} en moyenne, soit {pct:.0f} %."),
        "f1_x": "génération",
        "f1_y": "qualité trouvée / optimum exact",
        "f1_mean": "moyenne sur 10 tirages",
        "f1_worst": "pire tirage",
        "f1_opt": "optimum exact",
        "f1_badge": "{hits}/10 tirages\natteignent\nl'optimum",
        "f2_title": "Le critère optimisé et la qualité réelle se séparent",
        "f2_sub": ("À gauche ce que l'algorithme maximise, à droite ce qu'on veut vraiment.\n"
                   "Agrandir l'espace de recherche améliore l'un et dégrade l'autre."),
        "f2_kta": "KTA — critère maximisé (train)",
        "f2_r2": "R² — qualité de prédiction (test)",
        "f2_note": "un point = un tirage aléatoire ; la barre est la médiane",
        "f3_title": "Un circuit est un entier binaire",
        "f3_sub": ("Les portes commutent : l'ordre ne compte pas, seul le nombre de fois\n"
                   "que chaque porte apparaît. Ce vecteur tient dans des champs de bits."),
        "f3_circuit": "circuit",
        "f3_vector": "vecteur de multiplicités",
        "f3_bits": "champs de bits",
        "f3_int": "un entier",
        "f3_gates": "porte",
        "f3_caption": ("Chaque champ a une largeur fixe, donc tout entier est un circuit\n"
                       "valide : mutation et croisement ne peuvent pas sortir de l'espace."),
            "f4_title": "Les circuits trouvés par l'algorithme",
        "f4_sub": ("Le meilleur circuit de chaque famille, sur les 10 tirages.\n"
                   "Bleu : portes pilotées par les données. Rouge : portes de phase fixes."),
        "f4_wall": "H",
        "f4_metrics": "KTA {kta:+.3f}   R² {r2:+.3f}",
},
    "en": {
        "modes": ("1 layer\nrz, rzz", "2 layers\nrz, rzz", "2 layers\n+ T, CS gates"),
        "f1_title": "The genetic algorithm recovers the exact optimum",
        "f1_sub": ("A {space}-circuit space we can enumerate: the answer is known.\n"
                   "The algorithm evaluates only {evals} of them on average, i.e. {pct:.0f}%."),
        "f1_x": "generation",
        "f1_y": "score found / exact optimum",
        "f1_mean": "mean over 10 seeds",
        "f1_worst": "worst seed",
        "f1_opt": "exact optimum",
        "f1_badge": "{hits}/10 seeds\nreach the\noptimum",
        "f2_title": "The optimized criterion and the real quality come apart",
        "f2_sub": ("Left: what the algorithm maximizes. Right: what we actually want.\n"
                   "A larger search space improves one and degrades the other."),
        "f2_kta": "KTA — maximized criterion (train)",
        "f2_r2": "R² — predictive quality (test)",
        "f2_note": "one dot = one seed; the bar is the median",
        "f3_title": "A circuit is a binary integer",
        "f3_sub": ("The gates commute: order carries nothing, only how many times\n"
                   "each gate occurs. That vector fits in fixed bit fields."),
        "f3_circuit": "circuit",
        "f3_vector": "multiplicity vector",
        "f3_bits": "bit fields",
        "f3_int": "one integer",
        "f3_gates": "gate",
        "f3_caption": ("Every field has a fixed width, so any integer is a valid circuit:\n"
                       "mutation and crossover cannot leave the space."),
            "f4_title": "The circuits the algorithm found",
        "f4_sub": ("Best circuit of each family, over the 10 seeds.\n"
                   "Blue: data-driven gates. Red: fixed phase gates."),
        "f4_wall": "H",
        "f4_metrics": "KTA {kta:+.3f}   R² {r2:+.3f}",
},
}


def screen_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 13,
        "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def _save(fig, name: str, dpi: int) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure → {path}")
    return path


def fig_validation(curves, best, t, dpi: int) -> None:
    opt = best[best["mode"] == "valid_r1"].set_index("seed")["optimum"]
    sub = curves[curves["mode"] == "valid_r1"].copy()
    sub["frac"] = sub["best_fitness"] / sub["seed"].map(opt)
    agg = sub.groupby("generation")["frac"].agg(["mean", "min"])

    v = best[best["mode"] == "valid_r1"]
    hits = int((v["frac_of_opt"] >= 1 - 1e-9).sum())
    evals = int(v["n_evaluated"].median())
    space = 8008

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.plot(agg.index, agg["mean"], color=BLUE, lw=3.0, solid_capstyle="round",
            zorder=3)
    ax.plot(agg.index, agg["min"], color=BLUE, lw=1.8, ls="--", alpha=0.55,
            zorder=2)

    x_end = agg.index.max()
    ax.annotate(t["f1_mean"], (x_end, agg["mean"].iloc[-1]),
                xytext=(-8, 12), textcoords="offset points",
                ha="right", color=INK2, fontsize=12, fontweight="bold")
    ax.annotate(t["f1_worst"], (x_end, agg["min"].iloc[-1]),
                xytext=(-8, -20), textcoords="offset points",
                ha="right", color=MUTED, fontsize=12)
    ax.annotate(t["f1_opt"], (agg.index.min(), 1.0), xytext=(4, 6),
                textcoords="offset points", color=MUTED, fontsize=11)

    ax.text(0.97, 0.30, t["f1_badge"].format(hits=hits), transform=ax.transAxes,
            ha="right", va="center", fontsize=19, fontweight="bold",
            color=BLUE, linespacing=1.35)

    ax.set_xlabel(t["f1_x"])
    ax.set_ylabel(t["f1_y"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    fig.suptitle(t["f1_title"], x=0.01, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.895, t["f1_sub"].format(
        space=f"{space:,}".replace(",", "\u202f"), evals=evals,
        pct=100 * evals / space), ha="left", va="top", color=INK2,
        fontsize=12.5, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    _save(fig, "presentation_e11_1_validation", dpi)


def fig_kta_vs_r2(best, t, dpi: int) -> None:
    modes = [m for m in MODES if (best["mode"] == m).any()]
    labels = list(t["modes"])[:len(modes)]
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    rng = np.random.default_rng(0)
    for ax, (col, title, color) in zip(axes, [
            ("kta_train", t["f2_kta"], BLUE),
            ("r2_test", t["f2_r2"], RED)]):
        med = [best.loc[best["mode"] == m, col].median() for m in modes]
        ax.bar(x, med, width=0.56, color=color, alpha=0.90, zorder=2)
        tops = [best.loc[best["mode"] == m, col].max() for m in modes]
        head = max(tops) - min(min(v for v in best[col]), 0)
        for i, m in enumerate(modes):
            vals = best.loc[best["mode"] == m, col].to_numpy()
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=34,
                       facecolor=SURFACE, edgecolor=INK2, linewidth=1.0,
                       zorder=4, alpha=0.95)
            ax.annotate(f"{med[i]:+.2f}", (i, tops[i] + 0.045 * head),
                        ha="center", va="bottom", fontsize=14,
                        fontweight="bold", color=INK, zorder=5)
        ax.margins(y=0.16)
        ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", pad=12, fontsize=14)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    fig.suptitle(t["f2_title"], x=0.008, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(0.008, 0.905, t["f2_sub"], ha="left", va="top", color=INK2,
             fontsize=12.5, linespacing=1.4)
    fig.text(0.008, 0.012, t["f2_note"], ha="left", color=MUTED, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    _save(fig, "presentation_e11_2_kta_vs_r2", dpi)


def fig_genome(t, dpi: int) -> None:
    from qksearch.genome import iqp_genome_spec, to_int

    spec = iqp_genome_spec(2, 2, n_layers=1)
    names = ["rz(0)", "rz(1)", "rzz(0,1)"]
    mult = [3, 0, 2]
    caps = [spec.genes[i].cap for i in range(3)]
    widths = [spec.genes[i].width for i in range(3)]
    genome = tuple(mult) + (0,) * (len(spec.genes) - 3)
    value = to_int(genome, spec)

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    left, x0 = 0.15, 3.1

    def color_of(name):
        return BLUE

    ax.text(left, 6.85, t["f3_circuit"], fontsize=13, color=INK2, va="center")
    seq = [n for n, m in zip(names, mult) for _ in range(m)]
    w, gap = 1.44, 0.16
    for i, name in enumerate(seq):
        c = color_of(name)
        ax.add_patch(plt.Rectangle((x0 + i * (w + gap), 6.45), w, 0.82,
                                   facecolor=c, alpha=0.15, edgecolor=c,
                                   linewidth=1.5))
        ax.text(x0 + i * (w + gap) + w / 2, 6.86, name, ha="center",
                va="center", fontsize=12, color=INK)

    ax.text(left, 4.55, t["f3_vector"], fontsize=13, color=INK2, va="center")
    bw, bgap = 2.45, 0.30
    for i, (name, m, cap) in enumerate(zip(names, mult, caps)):
        c = color_of(name)
        cx = x0 + i * (bw + bgap) + bw / 2
        ax.add_patch(plt.Rectangle((x0 + i * (bw + bgap), 4.12), bw, 0.86,
                                   facecolor=SURFACE, edgecolor=c, linewidth=1.8))
        ax.text(cx, 4.55, f"{name}  →  {m}", ha="center", va="center",
                fontsize=13, color=INK)
        ax.text(cx, 3.70, f"0..{cap}   ({widths[i]} bits)", ha="center",
                va="center", fontsize=11, color=MUTED)

    ax.text(left, 2.05, t["f3_bits"], fontsize=13, color=INK2, va="center")
    sw, pos = 0.60, 0
    for name, m, width in zip(names, mult, widths):
        c = color_of(name)
        for b in format(m, f"0{width}b"):
            ax.add_patch(plt.Rectangle((x0 + pos * sw, 1.62), sw - 0.07, 0.86,
                                       facecolor=c, alpha=0.15, edgecolor=c,
                                       linewidth=1.2))
            ax.text(x0 + pos * sw + (sw - 0.07) / 2, 2.05, b, ha="center",
                    va="center", fontsize=13, color=INK, family="monospace")
            pos += 1
        ax.plot([x0 + pos * sw - 0.035] * 2, [1.55, 2.55], color=BASELINE,
                lw=1.2, zorder=1)

    terms = " + ".join(f"{m}×{2 ** sum(widths[:i])}"
                       for i, m in enumerate(mult) if m)
    ax.text(x0, 0.95, f"{terms}  =  {value}", fontsize=15, color=INK,
            va="center", fontweight="bold")
    ax.text(x0 + 0.02, 0.35, t["f3_int"], fontsize=11.5, color=MUTED,
            va="center")

    for y in (6.30, 3.45):
        ax.annotate("", xy=(1.85, y - 0.72), xytext=(1.85, y),
                    arrowprops=dict(arrowstyle="->", color=BASELINE, lw=1.8))

    fig.suptitle(t["f3_title"], x=0.012, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(0.012, 0.905, t["f3_sub"], ha="left", va="top", color=INK2,
             fontsize=12.5, linespacing=1.4)
    fig.text(0.012, 0.015, t["f3_caption"], ha="left", va="bottom", color=INK2,
             fontsize=12, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    _save(fig, "presentation_e11_3_genome", dpi)


PHASE_NAMES = frozenset({"t", "s", "t3", "z", "t5", "sdg", "tdg",
                        "ct", "cs", "ct3", "cz", "ct5", "csdg", "ctdg"})


def _gate_label(name: str, f, k: int = 1) -> str:
    subs = str.maketrans("0123", "₀₁₂₃")
    if f is None:
        return name.upper()
    angle = "".join(f"x{i}".translate(subs) for i in f)
    return name.upper() + "\n" + (f"{k}·{angle}" if k > 1 else angle)


def _columns(state, n_qubits: int):
    cols, i = [], 0
    while i < len(state):
        if state[i][0] == "h":
            j = i
            while j < len(state) and state[j][0] == "h":
                j += 1
            cols.append(("wall", None, 0))
            i = j
        else:
            j = i
            while j < len(state) and state[j] == state[i]:
                j += 1
            cols.append(("gate", state[i], j - i))
            i = j
    return cols


def _draw_circuit(ax, state, n_qubits: int, t, fontsize: float = 11) -> None:
    cols = _columns(state, n_qubits)
    n = len(cols) + 2
    ys = {q: 1.0 - (q + 0.5) / n_qubits for q in range(n_qubits)}
    ax.set_xlim(-0.75, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bh = 0.86 / n_qubits

    for q, y in ys.items():
        ax.plot([-0.15, n - 0.1], [y, y], color=BASELINE, lw=1.2, zorder=1)
        ax.text(-0.45, y, f"q{q}", fontsize=fontsize - 1, color=MUTED,
                ha="center", va="center")

    def box(col, q0, q1, label, color):
        y0, y1 = min(ys[q0], ys[q1]), max(ys[q0], ys[q1])
        if q0 != q1:
            ax.plot([col, col], [y0, y1], color=color, lw=1.6, zorder=2)
        ax.add_patch(plt.Rectangle((col - 0.36, y0 - bh / 2), 0.72,
                                   (y1 - y0) + bh, facecolor=SURFACE,
                                   edgecolor=color, linewidth=1.8, zorder=3))
        ax.text(col, (y0 + y1) / 2, label, ha="center", va="center",
                fontsize=fontsize, color=INK, zorder=4, linespacing=1.05)

    for q in range(n_qubits):
        box(0.5, q, q, t["f4_wall"], BASELINE)
    for idx, (kind, op, mult) in enumerate(cols):
        col = idx + 1.5
        if kind == "wall":
            for q in range(n_qubits):
                box(col, q, q, t["f4_wall"], BASELINE)
            continue
        name, qubits, f = op
        color = RED if name in PHASE_NAMES else BLUE
        box(col, min(qubits), max(qubits), _gate_label(name, f, mult), color)
    for q in range(n_qubits):
        box(n - 0.5, q, q, t["f4_wall"], BASELINE)


def fig_circuits(best, t, dpi: int) -> None:
    from qksearch.genome import decode, from_int, iqp_genome_spec

    specs = {"valid_r1": dict(n_layers=1, phases=()),
             "prod_r2": dict(n_layers=2, phases=()),
             "prod_r2_phase": dict(n_layers=2, phases=("t", "cs"))}
    modes = [m for m in MODES if (best["mode"] == m).any()]

    fig, axes = plt.subplots(len(modes), 1, figsize=(12.5, 2.35 * len(modes)))
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        row = best[best["mode"] == mode].sort_values("fitness").iloc[-1]
        spec = iqp_genome_spec(4, 4, **specs[mode])
        state = decode(from_int(int(row["genome_int"]), spec), spec)
        _draw_circuit(ax, state, 4, t)
        ax.set_title(
            t["f4_metrics"].format(kta=row["kta_train"], r2=row["r2_test"]),
            loc="left", fontsize=13, color=INK, pad=10)

    fig.tight_layout()
    _save(fig, "presentation_e11_4_circuits", dpi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=("fr", "en"), default="fr")
    p.add_argument("--figures", default="circuits",
                   help="figures à produire parmi validation,kta,genome,circuits "
                        "— ou 'all' pour les quatre")
    p.add_argument("--modes", default=",".join(MODES),
                   help="modes à afficher, séparés par des virgules")
    p.add_argument("--dpi", type=int, default=200,
                   help="200 suffit pour un projecteur, 300 pour l'impression")
    args = p.parse_args()

    for name in ("e11_genetic.csv", "e11_best.csv"):
        if not os.path.exists(os.path.join(RESULTS, name)):
            raise SystemExit(
                f"{name} manquant — lancer d'abord :\n"
                f"    python -m experiments 11")

    wanted = [m for m in args.modes.split(",") if m]
    unknown = [m for m in wanted if m not in MODES]
    if unknown:
        raise SystemExit(f"mode inconnu : {', '.join(unknown)} "
                         f"(attendu : {', '.join(MODES)})")

    screen_style()
    t = TEXT[args.lang]
    curves = pd.read_csv(os.path.join(RESULTS, "e11_genetic.csv"))
    best = pd.read_csv(os.path.join(RESULTS, "e11_best.csv"))
    curves = curves[curves["mode"].isin(wanted)]
    best = best[best["mode"].isin(wanted)]
    if best.empty:
        raise SystemExit(f"aucune donnée pour {args.modes} dans les CSV")

    known = ("validation", "kta", "genome", "circuits")
    figures = known if args.figures == "all" else tuple(
        f for f in args.figures.split(",") if f)
    unknown = [f for f in figures if f not in known]
    if unknown:
        raise SystemExit(f"figure inconnue : {', '.join(unknown)} "
                         f"(attendu : {', '.join(known)}, ou 'all')")

    if "validation" in figures and "valid_r1" in wanted:
        fig_validation(curves, best, t, args.dpi)
    if "kta" in figures:
        fig_kta_vs_r2(best, t, args.dpi)
    if "genome" in figures:
        fig_genome(t, args.dpi)
    if "circuits" in figures:
        fig_circuits(best, t, args.dpi)


if __name__ == "__main__":
    main()
