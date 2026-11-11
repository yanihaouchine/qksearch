# Experiments

Each experiment is a script in [`experiments/`](../experiments/), run with

```bash
python -m experiments N              # compute, write results/, draw the figure
python -m experiments N --plot-only  # redraw from results/
```

Every run writes its raw results to `results/` and records its parameters, git
commit and wall time in [`results/meta.json`](../results/meta.json). Only the
results of E16–E18 are versioned; the others are regenerated on demand: when
an experiment (or `--plot-only`) needs a result file that is missing, the
experiment that produces it runs first. For instance, `python -m experiments 8`
on a fresh clone runs E5 before drawing E8's circuit.
Unless stated otherwise, MOF experiments use 4 qubits, 100 training points
and 10 random splits.

## Summary

| # | Question | Finding |
| --- | --- | --- |
| E1 | Do random IQP kernels concentrate as qubits are added? | Off-diagonal Gram variance falls by orders of magnitude from 2 to 12 qubits. |
| E2 | How do KTA and test R² evolve with depth? | The KTA peaks at a few gates; beyond, training alignment rises while test R² falls. |
| E3 | Does expressibility predict test R²? | No: on the exhaustive MOF space the rank correlation is close to zero (ρ ≈ −0.06). |
| E4 | How efficient is MCTS against exhaustive enumeration? | MCTS recovers the exact optimum of the 8 008-circuit space at a fraction of its cost. Best-first search is not part of this comparison. |
| E5 | Searched quantum kernels against a cross-validated RBF | The RBF kernel stays ahead on MOF, but with the quantum encoding bandwidth frozen (see E13). |
| E6 | How do the DLA and Gram-variance regularizers trade off? | As tie-breakers they rarely change the selected circuit; when they dominate, the effect depends on the dataset. |
| E7 | Geometric difference between quantum and RBF kernels | Measures how much room a quantum kernel leaves to a classical one as the training set grows (Huang et al., 2021). |
| E8 | What does the selected MOF circuit look like? | Single-qubit rotations with repetitions (angle amplification) and no entangler. |
| E9 | Why does the search drop pore volume (x1)? | BET area and pore volume are collinear (r ≈ 0.91); the KTA keeps BET for marginal reasons (rank correlation, skewness), not predictive ones. |
| E10 | Do Havlíček-style entanglers help? | The rescaled `rzzhn` gate is kept in every selected circuit and raises the KTA; its R² gain is not significant. |
| E11 | Genetic algorithm on a binary genome | Recovers the exact optimum of the small space; in a larger two-layer space the KTA rises and test R² drops. Constant phase gates bring nothing. |
| E12 | Is the KTA invariant to the order of commuting gates? | Yes, to within 3 ULP, while a non-diagonal control spreads by 0.003 to 0.013 in KTA. This justifies the canonical gate order. |
| E13 | How does the encoding bandwidth change the selected circuit? | The KTA keeps rising with the bandwidth after test R² has peaked. |
| E14 | How many shots does the KTA ranking need? | About 30 shots per Gram entry; prediction breaks because of the fixed ridge parameter, not the measurement noise. |
| E15 | What breaks first when scaling to 8 qubits? | Neither the kernel contrast nor the shot budget, but the search: MCTS falls below uniform sampling from 6 qubits. |
| E16 | Closed-form leave-one-out ridge error as an objective | Ranks circuits twice as well as the KTA, but its argmax overfits. |
| E17 | Composite objective (KTA + LOO + variance + DLA) in a 3.4 × 10¹⁶-circuit space | Test R² 0.60 against 0.46 for the KTA, with 7-gate instead of 22-gate circuits. |
| E18 | Are the selected kernels classically simulable? | One layer: always certified. Two layers: structural certificates drop to 0–30 %, random Fourier features still match the quantum kernel on 50–70 % of splits. |

## Versioned results

| # | Figure | Data |
| --- | --- | --- |
| E16 | [`fig_e16_side_by_side_d6.png`](../report/figures/fig_e16_side_by_side_d6.png), [`d10`](../report/figures/fig_e16_side_by_side_d10.png) | [`e16_mse_heuristic_d6.csv`](../results/e16_mse_heuristic_d6.csv), [`d10`](../results/e16_mse_heuristic_d10.csv) |
| E17 | [`fig_e17_sum_heuristic.png`](../report/figures/fig_e17_sum_heuristic.png) | [`e17_sum_heuristic.csv`](../results/e17_sum_heuristic.csv) |
| E18 | [`fig_e18_simulability.png`](../report/figures/fig_e18_simulability.png) | [`e18_simulability.csv`](../results/e18_simulability.csv) |

### E16 in numbers

| Objective | Test R², 6 gates (8 008 circuits) | Spearman ρ with test R² | Test R², 10 gates (184 756 circuits) |
| --- | --- | --- | --- |
| KTA | 0.54 | 0.26 | 0.27 |
| KTA + DLA or variance | 0.54 | 0.26 | 0.27 |
| LOO ridge MSE | 0.35 | 0.52 | −0.46 |
| Oracle (best test R² in the space) | 0.81 | 1.00 | 0.82 |

### E17 in numbers

Budget of 8 000 evaluations, circuits of up to 40 gates from a 22-gate catalogue.

| Objective | Search | Test R² | Gates |
| --- | --- | --- | --- |
| KTA | best-first | 0.46 | 22.5 |
| KTA | uniform sampling | −0.16 | 38.5 |
| KTA + LOO + variance + DLA | best-first | **0.60** | 7 |
| KTA + LOO + variance + DLA | uniform sampling | 0.21 | 39 |

## Caveats

- **Best-first search is absent from E4, E5 and E15**, which compare MCTS to
  greedy search, exhaustive enumeration and uniform sampling. Since every
  node's score is exact, best-first search is the natural baseline; it is the
  strategy used in E17.
- **E1–E12 freeze the encoding bandwidth at π/4.** E13 shows this choice
  interacts with the circuit structure, so their quantum-versus-RBF
  comparisons are not tuned symmetrically.

## Calibration scripts

[`experiments/calibration/`](../experiments/calibration/) holds the short
exploratory scripts used to pick the regularizer weights
(`experiments.common.BALANCE_WEIGHTS` and `PRODUCTION_WEIGHTS`). They run in a
few seconds each, e.g. `python experiments/calibration/robust.py`.
