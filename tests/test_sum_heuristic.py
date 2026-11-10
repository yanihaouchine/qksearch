from __future__ import annotations

from collections import Counter

import numpy as np

from experiments.e17_sum_heuristic import sample_states, space_size
from qksearch.dla import DlaCalculator, pauli_closure, state_generators
from qksearch.gates import iqp_gate_set, pauli_gate_set
from qksearch.kernel import GramVarianceBonus
from qksearch.metric import GramTerm, RidgeLooMse, TargetAlignment, WeightedSum


def _grams(seed: int = 0, b: int = 5, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(b, n, 3))
    grams = np.exp(-((A[:, :, None] - A[:, None]) ** 2).sum(-1))
    return grams, rng.normal(size=n)


def test_weighted_sum_matches_its_terms() -> None:
    grams, y = _grams()
    terms = {"kta": (TargetAlignment(), 1.0), "mse": (RidgeLooMse(), 0.5),
             "var": (GramTerm(GramVarianceBonus(1.0)), 2.0)}
    total = WeightedSum(terms)
    expected = sum(w * m.batch(grams, y) for m, w in terms.values())
    assert np.allclose(total.batch(grams, y), expected)
    assert np.allclose([total(g, y) for g in grams], expected)
    assert np.allclose(sum(total.components(grams, y).values()), expected)


def test_pauli_closure_of_full_set_is_su() -> None:
    gs = pauli_gate_set(3, 3)
    assert DlaCalculator().dimension(gs.catalog) == 4 ** 3 - 1
    assert len(pauli_closure(state_generators((("rx", (0,), (0,)),)))) == 1


def test_space_size_counts_multisets() -> None:
    assert space_size(3, 2) == 1 + 3 + 6
    assert space_size(22, 40) > 1e16


def test_sample_states_is_uniform_over_canonical_states() -> None:
    gs = iqp_gate_set(2, 2)
    catalog = gs.catalog
    states = sample_states(np.random.default_rng(0), catalog, 60000, 2)
    canonical = {()}
    for _ in range(2):
        canonical |= {gs.insert(t, op) for t in canonical for op in gs.placements(t)}
    counts = Counter(states)
    assert set(counts) == canonical
    assert len(counts) == space_size(len(catalog), 2)
    freq = np.array(list(counts.values())) / len(states)
    assert np.allclose(freq, 1 / len(counts), rtol=0.25)
