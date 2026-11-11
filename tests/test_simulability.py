from __future__ import annotations

import numpy as np
import pytest

from qksearch.backend import NumpyBackend
from qksearch.dataset import AngleScaler, sine_dataset
from qksearch.gates import (
    HADAMARD_LAYER_OP,
    GateSpec,
    LayeredGateSet,
    iqp_gate_set,
    layer_count,
    pauli_gate_set,
)
from qksearch.iqp import IqpKernelHeuristic, hadamard_layer
from qksearch.metric import TargetAlignment
from qksearch.search import BestFirstSearch, search
from qksearch.simulability import (
    SimulabilityPenalty,
    certify,
    classical_gram,
    components,
    encoding_spectrum,
    rff_frequencies,
    rff_gram,
    stabilizer_renyi_entropy,
)

N = 4

IQP = (("rz", (0,), (0,)), ("rzz", (0, 1), (0, 1)), ("rzz", (1, 2), (1, 2)),
       ("rzz", (0, 2), (0, 2)), ("rz", (2,), (2,)))
MATCHGATE = (("rx", (0,), (0,)), ("rzz", (0, 2), (0, 2)), ("rx", (2,), (2,)),
             ("rzz", (1, 2), (1, 2)), ("rx", (1,), (1,)), ("rzz", (0, 2), (0, 3)),
             ("rzz", (1, 3), (1, 3)), ("rx", (3,), (3,)))
MIXED = (("rzz", (0, 1), (0, 1)), ("rz", (0,), (0,)), ("rzz", (2, 3), (2, 3)),
         ("rx", (3,), (3,)), ("rzz", (2, 3), (1, 2)), ("rx", (2,), (2,)))
UNCERTIFIED = (("rzz", (0, 1), (0, 1)), ("rx", (1,), (1,)), ("rzz", (1, 2), (1, 2)),
               ("rz", (1,), (0,)))


def _wrap(layer, n=N):
    h = hadamard_layer(n)
    return h + layer + h


def _inputs(n_samples=15, n_features=N, seed=0):
    return np.random.default_rng(seed).uniform(-1.5, 1.5, size=(n_samples, n_features))


def _random_iqp(rng, n, n_gates):
    ops = []
    for _ in range(n_gates):
        if rng.random() < 0.5:
            q = int(rng.integers(n))
            ops.append(("rz", (q,), (q,)))
        else:
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            ops.append(("rzz", (a, b), (a, b)))
    return tuple(ops)


def _random_matchgate(rng, n, n_gates):
    path = rng.permutation(n).tolist()
    edges = [tuple(sorted(e)) for e in zip(path, path[1:])]
    ops = [("rzz", e, e) for e in edges]
    for _ in range(n_gates):
        if rng.random() < 0.5:
            q = int(rng.integers(n))
            ops.append(("rx", (q,), (q,)))
        else:
            e = edges[int(rng.integers(len(edges)))]
            ops.append(("rzz", e, e))
    rng.shuffle(ops)
    return tuple(ops)


@pytest.mark.parametrize("layer, rules", [
    (IQP, {"iqp", "product"}),
    (MATCHGATE, {"matchgate"}),
    (MIXED, {"iqp", "matchgate"}),
    ((), {"product"}),
])
def test_certified_circuits_match_the_statevector(layer, rules):
    circuit = _wrap(layer)
    X = _inputs()
    assert certify(circuit, N) == rules
    np.testing.assert_allclose(classical_gram(circuit, X, N),
                               NumpyBackend(N).gram(circuit, X), atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_random_certified_circuits_match_the_statevector(seed):
    rng = np.random.default_rng(seed)
    X = _inputs(seed=seed)
    for layer, rule in ((_random_iqp(rng, N, 8), "iqp"),
                        (_random_matchgate(rng, N, 8), "matchgate")):
        circuit = _wrap(layer)
        assert rule in certify(circuit, N)
        np.testing.assert_allclose(classical_gram(circuit, X, N),
                                   NumpyBackend(N).gram(circuit, X), atol=1e-10)


def test_monte_carlo_iqp_stays_within_sampling_error():
    circuit = _wrap(IQP)
    X = _inputs()
    samples = 4000
    exact = NumpyBackend(N).gram(circuit, X)
    estimate = classical_gram(circuit, X, N, samples=samples, seed=1)
    assert np.abs(estimate - exact).max() < 5 * 2 / np.sqrt(samples)


def test_components_split_on_two_qubit_gates():
    assert sorted(components(_wrap(MIXED), N)) == [(0, 1), (2, 3)]
    assert sorted(components(_wrap(()), N)) == [(0,), (1,), (2,), (3,)]


@pytest.mark.parametrize("layer", [
    UNCERTIFIED,
    (("rzz", (0, 1), (0, 1)), ("rzz", (1, 2), (1, 2)), ("rzz", (0, 2), (0, 2)),
     ("rx", (0,), (0,))),
    (("rzz", (0, 1), (0, 1)), ("rx", (0,), (0,)), ("rz", (1,), (1,))),
    (("rzz", (0, 1), (0, 1)), ("ry", (0,), (0,))),
    (("rzz", (0, 1), (0, 1)), ("rzz", (0, 2), (0, 2)), ("rzz", (0, 3), (0, 3)),
     ("rx", (0,), (0,))),
])
def test_uncertified_circuits_are_rejected(layer):
    circuit = _wrap(layer)
    assert certify(circuit, N) == frozenset()
    with pytest.raises(ValueError):
        classical_gram(circuit, _inputs(), N)


def test_diagonal_gates_on_the_zero_state_are_inert():
    X = _inputs()
    assert certify(IQP, N) == {"product"}
    np.testing.assert_allclose(classical_gram(IQP, X, N), 1.0)
    np.testing.assert_allclose(NumpyBackend(N).gram(IQP, X), 1.0, atol=1e-12)


HL = (HADAMARD_LAYER_OP,)
PATH_ZZ = (("rzz", (0, 1), (0, 1)), ("rzz", (1, 2), (1, 2)), ("rzz", (2, 3), (2, 3)))
FIELDS = tuple(("rz", (q,), (q,)) for q in range(N))
PHASES = (("t", (1,), None), ("cs", (0, 2), None), ("cs", (1, 3), None))


def test_hadamard_layer_op_matches_per_qubit_hadamards():
    X = _inputs()
    backend = NumpyBackend(N)
    np.testing.assert_allclose(backend.states(HL + IQP, X),
                               backend.states(hadamard_layer(N) + IQP, X))


@pytest.mark.parametrize("layer, rules", [
    (PATH_ZZ + HL + FIELDS, {"matchgate"}),
    (PATH_ZZ + HL + FIELDS + HL + PATH_ZZ, {"matchgate"}),
    ((("rzz", (0, 1), (0, 1)),) + HL + (("rz", (3,), (3,)), ("rzz", (2, 3), (2, 3))),
     {"iqp", "product"}),
    (IQP + PHASES, {"iqp", "product"}),
])
def test_layered_certificates_match_the_statevector(layer, rules):
    circuit = _wrap(layer)
    X = _inputs()
    assert certify(circuit, N) == rules
    np.testing.assert_allclose(classical_gram(circuit, X, N),
                               NumpyBackend(N).gram(circuit, X), atol=1e-10)


@pytest.mark.parametrize("layer", [
    FIELDS + HL + PATH_ZZ,
    PATH_ZZ + FIELDS + HL + PATH_ZZ + FIELDS,
    IQP + HL + IQP,
    PATH_ZZ + PHASES + HL + FIELDS,
    PATH_ZZ + (("rzz", (0, 3), (0, 3)),) + HL + FIELDS,
])
def test_layered_circuits_without_certificate(layer):
    assert certify(_wrap(layer), N) == frozenset()


def test_constant_phases_cancel_in_a_single_layer():
    X = _inputs()
    backend = NumpyBackend(N)
    np.testing.assert_allclose(backend.gram(_wrap(IQP + PHASES), X),
                               backend.gram(_wrap(IQP), X), atol=1e-12)
    assert not np.allclose(backend.gram(_wrap(IQP + PHASES + HL + IQP), X),
                           backend.gram(_wrap(IQP + HL + IQP), X))


def test_constant_phases_in_the_last_layer_are_inert():
    X = _inputs()
    backend = NumpyBackend(N)
    circuit = _wrap(IQP + HL + PHASES)
    assert certify(circuit, N) == certify(_wrap(IQP), N)
    np.testing.assert_allclose(backend.gram(circuit, X), backend.gram(_wrap(IQP), X),
                               atol=1e-12)
    np.testing.assert_allclose(classical_gram(circuit, X, N), backend.gram(circuit, X),
                               atol=1e-10)
    assert certify(_wrap(IQP + HL + PHASES + (("rzz", (0, 3), (0, 3)),)), N) == frozenset()


def test_gates_touching_a_zero_qubit_are_reduced():
    X = _inputs()
    layer = ((("rz", (2,), (2,)), ("rz", (3,), (3,))) + HL
             + (("rzz", (0, 3), (0, 3)), ("cs", (1, 2), None), ("rz", (2,), (2,))))
    circuit = _wrap(layer)
    assert certify(circuit, N) == {"product"}
    np.testing.assert_allclose(classical_gram(circuit, X, N),
                               NumpyBackend(N).gram(circuit, X), atol=1e-10)


def test_layered_gate_set_limits_layers():
    gs = LayeredGateSet(iqp_gate_set(N, N), max_layers=2)
    assert HADAMARD_LAYER_OP not in gs.placements(())
    first = gs.placements(((("rz", (0,), (0,))),))
    assert HADAMARD_LAYER_OP in first
    split = (("rz", (0,), (0,)), HADAMARD_LAYER_OP)
    assert HADAMARD_LAYER_OP not in gs.placements(split)
    second = split + (("rzz", (0, 1), (0, 1)),)
    assert HADAMARD_LAYER_OP not in gs.placements(second)
    assert layer_count(second) == 2
    assert HADAMARD_LAYER_OP not in LayeredGateSet(iqp_gate_set(N, N), 1).catalog


def test_layered_gate_set_validates_arguments():
    with pytest.raises(ValueError):
        LayeredGateSet(iqp_gate_set(N, N), max_layers=0)
    with pytest.raises(ValueError):
        LayeredGateSet(pauli_gate_set(N, N), max_layers=2)


def test_layered_gate_set_accepts_extra_diagonal_gates():
    from qksearch.gates import ENTANGLER_SPECS, ProductGateSet

    base = ProductGateSet([GateSpec("rz", 1, uses_feature=True), ENTANGLER_SPECS["rzz"],
                           GateSpec("t", 1, uses_feature=False),
                           GateSpec("cs", 2, uses_feature=False, symmetric=True)], N, N)
    gs = LayeredGateSet(base, max_layers=2)
    assert len(gs.catalog) == 4 + 6 + 4 + 6 + 1


@pytest.mark.parametrize("rule", ["iqp", "matchgate"])
def test_simulators_scale_polynomially(rule):
    n = 40
    rng = np.random.default_rng(0)
    layer = _random_iqp(rng, n, 60) if rule == "iqp" else _random_matchgate(rng, n, 60)
    circuit = _wrap(layer, n)
    assert rule in certify(circuit, n)
    K = classical_gram(circuit, _inputs(5, n), n, samples=500)
    assert K.shape == (5, 5)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-8)
    assert np.all((K >= -1e-12) & (K <= 1.0 + 1e-8))


def test_magic_of_reference_states():
    zero = np.eye(16, dtype=complex)[:1]
    t_state = np.array([[1.0, np.exp(1j * np.pi / 4)]]) / np.sqrt(2.0)
    np.testing.assert_allclose(stabilizer_renyi_entropy(zero), [0.0], atol=1e-12)
    np.testing.assert_allclose(stabilizer_renyi_entropy(t_state), [np.log2(4 / 3)])
    plus = np.full((1, 16), 0.25, dtype=complex)
    np.testing.assert_allclose(stabilizer_renyi_entropy(plus), [0.0], atol=1e-12)


def test_penalty_marks_only_certified_circuits():
    penalty = SimulabilityPenalty(N, weight=3.0)
    assert penalty(_wrap(IQP)) == -3.0
    assert penalty(_wrap(UNCERTIFIED)) == 0.0


@pytest.mark.parametrize("kwargs", [{"n_qubits": 0}, {"n_qubits": 2, "weight": -1.0}])
def test_penalty_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        SimulabilityPenalty(**kwargs)


def test_penalty_steers_a_search_away_from_certified_circuits():
    n = 3
    raw = sine_dataset(n_samples=20, n_features=n, noise=0.05, seed=0)
    data = AngleScaler.fit(raw.X).apply(raw)

    def best(regularizer):
        heuristic = IqpKernelHeuristic(pauli_gate_set(n, n), NumpyBackend(n), data,
                                       TargetAlignment(), max_depth=4, n_qubits=n,
                                       regularizer=regularizer)
        state = search((), heuristic, BestFirstSearch(), budget=30).best_state
        return heuristic.circuit(state)

    assert certify(best(None), n)
    assert certify(best(SimulabilityPenalty(n, weight=10.0)), n) == frozenset()


def test_encoding_spectrum_groups_gates_by_encoded_variable():
    layer = IQP + PHASES + HL + (("rzz", (0, 1), (0, 1)), ("rzzh", (0, 1), (0, 1)))
    assert encoding_spectrum(layer) == {("prod", (0,)): 1, ("prod", (0, 1)): 2,
                                        ("prod", (1, 2)): 1, ("prod", (0, 2)): 1,
                                        ("prod", (2,)): 1, ("havlicek", (0, 1)): 1}


def test_rff_frequencies_stay_in_the_kernel_spectrum():
    layer = IQP + HL + IQP + (("rz", (0,), (0,)),)
    variables, omegas = rff_frequencies(_wrap(layer), 4096, seed=3)
    bounds = np.array([encoding_spectrum(layer)[v] for v in variables])
    assert omegas.shape == (4096, len(variables))
    assert np.all(np.abs(omegas) <= bounds)
    assert np.allclose(omegas.mean(axis=0), 0.0, atol=0.1)


def test_rff_gram_is_a_unit_diagonal_kernel():
    X = _inputs()
    K = rff_gram(_wrap(IQP + HL + IQP), X, n_features=256)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    np.testing.assert_allclose(rff_gram(_wrap(IQP), X[:5], X, n_features=256),
                               rff_gram(_wrap(IQP), X, n_features=256)[:5])


@pytest.mark.parametrize("layer", [
    FIELDS,
    (("rzz", (0, 1), (0, 1)), ("rz", (0,), (0,)), ("rz", (3,), (3,))),
])
def test_rff_converges_to_kernels_with_independent_parities(layer):
    circuit = _wrap(layer)
    X = _inputs()
    samples = 100000
    estimate = rff_gram(circuit, X, n_features=samples, seed=1)
    assert np.abs(estimate - NumpyBackend(N).gram(circuit, X)).max() < 5 / np.sqrt(samples)


def test_rff_gram_without_encoding_gates_is_constant():
    np.testing.assert_allclose(rff_gram(_wrap(PHASES), _inputs(), n_features=8), 1.0)
