import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.dataset import AngleScaler, RegressionDataset, sine_dataset
from qksearch.dla import DlaCalculator
from qksearch.enumerate_circuits import (
    DIM_MAX,
    MAX_DEPTH,
    N_FEATURES,
    N_QUBITS,
    N_TEST,
    N_TRAIN,
    NOISE_REG,
    enumerate_states,
    gate_set,
)
from qksearch.iqp import IqpKernelHeuristic
from qksearch.metric import TargetAlignment

gs = gate_set()
states = enumerate_states(gs)
dla = DlaCalculator()
print("seed | lam(ptp) lam(std) lam(iqr) | c(ptp) c(std) c(iqr)")
for seed in range(6):
    raw = sine_dataset(n_samples=N_TRAIN+N_TEST, n_features=N_FEATURES, noise=NOISE_REG, seed=seed)
    d0 = AngleScaler.fit(raw.X[:N_TRAIN]).apply(raw)
    train = RegressionDataset(d0.X[:N_TRAIN], d0.y[:N_TRAIN])
    h = IqpKernelHeuristic(gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
                           metric=TargetAlignment(), max_depth=MAX_DEPTH,
                           n_qubits=N_QUBITS, regularizer=None)
    iu = np.triu_indices(train.n_samples, k=1)
    kta = np.array([h.kta(s) for s in states])
    dd  = np.array([dla.dimension(h.circuit(s)) for s in states], float)/DIM_MAX
    b   = 4.0*np.array([h.evaluator.gram(s)[iu].var() for s in states])
    f = lambda a: (np.ptp(a), a.std(), np.subtract(*np.percentile(a,[75,25])))
    K, D, B = f(kta), f(dd), f(b)
    print(f"{seed:4d} | {K[0]/D[0]:8.2f} {K[1]/D[1]:8.2f} {K[2]/D[2]:8.2f} | "
          f"{K[0]/B[0]:6.2f} {K[1]/B[1]:6.2f} {K[2]/B[2]:6.2f}")
