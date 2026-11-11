import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.dla import DlaCalculator
from qksearch.enumerate_circuits import (
    DIM_MAX,
    MAX_DEPTH,
    N_QUBITS,
    enumerate_states,
    gate_set,
    sine_split,
)
from qksearch.iqp import IqpKernelHeuristic
from qksearch.metric import TargetAlignment

train, _ = sine_split()
gs = gate_set()
states = enumerate_states(gs)
h = IqpKernelHeuristic(gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
                       metric=TargetAlignment(), max_depth=MAX_DEPTH,
                       n_qubits=N_QUBITS, regularizer=None)
dla = DlaCalculator()

kta = np.array([h.kta(s) for s in states])
d = np.array([dla.dimension(h.circuit(s)) for s in states], float) / DIM_MAX

iqr_kta = np.subtract(*np.percentile(kta, [75, 25]))
iqr_d = np.subtract(*np.percentile(d, [75, 25]))
print(f"KTA   : min={kta.min():+.4f} max={kta.max():+.4f} "
      f"range={np.ptp(kta):.4f} std={kta.std():.4f} iqr={iqr_kta:.4f}")
print(f"d/dmax: min={d.min():.4f} max={d.max():.4f} "
      f"range={np.ptp(d):.4f} std={d.std():.4f} iqr={iqr_d:.4f}")
print(f"\nlambda (ranges)   = {np.ptp(kta)/np.ptp(d):.4f}")
print(f"lambda (std)      = {kta.std()/d.std():.4f}")
print(f"lambda (IQR)      = {iqr_kta / iqr_d:.4f}")
print(f"correlation KTA/DLA = {np.corrcoef(kta, d)[0,1]:+.3f}")

order0 = np.argsort(-kta)
print("\nlambda | top-3 (index) | rank of the best-KTA | #distinct dims in top-5")
for lam in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    s = kta - lam * d
    o = np.argsort(-s)
    print(f"{lam:5.2f}  | {o[:3]} | {list(o).index(order0[0])+1:2d} | "
          f"dims top5 = {[int(d[i]*DIM_MAX) for i in o[:5]]} "
          f"kta top5 = {[round(float(kta[i]),3) for i in o[:5]]}")
