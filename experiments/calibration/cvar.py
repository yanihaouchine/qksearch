import numpy as np

from qksearch.backend import NumpyBackend
from qksearch.enumerate_circuits import MAX_DEPTH, N_QUBITS, enumerate_states, gate_set, sine_split
from qksearch.iqp import IqpKernelHeuristic
from qksearch.metric import TargetAlignment

train, _ = sine_split()
gs = gate_set()
states = enumerate_states(gs)
h = IqpKernelHeuristic(gate_set=gs, backend=NumpyBackend(N_QUBITS), data=train,
                       metric=TargetAlignment(), max_depth=MAX_DEPTH,
                       n_qubits=N_QUBITS, regularizer=None)
iu = np.triu_indices(train.n_samples, k=1)

kta = np.array([h.kta(s) for s in states])
var = np.array([h.evaluator.gram(s)[iu].var() for s in states])
b = 4.0 * var

def spread(a):
    return np.ptp(a), a.std(), np.subtract(*np.percentile(a, [75, 25]))

for name, a in (("KTA    ", kta), ("Var(K) ", var), ("4·Var  ", b)):
    p, s, i = spread(a)
    print(f"{name}: min={a.min():+.4f} max={a.max():+.4f} range={p:.4f} "
          f"std={s:.4f} iqr={i:.4f}")

pk, sk, ik = spread(kta)
pb, sb, ib = spread(b)
print(f"\nc (ranges)   = {pk/pb:.4f}")
print(f"c (std)      = {sk/sb:.4f}")
print(f"c (IQR)      = {ik/ib:.4f}")
print(f"correlation KTA/Var = {np.corrcoef(kta, var)[0,1]:+.3f}")

best0 = int(np.argmax(kta))
print(f"\nbest KTA alone: idx={best0} kta={kta[best0]:+.4f} var={var[best0]:.4f}")
print("\n   c  | winner  | kta      | Var(K) | change | top-3")
for c in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    s = kta + c * b
    o = np.argsort(-s)
    g = int(o[0])
    print(f"{c:5.2f} | {g:7d} | {kta[g]:+.4f}  | {var[g]:.4f} | "
          f"{str(g != best0):6s} | {list(o[:3])}")
