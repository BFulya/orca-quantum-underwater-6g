"""Step 4 check: size and structure of the ORCA QUBO for one scenario."""
import numpy as np

from config import Config
from problem import build_problem
from qubo_builder import build_qubo, to_ising

prob = build_problem(Config(), seed=0)
qubo = build_qubo(prob)
kinds = [nm[0] for nm in qubo.names]
print(f"binary variables: {qubo.n}  =  x (paths) {kinds.count('x')} + z (placements) {kinds.count('z')}"
      f" + y (slices) {kinds.count('y')} + slack {kinds.count('s')}")
print(f"one-hot groups: {len(qubo.groups)} | capacity-constrained nodes: {len(qubo.cap)}"
      f" | quadratic couplers: {qubo.n_quadratic}"
      f" | density: {qubo.n_quadratic / (qubo.n * (qubo.n - 1) / 2) * 100:.1f} %")
core = build_qubo(prob, with_penalty=False)
c = np.abs(core.U[core.U != 0])
print(f"objective coefficients |c|, |Q|: min {c.min():.4f}  median {np.median(c):.3f}  max {c.max():.2f}")
print(f"one-hot penalties A_g: min {min(qubo.penalties):.1f}  median {np.median(qubo.penalties):.1f}"
      f"  max {max(qubo.penalties):.1f}")
h, J, const = to_ising(qubo)
allc = np.abs(np.concatenate([h, J[J != 0]]))
print(f"Ising: {len(h)} spins, {np.count_nonzero(J)} couplings, dynamic range max/min = "
      f"{allc.max() / allc[allc > 0].min():.0f}")