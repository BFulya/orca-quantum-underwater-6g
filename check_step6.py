"""Step 6 check: gate-model QAOA on a reduced ORCA instance (14 qubits)."""
from config import Config
from problem import build_problem
from qubo_builder import build_qubo
from qaoa import basis_energies, feasible_mask, run_qaoa
import solvers as S

# 1 service, 3 candidate paths, 3 NFs, 3 candidate hosts, 2 slices -> 3 + 9 + 2 = 14 qubits
cfg = Config(n_nodes=30, n_services=1, k_paths=3, chain_len=(3, 3), max_hosts_per_nf=3)
prob = build_problem(cfg, seed=0)
qubo = build_qubo(prob)
E, bits = basis_energies(qubo)
fm = feasible_mask(qubo, bits)
opt = prob.objective(S.ilp_solve(prob)["assignment"])

print(f"qubits: {qubo.n} | Hilbert space: {2 ** qubo.n} states | one-hot feasible: {fm.sum()}"
      f" | ground energy {E.min():.4f} (ILP optimum {opt:.4f})")
print(f"uniform superposition: P(optimum) = {(E <= E.min() + 1e-9).mean():.5f}, "
      f"P(feasible) = {fm.mean():.4f}\n")
print(f"{'p':>2}{'P(optimum)':>12}{'P(feasible)':>13}{'approx. ratio':>15}{'best of 1000 shots = opt':>27}{'time s':>8}")
for p in (1, 2, 3):
    r = run_qaoa(qubo, fm, p=p, restarts=4, seed=0)
    print(f"{p:>2}{r['p_ground']:12.5f}{r['p_feasible']:13.4f}{r['approx_ratio']:15.4f}"
          f"{str(r['shots_hit_optimum']):>27}{r['time_s']:8.1f}")

sqa = S.sqa_solve(prob, qubo, seed=0)
orca = S.orca_hybrid_solve(prob, qubo, seed=0)
print(f"\nSQA only : objective {prob.objective(sqa['assignment']):.4f}")
print(f"ORCA     : objective {prob.objective(orca['assignment']):.4f}")