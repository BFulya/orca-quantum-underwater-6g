"""Step 5 check: run every optimiser on the same scenario and compare with the exact optimum."""
from config import Config
from problem import build_problem
from qubo_builder import build_qubo
import solvers as S

prob = build_problem(Config(), seed=0)
qubo = build_qubo(prob)

runs = {
    "ILP (exact)": lambda: S.ilp_solve(prob),
    "ORCA (SQA + refinement)": lambda: S.orca_hybrid_solve(prob, qubo, seed=0),
    "SQA only (no refinement)": lambda: S.sqa_solve(prob, qubo, seed=0),
    "SA (classical, QUBO)": lambda: S.sa_solve(prob, qubo, seed=0),
    "Greedy": lambda: S.greedy_solve(prob, qubo),
    "GA": lambda: S.ga_solve(prob, qubo, seed=0),
    "PSO": lambda: S.pso_solve(prob, qubo, seed=0),
}
results = {name: fn() for name, fn in runs.items()}
opt = prob.objective(results["ILP (exact)"]["assignment"])

print(f"QUBO: {qubo.n} variables | exact optimum H* = {opt:.3f}\n")
print(f"{'method':<26}{'H_total':>9}{'gap %':>8}{'lat ms':>8}{'reliab.':>9}{'SLA %':>7}"
      f"{'cap.viol':>9}{'1-hot viol':>11}{'time s':>8}")
for name, r in results.items():
    m = prob.metrics(r["assignment"])
    gap = 100 * (m["objective"] - opt) / opt
    print(f"{name:<26}{m['objective']:9.3f}{gap:8.2f}{m['latency_ms']:8.2f}{m['reliability']:9.4f}"
          f"{m['sla_ratio']*100:7.0f}{m['cap_viol']:9d}{r['violations']:11d}{r['time_s']:8.2f}")