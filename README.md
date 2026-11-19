# ORCA Reproduction: Quantum Optimisation for Underwater 6G Networks

Independent implementation and evaluation of

> S. Sedaghat, "Osmotic-Driven Resilient Critical-Environment Awareness (ORCA): A 6G System Using Quantum Optimization for Underwater Survivability," *IEEE Access*, vol. 14, 2026. doi:10.1109/ACCESS.2026.3704385

Course project for **KBB5380 – Research Topics in Quantum Algorithm Design** (Başak Fulya Bilgeç).

## What is implemented

| File | Content | Paper |
|---|---|---|
| `config.py` | Simulation parameters (Table 3) and documented assumptions | Table 3 |
| `network.py` | 3-D underwater network, sub-THz path loss with molecular absorption, RIS, link reliability | Eq. 1–3, 10 |
| `problem.py` | SBA service chains, osmotic NF placement, full cost model H_total | Eq. 4–9, 12–16 |
| `qubo_builder.py` | QUBO with one-hot and slack-based capacity constraints, Ising mapping | Eq. 11, 17 |
| `solvers.py` | ORCA (simulated quantum annealing + hybrid refinement), SQA, SA, exact ILP, Greedy, GA, PSO, optional D-Wave | Sec. IV |
| `qaoa.py` | State-vector QAOA on reduced instances | Sec. III-L |
| `tests.py` | Proof that QUBO == objective == Ising and that the QUBO ground state is the true optimum | – |
| `experiments.py` | Main comparison (50 seeds), scaling, sensitivity, convergence, QAOA, ablation | Fig. 3–9 |
| `plots.py` | Figures and summary tables | – |

## Key results (50 random scenarios, N = 100 nodes)

| Method | Mean gap to exact optimum | SLA met |
|---|---|---|
| **ORCA** | **2.9 %** | 97 % |
| GA | 4.2 % | 97 % |
| Greedy | 7.7 % | 96 % |
| PSO | 24.2 % | 93 % |
| Classical SA | 64.8 % | 76 % |
| Annealing only (no refinement) | 94.3 % | 60 % |

* ORCA's near-optimality claim is confirmed and it scales better than the classical heuristics (3.6 % vs. 18.8 % for GA at 500 nodes).
* The ablation shows the quality comes from the classical refinement stage, not from the annealing stage.
* At the paper's absorption values (up to 2.5 dB/m) a sub-THz link reaches only ~12 m, so a 1 km³ network cannot be connected.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python tests.py                  # correctness checks
python experiments.py --quick    # fast version (a few minutes)
python experiments.py            # full version (~1 hour)
python plots.py                  # figures -> figures/, tables -> results/
```

The quantum annealer is emulated with path-integral Monte Carlo. `solvers.dwave_solve` runs the same QUBO on D-Wave Leap if `dwave-ocean-sdk` and an API token are available.