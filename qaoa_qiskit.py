"""
QAOA for ORCA built as a Qiskit circuit (Section III-L, gate-model route).

The Ising Hamiltonian of Eq. (17),  H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,
is turned into the standard QAOA ansatz:

    |+>^n  ->  [ RZ(2 gamma h_i / D) on every qubit,
                 RZZ(2 gamma J_ij / D) on every coupled pair,
                 RX(2 beta) on every qubit ]  x p layers

(D = E_max - E_min normalises the cost exactly as in qaoa.py.)

The script
  1. optimises (gamma, beta) with the NumPy simulator of qaoa.py,
  2. builds the same circuit in Qiskit and checks that both give the same state,
  3. samples 1000 shots and decodes the best measured bit string,
  4. reports circuit resources (logical gates, and after transpiling to a
     CX + single-qubit basis on all-to-all and on linear hardware),
  5. saves the circuit drawing to figures/fig7_qaoa_circuit.png.

Requires:  pip install qiskit pylatexenc
"""
import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from config import Config
from problem import build_problem
from qaoa import basis_energies, feasible_mask, qaoa_state, run_qaoa
from qubo_builder import build_qubo, decode, to_ising

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)


def qaoa_circuit(h, J, p, scale=1.0, measure=False):
    """Parameterised QAOA circuit for E = h.Z + sum_{i<j} J_ij Z_i Z_j (energies / scale)."""
    n = len(h)
    gamma = ParameterVector("γ", p)
    beta = ParameterVector("β", p)
    qc = QuantumCircuit(n, name="ORCA-QAOA")
    qc.h(range(n))
    pairs = [(int(i), int(j)) for i, j in zip(*np.nonzero(J))]
    for layer in range(p):
        qc.barrier()
        for i in range(n):                                  # cost layer, single-qubit part
            if h[i] != 0:
                qc.rz(2 * gamma[layer] * float(h[i] / scale), i)
        for i, j in pairs:                                  # cost layer, two-qubit part
            qc.rzz(2 * gamma[layer] * float(J[i, j] / scale), i, j)
        qc.barrier()
        for i in range(n):                                  # mixer layer
            qc.rx(2 * beta[layer], i)
    if measure:
        qc.measure_all()
    return qc, gamma, beta


def main():
    # ---------------------------------------------- reduced ORCA instance (14 qubits)
    cfg = Config(n_nodes=30, n_services=1, k_paths=3, chain_len=(3, 3), max_hosts_per_nf=3)
    prob = build_problem(cfg, seed=0)
    qubo = build_qubo(prob)
    E, bits = basis_energies(qubo)
    fm = feasible_mask(qubo, bits)
    h, J, _ = to_ising(qubo)
    D = E.max() - E.min()
    p = 2

    # 1) optimise the angles with the fast NumPy simulator
    res = run_qaoa(qubo, fm, p=p, restarts=4, seed=0)
    angles = np.array(res["params"])
    print(f"instance: {qubo.n} qubits | optimum energy {E.min():.4f} | p = {p}")
    print(f"optimised angles: gamma = {np.round(angles[:p], 4)}, beta = {np.round(angles[p:], 4)}")

    # 2) same circuit in Qiskit, compared with the NumPy state
    qc, gamma, beta = qaoa_circuit(h, J, p, scale=D)
    bound = qc.assign_parameters({gamma: angles[:p], beta: angles[p:]})
    psi_qiskit = Statevector(bound).data
    psi_numpy = qaoa_state((E - E.min()) / D, qubo.n, angles[:p], angles[p:])
    fidelity = abs(np.vdot(psi_numpy, psi_qiskit)) ** 2
    prob_q = np.abs(psi_qiskit) ** 2
    print(f"Qiskit vs NumPy state fidelity: {fidelity:.12f}")
    print(f"P(optimum) = {prob_q[np.isclose(E, E.min())].sum():.5f} | "
          f"P(feasible) = {prob_q[fm].sum():.4f}")

    # 3) sample 1000 shots and decode the best measured bit string
    counts = Statevector(bound).sample_counts(shots=1000)
    best_idx = min((int(s, 2) for s in counts), key=lambda k: E[k])
    a, viol = decode(prob, qubo, bits[best_idx])
    print(f"1000 shots: {len(counts)} distinct outcomes | best energy {E[best_idx]:.4f} "
          f"(optimum {E.min():.4f}) | one-hot violations {viol} | decoded: {a}")

    # 4) circuit resources
    ops = dict(bound.count_ops())
    print(f"logical circuit: depth {bound.depth()} | gates {ops}")
    for name, cmap in (("all-to-all", None), ("linear chain", CouplingMap.from_line(qubo.n))):
        t = transpile(bound, basis_gates=["cx", "rz", "sx", "x"], coupling_map=cmap,
                      optimization_level=1, seed_transpiler=0)
        o = dict(t.count_ops())
        print(f"transpiled ({name:>12}): depth {t.depth():4d} | CX {o.get('cx', 0):4d} | "
              f"1-qubit {o.get('rz', 0) + o.get('sx', 0) + o.get('x', 0):4d}")

    # full-size ORCA instance (N = 100): logical size of one QAOA layer
    full = build_qubo(build_problem(Config(), seed=0))
    hf, Jf, _ = to_ising(full)
    qf, _, _ = qaoa_circuit(hf, Jf, 1)
    of = dict(qf.count_ops())
    print(f"full ORCA instance (N = 100): {full.n} qubits | one layer needs "
          f"{of.get('rzz', 0)} RZZ + {of.get('rz', 0)} RZ + {of.get('rx', 0)} RX gates")

    # 5) circuit drawing (p = 1 for readability)
    qc1, _, _ = qaoa_circuit(h, J, 1, scale=D)
    path = os.path.join(FIG, "fig7_qaoa_circuit.png")
    try:
        fig = qc1.draw(output="mpl", fold=40, scale=0.6)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("wrote figures/fig7_qaoa_circuit.png")
    except Exception as exc:                                # pylatexenc missing, etc.
        print(f"(mpl drawing unavailable: {exc}); text drawing follows")
        print(qc1.draw(output="text", fold=120))


if __name__ == "__main__":
    main()