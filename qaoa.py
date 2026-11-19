"""
Gate-model route of Section III-L: QAOA on the ORCA Ising Hamiltonian.

A NumPy state-vector simulator (no external quantum SDK needed). For n qubits
the diagonal cost operator C = H_QUBO(q) is tabulated over all 2^n basis
states, so this is only for reduced instances (the paper likewise runs
10-50-variable instances directly on quantum hardware).

|psi(gamma, beta)> = prod_{l=1..p} exp(-i beta_l sum X) exp(-i gamma_l C) |+>^n
"""
import time

import numpy as np
from scipy.optimize import minimize

from qubo_builder import QUBO


def basis_energies(qubo: QUBO):
    n = qubo.n
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return qubo.energy(bits), bits


def _apply_mixer(psi, beta, n):
    """exp(-i beta X) on every qubit (RX(2 beta))."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for k in range(n):
        psi = np.moveaxis(psi, k, 0)
        a, b = psi[0].copy(), psi[1].copy()
        psi[0], psi[1] = c * a + s * b, s * a + c * b
        psi = np.moveaxis(psi, 0, k)
    return psi.reshape(-1)


def qaoa_state(cost, n, gammas, betas):
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * cost)
        psi = _apply_mixer(psi, b, n)
    return psi


def run_qaoa(qubo: QUBO, feasible_mask, p=1, restarts=8, seed=0, maxiter=300):
    """Optimise <C> with COBYLA from several random starts. Returns metrics of the
    best parameter set: probability of the ground state, of any one-hot-feasible
    state, and the approximation ratio of the expected energy."""
    rng = np.random.default_rng(seed)
    E, _ = basis_energies(qubo)
    n = qubo.n
    e_min, e_max = E.min(), E.max()
    cost = (E - e_min) / (e_max - e_min)          # normalised to [0, 1]
    ground = np.isclose(E, e_min)
    t0 = time.perf_counter()

    def f(x):
        psi = qaoa_state(cost, n, x[:p], x[p:])
        return float(np.real(np.vdot(psi, cost * psi)))

    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
        r = minimize(f, x0, method="COBYLA", options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    psi = qaoa_state(cost, n, best.x[:p], best.x[p:])
    prob = np.abs(psi) ** 2
    exp_e = float(prob @ E)
    # 1000-shot sampling: probability that the best of the shots is the optimum
    shots = rng.choice(len(prob), size=1000, p=prob / prob.sum())
    return dict(p=p, n_qubits=n, p_ground=float(prob[ground].sum()),
                p_feasible=float(prob[feasible_mask].sum()),
                expected_energy=exp_e, ground_energy=float(e_min),
                approx_ratio=float((e_max - exp_e) / (e_max - e_min)),
                shots_hit_optimum=bool(np.isclose(E[shots].min(), e_min)),
                params=best.x.tolist(), time_s=time.perf_counter() - t0,
                uniform_p_ground=float(ground.mean()),
                uniform_p_feasible=float(feasible_mask.mean()))


def feasible_mask(qubo: QUBO, bits):
    ok = np.ones(bits.shape[0], dtype=bool)
    for g in qubo.groups:
        ok &= bits[:, g].sum(axis=1) == 1
    return ok