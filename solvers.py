"""
Optimisers compared in the ORCA reproduction (Section IV-D).

QUBO-based (operate on H_QUBO):
  orca_hybrid_solve  ORCA: simulated quantum annealing (path-integral Monte
                 Carlo of the transverse-field Ising model, a classical
                 emulation of the annealer) + classical one-hot refinement,
                 mirroring the hybrid D-Wave Leap workflow of the paper.
  sqa_solve      the annealing stage alone (analogue of a direct QPU call).
  dwave_solve    the paper's actual backend (D-Wave Leap hybrid / QPU); used
                 only when dwave-ocean-sdk and an API token are available.
  sa_solve       classical simulated annealing on the same QUBO
                 (equivalent to D-Wave's `neal` SimulatedAnnealingSampler).
  ilp_solve      exact optimum: linearised binary program solved by HiGHS
                 (scipy.optimize.milp) with explicit one-hot constraints.
Direct-encoding baselines (one choice per decision group, always one-hot):
  greedy_solve, ga_solve (genetic algorithm), pso_solve (particle swarm).

Every solver returns a dict with the decoded assignment, number of violated
one-hot groups, wall-clock time and (where meaningful) a convergence trace.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from qubo_builder import QUBO, build_qubo, decode, encode, to_ising


# =============================================================== helpers
def _result(prob, qubo, q, t, trace=None, extra=None):
    a, viol = decode(prob, qubo, q)
    out = dict(assignment=a, violations=viol, time_s=t, trace=trace,
               qubo_energy=float(qubo.projected_energy(np.asarray(q)[: qubo.n_main])[0]))
    if extra:
        out.update(extra)
    return out


def _onehot_matrix(qubo, choice):
    """choice: (R, n_groups) integer -> (R, n) binary matrix."""
    R = choice.shape[0]
    q = np.zeros((R, qubo.n_main))
    for g, idx in enumerate(qubo.groups):
        q[np.arange(R), np.asarray(idx)[choice[:, g]]] = 1.0
    return q


# ============================================ simulated annealing (classical)
def auto_beta(U):
    """Beta range as in D-Wave neal: the hottest beta accepts the largest
    possible energy increase with prob. 1/2, the coldest accepts the smallest
    non-zero increase with prob. 1/100."""
    A = np.abs(U)
    rowsum = (A + A.T - np.diag(np.diag(A))).sum(axis=1)
    nz = A[A > 1e-3 * A.max()]
    return np.log(2.0) / rowsum.max(), np.log(100.0) / nz.min()


def sa_solve(prob, qubo: QUBO, reads=16, sweeps=300, seed=0, beta=None):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    n = qubo.n
    scale = np.abs(qubo.U).max()
    U = qubo.U / scale
    lin = np.diag(U).copy()
    W = np.triu(U, 1)
    W = W + W.T                                    # symmetric couplings
    q = (rng.random((reads, n)) < 0.5).astype(float)
    field = q @ W                                   # local fields per read
    beta = beta or auto_beta(U)
    betas = np.geomspace(beta[0], beta[1], sweeps)
    e = np.einsum("ri,ij,rj->r", q, np.triu(U, 1), q) + q @ lin
    best_e, best_q = e.copy(), q.copy()
    trace = np.empty(sweeps)
    for t, b in enumerate(betas):
        for i in rng.permutation(n):
            d = (1.0 - 2.0 * q[:, i]) * (lin[i] + field[:, i])
            acc = (d <= 0) | (rng.random(reads) < np.exp(-b * np.clip(d, 0, None)))
            if acc.any():
                sign = (1.0 - 2.0 * q[acc, i])[:, None]
                field[acc] += sign * W[i]
                q[acc, i] = 1.0 - q[acc, i]
                e[acc] += d[acc]
        better = e < best_e
        best_e[better], best_q[better] = e[better], q[better]
        trace[t] = best_e.min() * scale + qubo.offset
    r = int(np.argmin(best_e))
    return _result(prob, qubo, best_q[r], time.perf_counter() - t0, trace)


# ================================== simulated quantum annealing (ORCA core)
def _sqa_samples(qubo, rng, reads, slices, sweeps, temperature, gamma):
    """Path-integral Monte Carlo of H(t) = -Gamma(t) sum X_i + H_Ising.

    Trotter decomposition with P slices; replicas of the same spin are coupled
    by J_perp = -(P T / 2) ln tanh(Gamma / (P T)). Gamma is annealed from
    gamma[0] to gamma[1]; energies are normalised to max |coefficient| = 1.
    Returns the best slice of every read (as binary q) and a convergence trace.
    """
    h, J, _ = to_ising(qubo)
    scale = max(np.abs(h).max(), np.abs(J).max())
    h, J = h / scale, J / scale
    Js = J + J.T
    n, P, T = qubo.n, slices, temperature
    B = reads * P
    s = rng.choice([-1.0, 1.0], size=(reads, P, n))
    field = (s.reshape(B, n) @ Js).reshape(reads, P, n)
    gammas = np.geomspace(gamma[0], gamma[1], sweeps)
    best_e, best_s = np.inf, None
    trace = np.empty(sweeps)
    for t, G in enumerate(gammas):
        jperp = -0.5 * P * T * np.log(np.tanh(G / (P * T)))
        for i in rng.permutation(n):
            # checkerboard over Trotter slices: neighbours are fixed while a parity class updates
            for par in (0, 1):
                si = s[:, :, i]
                nb = np.roll(si, 1, axis=1) + np.roll(si, -1, axis=1)
                d = -2.0 * si * (h[i] + field[:, :, i]) / P + 2.0 * jperp * si * nb
                acc = (d <= 0) | (rng.random((reads, P)) < np.exp(-np.clip(d, 0, None) / T))
                acc[:, (1 - par)::2] = False
                if acc.any():
                    r_idx, p_idx = np.nonzero(acc)
                    delta = -2.0 * si[r_idx, p_idx]
                    field[r_idx, p_idx, :] += delta[:, None] * Js[i]
                    s[r_idx, p_idx, i] = -si[r_idx, p_idx]
        flat = s.reshape(B, n)
        e = flat @ h + 0.5 * np.einsum("bi,bi->b", flat, field.reshape(B, n))
        k = int(np.argmin(e))
        if e[k] < best_e:
            best_e, best_s = e[k], flat[k].copy()
        trace[t] = qubo.energy((1 - best_s) / 2)
    # final read-out: best slice of each read
    e = (s.reshape(B, n) @ h + 0.5 * np.einsum("bi,bi->b", s.reshape(B, n),
                                               field.reshape(B, n))).reshape(reads, P)
    picks = [(1 - s[r, int(np.argmin(e[r]))]) / 2 for r in range(reads)]
    picks.append((1 - best_s) / 2)
    return picks, trace


def sqa_solve(prob, qubo: QUBO, reads=8, slices=16, sweeps=300, seed=0,
              gamma=(3.0, 1e-3), temperature=0.01):
    """Pure (quantum-annealing-only) solver: best SQA sample, no classical refinement.
    Analogue of sending the QUBO directly to the QPU."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    samples, trace = _sqa_samples(qubo, rng, reads, slices, sweeps, temperature, gamma)
    q = min(samples, key=qubo.energy)
    return _result(prob, qubo, q, time.perf_counter() - t0, trace)


# ====================================== classical post-processing (hybrid)
def onehot_descent(qubo: QUBO, q, max_passes=50):
    """Steepest descent over one-hot group re-assignments (move the single active
    variable of a group to another option). Every move keeps the state one-hot,
    so the penalty barrier that freezes single-flip annealing is never crossed;
    slack registers are kept at their optimal value (capacity term evaluated
    exactly). This is the classical stage of the hybrid workflow."""
    nm = qubo.n_main
    q = np.asarray(q, dtype=float)[:nm].copy()
    lin = np.diag(qubo.U_core).copy()
    for g in qubo.groups:                     # start from a one-hot state
        on = [i for i in g if q[i] > 0.5]
        keep = min(on, key=lambda i: lin[i]) if on else min(g, key=lambda i: lin[i])
        q[g] = 0.0
        q[keep] = 1.0
    W = np.triu(qubo.U_core, 1)
    W = W + W.T
    field = W @ q
    # capacity bookkeeping: variable -> [(row, units)], row loads
    rows_of = {}
    for r, (idx, u, C, lam, _, _) in enumerate(qubo.cap):
        for i, w in zip(idx, u):
            rows_of.setdefault(int(i), []).append((r, w))
    load = np.array([float(q[idx] @ u) for idx, u, *_ in qubo.cap])
    Cs = np.array([c[2] for c in qubo.cap], dtype=float)
    lams = np.array([c[3] for c in qubo.cap], dtype=float)

    def pen(r, L):
        return lams[r] * max(0.0, L - Cs[r]) ** 2

    for _ in range(max_passes):
        improved = False
        for g in qubo.groups:
            g = np.asarray(g)
            i = int(g[np.argmax(q[g])])
            delta = (lin[g] + field[g] - W[i, g]) - (lin[i] + field[i])
            if qubo.cap:
                for k, j in enumerate(g):
                    j = int(j)
                    if j == i:
                        continue
                    ch = {}
                    for r, w in rows_of.get(i, []):
                        ch[r] = ch.get(r, 0.0) - w
                    for r, w in rows_of.get(j, []):
                        ch[r] = ch.get(r, 0.0) + w
                    delta[k] += sum(pen(r, load[r] + d) - pen(r, load[r]) for r, d in ch.items())
            delta[g == i] = 0.0
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                j = int(g[k])
                q[i], q[j] = 0.0, 1.0
                field += W[:, j] - W[:, i]
                for r, w in rows_of.get(i, []):
                    load[r] -= w
                for r, w in rows_of.get(j, []):
                    load[r] += w
                improved = True
        if not improved:
            break
    return qubo.with_slack(q)


def orca_hybrid_solve(prob, qubo: QUBO, seed=0, reads=8, slices=16, sweeps=300,
                      temperature=0.01, gamma=(3.0, 1e-3)):
    """ORCA solver: quantum-annealing stage (SQA emulation of the annealer)
    whose final samples are refined classically (one-hot descent), mirroring
    the hybrid quantum-classical workflow of D-Wave Leap used in the paper."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    samples, trace = _sqa_samples(qubo, rng, reads, slices, sweeps, temperature, gamma)
    best_q, best_e = None, np.inf
    for qs in samples:
        qq = onehot_descent(qubo, qs)
        e = float(qubo.projected_energy(qq)[0])
        if e < best_e:
            best_q, best_e = qq, e
    trace = np.append(trace, best_e)
    return _result(prob, qubo, best_q, time.perf_counter() - t0, trace,
                   dict(raw_best=float(min(qubo.energy(x) for x in samples))))


# ======================================================== D-Wave (optional)
def dwave_available():
    try:
        import dwave.system  # noqa: F401
        from dwave.cloud import Client
        with Client.from_config() as c:
            return c.config.token is not None
    except Exception:
        return False


def dwave_solve(prob, qubo: QUBO, solver="hybrid", time_limit=5, num_reads=100):
    """Submit the QUBO to D-Wave Leap (needs `pip install dwave-ocean-sdk` and
    `dwave setup` with an API token). Not executed in the reference runs."""
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    Q = qubo.to_dict()
    t0 = time.perf_counter()
    if solver == "hybrid":
        ss = LeapHybridSampler().sample_qubo(Q, time_limit=time_limit)
    else:
        ss = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=num_reads)
    best = ss.first.sample
    q = np.array([best.get(i, 0) for i in range(qubo.n)], dtype=float)
    info = dict(dwave_info=dict(ss.info))
    return _result(prob, qubo, q, time.perf_counter() - t0, None, info)


# ========================================================= exact (ILP)
def ilp_solve(prob, qubo_nopen: QUBO = None, time_limit=120.0):
    """Linearise q_i q_j with w_ij and solve exactly with HiGHS branch & bound.
    Positive coupler: w >= q_i + q_j - 1; negative coupler: w <= q_i, w <= q_j."""
    t0 = time.perf_counter()
    qb = qubo_nopen if qubo_nopen is not None else build_qubo(prob, with_penalty=False)
    n = qb.n
    pairs = list(zip(*np.nonzero(np.triu(qb.U, 1))))
    m = len(pairs)
    c = np.concatenate([np.diag(qb.U), [qb.U[i, j] for i, j in pairs]])
    capn = prob.cap_nodes()
    rows = len(qb.groups) + len(capn) + sum(1 if qb.U[i, j] > 0 else 2 for i, j in pairs)
    A = lil_matrix((rows, n + m))
    lb, ub = [], []
    r = 0
    for g in qb.groups:
        for i in g:
            A[r, i] = 1
        lb.append(1)
        ub.append(1)
        r += 1
    for j in capn:                     # node CPU capacity, hard constraint
        for s_, svc in enumerate(prob.services):
            for l in range(svc.n_nf):
                for mm, jj in enumerate(svc.hosts):
                    if jj == j:
                        A[r, qb.index[("z", s_, l, mm)]] = prob.units(s_, l)
        lb.append(-np.inf)
        ub.append(prob.cap_units(j))
        r += 1
    for k, (i, j) in enumerate(pairs):
        if qb.U[i, j] > 0:
            A[r, n + k] = 1
            A[r, i] = -1
            A[r, j] = -1
            lb.append(-1)
            ub.append(np.inf)
            r += 1
        else:
            for v in (i, j):
                A[r, n + k] = 1
                A[r, v] = -1
                lb.append(-np.inf)
                ub.append(0)
                r += 1
    integrality = np.concatenate([np.ones(n), np.zeros(m)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=integrality, bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit, disp=False))
    t = time.perf_counter() - t0
    if res.x is None:              # time limit reached before any feasible solution
        return None
    q = np.round(res.x[:n])
    out = _result(prob, qb, q, t, None, dict(optimal=(res.status == 0),
                                              mip_gap=getattr(res, "mip_gap", None)))
    return out


# ========================================================= greedy
def greedy_solve(prob, qubo: QUBO):
    """Sequential, myopic: service by service, path first, then NF by NF."""
    t0 = time.perf_counter()
    a = {"path": [], "host": [], "slice": []}
    placed = []
    load = {}
    lam = prob.cap_lambda()
    for s, svc in enumerate(prob.services):
        def path_cost(k):
            return prob.linear_path_cost(s, k) + sum(
                prob.path_pair_cost(s2, a["path"][s2], s, k) for s2 in range(s))
        k = min(range(len(svc.paths)), key=path_cost)
        sl = min(range(prob.cfg.n_slices), key=lambda x: prob.linear_slice_cost(s, x))
        hosts = []
        for l in range(svc.n_nf):
            def host_cost(m):
                j = svc.hosts[m]
                c = prob.linear_place_cost(s, l, m) + prob.path_place_cost(s, k, l, m)
                c += prob.slice_place_cost(s, sl, l, m)
                if l > 0 and hosts[-1] == m:
                    c -= prob.coloc_reward()
                c += sum(prob.node_pair_cost(s2, l2, s, l, j) for s2, l2, j2 in placed if j2 == j)
                L, C, u = load.get(j, 0), prob.cap_units(j), prob.units(s, l)
                c += lam * (max(0, L + u - C) ** 2 - max(0, L - C) ** 2)
                return c
            m = min(range(len(svc.hosts)), key=host_cost)
            hosts.append(m)
            placed.append((s, l, svc.hosts[m]))
            load[svc.hosts[m]] = load.get(svc.hosts[m], 0) + prob.units(s, l)
        a["path"].append(k)
        a["host"].append(hosts)
        a["slice"].append(sl)
    q = encode(prob, qubo, a)
    return _result(prob, qubo, q, time.perf_counter() - t0)


# ========================================================= genetic algorithm
def ga_solve(prob, qubo: QUBO, pop=40, gens=150, seed=0, pmut=None, elite=2):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    sizes = np.array([len(g) for g in qubo.groups])
    G = len(sizes)
    pmut = pmut or 1.0 / G
    P = (rng.random((pop, G)) * sizes).astype(int)
    fit = qubo.projected_energy(_onehot_matrix(qubo, P))
    trace = np.empty(gens)
    for t in range(gens):
        order = np.argsort(fit)
        P, fit = P[order], fit[order]
        trace[t] = fit[0]
        children = [P[:elite]]
        n_child = pop - elite

        def tour():
            i, j = rng.integers(pop, size=2)
            return P[i] if fit[i] < fit[j] else P[j]
        kids = np.empty((n_child, G), dtype=int)
        for c in range(n_child):
            p1, p2 = tour(), tour()
            mask = rng.random(G) < 0.5               # uniform crossover
            kids[c] = np.where(mask, p1, p2)
        mut = rng.random((n_child, G)) < pmut
        kids[mut] = (rng.random(mut.sum()) * np.broadcast_to(sizes, kids.shape)[mut]).astype(int)
        children.append(kids)
        P = np.vstack(children)
        fit = qubo.projected_energy(_onehot_matrix(qubo, P))
    b = int(np.argmin(fit))
    q = _onehot_matrix(qubo, P[b:b + 1])[0]
    return _result(prob, qubo, q, time.perf_counter() - t0, trace)


# ========================================================= particle swarm
def pso_solve(prob, qubo: QUBO, particles=40, iters=150, seed=0, w=0.7, c1=1.5, c2=1.5):
    """Random-key discrete PSO: one real score per (group, option); each group
    picks its arg-max option, so every particle is one-hot feasible."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    n = qubo.n_main

    def decode_keys(X):
        choice = np.empty((X.shape[0], len(qubo.groups)), dtype=int)
        for g, idx in enumerate(qubo.groups):
            choice[:, g] = np.argmax(X[:, idx], axis=1)
        return choice

    X = rng.random((particles, n))
    V = rng.normal(0, 0.1, (particles, n))
    fit = qubo.projected_energy(_onehot_matrix(qubo, decode_keys(X)))
    pbest, pfit = X.copy(), fit.copy()
    gb = int(np.argmin(pfit))
    trace = np.empty(iters)
    for t in range(iters):
        r1, r2 = rng.random((particles, n)), rng.random((particles, n))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (pbest[gb] - X)
        X = np.clip(X + V, 0.0, 1.0)
        fit = qubo.projected_energy(_onehot_matrix(qubo, decode_keys(X)))
        imp = fit < pfit
        pbest[imp], pfit[imp] = X[imp], fit[imp]
        gb = int(np.argmin(pfit))
        trace[t] = pfit[gb]
    q = _onehot_matrix(qubo, decode_keys(pbest[gb:gb + 1]))[0]
    return _result(prob, qubo, q, time.perf_counter() - t0, trace)