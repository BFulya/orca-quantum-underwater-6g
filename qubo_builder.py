"""
QUBO construction (Section III-F..K, Eq. 11-16) and Ising mapping (Eq. 17).

Binary variables (Section III-C):
  x[s,k]   choose candidate path k for service s
  z[s,l,m] place NF l of service s on candidate osmotic host m
  y[s,k]   map service s to SBA slice k
  s[j,b]   slack bits for the CPU capacity of node j (added here, see below)

H_QUBO(q) = q^T Q q + c^T q + const, stored as one upper-triangular matrix U
whose diagonal holds c (because q_i^2 = q_i for binary q).

Two constraint families are not written explicitly in the paper and are added
with the standard QUBO techniques:
  * one-hot (exactly one path / host / slice):  A_g (1 - sum_{i in g} q_i)^2
  * node capacity  sum_t u_t z_tj <= C_j:      lambda (sum_t u_t z_tj + sum_b c_b s_jb - C_j)^2
    with a bounded-binary slack register (c_b = 1, 2, 4, ..., remainder).
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from problem import Problem


@dataclass
class QUBO:
    U: np.ndarray          # (n, n) upper triangular, diagonal = linear terms
    offset: float          # constant term
    index: dict            # variable name -> position
    names: list            # position -> variable name
    groups: list           # one-hot groups (indices < n_main)
    penalty: float         # mean one-hot penalty A_g
    n_quadratic: int       # number of non-zero off-diagonal couplers
    n_main: int            # number of decision variables (x, z, y); slack bits follow
    U_core: np.ndarray     # (n_main, n_main) objective + one-hot penalties, no capacity
    core_offset: float
    cap: list = field(default_factory=list)    # [(z idx array, units, C, lambda, slack idx, coeffs)]
    penalties: list = None                     # per-group A_g

    @property
    def n(self):
        return len(self.names)

    @property
    def n_slack(self):
        return self.n - self.n_main

    def energy(self, q):
        """Full QUBO energy of one sample (1-D) or of a batch (2-D, rows = samples)."""
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            return float(q @ self.U @ q + self.offset)
        return np.einsum("ri,ij,rj->r", q, self.U, q) + self.offset

    def projected_energy(self, qm):
        """Energy of the decision variables with every slack register set optimally:
        core energy + lambda * max(0, load - C)^2 per capacity node."""
        qm = np.atleast_2d(np.asarray(qm, dtype=float))[:, : self.n_main]
        e = np.einsum("ri,ij,rj->r", qm, self.U_core, qm) + self.core_offset
        for idx, u, C, lam, _, _ in self.cap:
            e += lam * np.maximum(0.0, qm[:, idx] @ u - C) ** 2
        return e

    def with_slack(self, qm):
        """Complete a decision vector with optimal slack bits (full length n)."""
        q = np.zeros(self.n)
        q[: self.n_main] = np.asarray(qm, dtype=float)[: self.n_main]
        for idx, u, C, lam, sidx, coeffs in self.cap:
            v = int(max(0, C - round(float(q[idx] @ u))))
            q[sidx] = slack_bits(v, coeffs)
        return q

    def to_dict(self):
        """{(i, j): coeff} form used by D-Wave Ocean (dimod) samplers."""
        out = {}
        for i, j in zip(*np.nonzero(self.U)):
            out[(int(i), int(j))] = float(self.U[i, j])
        return out


def slack_coeffs(C):
    """Bounded binary encoding of an integer in [0, C]: 1, 2, 4, ..., remainder."""
    if C <= 0:
        return []
    k = int(np.floor(np.log2(C + 1)))
    coeffs = [2 ** b for b in range(k)]
    rest = C - (2 ** k - 1)
    if rest > 0:
        coeffs.append(rest)
    return coeffs


def slack_bits(v, coeffs):
    """Bits b with sum(b * coeffs) == v, for 0 <= v <= sum(coeffs)."""
    k = len(coeffs)
    has_rest = k > 0 and coeffs[-1] != 2 ** (k - 1)   # a remainder is never a power 2^(k-1)
    npow = k - 1 if has_rest else k
    bits = [0] * k
    if has_rest and v > 2 ** npow - 1:
        bits[-1] = 1
        v -= coeffs[-1]
    for b in range(npow):
        bits[b] = (v >> b) & 1
    return bits


def build_qubo(prob: Problem, with_penalty: bool = True, with_capacity: bool = None) -> QUBO:
    """with_penalty=False gives the bare objective used by the exact ILP, which
    then imposes one-hot and capacity as explicit linear constraints."""
    if with_capacity is None:
        with_capacity = with_penalty
    names, index, groups = [], {}, []

    def var(name):
        index[name] = len(names)
        names.append(name)
        return index[name]

    S = len(prob.services)
    for s, svc in enumerate(prob.services):
        groups.append([var(("x", s, k)) for k in range(len(svc.paths))])
        for l in range(svc.n_nf):
            groups.append([var(("z", s, l, m)) for m in range(len(svc.hosts))])
        groups.append([var(("y", s, k)) for k in range(prob.cfg.n_slices)])
    n_main = len(names)
    cap_nodes = prob.cap_nodes() if with_capacity else []
    slack = {}
    for j in cap_nodes:
        cs = slack_coeffs(prob.cap_units(j))
        slack[j] = [(var(("s", j, b)), c) for b, c in enumerate(cs)]

    n = len(names)
    U = np.zeros((n, n))
    offset = 0.0

    def add(i, j, v):
        if v == 0.0:
            return
        if i > j:
            i, j = j, i
        U[i, j] += v

    # ---------------------------------------------------- linear terms (c)
    for s, svc in enumerate(prob.services):
        offset += prob.sba_const(s)
        for k in range(len(svc.paths)):
            add(index[("x", s, k)], index[("x", s, k)], prob.linear_path_cost(s, k))
        for l in range(svc.n_nf):
            for m in range(len(svc.hosts)):
                i = index[("z", s, l, m)]
                add(i, i, prob.linear_place_cost(s, l, m))
        for k in range(prob.cfg.n_slices):
            i = index[("y", s, k)]
            add(i, i, prob.linear_slice_cost(s, k))

    # ------------------------------------------------- quadratic terms (Q)
    for s, svc in enumerate(prob.services):
        # Eq. (15) energy coupling + detour + data locality: path x placement
        for k in range(len(svc.paths)):
            for l in range(svc.n_nf):
                for m in range(len(svc.hosts)):
                    add(index[("x", s, k)], index[("z", s, l, m)],
                        prob.path_place_cost(s, k, l, m))
        # Eq. (16) co-location reward for consecutive NFs on the same host
        for l in range(svc.n_nf - 1):
            for m in range(len(svc.hosts)):
                add(index[("z", s, l, m)], index[("z", s, l + 1, m)], -prob.coloc_reward())
        # slice x placement compatibility
        for k in range(prob.cfg.n_slices):
            for l in range(svc.n_nf):
                for m in range(len(svc.hosts)):
                    add(index[("y", s, k)], index[("z", s, l, m)],
                        prob.slice_place_cost(s, k, l, m))
    # Eq. (14) interference + Eq. (7) RB contention between services
    for s1, s2 in combinations(range(S), 2):
        for k1 in range(len(prob.services[s1].paths)):
            for k2 in range(len(prob.services[s2].paths)):
                add(index[("x", s1, k1)], index[("x", s2, k2)],
                    prob.path_pair_cost(s1, k1, s2, k2))
    # battery coupling for NFs sharing a battery-powered host
    by_node = {}
    for s, svc in enumerate(prob.services):
        for l in range(svc.n_nf):
            for m, j in enumerate(svc.hosts):
                by_node.setdefault(j, []).append((s, l, m))
    for j, lst in by_node.items():
        if not np.isfinite(prob.net.battery_j[j]):
            continue
        for (s1, l1, m1), (s2, l2, m2) in combinations(lst, 2):
            if (s1, l1) != (s2, l2):
                add(index[("z", s1, l1, m1)], index[("z", s2, l2, m2)],
                    prob.node_pair_cost(s1, l1, s2, l2, j))

    # ------------------------------------------------- one-hot penalties
    # Group-specific A_g = scale * max_{i in g} (|c_i| + sum_g' max_j |U_ij|): enough
    # that leaving a group empty or doubly filled never pays off in the objective,
    # but no larger than that group needs (a global A makes the landscape rugged).
    penalties, A = [], 0.0
    if with_penalty:
        Um = U[:n_main, :n_main]
        absU = np.abs(Um) + np.abs(Um.T)
        lin_abs = np.abs(np.diag(Um))
        np.fill_diagonal(absU, 0.0)
        rows = lin_abs.copy()
        for g2 in groups:
            rows += absU[:, g2].max(axis=1)
        # largest capacity-penalty saving obtainable by dropping a placement
        lam_c = prob.cap_lambda()
        for j in cap_nodes:
            per_nf = {}
            for s, l, m in by_node[j]:
                per_nf[(s, l)] = prob.units(s, l)
            Lmax, C = sum(per_nf.values()), prob.cap_units(j)
            for s, l, m in by_node[j]:
                u = prob.units(s, l)
                rows[index[("z", s, l, m)]] += lam_c * (max(0, Lmax - C) ** 2 - max(0, Lmax - u - C) ** 2)
        for g in groups:
            Ag = prob.cfg.penalty_scale * float(rows[g].max())
            penalties.append(Ag)
            offset += Ag
            for i in g:
                U[i, i] -= Ag
            for i, j in combinations(g, 2):
                U[min(i, j), max(i, j)] += 2.0 * Ag
        A = float(np.mean(penalties))
    U_core = U[:n_main, :n_main].copy()
    core_offset = offset

    # ---------------------------------------- node capacity with slack bits
    cap = []
    lam = prob.cap_lambda()
    for j in cap_nodes:
        zt = [(index[("z", s, l, m)], prob.units(s, l)) for s, l, m in by_node[j]]
        terms = zt + slack[j]
        C = prob.cap_units(j)
        offset += lam * C * C
        for i, w in terms:
            add(i, i, lam * (w * w - 2 * C * w))
        for (i, wi), (k, wk) in combinations(terms, 2):
            add(i, k, 2 * lam * wi * wk)
        cap.append((np.array([i for i, _ in zt]), np.array([w for _, w in zt], dtype=float),
                    C, lam, np.array([i for i, _ in slack[j]]), [c for _, c in slack[j]]))

    nq = int(np.count_nonzero(np.triu(U, 1)))
    return QUBO(U=U, offset=offset, index=index, names=names, groups=groups,
                penalty=A, n_quadratic=nq, n_main=n_main, U_core=U_core,
                core_offset=core_offset, cap=cap, penalties=penalties)


# ------------------------------------------------------------- Ising (Eq. 17)
def to_ising(qubo: QUBO):
    """Map q = (1 - sigma)/2 and return (h, J, const) with
    E(sigma) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + const."""
    U = qubo.U
    lin = np.diag(U).copy()
    Q = np.triu(U, 1)
    h = -lin / 2.0 - (Q.sum(axis=1) + Q.sum(axis=0)) / 4.0
    J = Q / 4.0
    const = qubo.offset + lin.sum() / 2.0 + Q.sum() / 4.0
    return h, J, const


def ising_energy(h, J, const, sigma):
    sigma = np.asarray(sigma, dtype=float)
    return float(h @ sigma + sigma @ J @ sigma + const)


# ------------------------------------------------------------ encode/decode
def encode(prob: Problem, qubo: QUBO, a) -> np.ndarray:
    """Assignment -> full binary vector (slack bits set optimally)."""
    q = np.zeros(qubo.n)
    for s, svc in enumerate(prob.services):
        q[qubo.index[("x", s, a["path"][s])]] = 1
        for l, m in enumerate(a["host"][s]):
            q[qubo.index[("z", s, l, m)]] = 1
        q[qubo.index[("y", s, a["slice"][s])]] = 1
    return qubo.with_slack(q)


def decode(prob: Problem, qubo: QUBO, q):
    """Return (assignment, n_violated_groups). Broken one-hot groups are repaired
    by keeping the cheapest selected variable (or the cheapest one if none)."""
    q = np.asarray(q)
    lin = np.diag(qubo.U_core).copy()
    if qubo.penalties:
        for g, Ag in zip(qubo.groups, qubo.penalties):
            lin[g] += Ag
    viol = 0
    pick = {}
    for g in qubo.groups:
        on = [i for i in g if q[i] > 0.5]
        if len(on) != 1:
            viol += 1
        cand = on if on else g
        pick[g[0]] = min(cand, key=lambda i: lin[i])
    a = {"path": [], "host": [], "slice": []}
    for g in qubo.groups:
        name = qubo.names[pick[g[0]]]
        if name[0] == "x":
            a["path"].append(name[2])
            a["host"].append([])
        elif name[0] == "z":
            a["host"][name[1]].append(name[3])
        else:
            a["slice"].append(name[2])
    return a, viol