"""
Consistency checks for the ORCA reproduction.

1. For every one-hot-feasible assignment, the QUBO energy equals the directly
   computed objective H_total (Eq. 6) -> the QUBO is an exact encoding.
2. The Ising Hamiltonian (Eq. 17) gives the same energy as the QUBO.
3. Any one-hot violation costs at least as much as the cheapest feasible repair
   (the penalty A is large enough).
4. Brute force over all feasible assignments of a small instance agrees with
   brute force over all 2^n bit strings of the QUBO.
"""
import itertools

import numpy as np

from config import Config
from problem import build_problem
from qubo_builder import build_qubo, decode, encode, ising_energy, to_ising


def random_assignment(prob, rng):
    return {"path": [int(rng.integers(len(s.paths))) for s in prob.services],
            "host": [[int(rng.integers(len(s.hosts))) for _ in range(s.n_nf)]
                     for s in prob.services],
            "slice": [int(rng.integers(prob.cfg.n_slices)) for _ in prob.services]}


def all_assignments(prob):
    per = []
    for s in prob.services:
        per.append(list(itertools.product(range(len(s.paths)),
                                          itertools.product(range(len(s.hosts)), repeat=s.n_nf),
                                          range(prob.cfg.n_slices))))
    for combo in itertools.product(*per):
        yield {"path": [c[0] for c in combo], "host": [list(c[1]) for c in combo],
               "slice": [c[2] for c in combo]}


def test_encoding(seed=0, trials=200):
    prob = build_problem(Config(n_nodes=100), seed)
    qubo = build_qubo(prob)
    h, J, c = to_ising(qubo)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        a = random_assignment(prob, rng)
        q = encode(prob, qubo, a)
        e_q = qubo.energy(q)
        e_o = prob.objective(a)
        e_i = ising_energy(h, J, c, 1 - 2 * q)
        worst = max(worst, abs(e_q - e_o), abs(e_i - e_q))
        a2, v = decode(prob, qubo, q)
        assert v == 0 and a2 == a
    assert worst < 1e-6, worst
    print(f"[ok] QUBO == objective == Ising on {trials} random feasible assignments "
          f"(max abs diff {worst:.2e}); n={qubo.n}, couplers={qubo.n_quadratic}")


def test_penalty(seed=1, trials=2000):
    prob = build_problem(Config(n_nodes=100), seed)
    qubo = build_qubo(prob)
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        q = (rng.random(qubo.n) < 0.2).astype(float)
        a, v = decode(prob, qubo, q)
        if v and qubo.energy(q) < prob.objective(a) - 1e-9:
            bad += 1
        # slack bits can only make things worse than their optimal setting
        assert qubo.energy(q) >= qubo.energy(qubo.with_slack(q)) - 1e-9 or v
    print(f"[ok] penalty A={qubo.penalty:.1f}: infeasible strings cheaper than their "
          f"repair: {bad}/{trials}")
    assert bad == 0


def _ground(qubo, chunk=2 ** 18):
    n = qubo.n
    best = np.inf
    for start in range(0, 2 ** n, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** n))
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
        best = min(best, float(qubo.energy(bits).min()))
    return best


def test_bruteforce():
    try:                                   # available from step 5 on
        from solvers import ilp_solve
    except ImportError:
        ilp_solve = None
    cases = [(3, Config(n_nodes=30, n_services=1, k_paths=3, chain_len=(2, 2), max_hosts_per_nf=3)),
             (5, Config(n_nodes=30, n_services=2, k_paths=2, chain_len=(2, 2), max_hosts_per_nf=2, n_slices=1))]
    for seed, cfg in cases:
        prob = build_problem(cfg, seed)
        # make capacity bind: shrink the CPU of every candidate host
        for svc in prob.services:
            for j in svc.hosts:
                prob.net.cap[j] = min(prob.net.cap[j], 1.5)
        qubo = build_qubo(prob)
        best_feas = min(prob.objective(a) for a in all_assignments(prob))
        best_q = _ground(qubo)
        msg = (f"[ok] brute force n={qubo.n} ({qubo.n_slack} slack bits, {len(qubo.cap)} capacity rows): "
               f"best feasible objective {best_feas:.6f}, QUBO ground state {best_q:.6f}")
        assert abs(best_feas - best_q) < 1e-6
        if ilp_solve is not None:
            ilp = prob.objective(ilp_solve(prob)["assignment"])
            assert abs(ilp - best_feas) < 1e-6
            msg += f", ILP {ilp:.6f}"
        print(msg)


if __name__ == "__main__":
    test_encoding()
    test_penalty()
    test_bruteforce()