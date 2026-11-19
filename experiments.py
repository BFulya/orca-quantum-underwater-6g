"""
Experiments of the ORCA reproduction (Section IV / V of the paper).

  main         Fig. 3 / 4 / 9 analogue - all methods, N = 100, 50 seeds
  scaling      Fig. 5b / 6 / 8 analogue - N = 50 ... 500 nodes
  sensitivity  Fig. 5a / 5c / 5d analogue + SBA API overhead
  convergence  Fig. 7 analogue - energy vs iteration
  qaoa         gate-model QAOA on reduced (<= 14 qubit) instances
  ablation     contribution of each ORCA term and of the quantum stage

Usage:  python experiments.py [name ...] [--quick]
Results are written to results/<name>.csv
"""
import argparse
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import Config
from problem import build_problem
from qubo_builder import build_qubo, decode
import solvers as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
WORKERS = int(os.environ.get("ORCA_WORKERS", max(1, min(4, (os.cpu_count() or 2) - 1))))

METHODS = ["ORCA", "SQA-only", "SA", "ILP", "Greedy", "GA", "PSO"]


def pmap(fn, jobs, label):
    """Parallel map over independent scenarios with a progress counter."""
    rows = []
    with Pool(WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            rows += r
            print(f"\r  [{label}] {k}/{len(jobs)} scenarios done", end="", flush=True)
    print()
    return rows


def run_method(name, prob, qubo, seed, fast=False):
    kw = dict(sweeps=150, reads=4, slices=8) if fast else {}
    if name == "ORCA":
        return S.orca_hybrid_solve(prob, qubo, seed=seed, **kw)
    if name == "SQA-only":
        return S.sqa_solve(prob, qubo, seed=seed, **kw)
    if name == "SA":
        return S.sa_solve(prob, qubo, seed=seed, sweeps=kw.get("sweeps", 300))
    if name == "ILP":
        return S.ilp_solve(prob, time_limit=120.0)
    if name == "Greedy":
        return S.greedy_solve(prob, qubo)
    if name == "GA":
        return S.ga_solve(prob, qubo, seed=seed)
    if name == "PSO":
        return S.pso_solve(prob, qubo, seed=seed)
    raise ValueError(name)


def _record(prob, res, **tags):
    m = prob.metrics(res["assignment"])
    m.update(tags)
    m.update(time_s=res["time_s"], violations=res["violations"],
             optimal=res.get("optimal"))
    return m


# ---------------------------------------------------------------------- main
def _main_one(args):
    seed, cfg, methods, fast = args
    prob = build_problem(cfg, seed)
    t0 = time.perf_counter()
    qubo = build_qubo(prob)
    t_build = time.perf_counter() - t0
    rows = []
    for name in methods:
        res = run_method(name, prob, qubo, seed, fast)
        if res is None:
            continue
        rows.append(_record(prob, res, method=name, seed=seed, n_nodes=cfg.n_nodes,
                            n_services=cfg.n_services, n_vars=qubo.n,
                            n_couplers=qubo.n_quadratic, qubo_build_s=t_build))
    return rows


def _gap(df):
    """Optimality gap w.r.t. the ILP solution of the same scenario (exact unless
    the ILP hit its time limit; see column `optimal`)."""
    opt = df[df.method == "ILP"].set_index(["n_nodes", "seed"])["objective"]
    key = list(zip(df.n_nodes, df.seed))
    df["gap_pct"] = [100.0 * (o - opt.get(k, np.nan)) / abs(opt.get(k, np.nan))
                     for o, k in zip(df.objective, key)]
    return df


def exp_main(quick=False):
    seeds = range(8 if quick else 50)
    cfg = Config()
    rows = pmap(_main_one, [(s, cfg, METHODS, False) for s in seeds], "main")
    df = _gap(pd.DataFrame(rows))
    df.to_csv(os.path.join(OUT, "main.csv"), index=False)
    return df


# ------------------------------------------------------------------- scaling
def exp_scaling(quick=False):
    sizes = [50, 100, 200, 300, 500]
    seeds = range(2 if quick else 5)
    jobs = []
    for n in sizes:
        cfg = Config(n_nodes=n, n_services=max(3, round(6 * n / 100)))
        jobs += [(s, cfg, ["ORCA", "SA", "ILP", "Greedy", "GA", "PSO"], True) for s in seeds]
    rows = pmap(_main_one, jobs, "scaling")
    df = _gap(pd.DataFrame(rows))
    df.to_csv(os.path.join(OUT, "scaling.csv"), index=False)
    return df


# --------------------------------------------------------------- sensitivity
def _sens_one(args):
    seed, cfg, param, value = args
    try:
        prob = build_problem(cfg, seed)
    except RuntimeError:
        return [dict(param=param, value=value, seed=seed, method="none", built=False)]
    qubo = build_qubo(prob)
    rows = []
    for name in ["ORCA", "ILP", "Greedy", "GA"]:
        res = run_method(name, prob, qubo, seed, fast=True)
        if res is None:                      # ILP hit its time limit without a solution
            continue
        r = _record(prob, res, method=name, seed=seed, param=param, value=value, built=True)
        a = res["assignment"]
        # effective SNR of the links actually used, and co-location rate
        snr, coloc, pairs = [], 0, 0
        for s, svc in enumerate(prob.services):
            p = svc.paths[a["path"][s]]
            snr += [prob.net.G[x][y]["snr_db"] for x, y in p.links]
            hs = a["host"][s]
            coloc += sum(hs[l] == hs[l + 1] for l in range(len(hs) - 1))
            pairs += len(hs) - 1
        r.update(mean_snr_db=float(np.mean(snr)), coloc_rate=coloc / max(pairs, 1),
                 hops=float(np.mean([len(svc.paths[a["path"][s]].links)
                                     for s, svc in enumerate(prob.services)])))
        rows.append(r)
    return rows


def exp_sensitivity(quick=False):
    seeds = range(3 if quick else 10)
    grid = {
        "kappa_db_per_m": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        "auv_speed": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        "ris_gain_db": [0.0, 5.0, 10.0, 15.0, 20.0],
        "api_ms": [1.0, 2.0, 5.0, 10.0, 20.0],
        "traffic_load": [0.5, 1.0, 1.5, 2.0, 3.0],
    }
    jobs = []
    for param, values in grid.items():
        for v in values:
            if param in ("auv_speed", "ris_gain_db"):
                cfg = replace(Config(), **{param: (v, v)})
            else:
                cfg = replace(Config(), **{param: v})
            jobs += [(s, cfg, param, v) for s in seeds]
    rows = pmap(_sens_one, jobs, "sensitivity")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    return df


# --------------------------------------------------------------- convergence
def exp_convergence(quick=False):
    rows = []
    for seed in range(2 if quick else 5):
        prob = build_problem(Config(), seed)
        qubo = build_qubo(prob)
        ilp = S.ilp_solve(prob)
        opt = prob.objective(ilp["assignment"]) if ilp else np.nan
        for name, fn in [("ORCA (SQA stage)", lambda: S.sqa_solve(prob, qubo, seed=seed)),
                         ("SA", lambda: S.sa_solve(prob, qubo, seed=seed)),
                         ("GA", lambda: S.ga_solve(prob, qubo, seed=seed)),
                         ("PSO", lambda: S.pso_solve(prob, qubo, seed=seed))]:
            tr = fn()["trace"]
            for i, e in enumerate(tr):
                rows.append(dict(seed=seed, method=name, iter_frac=(i + 1) / len(tr),
                                 energy=float(e), optimum=opt))
        orca = S.orca_hybrid_solve(prob, qubo, seed=seed)
        rows.append(dict(seed=seed, method="ORCA (after refinement)", iter_frac=1.0,
                         energy=float(orca["trace"][-1]), optimum=opt))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "convergence.csv"), index=False)
    return df


# ---------------------------------------------------------------------- QAOA
def exp_qaoa(quick=False):
    from qaoa import basis_energies, feasible_mask, run_qaoa
    cfg = Config(n_nodes=30, n_services=1, k_paths=3, chain_len=(3, 3), max_hosts_per_nf=3)
    rows = []
    for seed in range(2 if quick else 5):
        prob = build_problem(cfg, seed)
        qubo = build_qubo(prob)
        E, bits = basis_energies(qubo)
        fm = feasible_mask(qubo, bits)
        opt = prob.objective(S.ilp_solve(prob)["assignment"])
        sqa = S.sqa_solve(prob, qubo, seed=seed)
        orca = S.orca_hybrid_solve(prob, qubo, seed=seed)
        for p in (1, 2, 3, 4):
            r = run_qaoa(qubo, fm, p=p, restarts=4 if quick else 8, seed=seed)
            r.pop("params")
            r.update(seed=seed, ilp_opt=opt, ground_is_feasible_opt=bool(abs(r["ground_energy"] - opt) < 1e-6),
                     sqa_hits_opt=bool(abs(prob.objective(sqa["assignment"]) - opt) < 1e-6),
                     orca_hits_opt=bool(abs(prob.objective(orca["assignment"]) - opt) < 1e-6))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "qaoa.csv"), index=False)
    return df


# ------------------------------------------------------------------ ablation
def _abl_one(args):
    seed, = args
    full = build_problem(Config(), seed)
    variants = {
        "ORCA (full)": Config(),
        "no co-location (Eq. 16)": Config(loc_ms=Config().api_ms),
        "no mobility (H_MOB)": replace(Config(), w={**Config().w, "MOB": 0.0}),
        "no energy coupling (Eq. 15)": Config(lambda_energy=0.0),
        "no interference/RB (Eq. 7, 14)": replace(Config(), w={**Config().w, "I": 0.0, "RAD": 0.0}),
    }
    rows = []
    for name, cfg in variants.items():
        plan = build_problem(cfg, seed)          # same topology, different objective
        qubo = build_qubo(plan)
        res = S.orca_hybrid_solve(plan, qubo, seed=seed, sweeps=150, reads=4, slices=8)
        m = full.metrics(res["assignment"])       # always judged by the full model
        m.update(variant=name, seed=seed, kind="terms")
        rows.append(m)
    # where does ORCA's quality come from: annealing stage vs classical refinement
    qubo = build_qubo(full)
    rng = np.random.default_rng(seed)
    for name in ["random + refinement", "SA + refinement", "SQA + refinement (ORCA)"]:
        t0 = time.perf_counter()
        if name.startswith("random"):
            starts = [(rng.random(qubo.n) < 0.5).astype(float) for _ in range(5)]
        elif name.startswith("SA"):
            starts = [S.sa_solve(full, qubo, seed=seed * 10 + k, reads=1, sweeps=150)["assignment"]
                      for k in range(5)]
            from qubo_builder import encode
            starts = [encode(full, qubo, a) for a in starts]
        else:
            starts, _ = S._sqa_samples(qubo, np.random.default_rng(seed), 4, 8, 150, 0.01, (3.0, 1e-3))
        best = min((S.onehot_descent(qubo, q) for q in starts), key=qubo.energy)
        a, _ = decode(full, qubo, best)
        m = full.metrics(a)
        m.update(variant=name, seed=seed, kind="stage", time_s=time.perf_counter() - t0)
        rows.append(m)
    ilp = S.ilp_solve(full)
    opt = full.objective(ilp["assignment"]) if ilp else np.nan
    for r in rows:
        r["gap_pct"] = 100.0 * (r["objective"] - opt) / abs(opt)
    return rows


def exp_ablation(quick=False):
    seeds = range(4 if quick else 20)
    rows = pmap(_abl_one, [(s,) for s in seeds], "ablation")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    return df


EXPS = dict(main=exp_main, scaling=exp_scaling, sensitivity=exp_sensitivity,
            convergence=exp_convergence, qaoa=exp_qaoa, ablation=exp_ablation)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for nm in a.names:
        t = time.perf_counter()
        df = EXPS[nm](a.quick)
        print(f"[{nm}] {len(df)} rows in {time.perf_counter() - t:.1f}s", flush=True)
        sys.stdout.flush()