"""
Figures and summary tables of the ORCA reproduction.

Reads results/*.csv written by experiments.py and writes figures/*.png plus
results/summary_*.csv. Usage:  python plots.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = ["ORCA", "SQA-only", "SA", "ILP", "Greedy", "GA", "PSO"]
COLOR = {"ORCA": "#1f5fbf", "SQA-only": "#7fa7e6", "SA": "#e08a2c", "ILP": "#333333",
         "Greedy": "#9a9a9a", "GA": "#3a9a5b", "PSO": "#b5524a"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load(name):
    p = os.path.join(RES, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/" + name)


# ------------------------------------------------------------------ main
def plot_main(df):
    methods = [m for m in ORDER if m in set(df.method)]
    g = df.groupby("method")
    summ = pd.DataFrame({
        "latency_ms": g.latency_ms.mean(), "latency_ci": g.latency_ms.apply(ci95),
        "reliability": g.reliability.mean(), "reliability_ci": g.reliability.apply(ci95),
        "energy_j": g.energy_j.mean(), "energy_ci": g.energy_j.apply(ci95),
        "sla_pct": 100 * g.sla_ratio.mean(),
        "gap_pct_mean": g.gap_pct.mean(), "gap_pct_median": g.gap_pct.median(),
        "optimal_hits_pct": g.gap_pct.apply(lambda x: 100 * np.mean(np.abs(x) < 1e-6)),
        "onehot_violation_pct": g.violations.apply(lambda x: 100 * np.mean(x > 0)),
        "resources_ok_pct": 100 * g.resources_ok.mean(),
        "time_s_mean": g.time_s.mean(), "time_s_median": g.time_s.median(),
    }).loc[methods]
    summ.round(4).to_csv(os.path.join(RES, "summary_main.csv"))
    print(summ.round(3).to_string())

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    panels = [("latency_ms", "Mean end-to-end latency (ms)", "latency_ci"),
              ("reliability", "Mean end-to-end reliability", "reliability_ci"),
              ("energy_j", "Network energy per epoch (J)", "energy_ci")]
    x = np.arange(len(methods))
    cols = [COLOR[m] for m in methods]
    for a, (col, title, ci) in zip(ax[0], panels):
        a.bar(x, summ[col], yerr=summ[ci], color=cols, capsize=3)
        a.set_xticks(x, methods, rotation=30)
        a.set_title(title)
        if col == "reliability":
            lo = (summ[col] - summ[ci]).min()
            a.set_ylim(lo - 0.005, 1.0)
    # optimality gap distribution
    data = [df[df.method == m].gap_pct.values for m in methods]
    bp = ax[1, 0].boxplot(data, patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c)
        p.set_alpha(0.8)
    ax[1, 0].set_xticks(x + 1, methods, rotation=30)
    ax[1, 0].set_title("Optimality gap vs exact ILP (%)")
    ax[1, 1].bar(x, summ.sla_pct, color=cols)
    ax[1, 1].set_xticks(x, methods, rotation=30)
    ax[1, 1].set_title("Services meeting SLA (D<30 ms, R>95 %) (%)")
    ax[1, 2].bar(x, summ.time_s_median, color=cols)
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_xticks(x, methods, rotation=30)
    ax[1, 2].set_title("Median run time on CPU (s, log)")
    fig.suptitle(f"Main comparison: N = 100 nodes, 6 services, {df.seed.nunique()} random scenarios "
                 "(bars = mean, error bars = 95 % CI)", y=1.01)
    save(fig, "fig1_main_comparison.png")


# --------------------------------------------------------------- scaling
def plot_scaling(df):
    methods = [m for m in ORDER if m in set(df.method)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for m in methods:
        d = df[df.method == m].groupby("n_nodes")
        ax[0].plot(d.n_vars.mean(), d.time_s.median(), "o-", color=COLOR[m], label=m)
        ax[2].plot(d.gap_pct.mean().index, d.gap_pct.mean(), "o-", color=COLOR[m], label=m)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("QUBO size (binary variables)")
    ax[0].set_ylabel("median run time (s)")
    ax[0].set_title("Run time vs problem size")
    ax[0].legend(fontsize=8)
    v = df[df.method == "ORCA"].groupby("n_nodes")[["n_vars", "n_couplers"]].mean()
    ax[1].plot(v.index, v.n_vars, "o-", color="#1f5fbf", label="binary variables")
    ax[1].plot(v.index, v.n_couplers / 10, "s--", color="#7fa7e6", label="couplers / 10")
    ax[1].set_xlabel("network size N (nodes)")
    ax[1].set_title("ORCA QUBO size vs network size")
    ax[1].legend(fontsize=8)
    ax[2].set_xlabel("network size N (nodes)")
    ax[2].set_ylabel("mean gap to ILP (%)")
    ax[2].set_title("Solution quality vs network size")
    save(fig, "fig2_scaling.png")
    tab = df.groupby(["n_nodes", "method"]).agg(
        n_vars=("n_vars", "mean"), time_s=("time_s", "median"), gap_pct=("gap_pct", "mean"),
        ilp_optimal=("optimal", lambda x: np.nan if x.isna().all() else 100 * np.mean(x == True)))
    tab.round(3).to_csv(os.path.join(RES, "summary_scaling.csv"))
    print(tab.round(3).to_string())


# ----------------------------------------------------------- sensitivity
def plot_sensitivity(df):
    built = df[df.method != "none"]
    methods = ["ORCA", "ILP", "Greedy", "GA"]
    specs = [("kappa_db_per_m", "latency_ms", "absorption kappa (dB/m @100 GHz)", "latency (ms)"),
             ("auv_speed", "reliability", "AUV speed (m/s)", "reliability"),
             ("ris_gain_db", "mean_snr_db", "RIS gain (dB)", "mean SNR of used links (dB)"),
             ("api_ms", "latency_ms", "SBA API overhead per remote call (ms)", "latency (ms)"),
             ("api_ms", "coloc_rate", "SBA API overhead per remote call (ms)", "co-located NF pairs"),
             ("traffic_load", "latency_ms", "traffic load (x queueing)", "latency (ms)")]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, (param, metric, xl, yl) in zip(ax.flat, specs):
        d = built[built.param == param]
        for m in methods:
            dm = d[d.method == m].groupby("value")[metric]
            a.errorbar(dm.mean().index, dm.mean(), yerr=dm.apply(ci95), fmt="o-", ms=4,
                       color=COLOR[m], label=m, capsize=2)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
    ax[0, 0].legend(fontsize=8)
    # connectivity for the absorption sweep
    k = df[df.param == "kappa_db_per_m"].groupby("value").apply(
        lambda x: x[["seed", "built"]].drop_duplicates().built.mean())
    a2 = ax[0, 0].twinx()
    a2.plot(k.index, 100 * k.values, "k:", lw=1)
    a2.set_ylabel("scenarios buildable (%)", fontsize=8)
    a2.set_ylim(0, 105)
    a2.grid(False)
    fig.suptitle("Sensitivity analysis (mean +/- 95 % CI over random scenarios)")
    save(fig, "fig3_sensitivity.png")
    tab = built.groupby(["param", "value", "method"])[
        ["latency_ms", "reliability", "mean_snr_db", "coloc_rate", "sla_ratio"]].mean()
    tab.round(4).to_csv(os.path.join(RES, "summary_sensitivity.csv"))


# ----------------------------------------------------------- convergence
def plot_convergence(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    col = {"ORCA (SQA stage)": COLOR["SQA-only"], "SA": COLOR["SA"], "GA": COLOR["GA"],
           "PSO": COLOR["PSO"]}
    grid = np.linspace(0.01, 1, 100)
    for m, c in col.items():
        curves = []
        for s, d in df[df.method == m].groupby("seed"):
            gap = 100 * (d.energy.values - d.optimum.values) / np.abs(d.optimum.values)
            curves.append(np.interp(grid, d.iter_frac.values, gap))
        ax.plot(100 * grid, np.median(curves, axis=0), color=c, label=m)
    o = df[df.method == "ORCA (after refinement)"]
    og = np.median(100 * (o.energy - o.optimum) / np.abs(o.optimum))
    ax.plot([100], [og], "*", ms=14, color=COLOR["ORCA"], label="ORCA after classical refinement")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("progress (% of iterations / sweeps)")
    ax.set_ylabel("median gap to optimum (%)")
    ax.set_title("Convergence of the QUBO energy")
    ax.legend(fontsize=8)
    save(fig, "fig4_convergence.png")


# ------------------------------------------------------------------- QAOA
def plot_qaoa(df):
    g = df.groupby("p")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].errorbar(g.p_ground.mean().index, g.p_ground.mean(), yerr=g.p_ground.apply(ci95),
                   fmt="o-", color=COLOR["ORCA"], label="QAOA")
    ax[0].axhline(df.uniform_p_ground.mean(), ls="--", color="gray", label="uniform superposition")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("QAOA depth p")
    ax[0].set_title("P(measure the optimum)")
    ax[0].legend(fontsize=8)
    ax[1].errorbar(g.p_feasible.mean().index, g.p_feasible.mean(), yerr=g.p_feasible.apply(ci95),
                   fmt="o-", color=COLOR["GA"], label="QAOA")
    ax[1].axhline(df.uniform_p_feasible.mean(), ls="--", color="gray", label="uniform superposition")
    ax[1].set_xlabel("QAOA depth p")
    ax[1].set_title("P(measure a one-hot-feasible state)")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"QAOA on reduced ORCA instances ({int(df.n_qubits.iloc[0])} qubits, "
                 f"{df.seed.nunique()} instances)")
    save(fig, "fig5_qaoa.png")
    tab = g[["p_ground", "p_feasible", "approx_ratio", "shots_hit_optimum"]].mean()
    tab["uniform_p_ground"] = df.uniform_p_ground.mean()
    tab["uniform_p_feasible"] = df.uniform_p_feasible.mean()
    tab["sqa_hits_opt"] = df.sqa_hits_opt.mean()
    tab["orca_hits_opt"] = df.orca_hits_opt.mean()
    tab.round(5).to_csv(os.path.join(RES, "summary_qaoa.csv"))
    print(tab.round(5).to_string())


# --------------------------------------------------------------- ablation
def plot_ablation(df):
    tab = df.groupby(["kind", "variant"]).agg(
        gap_pct=("gap_pct", "mean"), gap_ci=("gap_pct", ci95), latency_ms=("latency_ms", "mean"),
        reliability=("reliability", "mean"), energy_j=("energy_j", "mean"),
        sla_pct=("sla_ratio", lambda x: 100 * x.mean()))
    tab.round(4).to_csv(os.path.join(RES, "summary_ablation.csv"))
    print(tab.round(3).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, kind, title in [(ax[0], "terms", "Removing one QUBO term (judged by full model)"),
                           (ax[1], "stage", "Which stage gives ORCA its quality")]:
        t = tab.loc[kind].sort_values("gap_pct")
        c = [COLOR["ORCA"] if ("full" in v or "ORCA" in v) else "#9a9a9a" for v in t.index]
        a.barh(t.index, t.gap_pct, xerr=t.gap_ci, color=c, capsize=3)
        a.set_xlabel("mean gap to exact optimum (%)")
        a.set_title(title)
    save(fig, "fig6_ablation.png")


if __name__ == "__main__":
    for name, fn in [("main", plot_main), ("scaling", plot_scaling),
                     ("sensitivity", plot_sensitivity), ("convergence", plot_convergence),
                     ("qaoa", plot_qaoa), ("ablation", plot_ablation)]:
        df = load(name)
        if df is None:
            print(f"[skip] results/{name}.csv not found")
            continue
        print(f"[{name}]")
        fn(df)