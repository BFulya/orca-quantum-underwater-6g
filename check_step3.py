"""Step 3 check: build one ORCA scenario (network + SBA services) and evaluate two placements."""
from config import Config
from problem import build_problem

prob = build_problem(Config(), seed=0)
net = prob.net

for s, svc in enumerate(prob.services):
    best = svc.paths[0]
    print(f"service {s}: source {net.ntype[svc.src][:3]} {svc.src:>2} -> gateway {svc.dst:>2} | {svc.n_nf} NFs | "
          f"{'URLLC alarm' if svc.critical else 'telemetry  '} | {len(svc.paths)} paths "
          f"(best: {len(best.links)} hops, {best.delay_ms:.1f} ms, P={best.psucc:.4f}) | "
          f"hosts: {[net.ntype[j][:3] for j in svc.hosts]}")

# placement A: every NF on the destination gateway (host index 0), shortest path, URLLC slice
a_gw = {"path": [0] * len(prob.services),
        "host": [[0] * svc.n_nf for svc in prob.services],
        "slice": [0] * len(prob.services)}
# placement B: every NF on the second candidate host (the source AUV when it can host, else an edge node)
a_src = {"path": [0] * len(prob.services),
         "host": [[1] * svc.n_nf for svc in prob.services],
         "slice": [0] * len(prob.services)}

for name, a in (("all NFs on gateway", a_gw), ("all NFs on host #1 ", a_src)):
    m = prob.metrics(a)
    print(f"{name}: H_total={m['objective']:7.2f} | latency {m['latency_ms']:5.2f} ms | "
          f"reliability {m['reliability']:.4f} | energy {m['energy_j']:6.2f} J | "
          f"SLA {m['sla_ratio']*100:3.0f}% | capacity violations {m['cap_viol']}")