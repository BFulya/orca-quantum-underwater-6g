"""
SBA service chains, osmotic NF placement candidates and the ORCA cost model
(Section III-B/E/F/G/H/I/J).

The Problem object holds every coefficient the QUBO needs, and an
`objective()` that evaluates the same weighted cost H_total (Eq. 6) directly
from a decoded assignment. For any one-hot-feasible assignment,
objective(a) == H_QUBO(q(a)) + const, which is checked in tests.py.
"""
from dataclasses import dataclass, field
from itertools import combinations, islice
import math

import networkx as nx
import numpy as np

from config import Config
from network import Network, build_network

D_REF_MS = 10.0     # normalisation of delay-like terms
E_REF_J = 1.0       # normalisation of energy terms
R_REF = 0.01        # normalisation of -ln(P_succ)
RB_REF = 10.0       # normalisation of RB overload
LOW_BATTERY_J = 2.0  # node counts as low-energy for Eq. (15)
SLICE_PRICE = (0.6, 0.2)  # URLLC is the more expensive slice
LOAD_UNIT = 0.25    # NF CPU load is discretised in quarters of a reference CPU
CAP_LAMBDA = 2.0    # penalty per (overloaded unit)^2 for node capacity


@dataclass
class Path:
    nodes: list
    links: list          # list of (a, b) with a < b
    delay_ms: float
    neglogp: float       # -ln(P_succ_p), Eq. (3)
    psucc: float
    e_tx_j: float        # transmission energy for the epoch
    exp_ho: float        # E[N_HO(p)]
    instab: float        # sum of I_instab over links


@dataclass
class Service:
    src: int
    dst: int
    critical: bool       # alarm service -> must use URLLC slice
    work_ms: list        # per NF processing work at reference CPU
    load: list           # per NF CPU load share
    rate_bps: float
    rb: int
    paths: list = field(default_factory=list)
    hosts: list = field(default_factory=list)   # candidate osmotic hosts (shared by NFs)
    detour_ms: np.ndarray = None                # (K, M) extra delay if host off path
    raw_ms: np.ndarray = None                   # (K, M) delay to ship raw data to NF_1 on host m
    raw_e: np.ndarray = None                    # (K, M) energy to ship raw data to NF_1 on host m

    @property
    def n_nf(self):
        return len(self.work_ms)


@dataclass
class Problem:
    cfg: Config
    net: Network
    services: list
    c_ho_ms: float

    # ---------------------------------------------------------------- helpers
    def proc_ms(self, s, l, j):
        return self.services[s].work_ms[l] / self.net.cpu[j]

    def proc_energy_j(self, s, l, j):
        busy_s = self.cfg.event_rate_hz * self.cfg.epoch_s * self.proc_ms(s, l, j) / 1e3
        return self.net.power_w[j] * busy_s

    def units(self, s, l):
        return int(round(self.services[s].load[l] / LOAD_UNIT))

    def cap_units(self, j):
        return int(round(self.net.cap[j] / LOAD_UNIT))

    def cap_lambda(self):
        return self.cfg.w["C"] * CAP_LAMBDA

    def cap_nodes(self):
        """Nodes whose capacity can actually be exceeded by the candidate NFs."""
        demand = {}
        for s, svc in enumerate(self.services):
            for l in range(svc.n_nf):
                for j in svc.hosts:
                    demand[j] = demand.get(j, 0) + self.units(s, l)
        return sorted(j for j, d in demand.items() if d > self.cap_units(j))

    def cap_penalty(self, a):
        """Node capacity: lambda * sum_j max(0, load_j - cap_j)^2 (load in units).
        Equals the QUBO slack-variable penalty with the slack chosen optimally."""
        load = {}
        for s, svc in enumerate(self.services):
            for l, m in enumerate(a["host"][s]):
                j = svc.hosts[m]
                load[j] = load.get(j, 0) + self.units(s, l)
        return self.cap_lambda() * sum(max(0, v - self.cap_units(j)) ** 2 for j, v in load.items())

    def slice_mismatch(self, k, j):
        """delta_slice(k, j): URLLC NFs should not run on slow sensor nodes."""
        return 1.0 if (k == 0 and self.net.ntype[j] == "sensor") else 0.0

    # ------------------------------------------------------- objective pieces
    def linear_path_cost(self, s, k):
        """Eq. (12): c_xp = wE Ep + wD Dp + wI sum I + wSBA ... + wMOB E[N_HO]."""
        w, p = self.cfg.w, self.services[s].paths[k]
        c = w["E"] * p.e_tx_j / E_REF_J
        c += w["D"] * p.delay_ms / D_REF_MS
        c += w["R"] * p.neglogp / R_REF
        c += w["I"] * p.instab
        c += w["MOB"] * p.exp_ho * self.c_ho_ms / D_REF_MS
        over = max(0, self.services[s].rb - min(self.net.G[a][b]["rb_cap"] for a, b in p.links))
        c += w["RAD"] * over ** 2 / RB_REF
        return c

    def linear_place_cost(self, s, l, m):
        """Eq. (13): c_ztj = wE cost_proc + wS delta_slice + wC capacity_penalty."""
        w, j = self.cfg.w, self.services[s].hosts[m]
        c = w["E"] * self.proc_energy_j(s, l, j) / E_REF_J
        c += w["D"] * self.proc_ms(s, l, j) / D_REF_MS
        c += w["R"] * (-math.log(self.net.avail[j])) / R_REF
        bat = self.net.battery_j[j]
        if np.isfinite(bat):
            c += w["C"] * max(0.0, self.proc_energy_j(s, l, j) - bat) / E_REF_J
        return c

    def linear_slice_cost(self, s, k):
        w = self.cfg.w
        c = w["S"] * SLICE_PRICE[k]
        if self.services[s].critical and k != 0:
            c += w["S"] * 5.0                     # critical alarm forced to URLLC
        return c

    def sba_const(self, s):
        """Eq. (8) when every consecutive NF pair is remote, plus D_disc."""
        L = self.services[s].n_nf
        return self.cfg.w["SBA"] * ((L - 1) * self.cfg.api_ms + self.cfg.disc_ms) / D_REF_MS

    def coloc_reward(self):
        """-lambda_coloc of Eq. (16): saving of a co-located call vs a remote call."""
        return self.cfg.w["SBA"] * (self.cfg.api_ms - self.cfg.loc_ms) / D_REF_MS

    def path_pair_cost(self, s1, k1, s2, k2):
        """Eq. (14) interference + pairwise RB over-allocation proxy of Eq. (7)."""
        w = self.cfg.w
        p1, p2 = self.services[s1].paths[k1], self.services[s2].paths[k2]
        shared_links = set(p1.links) & set(p2.links)
        shared_nodes = (set(p1.nodes) & set(p2.nodes)) - {self.services[s1].dst, self.services[s2].dst}
        c = w["I"] * (len(shared_links) + 0.5 * len(shared_nodes))
        r = self.services[s1].rb + self.services[s2].rb
        over = sum(max(0, r - self.net.G[a][b]["rb_cap"]) ** 2 for a, b in shared_links)
        c += w["RAD"] * over / RB_REF
        return c

    def path_place_cost(self, s, k, l, m):
        """Detour delay for an off-path host + Eq. (15) energy coupling."""
        svc = self.services[s]
        j = svc.hosts[m]
        c = self.cfg.w["D"] * svc.detour_ms[k, m] / D_REF_MS
        if l == 0:   # data-locality pressure: raw data must reach the ingestion NF
            c += self.cfg.w["D"] * svc.raw_ms[k, m] / D_REF_MS
            c += self.cfg.w["E"] * svc.raw_e[k, m] / E_REF_J
        if j in svc.paths[k].nodes and self.net.battery_j[j] < LOW_BATTERY_J:
            c += self.cfg.lambda_energy
        return c

    def node_pair_cost(self, s1, l1, s2, l2, j):
        """Battery coupling: two NFs on the same battery node beyond its budget.
        (CPU capacity is handled exactly with slack variables, see qubo_builder.)"""
        c = 0.0
        bat = self.net.battery_j[j]
        if np.isfinite(bat):
            e = self.proc_energy_j(s1, l1, j) + self.proc_energy_j(s2, l2, j)
            c += self.cfg.w["C"] * max(0.0, e - bat) / E_REF_J
        return c

    def slice_place_cost(self, s, k, l, m):
        return self.cfg.w["S"] * self.slice_mismatch(k, self.services[s].hosts[m])

    # --------------------------------------------------------- full objective
    def objective(self, a):
        """H_total (Eq. 6) of a one-hot assignment a = dict(path, host, slice)."""
        S = len(self.services)
        H = 0.0
        for s, svc in enumerate(self.services):
            k, hs, sl = a["path"][s], a["host"][s], a["slice"][s]
            H += self.linear_path_cost(s, k) + self.linear_slice_cost(s, sl) + self.sba_const(s)
            for l, m in enumerate(hs):
                H += self.linear_place_cost(s, l, m)
                H += self.path_place_cost(s, k, l, m)
                H += self.slice_place_cost(s, sl, l, m)
            for l in range(svc.n_nf - 1):
                if hs[l] == hs[l + 1]:
                    H -= self.coloc_reward()
        for s1, s2 in combinations(range(S), 2):
            H += self.path_pair_cost(s1, a["path"][s1], s2, a["path"][s2])
        placed = [(s, l, self.services[s].hosts[m]) for s in range(S)
                  for l, m in enumerate(a["host"][s])]
        for (s1, l1, j1), (s2, l2, j2) in combinations(placed, 2):
            if j1 == j2:
                H += self.node_pair_cost(s1, l1, s2, l2, j1)
        return H + self.cap_penalty(a)

    # ------------------------------------------------------ physical metrics
    def metrics(self, a):
        """Physical KPIs of an assignment (latency, reliability, energy, SLA)."""
        cfg, net = self.cfg, self.net
        lat, rel, sla = [], [], []
        energy = 0.0
        node_load, node_energy, link_rb = {}, {}, {}
        for s, svc in enumerate(self.services):
            k, hs, sl = a["path"][s], a["host"][s], a["slice"][s]
            p = svc.paths[k]
            hosts = [svc.hosts[m] for m in hs]
            # Eq. (4): sum link delays + sum (t_proc + t_API) + D_disc (+ detour)
            d = p.delay_ms + cfg.disc_ms
            d += sum(self.proc_ms(s, l, j) for l, j in enumerate(hosts))
            d += sum(svc.detour_ms[k, m] for m in hs) + svc.raw_ms[k, hs[0]]
            d += sum(cfg.loc_ms if hosts[l] == hosts[l + 1] else cfg.api_ms
                     for l in range(svc.n_nf - 1))
            r = p.psucc * float(np.prod([net.avail[j] for j in set(hosts)]))
            e = p.e_tx_j + svc.raw_e[k, hs[0]] + sum(self.proc_energy_j(s, l, j) for l, j in enumerate(hosts))
            energy += e
            lat.append(d)
            rel.append(r)
            slice_ok = (sl == 0) or not svc.critical
            sla.append(d <= cfg.d_max_ms and r >= cfg.eta_req and slice_ok)
            for l, j in enumerate(hosts):
                node_load[j] = node_load.get(j, 0.0) + svc.load[l]
                node_energy[j] = node_energy.get(j, 0.0) + self.proc_energy_j(s, l, j)
            for lk in p.links:
                link_rb[lk] = link_rb.get(lk, 0) + svc.rb
        cap_viol = sum(v > net.cap[j] + 1e-9 for j, v in node_load.items())
        bat_viol = sum(v > net.battery_j[j] + 1e-9 for j, v in node_energy.items())
        rb_viol = sum(v > net.G[x][y]["rb_cap"] for (x, y), v in link_rb.items())
        return dict(latency_ms=float(np.mean(lat)), latency_max_ms=float(np.max(lat)),
                    reliability=float(np.mean(rel)), reliability_min=float(np.min(rel)),
                    energy_j=float(energy), sla_ratio=float(np.mean(sla)),
                    resources_ok=bool(cap_viol == 0 and bat_viol == 0 and rb_viol == 0),
                    cap_viol=int(cap_viol), bat_viol=int(bat_viol), rb_viol=int(rb_viol),
                    objective=self.objective(a))


# ------------------------------------------------------------- construction
def _link_key(a, b):
    return (a, b) if a < b else (b, a)


def _make_path(net, nodes, svc, cfg):
    G = net.G
    links = [_link_key(a, b) for a, b in zip(nodes[:-1], nodes[1:])]
    bits = svc.rate_bps * cfg.epoch_s * cfg.reduction
    psucc = float(np.prod([G[a][b]["psucc"] for a, b in links]))
    return Path(nodes=list(nodes), links=links,
                delay_ms=sum(G[a][b]["delay_ms"] for a, b in links),
                neglogp=-math.log(max(psucc, 1e-300)), psucc=psucc,
                e_tx_j=sum(G[a][b]["e_bit"] * bits for a, b in links),
                exp_ho=sum(G[a][b]["exp_ho"] for a, b in links),
                instab=sum(G[a][b]["instab"] for a, b in links))


def build_problem(cfg: Config, seed: int, max_tries: int = 50) -> Problem:
    """Random scenario: topology + S services with K paths and M hosts each."""
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        net = build_network(cfg, rng)
        comp = max(nx.connected_components(net.G), key=len)
        gws = [i for i in comp if net.ntype[i] == "gateway"]
        srcs = [i for i in comp if net.ntype[i] in ("sensor", "auv")]
        if gws and len(srcs) >= cfg.n_services:
            break
    else:
        raise RuntimeError("could not build a connected scenario")

    G = net.G
    # routing weight for candidate generation: delay plus a reliability term
    for a, b, e in G.edges(data=True):
        e["w"] = e["delay_ms"] + 2.0 * (-math.log(e["psucc"])) / R_REF * 0.1
    services = []
    chosen = rng.choice(srcs, size=cfg.n_services, replace=False)
    for src in chosen:
        src = int(src)
        dist = nx.single_source_dijkstra_path_length(G, src, weight="w")
        dst = min(gws, key=lambda g: dist.get(g, math.inf))
        L = int(rng.integers(cfg.chain_len[0], cfg.chain_len[1] + 1))
        svc = Service(src=src, dst=dst, critical=bool(rng.random() < 0.5),
                      work_ms=list(rng.uniform(0.2, 1.0, size=L)),
                      load=list(rng.integers(1, 6, size=L) * LOAD_UNIT),
                      rate_bps=float(rng.uniform(0.5e6, 5e6)),
                      rb=int(rng.integers(cfg.rb_demand[0], cfg.rb_demand[1] + 1)))
        gen = nx.shortest_simple_paths(G, src, dst, weight="w")
        svc.paths = [_make_path(net, p, svc, cfg) for p in islice(gen, cfg.k_paths)]
        services.append(svc)

    prob = Problem(cfg, net, services, c_ho_ms=float(rng.uniform(*cfg.c_ho_ms)))
    for s, svc in enumerate(services):
        _choose_hosts(prob, s)
    return prob


def _choose_hosts(prob, s):
    """Osmotic candidate hosts: nodes on the candidate paths ranked by pressure
    (latency + energy), always including the destination gateway."""
    cfg, net, svc = prob.cfg, prob.net, prob.services[s]
    on_path = {j for p in svc.paths for j in p.nodes}
    pool = set(on_path)
    for j in on_path:                       # osmotic diffusion: one hop off the routes
        pool.update(net.G.neighbors(j))
    max_load = max(svc.load)

    def pressure(j):
        lat = np.mean(svc.work_ms) / net.cpu[j]
        en = net.power_w[j] / net.cpu[j]
        return lat + 0.5 * en

    pool = [j for j in pool if net.cap[j] >= max_load]
    ranked = sorted(pool, key=pressure)
    first = [svc.dst] + ([svc.src] if net.cap[svc.src] >= max_load else [])
    hosts = first + [j for j in ranked if j not in first]
    svc.hosts = hosts[: cfg.max_hosts_per_nf]
    # detour: round trip from the nearest node on path k to host j
    K, M = len(svc.paths), len(svc.hosts)
    svc.detour_ms = np.zeros((K, M))
    svc.raw_ms = np.zeros((K, M))
    svc.raw_e = np.zeros((K, M))
    G = net.G
    raw_bits_epoch = svc.rate_bps * cfg.epoch_s * (1.0 - cfg.reduction)

    def seg(nodes):  # raw-data transfer delay and energy over a node sequence
        dl = sum(cfg.snapshot_bits / G[a][b]["rate"] * 1e3 for a, b in zip(nodes[:-1], nodes[1:]))
        en = sum(G[a][b]["e_bit"] * raw_bits_epoch for a, b in zip(nodes[:-1], nodes[1:]))
        return dl, en

    for k, p in enumerate(svc.paths):
        dist, paths = nx.multi_source_dijkstra(G, set(p.nodes), weight="delay_ms")
        pos = {v: i for i, v in enumerate(p.nodes)}
        for m, j in enumerate(svc.hosts):
            if j in pos:
                svc.detour_ms[k, m] = 0.0
                d, e = seg(p.nodes[: pos[j] + 1])
            else:
                svc.detour_ms[k, m] = 2.0 * dist[j]
                attach = paths[j][0]
                d0, e0 = seg(p.nodes[: pos[attach] + 1])
                d1, e1 = seg(paths[j])
                d, e = d0 + d1, e0 + e1
            svc.raw_ms[k, m], svc.raw_e[k, m] = d, e