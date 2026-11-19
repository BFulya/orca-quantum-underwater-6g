"""
Physical layer and topology of the underwater 6G IoCT (Section III-A/C/D).

Implements:
  Eq. (1)  frequency-dependent path loss with molecular absorption
  Eq. (2)  link success probability from SNR, BER and channel instability
  Eq. (3)  end-to-end path success probability (product of links)
  Eq. (10) RIS-enhanced effective SNR
"""
from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
from scipy.special import erfc

from config import Config

C0 = 3.0e8                 # speed of light in vacuum (m/s)
V_WATER = C0 / 9.0         # EM phase velocity in water, n ~ 9 (relative permittivity ~ 81)

NODE_TYPES = ("gateway", "auv", "edge", "sensor")
# relative CPU speed, processing power (W), availability per epoch
NODE_PROFILE = {
    "gateway": dict(cpu=3.0, power_w=4.0, avail=0.9999, cap=4.0),
    "edge":    dict(cpu=1.5, power_w=2.4, avail=0.999,  cap=4.0),
    "auv":     dict(cpu=1.0, power_w=2.5, avail=0.995,  cap=2.5),
    "sensor":  dict(cpu=0.3, power_w=0.6, avail=0.999,  cap=1.0),
}


@dataclass
class Network:
    cfg: Config
    pos: np.ndarray            # (N, 3) coordinates (m); z = depth below surface
    ntype: list                # node type per node
    speed: np.ndarray          # m/s (0 for static nodes)
    cpu: np.ndarray            # relative CPU speed
    power_w: np.ndarray        # processing power draw
    battery_j: np.ndarray      # energy budget per epoch (inf = mains powered)
    avail: np.ndarray          # node availability probability
    cap: np.ndarray            # concurrent NF load capacity
    G: nx.Graph                # links with physical attributes
    kappa_db: float            # scenario absorption coefficient at 100 GHz (dB/m)

    @property
    def n(self):
        return len(self.ntype)


# --------------------------------------------------------------------- channel
def path_loss_db(f_hz, d, kappa_db_100, cfg, shadow_db=0.0):
    """Eq. (1): PL = PL0(f) + 10 n log10(d/d0) + A_mol(f,d) + X_sigma (d0 = 1 m)."""
    pl0 = 20.0 * np.log10(4.0 * math.pi * f_hz / C0)
    kappa_f = kappa_db_100 * (f_hz / 100e9) ** cfg.kappa_freq_exp
    # Beer-Lambert: A_mol = 10 log10(exp(kappa d)) -> linear in d when kappa is in dB/m
    a_mol = kappa_f * d
    return pl0 + 10.0 * cfg.pathloss_exp * np.log10(np.maximum(d, 1.0)) + a_mol + shadow_db


def snr_db(f_hz, b_hz, d, pt_dbm, g_db, kappa_db_100, cfg, shadow_db=0.0, ris_db=0.0):
    """SNR = Pr / (N0 B), with Eq. (10) SNR_eff = SNR + g_RIS."""
    pr = pt_dbm - path_loss_db(f_hz, d, kappa_db_100, cfg, shadow_db) + g_db
    noise = cfg.n0_dbm_hz + 10.0 * np.log10(b_hz)
    return pr - noise + ris_db


def link_success(snr_eff_db, instab, cfg):
    """Eq. (2): P_succ = Phi(SNR, BER_target, I_instab).

    Phi = BPSK BER (erfc), packet error over L bits, base packet loss,
    channel instability, and a bounded ARQ retransmission.
    """
    snr_lin = 10.0 ** (snr_eff_db / 10.0)
    ber = np.maximum(0.5 * erfc(np.sqrt(snr_lin)), 1e-15)
    p_pkt = (1.0 - ber) ** cfg.packet_bits
    p0 = p_pkt * (1.0 - cfg.packet_loss_rate) * (1.0 - instab)
    return 1.0 - (1.0 - p0) ** cfg.arq_attempts, ber


def max_range(f_hz, b_hz, pt_dbm, g_db, kappa, cfg, ris_db):
    """Distance at which SNR drops to snr_min (bisection)."""
    lo, hi = 1.0, 5000.0
    if snr_db(f_hz, b_hz, hi, pt_dbm, g_db, kappa, cfg, 0.0, ris_db) >= cfg.snr_min_db:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if snr_db(f_hz, b_hz, mid, pt_dbm, g_db, kappa, cfg, 0.0, ris_db) >= cfg.snr_min_db:
            lo = mid
        else:
            hi = mid
    return lo


# -------------------------------------------------------------------- topology
def build_network(cfg: Config, rng: np.random.Generator) -> Network:
    n = cfg.n_nodes
    n_gw = max(2, int(round(cfg.frac_gateway * n)))
    n_auv = int(round(cfg.frac_auv * n))
    n_edge = int(round(cfg.frac_edge * n))
    n_sen = n - n_gw - n_auv - n_edge
    ntype = ["gateway"] * n_gw + ["auv"] * n_auv + ["edge"] * n_edge + ["sensor"] * n_sen

    L = np.array(cfg.volume_m)
    pos = rng.uniform(0.0, 1.0, size=(n, 3)) * L
    pos[:n_gw, 2] = rng.uniform(0.0, 20.0, size=n_gw)          # gateways near the surface
    speed = np.array([rng.uniform(*cfg.auv_speed) if t == "auv" else 0.0 for t in ntype])

    prof = [NODE_PROFILE[t] for t in ntype]
    cpu = np.array([p["cpu"] for p in prof])
    power = np.array([p["power_w"] for p in prof])
    avail = np.array([p["avail"] for p in prof])
    cap = np.array([p["cap"] for p in prof])
    battery = np.array([np.inf if t in ("gateway", "edge") else rng.uniform(1.0, 5.0)
                        for t in ntype])                          # Table 3: budget ~5 J

    kappa = cfg.kappa_db_per_m * rng.uniform(1 - cfg.kappa_variation, 1 + cfg.kappa_variation)

    # candidate neighbours: nearest nodes, kept if the link budget closes
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    order = np.argsort(D, axis=1)
    for i in range(n):
        kept = 0
        for j in order[i, 1:]:
            if kept >= cfg.max_neighbors:
                break
            j = int(j)
            if G.has_edge(i, j):
                kept += 1
                continue
            attrs = _make_link(i, j, D[i, j], cfg, rng, kappa, speed)
            if attrs is not None:
                G.add_edge(i, j, **attrs)
                kept += 1
    return Network(cfg, pos, ntype, speed, cpu, power, battery, avail, cap, G, kappa)


def _make_link(i, j, d, cfg, rng, kappa, speed):
    f = float(rng.choice(cfg.carriers_ghz)) * 1e9
    # wider channels at higher carriers (Table 3: 1 - 10 GHz)
    frac = (f / 1e9 - min(cfg.carriers_ghz)) / (max(cfg.carriers_ghz) - min(cfg.carriers_ghz))
    b = (cfg.bandwidth_ghz[0] + frac * (cfg.bandwidth_ghz[1] - cfg.bandwidth_ghz[0])) * 1e9
    pt_dbm = 10.0 * math.log10(rng.uniform(*cfg.pt_mw))
    g_db = rng.uniform(*cfg.gain_dbi) + rng.uniform(*cfg.gain_dbi)
    shadow = rng.normal(0.0, cfg.shadow_sigma_db)
    has_ris = rng.random() < cfg.ris_fraction
    ris = rng.uniform(*cfg.ris_gain_db) if has_ris else 0.0

    s = snr_db(f, b, d, pt_dbm, g_db, kappa, cfg, shadow, ris)
    if s < cfg.snr_min_db:
        return None
    r_max = max_range(f, b, pt_dbm, g_db, kappa, cfg, ris)
    v = max(speed[i], speed[j])
    margin = max(r_max - d, 1.0)
    instab = 1.0 - math.exp(-v * cfg.horizon_s / margin) if v > 0 else 0.0
    instab = min(instab, 0.5)
    exp_ho = v * cfg.epoch_s / max(r_max, 1.0) if v > 0 else 0.0     # E[N_HO] contribution
    p, ber = link_success(s, instab, cfg)
    rate = b * math.log2(1.0 + 10 ** (s / 10.0))
    delay_ms = (d / V_WATER + cfg.packet_bits / rate) * 1e3 \
        + rng.uniform(*cfg.queue_ms) * cfg.traffic_load
    pt_w = 10 ** (pt_dbm / 10.0) / 1e3
    return dict(d=float(d), f=f, b=b, pt_dbm=pt_dbm, g_db=g_db, shadow=shadow,
                ris_db=ris, snr_db=float(s), snr_noris_db=float(s - ris), ber=float(ber),
                instab=instab, exp_ho=exp_ho, psucc=float(p), rate=rate,
                delay_ms=float(delay_ms), e_bit=(2.0 * pt_w) / rate + 2.0 * cfg.e_elec_j_bit,
                rb_cap=int(rng.integers(cfg.rb_cap[0], cfg.rb_cap[1] + 1)))


def path_success(G, path_nodes):
    """Eq. (3): product of link success probabilities along the path."""
    p = 1.0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        p *= G[a][b]["psucc"]
    return p