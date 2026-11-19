"""
ORCA reproduction - global configuration.

All values follow Table 3 of Sedaghat, "ORCA: A 6G System Using Quantum
Optimization for Underwater Survivability", IEEE Access, vol. 14, 2026.
Where the paper only gives a range, the range is sampled per scenario.
Where the paper gives no number at all, the chosen value is marked
ASSUMPTION and is discussed in the report.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ topology
    n_nodes: int = 100                     # Table 3: 50 - 1,000
    volume_m: tuple = (1000.0, 1000.0, 500.0)  # Table 3: 1 km x 1 km x 500 m
    frac_gateway: float = 0.05             # ASSUMPTION: node-type mix
    frac_auv: float = 0.15
    frac_edge: float = 0.10                # edge relays / cloudlets
    # remainder are seabed sensors (extreme edge)
    max_neighbors: int = 8                 # ASSUMPTION: keep graph sparse
    k_paths: int = 4                       # K-shortest candidate paths per service

    # ------------------------------------------------------------ 6G radio (PHY)
    carriers_ghz: tuple = (100.0, 140.0, 220.0, 300.0)  # Table 3: 100 GHz - 1 THz
    bandwidth_ghz: tuple = (1.0, 10.0)     # Table 3: 1 - 10 GHz
    pt_mw: tuple = (10.0, 100.0)           # Table 3: 10 - 100 mW
    gain_dbi: tuple = (10.0, 20.0)         # Table 3: Gtx, Grx 10 - 20 dBi
    n0_dbm_hz: float = -174.0              # Table 3
    pathloss_exp: float = 2.0              # n(f), ASSUMPTION (LoS)
    shadow_sigma_db: float = 2.0           # X_sigma, ASSUMPTION
    # Beer-Lambert molecular absorption, dB per metre at 100 GHz.
    # Paper sweeps 0.1 - 2.5 dB/m (Fig. 5a) with +/-40 % variation (Table 3).
    # Default chosen low enough that a 1 km^3 network can be connected.
    kappa_db_per_m: float = 0.02           # ASSUMPTION (see report)
    kappa_variation: float = 0.40          # Table 3: +/- 40 %
    kappa_freq_exp: float = 1.0            # kappa(f) ~ f^1, ASSUMPTION
    ris_gain_db: tuple = (5.0, 15.0)       # Table 3: 5 - 15 dB SNR improvement
    ris_fraction: float = 0.5              # share of links assisted by RIS, ASSUMPTION
    snr_min_db: float = 8.0                # link exists above this SNR, ASSUMPTION
    ber_target: float = 1e-6               # Table 3
    packet_bits: int = 1024                # ASSUMPTION
    packet_loss_rate: float = 0.02         # Table 3: 2 %
    arq_attempts: int = 2                  # ASSUMPTION: one link-layer retransmission
    e_elec_j_bit: float = 1e-9             # radio electronics energy (first-order radio model), ASSUMPTION

    # ---------------------------------------------------------------- latency
    queue_ms: tuple = (0.8, 2.5)           # per-hop queueing/forwarding, ASSUMPTION
    traffic_load: float = 1.0              # Table 3: low - extreme (multiplier)
    api_ms: float = 2.0                    # gamma_API per remote NF hop (Table 3: 1-20 ms)
    loc_ms: float = 0.1                    # gamma_loc, co-located call
    disc_ms: float = 0.5                   # D_disc service discovery
    d_max_ms: float = 30.0                 # Table 3: D_max < 30 ms
    eta_req: float = 0.95                  # Table 3: eta_req > 95 %
    c_ho_ms: tuple = (5.0, 10.0)           # Table 3: C_HO 5 - 10 ms

    # ---------------------------------------------------------------- mobility
    auv_speed: tuple = (0.5, 5.0)          # Table 3: 0.5 - 5 m/s
    horizon_s: float = 10.0                # link-holding horizon, ASSUMPTION

    # ---------------------------------------------------------------- services
    n_services: int = 6
    chain_len: tuple = (2, 6)              # Table 3 allows 2-16; reduced for QUBO size
    max_hosts_per_nf: int = 6              # osmotic candidate hosts per NF
    epoch_s: float = 60.0                  # energy accounting window
    event_rate_hz: float = 10.0            # NF invocations per second
    snapshot_bits: float = 4e6             # raw sensor window shipped to the ingestion NF, ASSUMPTION
    reduction: float = 0.05                # data left after NF_1 (ingestion/feature extraction), ASSUMPTION
    rb_cap: tuple = (12, 20)               # resource blocks per link
    rb_demand: tuple = (3, 8)              # RBs per service flow
    n_slices: int = 2                      # 0 = URLLC, 1 = mMTC/eMBB

    # ------------------------------------------------------------ QUBO weights
    # w_l in [0.1, 1.0] (Table 3)
    w: dict = field(default_factory=lambda: {
        "E": 0.6, "D": 1.0, "R": 1.0, "I": 0.3, "S": 0.4,
        "C": 1.0, "RAD": 0.5, "SBA": 1.0, "MOB": 0.5})
    lambda_energy: float = 1.0             # Eq. (15)
    penalty_scale: float = 1.0             # one-hot penalty A_g = scale * (safe bound)