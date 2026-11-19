"""Step 2 check: build one underwater network and print its physical-layer statistics."""
import networkx as nx
import numpy as np

from config import Config
from network import build_network, max_range

cfg = Config()
net = build_network(cfg, np.random.default_rng(0))
G = net.G
links = [e for _, _, e in G.edges(data=True)]
largest = max(nx.connected_components(G), key=len)

print(f"nodes: {net.n}  links: {G.number_of_edges()}  largest component: {len(largest)}/{net.n}")
print("node types:", {t: net.ntype.count(t) for t in ("gateway", "auv", "edge", "sensor")})
print(f"absorption kappa (this scenario): {net.kappa_db:.4f} dB/m at 100 GHz")
print(f"link length   median {np.median([e['d'] for e in links]):6.1f} m")
print(f"SNR (eff.)    median {np.median([e['snr_db'] for e in links]):6.1f} dB")
print(f"RIS links     {np.mean([e['ris_db'] > 0 for e in links]) * 100:5.1f} %")
print(f"P_succ link   median {np.median([e['psucc'] for e in links]):.5f}   10th pct {np.quantile([e['psucc'] for e in links], 0.1):.4f}")
print(f"link delay    median {np.median([e['delay_ms'] for e in links]):6.2f} ms")

# how far can a 100 GHz link reach for the absorption values swept in the paper (Fig. 5a)?
print("max range of a typical 100 GHz link (Pt = 50 mW, 30 dBi total gain, no RIS):")
for k in (0.02, 0.1, 0.5, 1.0, 2.5):
    r = max_range(100e9, 1e9, 10 * np.log10(50), 30.0, k, cfg, 0.0)
    print(f"   kappa = {k:4.2f} dB/m  ->  {r:7.1f} m")