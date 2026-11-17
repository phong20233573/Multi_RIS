"""
compare_schemes.py

End-to-end comparison of four transmission schemes (A/B/C/D) for the
multi-RIS UWOC system, reporting throughput [bit/s], average delay and
packet drop probability versus average SNR.

Run
---
    python src/compare_schemes.py            # full sweep
    python src/compare_schemes.py --quick    # fast sanity check
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import (
    SystemConfig, RISConfig, TurbulenceConfig, PointingErrorConfig, ReceiverConfig
)
from src.channel import generate_composite_fading
from src.per_model import amc_thresholds
from src import harq


# ----------------------------------------------------------------------------
# Tunable simulation parameters
# ----------------------------------------------------------------------------
RS = dict(n=255, k=223, s=8)     # Reed-Solomon RS(255,223) -> t = 16
R_SYM = 1e9                       # symbol rate [baud] -> throughput in bit/s
M_MAX = 3                         # max retransmissions (up to 4 attempts)
PER_QOS = 1e-2                    # per-transmission packet-error target for AMC
N_PACKETS = 100_000               # packets to deliver per scheme
M_LIST = [2, 4, 8, 16]            # AMC constellation orders
SNR_DB = np.arange(0, 31, 2)      # average SNR sweep: 0, 2, ..., 30 dB
SEED = 0


def throughput(res):
    """
    Throughput [bit/s] from a simulation result dict.

    Parallel streams (two-stream scheme) transmit on the SAME symbol clock, so
    the total throughput is the SUM of the per-stream throughputs.
    """
    L = RS["k"] * RS["s"]         # info bits per packet
    total = 0.0
    for dk, sk in zip(res["stream_delivered"], res["stream_symbols"]):
        if sk > 0:
            total += dk * L * R_SYM / sk
    return total


def drop_rate(res, N):
    return res["dropped"] / N


def total_delay_ms(res):
    """Total elapsed transmission time for the complete simulation [ms].

    Dropped packets are included because their transmission attempts consume
    channel time. For two parallel streams, elapsed time is the slower stream.
    """
    return res["elapsed_symbols"] / R_SYM * 1e3


def build_config():
    turb = TurbulenceConfig(alpha=3.35, beta=3.19)
    pe = PointingErrorConfig(sigma_theta=2.0e-3, sigma_phi=2.0e-3)
    rec = ReceiverConfig(gamma_th_db=13.0)
    cfg = SystemConfig(
        ris=RISConfig(K=2, N=[6, 6]),
        turbulence=turb,
        pointing_error=pe,
        receiver=rec,
    )
    cfg.validate()
    return cfg


def main():
    quick = "--quick" in sys.argv
    n_packets = 5000 if quick else N_PACKETS
    snr_db = SNR_DB

    # 1. AMC selection thresholds. HARQ success itself is decided by the
    # physical M-PAM samples and the real RS decoder in harq.py.
    theta_db = amc_thresholds(
        M_LIST, RS["n"], RS["k"], RS["s"], PER_QOS
    )
    print("AMC thresholds (dB) at PER target = %.4f:" % PER_QOS)
    for M in M_LIST:
        print(f"  {M:2d}-PAM : {theta_db[M]:7.2f} dB")

    # 2. Generate the channel ONCE (independent of average SNR)
    cfg = build_config()
    num_ts = int(n_packets * (M_MAX + 1))   # worst-case timeslots (single stream)
    print(f"Generating channel matrix ({num_ts} timeslots, K=2)...")
    np.random.seed(SEED)
    h = generate_composite_fading(cfg, num_ts)
    print("Channel generated. shape =", h.shape)

    K = cfg.ris.K
    base_full = (h[:, 0] + h[:, 1]) ** 2 / (K ** 2)
    base_sel = np.maximum(h[:, 0] ** 2, h[:, 1] ** 2)
    base_b1 = 0.5 * h[:, 0] ** 2
    base_b2 = 0.5 * h[:, 1] ** 2

    names = [
        "A (Full+ARQ+4PAM)",
        "B (Sel+ARQ+4PAM)",
        "C (2stream+ARQ+AMC)",
        "D (2stream+IR+AMC)",
    ]
    out = {n: {"thr": [], "delay": [], "drop": []} for n in names}

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for gdb in snr_db:
        gbar = 10.0 ** (gdb / 10.0)
        # h is the channel amplitude used by y = h*x + n.
        # Its square is the linear SNR used by AMC.
        hf = np.sqrt(gbar * base_full)
        hs = np.sqrt(gbar * base_sel)
        h1 = np.sqrt(gbar * base_b1)
        h2 = np.sqrt(gbar * base_b2)

        rA = harq.simulate_arq(
            hf, RS, theta_db, M_fixed=4, M_max=M_MAX,
            N=n_packets, rng=rng,
        )
        rB = harq.simulate_arq(hs, RS, theta_db, M_fixed=4, M_max=M_MAX, N=n_packets, rng=rng)
        rC = harq.simulate_two_stream(h1, h2, RS, theta_db, M_max=M_MAX, N=n_packets, mode="arq", rng=rng)
        rD = harq.simulate_two_stream(h1, h2, RS, theta_db, M_max=M_MAX, N=n_packets, mode="ir", rng=rng)

        for n, r in zip(names, [rA, rB, rC, rD]):
            out[n]["thr"].append(throughput(r))
            out[n]["delay"].append(total_delay_ms(r))
            out[n]["drop"].append(drop_rate(r, n_packets))

        print(
            f"SNR {gdb:4.0f} dB | thr(A/B/C/D) = "
            f"{out[names[0]]['thr'][-1]/1e9:.3f} / {out[names[1]]['thr'][-1]/1e9:.3f} / "
            f"{out[names[2]]['thr'][-1]/1e9:.3f} / {out[names[3]]['thr'][-1]/1e9:.3f} Gbit/s"
        )
    print(f"Simulation done in {time.time() - t0:.1f} s")
    plot_results(snr_db, names, out)


def plot_results(snr_db, names, out):
    colors = ["red", "blue", "green", "orange"]
    markers = ["o", "s", "^", "d"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for n, c, mk in zip(names, colors, markers):
        ax.plot(snr_db, np.array(out[n]["thr"]) / 1e9, marker=mk, color=c, linestyle="-", label=n)
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("Throughput (Gbit/s)")
    ax.set_title("Average throughput")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[1]
    for n, c, mk in zip(names, colors, markers):
        ax.plot(snr_db, out[n]["delay"],
                marker=mk, color=c, linestyle="-", label=n)
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("Total transmission time (ms)")
    ax.set_title("Total time to process all packets")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[2]
    for n, c, mk in zip(names, colors, markers):
        ax.plot(snr_db, out[n]["drop"], marker=mk, color=c, linestyle="-", label=n)
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("Packet drop probability")
    ax.set_yscale("log")
    ax.set_title("Packet drop probability (must stay below QoS)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "comparison_harq.png",
    )
    fig.savefig(out_png, dpi=150)
    print("Figure saved to", out_png)

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'SNR(dB)':>7} | " + " | ".join(f"{n:>22}" for n in names))
    print("-" * 78)
    for i, gdb in enumerate(snr_db):
        row = f"{gdb:7.0f} | " + " | ".join(f"{out[n]['thr'][i]/1e9:>21.3f}G" for n in names)
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    main()
