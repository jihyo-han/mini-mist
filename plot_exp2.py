"""
실험 2 그래프: TTFT violation rate vs rps
4조합 × subplot, 각각 single vs disaggregated 2개 선
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import generate_poisson_requests_timed, simulate_online
from device_config import DEVICES

def theoretical_rps_max(lin):
    """Disagg A100 prefill 처리 한계: 1 / (Lin × prefill_sec_per_token)"""
    return 1.0 / (lin * DEVICES["A100"]["prefill_sec_per_token"])

# ── 파라미터 ───────────────────────────────────────────────────────────

T        = 300
SEED     = 42
TTFT_SLO = 2.0

RPS_LIST   = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
LEN_COMBOS = [
    ("Lin=128,  Lout=64",   128,  64),
    ("Lin=128,  Lout=256",  128, 256),
    ("Lin=1024, Lout=64",  1024,  64),
    ("Lin=1024, Lout=256", 1024, 256),
]

# ── 데이터 수집 ────────────────────────────────────────────────────────

data = {}
for label, lin, lout in LEN_COMBOS:
    for mode in ["single", "disaggregated"]:
        vals = []
        for rps in RPS_LIST:
            timed = generate_poisson_requests_timed(
                duration_sec=T, rps=rps, seed=SEED,
                fixed_input_len=lin, fixed_output_len=lout
            )
            if not timed:
                vals.append(float("nan"))
                continue
            r = simulate_online(mode, timed, ttft_slo=TTFT_SLO)
            vals.append(r["ttft_violation_rate"] * 100)
        data[(label, mode)] = vals

# ── 그래프 ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
axes = axes.flatten()

for ax, (label, lin, lout) in zip(axes, LEN_COMBOS):
    ax.plot(RPS_LIST, data[(label, "single")],
            marker="o", color="#d62728", label="Single (A100×2)")
    ax.plot(RPS_LIST, data[(label, "disaggregated")],
            marker="s", color="#1f77b4", linestyle="--", label="Disagg (A100+L4)")

    ax.axhline(50, color="gray", linestyle=":", linewidth=0.9)

    rps_theory = theoretical_rps_max(lin)
    ax.axvline(rps_theory, color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"Theory: {rps_theory:.1f} req/s")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Request Rate (req/s)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 60)
    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

axes[0].set_ylabel("TTFT Violation Rate (%)", fontsize=10)
axes[2].set_ylabel("TTFT Violation Rate (%)", fontsize=10)

fig.suptitle(f"TTFT SLO={TTFT_SLO}s Violation Rate vs Request Rate\n"
             f"(T={T}s, Poisson arrival, LLaMA-2 70B)", fontsize=13)

plt.tight_layout()
output_path = "exp2_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"저장 완료: {output_path}")
