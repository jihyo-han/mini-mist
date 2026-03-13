import matplotlib
import matplotlib.pyplot as plt
import random
from runtime_model import get_latency_sec
from device_config import DEVICES
from simulator import generate_poisson_requests_timed, simulate_online

# ── 데이터 수집 ────────────────────────────────────────────────────────
rps_list = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]

results = {"single": [], "disaggregated": []}
for rps in rps_list:
    timed = generate_poisson_requests_timed(duration_sec=300, rps=rps, seed=42)
    for mode in ["single", "disaggregated"]:
        r = simulate_online(mode, timed)
        results[mode].append(r)

mean_single  = [r["mean_latency_sec"]    for r in results["single"]]
mean_disagg  = [r["mean_latency_sec"]    for r in results["disaggregated"]]
p99_single   = [r["p99_latency_sec"]     for r in results["single"]]
p99_disagg   = [r["p99_latency_sec"]     for r in results["disaggregated"]]
cost_single  = [r["cost_per_token"]*1e6  for r in results["single"]]
cost_disagg  = [r["cost_per_token"]*1e6  for r in results["disaggregated"]]
qwait_single = [r["mean_queue_wait_sec"] for r in results["single"]]
qwait_disagg = [r["mean_queue_wait_sec"] for r in results["disaggregated"]]
energy_single = [r["joule_per_token"] for r in results["single"]]
energy_disagg = [r["joule_per_token"] for r in results["disaggregated"]]

# ── 플롯 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("mini-MIST: Heterogeneous GPU Disaggregation\n"
             "A100 (Prefill) + L4 (Decode)  vs  A100-only Baseline",
             fontsize=13, fontweight='bold', y=1.02)

C_S = "#E74C3C"
C_D = "#2980B9"

# subplot 1: Latency
ax = axes[0]
ax.plot(rps_list, mean_single, 'o-',  color=C_S, label='Single (mean)', lw=2)
ax.plot(rps_list, mean_disagg, 'o-',  color=C_D, label='Disagg (mean)', lw=2)
ax.plot(rps_list, p99_single,  's--', color=C_S, label='Single (P99)',  lw=1.5, alpha=0.7)
ax.plot(rps_list, p99_disagg,  's--', color=C_D, label='Disagg (P99)',  lw=1.5, alpha=0.7)
ax.set_xlabel("Request Rate (req/s)", fontsize=11)
ax.set_ylabel("Latency (s)", fontsize=11)
ax.set_title("E2E Latency vs Request Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
for i, rps in enumerate(rps_list):
    if mean_disagg[i] < mean_single[i]:
        ax.axvline(x=rps, color='gray', linestyle=':', alpha=0.5)
        ax.text(rps+0.2, max(mean_single)*0.85,
                f'disagg\nbetter\n→ rps≥{rps}', fontsize=8, color='gray')
        break

# subplot 2: Cost
ax = axes[1]
ax.plot(rps_list, cost_single, 'o-', color=C_S, label='Single (A100)',    lw=2)
ax.plot(rps_list, cost_disagg, 'o-', color=C_D, label='Disagg (A100+L4)', lw=2)
ax.set_xlabel("Request Rate (req/s)", fontsize=11)
ax.set_ylabel("Cost per Token (μ$)", fontsize=11)
ax.set_title("Cost Efficiency vs Request Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# subplot 3: Queue Wait
ax = axes[2]
ax.plot(rps_list, qwait_single, 'o-', color=C_S, label='Single', lw=2)
ax.plot(rps_list, qwait_disagg, 'o-', color=C_D, label='Disagg', lw=2)
ax.set_xlabel("Request Rate (req/s)", fontsize=11)
ax.set_ylabel("Mean Queue Wait (s)", fontsize=11)
ax.set_title("Queuing Delay vs Request Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# subplot 4: Energy
ax = axes[3]
ax.plot(rps_list, energy_single, 'o-', color=C_S, label='Single (A100)',    lw=2)
ax.plot(rps_list, energy_disagg, 'o-', color=C_D, label='Disagg (A100+L4)', lw=2)
ax.set_xlabel("Request Rate (req/s)", fontsize=11)
ax.set_ylabel("Energy per Token (J)", fontsize=11)
ax.set_title("Energy Efficiency vs Request Rate", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
print("저장 완료: results.png")
