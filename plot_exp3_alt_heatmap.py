"""
실험 3 (대체): Energy 2D Crossover Heatmap 시각화 [수정본]
- exp3_alt_results.csv 읽어서 Energy 히트맵 1개 생성 (Cost 제거)
- 음수(파랑) = Disagg 에너지 우위, 양수(빨강) = Single 에너지 우위
- x축: rps, y축: Lin (위쪽이 큰 값)
- crossover 이론값: g() = 1 / (Lin × prefill_sec_per_token["A100"])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from device_config import DEVICES

df = pd.read_csv("exp3_alt_results.csv")

RPS_LIST     = [1, 2, 5, 10, 20, 50, 100]
LIN_LIST     = [128, 256, 512, 1024]
LIN_LIST_REV = list(reversed(LIN_LIST))   # y축: 위=1024, 아래=128

# ── g() 공식: RPS* = 1 / (Lin × prefill_sec_per_token["A100"]) ──────
# device_config에서 직접 가져옴 (하드코딩 없음)
PREFILL_A100  = DEVICES["A100"]["prefill_sec_per_token"]
crossover_rps = {lin: 1.0 / (lin * PREFILL_A100) for lin in LIN_LIST}

# ── 매트릭스 생성 ──────────────────────────────────────────────────────

def make_matrix(col):
    mat = np.zeros((len(LIN_LIST_REV), len(RPS_LIST)))
    for i, lin in enumerate(LIN_LIST_REV):
        for j, rps in enumerate(RPS_LIST):
            row = df[(df["rps"] == rps) & (df["lin"] == lin)]
            if len(row) > 0:
                mat[i, j] = row[col].values[0]
    return mat

joule_mat = make_matrix("joule_diff_pct")

# ── 플롯 ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))

vmax = max(abs(joule_mat.min()), abs(joule_mat.max()))
im   = ax.imshow(joule_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

ax.set_xticks(range(len(RPS_LIST)))
ax.set_xticklabels([str(r) for r in RPS_LIST])
ax.set_yticks(range(len(LIN_LIST_REV)))
ax.set_yticklabels([str(l) for l in LIN_LIST_REV])
ax.set_xlabel("RPS (requests/sec)", fontsize=12)
ax.set_ylabel("Lin (input length)", fontsize=12)
ax.set_title("Total Energy: (Disagg − Single) / Single × 100 (%)", fontsize=13, fontweight="bold")

# 셀 값 표시
for i in range(len(LIN_LIST_REV)):
    for j in range(len(RPS_LIST)):
        val   = joule_mat[i, j]
        txt   = f"{val:+.1f}%"
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

# g() crossover 이론값 (★)
rps_arr = np.array(RPS_LIST, dtype=float)
plotted = False
for i, lin in enumerate(LIN_LIST_REV):
    crps = crossover_rps[lin]
    if crps < rps_arr[0] or crps > rps_arr[-1]:
        continue
    idx = np.interp(crps, rps_arr, np.arange(len(rps_arr)))
    ax.plot(idx, i, marker="*", color="lime", markersize=13,
            markeredgecolor="black", zorder=5)
    plotted = True

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("(Disagg − Single) / Single × 100 (%)\n← Disagg 에너지 절약  |  Single 에너지 절약 →",
               fontsize=10)

# 범례
legend_elements = [
    Line2D([0], [0], marker="*", color="lime", markeredgecolor="black",
           linestyle="None", markersize=12,
           label=f"g() crossover RPS* (A100 prefill, prefill_sec={PREFILL_A100:.7f})"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

fig.suptitle(
    "Experiment 3 (Alt): Energy Crossover Map\n"
    "LLaMA-2 70B | A100×2 (Single) vs A100×2+L4×8 (Disagg) | Lout=128, SLO=2.0s",
    fontsize=12
)
plt.tight_layout()
plt.savefig("exp3_alt_fig1_energy_heatmap.png", dpi=150, bbox_inches="tight")
print("저장 완료: exp3_alt_fig1_energy_heatmap.png")
print(f"g() 공식 사용값: PREFILL_A100 = {PREFILL_A100}")
print("crossover RPS* 이론값:")
for lin in LIN_LIST:
    print(f"  Lin={lin:4d}: RPS* = {crossover_rps[lin]:.2f}")
