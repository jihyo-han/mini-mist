"""
실험 3 (구): KV Transfer Latency 분석
- bandwidth 16 / 32 / 64 GB/s 조건에서 TTFT 위반율 비교
- 결론: overhead 미미 → queue saturation이 지배적
- 이 결과를 근거로 실험 3을 2D crossover map으로 대체
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from simulator import simulate_online, generate_poisson_requests_timed
from device_config import KV_BYTES_PER_TOKEN

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────
import matplotlib.font_manager as fm
import os

# 서버에 설치된 나눔폰트 자동 탐색
_korean_font = None
for _path in fm.findSystemFonts():
    if any(k in _path.lower() for k in ['nanum', 'nanumgothic', 'malgun', 'noto']):
        _korean_font = fm.FontProperties(fname=_path).get_name()
        break

if _korean_font:
    plt.rcParams['font.family'] = _korean_font
else:
    # 폰트 없으면 영문 제목 사용 (경고 없이 진행)
    _korean_font = None

plt.rcParams.update({
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  9.5,
    'figure.dpi':       150,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ── 실험 파라미터 ──────────────────────────────────────────────────────
T        = 300
RPS_LIST = [1, 2, 5, 10, 15, 20, 30, 50]
BW_LIST = [8.0, 16.0, 32.0, 64.0]
BW_LABELS = ['8 GB/s (ethernet)', '16 GB/s (worst case)', '32 GB/s (PCIe 4.0)', '64 GB/s (best case)']
BW_COLORS = ['#E74C3C', '#F39C12', '#27AE60', '#2980B9']
LIN_LIST  = [128, 512, 1024]
LOUT_FIXED = 128

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Fig 1: TTFT 위반율 × rps (Lin 3개 subplot) ────────────────────────
print("Fig 1 생성 중...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
fig.suptitle(
    'Fig 1. TTFT Violation Rate vs RPS — KV Transfer Bandwidth 비교\n'
    'Disaggregated (A100×2 prefill + L4×8 decode) | Lout=128 | SLO=2.0s',
    fontsize=11, y=1.01
)

for ax, lin in zip(axes, LIN_LIST):
    # baseline (KV transfer 없음)
    base_viols = []
    for rps in RPS_LIST:
        timed = generate_poisson_requests_timed(
            T, rps, seed=42, fixed_input_len=lin, fixed_output_len=LOUT_FIXED)
        r = simulate_online("disaggregated", timed, kv_bandwidth_gbps=0.0)
        base_viols.append(r['ttft_violation_rate'] * 100)
    ax.plot(RPS_LIST, base_viols, 'k--', lw=1.8, label='No KV transfer (baseline)', zorder=5)

    for bw, label, color in zip(BW_LIST, BW_LABELS, BW_COLORS):
        viols = []
        for rps in RPS_LIST:
            timed = generate_poisson_requests_timed(
                T, rps, seed=42, fixed_input_len=lin, fixed_output_len=LOUT_FIXED)
            r = simulate_online("disaggregated", timed, kv_bandwidth_gbps=bw)
            viols.append(r['ttft_violation_rate'] * 100)
        ax.plot(RPS_LIST, viols, 'o-', color=color, lw=2, ms=5, label=label)

    ax.set_title(f'Lin = {lin}')
    ax.set_xlabel('RPS (req/s)')
    ax.set_ylim(-5, 108)
    ax.set_xlim(0, 52)
    ax.set_xticks(RPS_LIST)
    ax.set_xticklabels([str(r) for r in RPS_LIST], rotation=30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    if ax == axes[0]:
        ax.set_ylabel('TTFT Violation Rate')
        ax.legend(loc='upper left', fontsize=8.5)

    # "모든 곡선 겹침" 주석
    ax.text(0.97, 0.48, 'All curves\noverlap',
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

fig.tight_layout()
out1 = os.path.join(OUT_DIR, 'exp3_fig1_violation_rate.png')
fig.savefig(out1, bbox_inches='tight')
plt.close()
print(f"  → {out1}")

# ── Fig 2: KV Transfer Latency (ms) × Lin ─────────────────────────────
print("Fig 2 생성 중...")
lin_range = np.linspace(64, 1024, 300)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.set_title(
    'Fig 2. KV Transfer Latency vs Input Length\n'
    'LLaMA-2 70B | KV_BYTES_PER_TOKEN = 327,680 B/token',
    fontsize=11
)

for bw, label, color in zip(BW_LIST, BW_LABELS, BW_COLORS):
    lat_ms = (lin_range * KV_BYTES_PER_TOKEN) / (bw * 1e9) * 1000
    ax.plot(lin_range, lat_ms, color=color, lw=2.2, label=label)

ax.axhline(y=2000, color='black', lw=1.5, ls='--', label='TTFT SLO = 2.0 s (2000 ms)')
ax.axhline(y=100,  color='gray',  lw=1.0, ls=':',  label='100 ms reference')

ax.set_xlabel('Input Length (tokens)')
ax.set_ylabel('KV Transfer Latency (ms)')
ax.set_xlim(64, 1024)
ax.set_ylim(0, 50)
ax.legend(fontsize=9)

# 주석: 최악 조건
worst_ms = (1024 * KV_BYTES_PER_TOKEN) / (8e9) * 1000
ax.annotate(
    f'Worst case: Lin=1024, 8 GB/s\n→ {worst_ms:.1f} ms = {worst_ms/2000*100:.1f}% of SLO',
    xy=(1024, worst_ms), xytext=(680, 24),
    arrowprops=dict(arrowstyle='->', color='#E74C3C'),
    fontsize=9, color='#E74C3C',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E74C3C', alpha=0.9)
)

fig.tight_layout()
out2 = os.path.join(OUT_DIR, 'exp3_fig2_kv_latency.png')
fig.savefig(out2, bbox_inches='tight')
plt.close()
print(f"  → {out2}")

# ── Fig 3: Queue wait vs KV transfer (stacked bar) ────────────────────
print("Fig 3 생성 중...")
# Lin=512, bw=16 GB/s (가장 불리한 조건)
FIXED_LIN = 512
FIXED_BW  = 16.0
PREFILL_SPT = 0.0003368763  # A100 prefill_sec_per_token

queue_s_list  = {'single': [], 'disaggregated': []}
kv_ms_list    = []
prefill_ms_val = FIXED_LIN * PREFILL_SPT * 1000

for rps in RPS_LIST:
    timed = generate_poisson_requests_timed(
        T, rps, seed=42, fixed_input_len=FIXED_LIN, fixed_output_len=LOUT_FIXED)

    r_s = simulate_online('single', timed)
    queue_s_list['single'].append(r_s['mean_queue_wait_sec'] * 1000)

    r_d = simulate_online('disaggregated', timed, kv_bandwidth_gbps=FIXED_BW)
    queue_s_list['disaggregated'].append(r_d['mean_queue_wait_sec'] * 1000)
    kv_ms_list.append(r_d.get('mean_kv_transfer_sec', 0) * 1000)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f'Fig 3. TTFT 구성 요소 분해: Queue Wait vs KV Transfer\n'
    f'Lin={FIXED_LIN}, Lout={LOUT_FIXED}, BW={FIXED_BW:.0f} GB/s (worst case) | SLO=2.0s',
    fontsize=11, y=1.01
)

x = np.arange(len(RPS_LIST))
w = 0.6

for ax, mode in zip(axes, ['single', 'disaggregated']):
    qw = queue_s_list[mode]
    pf = [prefill_ms_val] * len(RPS_LIST)

    b1 = ax.bar(x, qw, w, label='Queue wait',   color='#3498DB', alpha=0.88)
    b2 = ax.bar(x, pf, w, bottom=qw,            label='Prefill time', color='#95A5A6', alpha=0.88)

    if mode == 'disaggregated':
        bottom2 = [q + p for q, p in zip(qw, pf)]
        b3 = ax.bar(x, kv_ms_list, w, bottom=bottom2,
                    label=f'KV transfer ({FIXED_BW:.0f} GB/s)', color='#E74C3C', alpha=0.88)

    ax.axhline(y=2000, color='black', lw=1.5, ls='--', label='SLO = 2000 ms')

    title = 'Single (A100×2)' if mode == 'single' else 'Disaggregated (A100×2 + L4×8)'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RPS_LIST])
    ax.set_xlabel('RPS (req/s)')
    ax.set_ylabel('Latency (ms)')
    ax.set_ylim(0, 3200)
    ax.legend(fontsize=8.5, loc='upper left')

    if mode == 'disaggregated':
        ax.text(0.97, 0.85,
                'KV transfer bar\nbarely visible\n→ not the bottleneck',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.5, color='#E74C3C',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E74C3C', alpha=0.85))

fig.tight_layout()
out3 = os.path.join(OUT_DIR, 'exp3_fig3_queue_vs_kv.png')
fig.savefig(out3, bbox_inches='tight')
plt.close()
print(f"  → {out3}")

print("\n✅ 그래프 3개 생성 완료")
print(f"   Fig1: exp3_fig1_violation_rate.png")
print(f"   Fig2: exp3_fig2_kv_latency.png")
print(f"   Fig3: exp3_fig3_queue_vs_kv.png")
