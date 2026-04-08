"""
실험 2: TTFT SLO 기반 Crossover 탐색
Lin/Lout 4가지 조합 × rps sweep × single/disaggregated 비교
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import generate_poisson_requests_timed, simulate_online

# ── 실험 파라미터 ──────────────────────────────────────────────────────

T        = 300      # 시뮬레이션 시간 (초)
SEED     = 42
TTFT_SLO = 2.0      # seconds

RPS_LIST = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

LEN_COMBOS = [
    ("짧+짧", 128,  64),
    ("짧+길", 128, 256),
    ("길+짧", 1024,  64),
    ("길+길", 1024, 256),
]

# ── sweep 실행 ─────────────────────────────────────────────────────────

print("=" * 90)
print(f"실험 2: TTFT SLO={TTFT_SLO}s 위반율 — Lin/Lout 조합별 rps sweep (T={T}s)")
print("=" * 90)

header = (f"{'조합':<8} {'Lin':>5} {'Lout':>5} {'모드':<16} "
          + "  ".join(f"rps={r:<5.1f}" for r in RPS_LIST))
print(header)
print("-" * len(header))

results = {}  # (combo_name, mode) -> [violation_rate per rps]

for combo_name, lin, lout in LEN_COMBOS:
    for mode in ["single", "disaggregated"]:
        row_vals = []
        for rps in RPS_LIST:
            timed = generate_poisson_requests_timed(
                duration_sec=T, rps=rps, seed=SEED,
                fixed_input_len=lin, fixed_output_len=lout
            )
            if len(timed) == 0:
                row_vals.append(float("nan"))
                continue
            r = simulate_online(mode, timed, ttft_slo=TTFT_SLO)
            row_vals.append(r["ttft_violation_rate"])

        results[(combo_name, mode)] = row_vals
        viol_str = "  ".join(f"{v*100:>8.1f}%" for v in row_vals)
        print(f"{combo_name:<8} {lin:>5} {lout:>5} {mode:<16} {viol_str}")

    print()

# ── Crossover point 요약 ───────────────────────────────────────────────

print("=" * 90)
print("Crossover 요약: Disagg TTFT violation < 50% 를 만족하는 최대 rps")
print("-" * 90)

for combo_name, lin, lout in LEN_COMBOS:
    vals = results[(combo_name, "disaggregated")]
    crossover_rps = None
    for rps, v in zip(RPS_LIST, vals):
        if v < 0.5:
            crossover_rps = rps
    label = f"{crossover_rps}" if crossover_rps is not None else "없음 (전 구간 위반)"
    print(f"  {combo_name} (Lin={lin}, Lout={lout}): 최대 rps ≈ {label}")

print()
print("※ 그래프는 별도 생성 예정")
