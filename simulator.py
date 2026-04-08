import random
from runtime_model import get_latency_sec
from device_config import DEVICES, KV_BYTES_PER_TOKEN

TTFT_SLO = 2.0  # seconds


# ── 기존 순차 시뮬레이션 (변경 없음) ──────────────────────────────────

def simulate(mode, requests):
    total_time = 0.0
    total_cost = 0.0
    total_energy = 0.0
    total_tokens = 0

    if mode == "single":
        for input_len, output_len in requests:
            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("A100", "decode", 1, input_len) * output_len
            req_time = p_time + d_time
            total_time   += req_time
            total_cost   += DEVICES["A100"]["price_per_hr"] * (req_time / 3600)
            total_energy += DEVICES["A100"]["tdp_watt"] * req_time
            total_tokens += input_len + output_len

    else:  # disaggregated
        for input_len, output_len in requests:
            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("L4",   "decode",  1, input_len) * output_len
            req_time = p_time + d_time
            total_time   += req_time
            total_cost   += (DEVICES["A100"]["price_per_hr"] * (p_time / 3600)
                           + DEVICES["L4"]["price_per_hr"]   * (d_time / 3600))
            total_energy += (DEVICES["A100"]["tdp_watt"] * p_time
                           + DEVICES["L4"]["tdp_watt"]   * d_time)
            total_tokens += input_len + output_len

    return {
        "cost_per_token":  total_cost   / total_tokens,
        "joule_per_token": total_energy / total_tokens,
        "total_time_sec":  total_time,
    }


# ── 동시 요청 도착 시뮬레이션 ──────────────────────────────────────────

def generate_poisson_requests(n, rps, seed=42):
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for _ in range(n):
        t += rng.expovariate(rps)
        input_len  = rng.randint(128, 1024)
        output_len = rng.randint(64, 256)
        reqs.append((t, input_len, output_len))
    return reqs


def generate_poisson_requests_timed(
    duration_sec, rps, seed=42,
    fixed_input_len=None, fixed_output_len=None
):
    """
    시간 T초 동안 Poisson 도착 프로세스로 요청 생성.

    fixed_input_len  : None이면 randint(128, 1024), 정수 지정 시 고정값 사용
    fixed_output_len : None이면 randint(64, 256),   정수 지정 시 고정값 사용
    """
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    while t < duration_sec:
        t += rng.expovariate(rps)
        if t >= duration_sec:
            break
        input_len  = fixed_input_len  if fixed_input_len  is not None else rng.randint(128, 1024)
        output_len = fixed_output_len if fixed_output_len is not None else rng.randint(64, 256)
        reqs.append((t, input_len, output_len))
    return reqs


def simulate_online(mode, timed_requests, ttft_slo=TTFT_SLO, kv_bandwidth_gbps=0.0):
    """
    동시 요청 도착을 고려한 이벤트 기반 시뮬레이션.

    Parameters
    ----------
    mode               : "single" or "disaggregated"
    timed_requests     : [(arrival_time, input_len, output_len), ...]
    ttft_slo           : TTFT SLO 기준 (초), 기본값 2.0s
    kv_bandwidth_gbps  : KV cache transfer 대역폭 (GB/s).
                         0.0이면 transfer latency 없음 (실험 1~2 호환).
                         disaggregated 모드에서만 적용됨.
                         예) PCIe: 32.0, NVLink: 600.0

    Returns
    -------
    dict: cost_per_token, joule_per_token, total_time_sec,
          mean_latency_sec, p99_latency_sec, mean_queue_wait_sec,
          ttft_violation_rate, mean_kv_transfer_sec (disaggregated 전용)
    """
    total_tokens  = sum(il + ol for _, il, ol in timed_requests)
    e2e_latencies = []
    queue_waits   = []
    ttft_list     = []
    kv_transfer_times = []

    a100_free_at = 0.0
    l4_free_at   = 0.0
    a100_busy    = 0.0
    l4_busy      = 0.0

    if mode == "single":
        # A100이 prefill + decode 모두 처리 (KV transfer 없음)
        for arrival_time, input_len, output_len in timed_requests:
            start      = max(arrival_time, a100_free_at)
            queue_wait = start - arrival_time

            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("A100", "decode",  1, input_len) * output_len

            finish       = start + p_time + d_time
            a100_free_at = finish
            a100_busy   += p_time + d_time

            ttft = queue_wait + p_time

            e2e_latencies.append(finish - arrival_time)
            queue_waits.append(queue_wait)
            ttft_list.append(ttft)

        wall_time    = a100_free_at
        total_cost   = DEVICES["A100"]["price_per_hr"] * (a100_free_at / 3600)
        total_energy = DEVICES["A100"]["tdp_watt"] * a100_busy

    else:  # disaggregated
        # KV transfer 대역폭 (bytes/sec)
        bw_bytes_per_sec = kv_bandwidth_gbps * 1e9 if kv_bandwidth_gbps > 0.0 else float("inf")

        for arrival_time, input_len, output_len in timed_requests:
            prefill_start = max(arrival_time, a100_free_at)
            queue_wait    = prefill_start - arrival_time

            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("L4",   "decode",  1, input_len) * output_len

            # KV transfer: prefill 완료 후 A100 → L4 전송
            # transfer 크기 = input_len × KV_BYTES_PER_TOKEN
            kv_transfer_sec = (input_len * KV_BYTES_PER_TOKEN) / bw_bytes_per_sec

            prefill_end  = prefill_start + p_time
            a100_free_at = prefill_end          # A100은 prefill 완료 후 즉시 다음 요청 가능
            a100_busy   += p_time

            # decode는 (prefill 완료 + KV transfer 완료) 이후에 시작 가능
            transfer_end = prefill_end + kv_transfer_sec
            decode_start = max(transfer_end, l4_free_at)
            decode_end   = decode_start + d_time
            l4_free_at   = decode_end
            l4_busy     += d_time

            # TTFT = queue_wait + prefill_time + kv_transfer_time
            # (decode 시작 전까지 첫 토큰이 나오지 않으므로 transfer도 포함)
            ttft = queue_wait + p_time + kv_transfer_sec

            e2e_latencies.append(decode_end - arrival_time)
            queue_waits.append(queue_wait)
            ttft_list.append(ttft)
            kv_transfer_times.append(kv_transfer_sec)

        wall_time    = max(a100_free_at, l4_free_at)
        total_cost   = (DEVICES["A100"]["price_per_hr"] * (a100_free_at / 3600)
                      + DEVICES["L4"]["price_per_hr"]   * (l4_free_at   / 3600))
        total_energy = (DEVICES["A100"]["tdp_watt"] * a100_busy
                      + DEVICES["L4"]["tdp_watt"]   * l4_busy)

    e2e_sorted = sorted(e2e_latencies)
    p99_idx    = max(0, int(len(e2e_sorted) * 0.99) - 1)
    n_violations = sum(1 for t in ttft_list if t > ttft_slo)

    result = {
        "cost_per_token":      total_cost   / total_tokens,
        "joule_per_token":     total_energy / total_tokens,
        "total_time_sec":      wall_time,
        "mean_latency_sec":    sum(e2e_latencies) / len(e2e_latencies),
        "p99_latency_sec":     e2e_sorted[p99_idx],
        "mean_queue_wait_sec": sum(queue_waits)   / len(queue_waits),
        "ttft_violation_rate": n_violations / len(ttft_list),
    }

    # disaggregated 전용 출력값
    if kv_transfer_times:
        result["mean_kv_transfer_sec"] = sum(kv_transfer_times) / len(kv_transfer_times)

    return result


# ── 실행 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    requests = [
        (200, 50),
        (300, 80),
        (150, 40),
    ]

    print("=== Single A100 (순차) ===")
    print(simulate("single", requests))

    print("\n=== Disaggregated A100+L4 (순차) ===")
    print(simulate("disaggregated", requests))

    T = 300
    print("\n" + "=" * 65)
    print(f"[ Poisson arrival 시뮬레이션 — T={T}s 시간 기반 ]")
    print(f"{'모드':<22} {'rps':>4}  {'mean_lat':>9} {'p99_lat':>9} "
          f"{'queue_wait':>11} {'cost/tok':>12} {'joule/tok':>12} {'ttft_viol':>10}")
    print("=" * 65)

    for rps in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        timed = generate_poisson_requests_timed(duration_sec=T, rps=rps, seed=42)

        for mode in ["single", "disaggregated"]:
            r = simulate_online(mode, timed)
            print(f"  {mode:<20} {rps:>4.0f}  "
                  f"{r['mean_latency_sec']:>9.3f}s "
                  f"{r['p99_latency_sec']:>9.3f}s "
                  f"{r['mean_queue_wait_sec']:>11.3f}s "
                  f"{r['cost_per_token']:>12.9f} "
                  f"{r['joule_per_token']:>12.6f} "
                  f"{r['ttft_violation_rate']:>9.1%}")
        print()

    # ── 실험 3 미리보기: KV transfer 대역폭 비교 ──────────────────────
    print("\n" + "=" * 75)
    print("[ 실험 3 미리보기: KV Transfer Bandwidth 비교 (Lin=512, Lout=128) ]")
    print(f"{'bandwidth':>12}  {'rps':>4}  {'ttft_viol':>10}  {'mean_kv_xfer':>14}")
    print("=" * 75)

    for bw in [16.0, 32.0, 64.0, 0.0]:
        label = f"{bw:.0f} GB/s" if bw > 0 else "∞ (no xfer)"
        for rps in [1, 5, 10, 20]:
            timed = generate_poisson_requests_timed(
                duration_sec=T, rps=rps, seed=42,
                fixed_input_len=512, fixed_output_len=128
            )
            r = simulate_online("disaggregated", timed, kv_bandwidth_gbps=bw)
            kv_str = f"{r.get('mean_kv_transfer_sec', 0.0)*1000:.2f} ms"
            print(f"  {label:>12}  {rps:>4}  {r['ttft_violation_rate']:>9.1%}  {kv_str:>14}")
        print()
