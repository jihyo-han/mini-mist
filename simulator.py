import random
from runtime_model import get_latency_sec
from device_config import DEVICES


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


# ── 동시 요청 도착 시뮬레이션 (신규) ──────────────────────────────────

def generate_poisson_requests(n, rps, seed=42):
    """
    Poisson 도착 프로세스로 요청 생성.
    도착 간격 ~ Exponential(1/rps)

    반환: [(arrival_time, input_len, output_len), ...]  arrival_time 오름차순
    """
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for _ in range(n):
        t += rng.expovariate(rps)          # 다음 요청까지 대기 시간
        input_len  = rng.randint(128, 1024)
        output_len = rng.randint(64, 256)
        reqs.append((t, input_len, output_len))
    return reqs




def generate_poisson_requests_timed(duration_sec, rps, seed=42):
    """
    시간 T초 동안 Poisson 도착 프로세스로 요청 생성.
    """
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    while t < duration_sec:
        t += rng.expovariate(rps)
        if t >= duration_sec:
            break
        input_len  = rng.randint(128, 1024)
        output_len = rng.randint(64, 256)
        reqs.append((t, input_len, output_len))
    return reqs

def simulate_online(mode, timed_requests):
    """
    동시 요청 도착을 고려한 이벤트 기반 시뮬레이션.

    Parameters
    ----------
    mode            : "single" or "disaggregated"
    timed_requests  : [(arrival_time, input_len, output_len), ...]
                       arrival_time 오름차순 정렬 필요

    Returns
    -------
    dict: cost_per_token, joule_per_token, total_time_sec,
          mean_latency_sec, p99_latency_sec, mean_queue_wait_sec
    """
    total_tokens  = sum(il + ol for _, il, ol in timed_requests)
    e2e_latencies = []
    queue_waits   = []

    a100_free_at = 0.0
    l4_free_at   = 0.0
    a100_busy    = 0.0
    l4_busy      = 0.0

    if mode == "single":
        # A100이 prefill + decode 모두 처리
        # 이전 요청이 끝날 때까지 다음 요청은 큐에서 대기
        for arrival_time, input_len, output_len in timed_requests:
            start      = max(arrival_time, a100_free_at)
            queue_wait = start - arrival_time

            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("A100", "decode",  1, input_len) * output_len

            finish       = start + p_time + d_time
            a100_free_at = finish
            a100_busy   += p_time + d_time

            e2e_latencies.append(finish - arrival_time)
            queue_waits.append(queue_wait)

        wall_time    = a100_free_at
        total_cost   = DEVICES["A100"]["price_per_hr"] * (a100_free_at / 3600)
        total_energy = DEVICES["A100"]["tdp_watt"] * a100_busy

    else:  # disaggregated
        # A100: prefill 전담 / L4: decode 전담
        # prefill 끝나는 즉시 L4로 넘김 → 두 GPU가 파이프라인으로 돌아감
        for arrival_time, input_len, output_len in timed_requests:
            # prefill: 도착 후 A100이 빌 때까지 대기
            prefill_start = max(arrival_time, a100_free_at)
            queue_wait    = prefill_start - arrival_time

            p_time = get_latency_sec("A100", "prefill", 1, input_len)
            d_time = get_latency_sec("L4",   "decode",  1, input_len) * output_len

            prefill_end  = prefill_start + p_time
            a100_free_at = prefill_end
            a100_busy   += p_time

            # decode: prefill 끝난 직후, L4가 비면 바로 시작
            decode_start = max(prefill_end, l4_free_at)
            decode_end   = decode_start + d_time
            l4_free_at   = decode_end
            l4_busy     += d_time

            e2e_latencies.append(decode_end - arrival_time)
            queue_waits.append(queue_wait)

        wall_time    = max(a100_free_at, l4_free_at)
        total_cost   = (DEVICES["A100"]["price_per_hr"] * (a100_free_at / 3600)
                      + DEVICES["L4"]["price_per_hr"]   * (l4_free_at   / 3600))
        total_energy = (DEVICES["A100"]["tdp_watt"] * a100_busy
                      + DEVICES["L4"]["tdp_watt"]   * l4_busy)

    e2e_sorted = sorted(e2e_latencies)
    p99_idx    = max(0, int(len(e2e_sorted) * 0.99) - 1)

    return {
        "cost_per_token":      total_cost   / total_tokens,
        "joule_per_token":     total_energy / total_tokens,
        "total_time_sec":      wall_time,
        "mean_latency_sec":    sum(e2e_latencies) / len(e2e_latencies),
        "p99_latency_sec":     e2e_sorted[p99_idx],
        "mean_queue_wait_sec": sum(queue_waits)   / len(queue_waits),
    }


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

    T = 300  # 시뮬레이션 시간 300초 (5분)
    print("\n" + "=" * 65)
    print(f"[ Poisson arrival 시뮬레이션 — T={T}s 시간 기반 ]")
    print(f"{'모드':<22} {'rps':>4}  {'mean_lat':>9} {'p99_lat':>9} "
          f"{'queue_wait':>11} {'cost/tok':>12} {'joule/tok':>12}")
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
                  f"{r['joule_per_token']:>12.6f}")
        print()