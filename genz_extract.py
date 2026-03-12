from GenZ.system import System
from GenZ.unit import Unit
from GenZ.LLM_inference import llm_prefill, llm_decode
import pandas as pd

u = Unit()

HW_CONFIGS = [
    ("A100", System(unit=u, flops=312,   offchip_mem_bw=2000, off_chip_mem_size=81920, bits='bf16'), 2),
    ("L4",   System(unit=u, flops=121.3, offchip_mem_bw=300,  off_chip_mem_size=24576, bits='bf16'), 8),
]

INPUT_LENS = [128, 256, 512, 1024]
rows = []

for hw_name, system, tp in HW_CONFIGS:
    print(f"\n[{hw_name}] (tensor_parallel={tp})")
    for input_len in INPUT_LENS:
        try:
            p = llm_prefill.prefill_moddeling(
                model='llama-2-70b', batch_size=1,
                input_tokens=input_len,
                system_name=system, bits='bf16', tensor_parallel=tp,
            )
            d = llm_decode.decode_moddeling(
                model='llama-2-70b', batch_size=1,
                input_tokens=input_len,
                system_name=system, bits='bf16', tensor_parallel=tp,
            )
            prefill_sec = float(p['Latency']) / 1000
            decode_sec  = float(d['Latency']) / 1000

            print(f"  input={input_len:4d}  prefill={prefill_sec*1000:.2f}ms  decode/tok={decode_sec*1000:.3f}ms")
            rows.append({
                "hw": hw_name, "input_len": input_len,
                "prefill_sec": prefill_sec,
                "prefill_sec_per_token": prefill_sec / input_len,
                "decode_sec_per_token": decode_sec,
            })
        except Exception as e:
            print(f"  input={input_len}: 오류 → {e}")

if not rows:
    print("\n결과 없음")
else:
    df = pd.DataFrame(rows)
    df.to_csv("genz_results.csv", index=False)
    print("\n결과 저장: genz_results.csv")
    print("\n" + "=" * 55)
    print("device_config.py 업데이트용 값")
    print("=" * 55)
    for hw_name in ["A100", "L4"]:
        sub = df[df["hw"] == hw_name]
        if len(sub) == 0: continue
        avg_p = sub["prefill_sec_per_token"].mean()
        avg_d = sub["decode_sec_per_token"].mean()
        print(f"\n{hw_name}:")
        print(f'  "prefill_sec_per_token": {avg_p:.10f},')
        print(f'  "decode_sec_per_token":  {avg_d:.10f},')
