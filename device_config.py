DEVICES = {
    "A100": {
        "price_per_hr": 3.0,
        "tdp_watt": 600,                      # 300W × TP=2
        "prefill_sec_per_token": 0.0003368763,
        "decode_sec_per_token":  0.0338472114
    },
    "L4": {
        "price_per_hr": 1.2,
        "tdp_watt": 576,                      # 72W × TP=8
        "prefill_sec_per_token": 0.0004344148,
        "decode_sec_per_token":  0.0591576248
    }
}

# LLaMA-2 70B KV cache 크기 (fp16, GQA)
# 계산: 2(K+V) × 8(kv_heads) × 128(head_dim) × 2(fp16 bytes) × 80(layers)
# = 327,680 bytes/token
KV_BYTES_PER_TOKEN = 327_680  # bytes per input token
