from device_config import DEVICES

def get_latency_sec(device_name, stage, batch_size, input_len):
    """
    아주 단순한 latency 모델
    - prefill: input_len에 비례
    - decode: token 1개당 고정 시간
    """

    device = DEVICES[device_name]

    if stage == "prefill":
        # 초 단위
        return input_len * device["prefill_sec_per_token"]

    elif stage == "decode":
        return device["decode_sec_per_token"]

    else:
        raise ValueError("Unknown stage")
