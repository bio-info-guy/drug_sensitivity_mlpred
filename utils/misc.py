import gc
from names_generator import generate_name
import hashlib
import os
import logging
# dummy gc callback for optuna, its probably useless for memory management
def dummy_gc(input1, input2):
    gc.collect()

# random name generator

def hash_string_to_8_digits(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    return int(hex_digest, 16) % (10**8)

def random_name(config, X, y):
    config_hash = hash_string_to_8_digits(str(config))
    X_hash = hash_string_to_8_digits(str(X.sum().sum()))
    y_hash = hash_string_to_8_digits(str(y.sum().sum()))
    return(generate_name(seed=config_hash+X_hash+y_hash))


def set_cuda_device():
    try:
        from pynvml import (
        nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlShutdown
        )
        nvmlInit()
        free_list = []
        for i in range(nvmlDeviceGetCount()):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem = nvmlDeviceGetMemoryInfo(handle).free
            free_list.append((mem, i))
        nvmlShutdown()
        gpu_to_use = max(free_list)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
        print(f"[Auto-GPU] using GPU #{gpu_to_use}", flush=True)
    except Exception:
    # If pynvml isn’t installed, fall back to whatever SLURM set (or default 0)
        print(f"[Auto-GPU] NVML unavailable, using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','0')}", flush=True)
