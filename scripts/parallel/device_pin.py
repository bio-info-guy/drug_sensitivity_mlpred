# cpu_mem_pin.py
import os
import atexit
import fcntl
import multiprocessing
import time
try:
    import psutil
except ImportError:
    psutil = None
_locks = []
def pick_resources(
    cores_required=16,
    mem_required_gb=64,
    lock_dir="/tmp",
    wait_interval=5
):
    """
    Reserve a block of `cores_required` CPUs and ~`mem_required_gb` GB RAM.
    If all slots are taken, wait in a loop (sleeping `wait_interval` seconds)
    until one becomes free.
    """
    # 1) Discover total resources
    total_cores = multiprocessing.cpu_count()
    if psutil:
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
    else:
        total_mem_gb = float(os.environ.get("TOTAL_MEM_GB", 0))
    # 2) Compute how many slots fit on this node
    cpu_slots = total_cores // cores_required
    mem_slots = int(total_mem_gb // mem_required_gb)
    max_slots = min(cpu_slots, mem_slots)
    if max_slots < 1:
        raise RuntimeError(
            f"Not enough resources: {total_cores} cores, {total_mem_gb:.1f} GB RAM"
        )
    # 3) Loop until we successfully lock one slot
    while True:
        for slot in range(max_slots):
            lock_path = os.path.join(lock_dir, f"slot_{slot}.lock")
            lf = open(lock_path, "w")
            try:
                # non-blocking attempt
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _locks.append(lf)
                # pin to cores [slot*cores_required … slot*cores_required+cores_required-1]
                start = slot * cores_required
                cpus = list(range(start, start + cores_required))
                try:
                    os.sched_setaffinity(0, cpus)
                except AttributeError:
                    # expose for external taskset wrappers
                    os.environ["TASKSET_CPUS"] = ",".join(map(str, cpus))
                # hint to threaded libs
                os.environ["OMP_NUM_THREADS"] = str(cores_required)
                print(
                    f"[Auto-pin] acquired slot {slot}: CPUs {cpus[0]}–{cpus[-1]}, "
                    f"reserving ~{mem_required_gb} GB RAM",
                    flush=True
                )
                return
            except BlockingIOError:
                lf.close()
                continue
        # no free slot yet—sleep, then retry
        time.sleep(wait_interval)


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

def pick_free_gpu(lock_dir="/tmp/"):
    global locks
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown
        nvmlInit()
        gpu_count = nvmlDeviceGetCount(); nvmlShutdown()
    except ImportError:
        gpu_count = int(os.environ.get("NUM_GPUS", "1"))    
    for idx in range(gpu_count):
        lock_path = f"{lock_dir}/gpu{idx}.lock"
        try:
            lf = open(lock_path, "w")
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"[Auto-pin] using GPU {idx}", flush=True)
            locks = [lf]
            return
        except:
            continue
    raise RuntimeError("No free GPU lock found!")

def _cleanup():
    for lf in _locks:
        try:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()
        except:
            pass
        
