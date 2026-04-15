import torch
import gc


def emergency_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("ok")

# emergency_memory_cleanup()