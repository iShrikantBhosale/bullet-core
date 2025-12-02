import time
import math
import psutil
try:
    import torch
except ImportError:
    torch = None

class BulletLogger:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_tokens = 0

    def format_time(self, secs):
        if secs < 60: return f"{secs:.0f}s"
        if secs < 3600: return f"{secs/60:.1f}m"
        return f"{secs/3600:.1f}h"

    def log(self, step, loss, lr, total_steps, tokens_processed):
        now = time.time()
        step_time = now - self.last_time
        self.last_time = now

        # tokens/sec
        tps = (tokens_processed - self.last_tokens) / step_time if step_time > 0 else 0
        self.last_tokens = tokens_processed

        # ETA
        progress = step / total_steps if total_steps > 0 else 0
        eta = (time.time() - self.start_time) * (1 - progress) / max(progress, 1e-6)

        # memory
        mem = psutil.virtual_memory().percent
        
        # GPU memory if available
        gpu_mem = ""
        if torch is not None and torch.cuda.is_available():
            gpu_mem = f" gpu_mem={torch.cuda.memory_allocated() / 1024**3:.2f}GB"

        print(
            f"[Step {step:6d}] "
            f"loss={loss:.4f} "
            f"lr={lr:.2e} "
            f"eta={self.format_time(eta)} "
            f"tokens/s={tps:.0f} "
            f"mem={mem}%{gpu_mem}"
        )
