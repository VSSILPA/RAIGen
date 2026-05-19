"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys
import torch
from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cache_activations_runner import CacheActivationsRunner
from config import CacheActivationsRunnerConfig

def run():
    print("Visible devices:", torch.cuda.device_count())
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[Init] Rank {local_rank} assigned to cuda:{local_rank}", flush=True)
    else:
        # Fallback for non-distributed runs
        torch.cuda.set_device(0)
    args = parse(CacheActivationsRunnerConfig)
    CacheActivationsRunner(args).run()


if __name__ == "__main__":
    run()
