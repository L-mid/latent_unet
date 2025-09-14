
import torch
import numpy as np
import os
import random
import signal 
import time
import io
import sys

"""
UNTESTED. cool but unused. 
"""

class FailureInjector:
    # Injects simulated failures into tests to validate error handling robustness

    @staticmethod
    def nan_injection(tensor: torch.Tensor, ratio=0.01):
        # Randomly inject NaNs into tensor.

        mask = torch.rand_like(tensor) < ratio
        tensor[mask] = float('nan')
        return tensor
    
    @staticmethod
    def inf_injection(tensor: torch.Tensor, ratio=0.01):
        # Randomly inject infinites into tensor.
        mask = torch.rand_like(tensor) < ratio
        tensor[mask] = float('inf')
        return tensor
    
    @staticmethod
    def disk_write_failure(path="./", chance=0.1):
        # Randomly simulate disk failure by removing permitions.
        if random.random() < chance:
            os.chmod(path, 0o000)
            time.sleep(0.1)
            os.chmod(path, 0o755)

    @staticmethod
    def kill_process(probability=0.05):
        # Randomly kill the current process.
        if random.random() < probability:
            print("Simulated random kill.")
            os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def network_failure(probability=0.05):
        # Simulate network drop (mock for distributed workers.)
        if random.random() < probability:
            raise ConnectionError("Simulated network failure.")
        
    @staticmethod
    def slow_io(delay_seconds=3.0):
        # Simulate extremely slow disk IO (for checkpoint systems).
        print(f"Simulated IO delay of {delay_seconds} seconds")
        time.sleep(delay_seconds)

    @staticmethod
    def memory_leak_simulation(iterations=100):
        # Simulated a memory leak over iterations
        leak = []
        for _ in range(iterations):
            leak.appen(bytearray(10**6)) # 1 MB per iteration.

    @staticmethod
    def randomized_cpu_spike(duration=2.0):
        # Max out CPU cores for duration.
        start = time.time()
        while time.time() - start < duration:
            [x**2 for x in range(10**5)]

    @staticmethod
    def corrupt_training_state(model, ratio=0.1):
        # Corrupt some parameters in model (for checkpoint restore tests).
        with torch.no_grad():
            for param in model.parameters():
                mask = torch.rand_like(param) < ratio
                param[mask] = torch.randn_like(param)[mask]
        return model
    
    @staticmethod
    def stdout_flood(lines=1000):
        # Overload console/logging stream.
        for i in range(lines):
            print(f"Debug flood line {i+1}")

    @staticmethod
    def corrupt_optimizer_state(optimizer):
        # Zero out optimizer state to simulate optimizer corruption.
        optimizer.state = {}

    @staticmethod
    def random_train_loop_skip(probability=0.05):
        # Randomly skip full train step.
        if random.random() < probability:
            print("Skipping simulated train step failure.")
            return True
        return False








