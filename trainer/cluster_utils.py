
import os
import logging
import torch
import torch.distributed as dist
import time

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Enviroment dectection
# ----------------------------------------------------------------------------------

def discover_cluster_env() -> dict:
    # Detects cluster environment configuration from known variables.

    env = {}

    # Distributed backend
    env["backend"] = os.environ.get("DIST_BACKEND", "nccl")

    # Master node adress/port
    env["master_addr"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    env["master_port"] = os.environ.get("MASTER_PORT", 29500)

    # World info
    env["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
    env["rank"] = int(os.environ.get("RANK", 0))
    env["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

    # Optional SLURM support
    if "SLURM_PROCID" in os.environ:
        env["rank"] = int(os.environ["SLURM_PROCID"])
    if "SLURM_NTASKS" in os.environ:
        env["world_size"] = int(os.environ["SLURM_NTASKS"])
    if "SLURM_LOCALID" in os.environ:
        env["local_rank"] = int(os.environ["SLURM_LOCALID"])

    return 


# ----------------------------------------------------------------------------------
# Distributed init (cluster-safe)
# ----------------------------------------------------------------------------------

def initalize_distributed(timeout_seconds: int = 1800):
    # Safe, idemopotent distributed initialization using cluster enviroment

    if not torch.distributed.is_available():
        logger.warning("[CLUSTER] torch.distributed not available.")

    if dist.is_initialized():
        logger.info("[CLUSTER] torch.distributed already initialized.")

    env = discover_cluster_env()

    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]

    logger.info(f" [CLUSTER] Initializing distributed:")
    logger.info(f"  - backend:      {env['backend']}")
    logger.info(f"  - world_size:   {env['world_size']}")
    logger.info(f"  - rank:         {env['rank']}")
    logger.info(f"  - master_addr:  {env['master_addr']}")
    logger.info(f"  - master_port:  {env['master_port']}")

    torch.cuda.set_device(env["local_rank"])

    dist.init_process_group(
        backend=env["backend"],
        rank=env["rank"],
        world_size=env["world_size"],
        timeout=torch.distributed.timedelta(seconds=timeout_seconds),
        init_method="env://"
    )


# --------------------------------------------------------------------------------
# Rank utilities
# --------------------------------------------------------------------------------

def get_rank() -> int:
    if torch.is_distributed():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    if torch.is_distributed():
        return dist.get_world_size()
    return 1

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_rank0() -> bool:
    return get_rank() == 0


# -----------------------------------------------------------------------------------
# Synchronization helpers
# -----------------------------------------------------------------------------------

def global_barrier_sync():
    # Force all processes to sync (e.g., after loading model weights).

    if is_distributed():
        dist.barrier()
        logger.info(f"[CLUSTER] Barrier sync complete (rank {get_rank}).")



# ------------------------------------------------------------------------------------
# Rank-aware printing
# ------------------------------------------------------------------------------------

def rank_print(*args, force: bool = False, **kwargs):
    # Print only from rank 0 by default.

    if is_rank0() or force:
        print(*args, **kwargs)


# -------------------------------------------------------------------------------------
# Optional: Retry-safe init wrapper (e.g., for flaky jobs)
# --------------------------------------------------------------------------------------

def retry_distributed_init(
        retries: int = 3,
        wait_seconds: int = 5,
        timeout_seconds: int = 1800
):
    for attempt in range(retries):
        try:
            initalize_distributed(timeout_seconds=timeout_seconds)
            return
        except Exception as e:
            logger.warning(f"[CLUSTER] Init failed (attempted {attempt+1}/retries): {e}")
            time.sleep(wait_seconds)
        raise RuntimeError("[CLUSTER] Failed to initalize distributed after multiple attempts")











