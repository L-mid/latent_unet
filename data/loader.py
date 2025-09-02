
import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    # Safe random seed initialization per dataloader worker.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"[DATALOADER] Worker {worker_id} seeded with {seed}")


def create_dataloader(dataset, batch_size, shuffle=None, sampler=None, num_workers=4, collate_fn=None):  # many of these need defaults set (example sampler=none)
    # Centralized DataLoader builder (not cfg controlled).

    loader = torch.utils.data.DataLoader(       
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return loader



















