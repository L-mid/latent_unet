
import math
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# === Notes:
"""
Upgraded chunker not currently used.
"""

# -----------------------------------------------------------------------------------
# Main chunk config generator
# -----------------------------------------------------------------------------------

def get_chunk_config(
    weight_shapes: Dict[str, Tuple],
    strategy: str = "auto",
    target_elements: int = 128*1024,
    per_key_override: Optional[Dict[str, Tuple]] = None
) -> Dict[str, Tuple]:
    # Generate chunk shapes for each tensor.

    chunks = {}

    for name, shape in weight_shapes.items():
        if per_key_override and name in per_key_override:
            chunks[name] = per_key_override[name]
        elif strategy == "fixed":
            chunks[name] = fixed_chunk(shape, target_elements)
        elif strategy == "auto":
            chunks[name] = real_auto_chunk(shape, target_elements)
        elif strategy == "none":
            chunks[name] = None
        else:
            raise ValueError(f"[CHUNK] Unknown strategy: {strategy}")
        
        logger.debug(f"[CHUNK] {name}: shape={shape} -> chunk={chunks[name]}")

        return chunks


# ------------------------------------------------------------------------------------
# Fixed chunk (uniform first dimension)
# ------------------------------------------------------------------------------------

def fixed_chunk(shape: Tuple, target_elements: int) -> Tuple:
    # Fixed chunk: try to preserve full dims, but limit first dim. 

    if len(shape) == 0:
        return ()
    
    product_other_dims = product(shape[1:])
    first_dim_chunk = max(1, target_elements // max(1, product_other_dims))

# -----------------------------------------------------------------------------------
# Auto chunk (scale across all dims)
# -----------------------------------------------------------------------------------

def auto_chunk(
        shape: Tuple[int, ...], 
        target_elements: int,
) -> Tuple[int, ...]:  
    # Scalars: no chunks
    if not shape:
        return ()

    total = product(shape)
    # If the array is tiny, just one chunk = whole array
    if target_elements >= total:
        return tuple(shape)  
    
    # Inital uniform scale
    scale = (target_elements / total) ** (1.0 / len(shape))
    chunk = [max(1, int(math.floor(d * scale))) for d in shape]

    # Clamp to <= dimension size (and >=1)
    chunk = [min(max(c, 1), d) for c, d in zip(chunk, shape)]

    # If we undershot (common due to floor), greedily grow axes with most headroom
    def prod(lst):
        p = 1
        for v in lst: p *= v  
        return p

    while prod(chunk) < target_elements:
        # pick the axis with the largest possible relative growth
        i = max(range(len(shape)), key=lambda j: (shape[j] / chunk[j]) if chunk[j] < shape[j] else 0)  
        if chunk[i] >= shape[i]:
            break
        chunk[i] += 1

    return tuple(chunk)


# -----------------------------------------------------------------------------------
# Auto chunk for real work loads (not currently implimented)
# -----------------------------------------------------------------------------------

def real_auto_chunk(
        shape: Tuple[int, ...],
        dtype_itemsize: int,
        target_chunk_bytes: int = 2 * 1024 * 1024,  # ~2 MiB
        order: str = "C",   # "C" (row-major) or "F" (col-major)
        small_axis_threshold: int = 64,
        snap_multiples = (8, 4, 2),
) -> Tuple[int, ...]:
    if not shape:
        return ()
    
    target_elems = max(1, target_chunk_bytes // max(1, dtype_itemsize))
    n = len(shape)
    chunk = [1] * n
    locked = [False]*n

    # Pre-take tiny axes whole
    axes = (range(n - 1, -1, -1) if order == "C" else range(n))
    for i in axes:
        if shape[i] <= small_axis_threshold:
            chunk[i] = shape[i]
            locked[i] = True

    # Grow greedily toward target_elems, preferring fastest-varying axis (don't touch locked or full dims)
    def prod(lst):
        p = 1
        for v in lst: p *= v
        return p
    
    while prod(chunk) < target_elems:
        grew = False
        for i in axes:
            if locked[i] or chunk[i] >= shape[i]:
                continue
            new_size = min(shape[i], max(chunk[i] * 2, chunk[i] + 1))
            # try doubling; if already big, at least +1
            trial = (prod(chunk) // chunk[i]) * new_size
            if trial <= target_elems * 1.25 or new_size == shape[i]:
                chunk[i] = new_size
                grew = True
                if prod(chunk) >= target_elems:
                    break
        if not grew:
            break

    # snap to nice multiples without exceeding dim. Never shrink
    for i in range(n):
        if locked[i]:
            continue
        base = chunk[i]
        for m in snap_multiples:
            # ciel to multipule, capped by dim size
            snapped = min(shape[i], ((base + m - 1) // m) * m)
            if snapped >= base:
                chunk[i] = snapped
                break
        chunk[i] = min(max(chunk[i], 1), shape[i])

    return tuple(chunk)


# -----------------------------------------------------------------------------------
# Per-layer manual override helper
# -----------------------------------------------------------------------------------

def layerwize_override(**kwargs) -> Dict[str, Tuple]:
    """
    Example usage:
        layerwize_override(
            linear__weight=(128, 5),
            conv1__weight=(32, 3, 3, 3)
        )
    """
    return kwargs


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def product(shape: Tuple) -> int:
    p = 1
    for d in shape: p *= d
    return p

