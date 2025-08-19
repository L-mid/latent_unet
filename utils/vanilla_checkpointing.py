
import torch
import os, io, json, tempfile, time, hashlib
import yaml
from pathlib import Path

# === NOTES ===
"""
atomic checkpointing and non are very different and i don't know why. 
I will never use non atomic checkpointing so it should be merged and removed.

Test: the test does not test atomic checkpointing well, only non. 
"""


def save_checkpoint(model, optimizer, ema=None, epoch=0, step=0, loss=None,
                    path="checkpoint.pt", config=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }

    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)    # Or rewrite into atomic write for safety


    # Optionally save YAML config snapshot for reproducibility
    if config:
        config_path = str(Path(path).with_suffix("yaml"))
        with open(config_path, "w") as f:
            yaml.dump(config, f)


def load_checkpoint(path, model, optimizer=None, ema=None, scaler=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if ema and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"]) 

    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", None),
    }

def _fsync_dir(dirpath: str):
    # Extra durability: fsync the directory so the rename is persisted.
    try:
        dirfd = os.open(dirpath, os.O_RDONLY)
        try: 
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except (PermissionError, OSError):
        # Not supported on Windows / some filesystems.
        # Safe to skip - only weakens crash-duribility, not atomicuty
        return


def save_checkpoint_atomic(obj, path: str, with_hash: bool = False):
    """
    Atomically save 'obj' to 'path' using torch.save, guarding against partial writes.
    Optionally writes a sidecar .sha256 file for integrity checks.
    """
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    base = os.path.basename(path)

    # Make sure target dir exists
    os.makedirs(d, exist_ok=True)

    import tempfile as _stdlib_tempfile     # avoid collisions

    # Temp file in the same directory (required for atomic rename)
    fd, tmp_path = _stdlib_tempfile.mkstemp(prefix=base + ".tmp", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            # Write
            torch.save(obj, f)
            # Flush Python buffers
            f.flush()
            # Ensure bytes reach disk
            os.fsync(f.fileno())

        # Optional: integrity sidecar (lets you detect corruption later)
        if with_hash:
            h = hashlib.sha256()
            with open(tmp_path, "rb") as tf:
                for chunk in iter(lambda: tf.read(1024 * 1024), b""):
                    h.update(chunk)
            with open(path + ".sha256.tmp", "w", encoding="utf-8") as hf:
                hf.write(json.dumps({"sha256": h.hexdigest(), "size": os.path.getsize(tmp_path)}))
                hf.flush()
                os.fsync(hf.fileno())
            os.replace(path + ".sha256.tmp", path + ".sha256")

        # Atomic swap
        os.replace(tmp_path, path)  # atomic within same filesystem
        _fsync_dir(d)               # make the rename durable
    except BaseException:
        # Clean up the temp if something went wrong (including KeyboardInterrupt)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def load_checkpoint_verified(path: str):
    """
    Optional loader that verifies the sidecar hash if present.
    """
    sidecar = path + ".sha256"
    if os.path.exists(sidecar):
        meta = json.load(open(sidecar, "r", encoding="utf-8"))
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != meta.get("sha256"):
            raise RuntimeError("Checkpoint hash mismatch - file may be corrupted.")
    return torch.load(path, map_location="cpu")


def auto_resume(path_or_dir):
    # Automatically find latest checkpoint in a directory.
    ckpts = sorted(Path(path_or_dir).glop("*.pt"), key=os.path.getmtime)
    return str(ckpts[-1]) if ckpts else None

def rotate_checkpoints(dir_path, keep_last_n=3):
    # Keep only the last N checkpoints in a directory.
    ckpts = sorted(Path(dir_path).glob("*.pt"), key=os.path.getmtime)
    for ckpts in ckpts[:-keep_last_n]:
        os.remove(ckpts)























