
from __future__ import annotations
from pathlib import Path
import os, re, json, uuid, time, shutil, stat

# === NOTES:
"""
checkpoint_io.
"""

# ---------- Basics

def _is_zarr_root(p: Path) -> bool:
    # v3: zarr.json; v2: .zgroup; accept either
    return (p / "zarr.json").exists() # or (p / ".zgroup").exists()


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def update_latest_pointer(ckpt_root: Path, ckpt_dir: Path, epoch: int, step: int):
    latest = ckpt_root / "LATEST.json"
    payload = {"dir": ckpt_dir.name, "epoch": epoch, "step": step}
    _atomic_write_text(latest, json.dumps(payload))

def resolve_from_pointer(ckpt_root: Path) -> Path | None:
    f = ckpt_root / "LATEST.json"
    if not f.exists():
        return None
    info = json.loads(f.read_text(encoding="utf-8"))
    p = ckpt_root / info["dir"]
    return p if p.exists() else None

def mark_complete(ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True) 
    (ckpt_dir / ".complete").write_text("ok", encoding="utf-8")  

# ----------- Name scan

PAT = re.compile(r"^epoch_(\d+)_step_(\d+)$")  # numbers can be zero-padded or not

def _parse_dirname(d: Path) -> tuple[int,int] | None:
    m = PAT.match(d.name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def _list_named_dirs(ckpt_root: Path):
    out = []
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        parsed = _parse_dirname(d)
        if parsed:
            out.append((*parsed, d))
    return out


# --------------- Windows-safe delete

def _rmtree_retry(path: Path, retries=8, base_delay=0.05):
    def _onexc(func, p, exc):
        # Python 3.13 passes (func, path, exc)
        # Try making writable, then re-raise to trigger retry
        if isinstance(exc, PermissionError):
            try: os.chmod(p, stat.S_IWRITE)
            except Exception: pass
        raise exc
    for i in range(retries):
        try:
            shutil.rmtree(path, onexc=_onexc)
            return
        except PermissionError:
            time.sleep(base_delay * (2 ** i))
    shutil.rmtree(path, ignore_errors=True)


# ---------- Versioned save staging

def begin_versioned_save(ckpt_root: Path, epoch: int, step: int) -> tuple[Path, Path]:
    """
    Returns (tmp_dir, final_dir). Caller writes into tmp_dir and then calls finalize_versioned_save.
    """
    ckpt_root.mkdir(parents=True, exist_ok=True)
    final_dir = ckpt_root / f"epoch_{epoch:06d}_step_{step:09d}"
    tmp_dir = ckpt_root / (final_dir.name + f".tmp-{uuid.uuid4().hex}")
    if tmp_dir.exists():
        _rmtree_retry(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir, final_dir

def finalize_versioned_save(tmp_dir: Path, final_dir: Path):
    if final_dir.exists():
        _rmtree_retry(final_dir)
    os.replace(tmp_dir, final_dir)  # rename tmp -> final

def keep_last_n(ckpt_root: Path, n: int = 3):
    items = [(e, s, d) for (e, s, d) in _list_named_dirs(ckpt_root) if (d / ".complete").exists()]
    if len(items) <= n:
        return
    items.sort(key=lambda t: (t[0], t[1]))
    for _, _, d in items[:-n]:
        _rmtree_retry(d)


# ------------- Unified resolver 

def resolve_latest_checkpoint(ckpt_root: str | Path) -> Path | None:
    """
    Find a checkpoint dir WITHOUT opening Zarr metadata:
      1) if ckpt_root itself is a Zarr root, return it (single-checkpoint style).
      2) pointer file LATEST.json (if complete).
      3) newest complete in name order epoch_XXXX_step_YYYY.
      4) newest complete by mtime.
    """
    if ckpt_root == None:
        return None
    
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        return None

    # (1) Monolithic root (your older style)
    if _is_zarr_root(ckpt_root):
        # optional: also require .complete for safety
        if (ckpt_root / ".complete").exists() or True:
            return ckpt_root

    # (2) Pointer
    p = resolve_from_pointer(ckpt_root)
    if p and (p / ".complete").exists():
        return p

    # (3) Name order
    items = [(e, s, d) for (e, s, d) in _list_named_dirs(ckpt_root) if (d / ".complete").exists()]
    if items:
        items.sort(key=lambda t: (t[0], t[1]))
        return items[-1][2]

    # (4) mtime fallback
    completed = [d for d in ckpt_root.iterdir() if d.is_dir() and (d / ".complete").exists()]
    if completed:
        return max(completed, key=lambda d: d.stat().st_mtime)

    return None