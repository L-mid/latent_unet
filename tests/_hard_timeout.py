
from __future__ import annotations
import multiprocessing as mp
import sys
import logging
from logging.handlers import QueueHandler
from queue import Empty
import os
import pickle, multiprocessing as mp, multiprocessing.queues as mpq
from multiprocessing.connection import Connection, PipeConnection
import multiprocessing


def _init_child_logging(log_queue, level=logging.INFO):
    # CHILD: send logs to parent via QueueHandler
    root = logging.getLogger(__name__)      # or whatever finds names
    root.handlers.clear()           # avoid duplicate handlers between retries
    root.setLevel(level)
    if log_queue is not None and hasattr(log_queue, "put"):
        root.addHandler(QueueHandler(log_queue))
    # now all logging.* in child flows to parent listener

    else:
        # fallback so something is visible even without bridge
        root.addHandler(logging.StreamHandler(sys.stderr))  # cannot be turned off?
    


# --- TOP-LEVEL worker (must be importable) ---
def _worker_entry(*, res_q, fn, args, kwargs, timeout, stackdump_path, log_q, stderr_output=False):
    # Arm faulthandler in the **child** only if requested
    fh = None
    singleshot = False
    print("[Worker Entry: Started!]")

    _init_child_logging(log_q)

    # faulthandler to file/stderr here...
    try:
        if stackdump_path:
            # Unique file per child
            path = f"{stackdump_path}_{os.getpid()}.txt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Unbuffered binary to reduce lost writes on kill
            fh = open(path, "wb", buffering=0)      # unbuffered write
            # Banner so you know we armed
            fh.write(b"== hard-timeout armed ==\n");    # immediately visible

            import faulthandler
            faulthandler.enable(file=fh)

            # (A) Single shot before kill:
            if singleshot != False:
                lead = min(10.0, max(2.0, 0.2 * float(timeout)))
                fire_at = max(1.0, float(timeout) - lead)
                faulthandler.dump_traceback_later(timeout=5, file=fh, repeat=True)

            # (B) Default: periodic snapshots (no A if B):
            intervel = max(10.0, min(30.0, 0.2 * float(timeout)))
            faulthandler.dump_traceback_later(timeout=intervel, repeat=True, file=fh)

        elif stderr_output:
            import faulthandler
            faulthandler.dump_traceback_later(timeout=30, file=None, repeat=True)  # to stderr, doesn't work rn 
        
        res = fn(*args, **kwargs)
        res_q.send(("ok", res))
    except BaseException as e:
        # Keep it simple to avoid pickling complex exceptions
        res_q.send(("err", f"{type(e).__name__}: {e}"))
        print("Worker Entry: Finished (or timed out).")
    finally:
        try:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        if fh:
            try:
                # best-effort fsync; fh is unbuffered so this is conservative
                os.fsync(fh.fileno())
            except Exception:
                pass
            fh.close()
        try: res_q.close()
        except Exception: pass


def _kill_tree(pid:int):
    try:
        import psutil
        proc = psutil.Process(pid)
        for c in proc.children(recursive=True):
            c.kill()
        proc.kill()
    except Exception:
        try:
            import os, signal
            os.kill(pid, 9)
        except Exception:
            pass


def _check_picklable(**named_objs):
    import pickle
    for name, obj in named_objs.items():
        try:
            pickle.dumps(obj)
        except Exception as e:
            raise TypeError(f"Object '{name} is not pickable: {e!r}") from e


def run_with_hard_timeout(fn, /, *args, timeout: float, 
                        stackdump_path: str | None = None,
                        bridge_logs: bool = True,
                        stderr_output: bool = False,
                        start_method: str = "spawn", **kwargs):
    """
    Run `fn(*arg, **kwargs)` in a *separate process* and enforce a hard timeout.
    One timeout, kill the process (and its children if psutil is available),
    then raise TimeoutError. Returns the fuction's return value otherwise.
    """
    from logging.handlers import QueueListener, QueueHandler
    import logging, sys, multiprocessing as mp

    if not callable(fn):
        raise TypeError(f"`fn` must be callable, got {type(fn).__name__}")
    
    # logs & chrash outputs
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    log_q = ctx.Queue() if bridge_logs else None
    # (avoid: from queue import Queue; log_q = Queue())

    # PARENT: listener forwards child logs to console (and file if you like)
    listener = None
    if log_q is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s"))
        listener = QueueListener(log_q, console)  # add a FileHandler here too if desired
        listener.start()


    # check if pickelable before running:
    try: 
        _check_picklable(
        fn=fn,
        args_tuple=args,        # try tuple as whole
        kwargs_dict=kwargs,     # try dict as whole
        ) 
    except Exception as e:
        raise RuntimeError(f"Contents not picklable: error:{e}")  
    print("[TIMEOUT] Base contents pickleable.")
    
    try:
        for i, a in enumerate(args):
            _check_picklable(**{f"args[{i}]": a})
        for k, v in kwargs.items():
            _check_picklable(**{f"kwargs[{k}]": v})
    except Exception as e:
        raise RuntimeError(f"Sub-contents not picklable: error:{e}")     
    print("[TIMEOUT] Sub-contents pickleable.")

    # sanity type checks (no pickling!)
    if log_q is None and not isinstance(log_q, mpq.Queue):
        raise TypeError(f"log_queue must be multiprocessing.Queue, got {type(log_q)}")
    if not isinstance(child_conn, multiprocessing.connection.PipeConnection):
        raise TypeError(f"child_conn must be a multiprocessing.Connection, got {type(child_conn)}")
    print("[TIMEOUT] Type checks passed.")

    p = ctx.Process(
        target=_worker_entry, 
                kwargs=dict(
                    res_q=child_conn,   # <-- Pipe end for child to send result
                    log_q=log_q,             # <-- Queue for logs (or None)
                    fn=fn, 
                    args=args, 
                    kwargs=kwargs, 
                    timeout=timeout, 
                    stackdump_path=stackdump_path, 
                    stderr_output=stderr_output
                ), 
                daemon=False,
    )
    
    try:
        p.start()
        # Close the child's end in the parent so EOF is detectable
        child_conn.close()

        p.join(timeout)
        if p.is_alive():
            # Try to kill the entire process tree (best effort).
            _kill_tree(p.pid)
            p.join(5)
            raise TimeoutError(f"Hard timeout exceeded: {timeout}s")
        

        # block briefly for the result to arrive; avoid get_nowait()
        if parent_conn.poll(2.0):       # <- timed poll
            status, payload = parent_conn.recv()
        else:
            raise RuntimeError(f"Child exited ((exitcode={p.exitcode})) without posting a result")
        

        # Child finished; fetch its message.
        if status == "err":
            raise RuntimeError(f"Child raised: {payload}")
        return payload
    finally:
        try: parent_conn.close()
        except Exception: pass

        if listener is not None:
            try: listener.stop()
            except Exception: pass

        if log_q is not None:
            try: log_q.close(); log_q.join_thread()
            except Exception: pass

       


