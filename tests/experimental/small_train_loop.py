
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import re
from utils.debug import debug_log, debug_section  

from utils.checkpointing.zarr_checkpointing.zarr_wrapper import load_model, save_model
from utils.visualizer import visualize_everything   # I do not trust this
from trainer.logger import ExperimentLogger     # learn how to use
from trainer.optim_utils import build_optimizer     # no scheduler
from trainer.optim_utils import EMA   
from trainer.losses import get_loss_fn  # mse
from diffusion.forward_process import ForwardProcess
from tqdm import tqdm   # bars?

# === NOTES:
"""
Designed small.
"""

def train_loop(cfg, model, dataset, logger=None):

    device = torch.device(cfg.device)
    model.to(device)
    diffusion = ForwardProcess(cfg).to(device)   
    
    logger = logger  # ExperimentLogger(cfg) or NoopLogger() (others too)

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    optimizer = build_optimizer(model.parameters(), cfg)
    #ema = EMA(model, decay=0.999)   # internal issues as untested

    loss_fn = get_loss_fn(cfg)

    start_epoch = 0
    if cfg.resume_path:

        def _find_latest_checkpoint(dirpath: str | Path): 
            p = Path(dirpath)
            print(p, "| this is p (finder)") 
            assert p.exists()
            if not p.exists():  
                return None # OHH
            # ok. 
            # the fn is passing none itself because of a failure. HAHAH
            entries = list(p.iterdir())
            print(f"[finder] entries={len(entries)} -> {[e.name for e in entries]}")

            # Example fallback matcher; treak to your naming scheme
            canadiates = []
            for child in entries:
                m = re.search(r'(\d+)', child.name)     # e.g., epoch_0001, ckpt-1, etc.
                if m:
                    canadiates.append((int(m.group(1)), child))

            if not canadiates:
                print("[finder] REASON: no canadates macthed pattern (wrong naming schema between saver and finder)")
                raise RuntimeError
                # possible answers:
                # somewhere i'm deleating next runs files. (most likely)
            
            canadiates.sort()
            found = canadiates[-1][1]
            print(f"[finder] FOUND: {found}")
            return found

            
        resume_root = _find_latest_checkpoint(cfg.resume_path) 
        if resume_root is None:
            logger.print(f"No checkpoint found in %s; starting fresh. Attempted path: {cfg.resume_path}", level="info")
            start_epoch = 0
            print(start_epoch, "| this is what start_epoch is. [SMALL TRAIN]. No ckpt found.") 
        else:
            state = load_model(model, optimizer, scheduler=None, path=cfg.resume_path)  
            start_epoch = state["epoch_next"]   
            print(start_epoch, "| this is what start_epoch is. [SMALL TRAIN]. ckpt found.")  

            # in meta, store:
            """{"epoch_completed": N, "epoch_next": N+1, "global_step": S, ...}"""
            # make sure the saver closes checkpointer.
    
    else: print("NO RESUME PATH GIVEN in cfg. Starting from sctrach.")
 

    # --------------- Training Loop ----------------------
    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")      # possible datetime warning

        for step, batch in enumerate(pbar):
            with debug_section("train_step"):
                x = batch["image"].to(device)
                t = torch.randint(low=0, high=cfg.schedule.timesteps, size=(x.size(0),), device=x.device, dtype=torch.long)
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise=noise)

                noise_pred = model(x_noisy, t)
                loss, _ = loss_fn(noise_pred, noise, t)     # _ = weight

                debug_log(f"Step {step} | Loss: {loss.item():.4f}", name="train_loop")

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                #if ema: ema.update(model)       # should the ema state be saved in the model?

                # Log to console and optionally W&B/tensorboard
                loss_value = float(loss.item())
                logs = {"loss": loss_value, "start_epoch": start_epoch}

                if logger: logger.log_scalar("loss", loss.item(), step=epoch * len(dataloader) + step)

        
        # ---------------- Evaluation & Logging ------------------------

        if (epoch+1) % cfg.training.vis_interval == 0:
            model.eval()
            vis_imgs = visualize_everything(model, cfg=cfg)
            if logger:
                if vis_imgs == None: pass   # make this a warning or smth later
                else: logger.log_image("samples", vis_imgs, step=epoch)

        # -------------- Checkpointing -------------------

        if (epoch+1) % cfg.training.ckpt_interval == 0:
            print("checkpoint saving actually works")
            if cfg.checkpoint.backend in {"noop", None}:        # this won't raise if catagory not present
                pass    # skip saving in tests
            else:
                save_model(
                    model, optimizer, epoch=epoch, scheduler=None, step=step, path=cfg.checkpoint.out_dir, 
                           extra={"epoch_completed": epoch, "epoch_next": epoch+1, "global_step": step}
                           )       # conisder adding EMA, mabye metadata.
            assert Path(cfg.checkpoint.out_dir).exists()
            print(Path(cfg.checkpoint.out_dir), "| This is what's saved.")

    
    logger.finish()
    return logs




 


