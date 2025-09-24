
import os, torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.debug import debug_log, debug_section 

from utils.checkpointing.zarr_checkpointing.zarr_wrapper import save_model, load_model
from utils.visualizer_refactor import build_visualizer
from trainer.logger import build_logger, log_ckpt_artifact
from trainer.optim_utils import build_optimizer, build_scheduler
from trainer.optim_utils import EMA
from trainer.losses import get_loss_fn
from diffusion.forward_process import ForwardProcess
from diffusion.ddim import ddim_sample, sample_ddim_grid
from utils.checkpointing.ckpt_io import resolve_latest_checkpoint
from tqdm import tqdm

# === NOTES:
"""
All these calls (like build_optimizer, EMA) are weird, not the same, and mabye should be crunched down.
)

Weird noop logger mechanics.

heres something:

# train_loop.py
ckpt_root = Path(cfg.checkpoint.out_dir)
ckpt_dir  = ckpt_root / f"epoch_{epoch:06d}_step_{step:09d}"
group = zarr_core.open_store(ckpt_dir, mode="w")  # no rmtree on root

This gives versioned ckpts consider it


theres more too, including atomacy (great!)

"""


def train_loop(cfg, model, dataset):
    

    # --------- Setup -----------
    device = torch.device(cfg.device)
    model.to(device)

    diffusion = ForwardProcess().to(device)               # does not take cfg   
    logger = build_logger(cfg)

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    optimizer = build_optimizer(model.parameters(), cfg)                       
    loss_scheduler = build_scheduler(optimizer, cfg)    # should rename to 'build_loss_scheduler'?              
    ema = EMA(model)

    loss_fn = get_loss_fn(cfg)

    scaler = torch.amp.GradScaler(enabled=cfg.training.amp)

    viz = build_visualizer(cfg.viz)


    # --- Resolve resume path ---
    resume_dir = resolve_latest_checkpoint(cfg.resume_path)     # resume_path = output_dir, unless not loading from outputs

    start_epoch = 0
    global_step = 0
    if resume_dir:
        state = load_model(model, optimizer, loss_scheduler, ema, path=resume_dir)
        # Expect your loader to return attrs like epoch_next/global_step; if not, read group.attrs
        start_epoch = int(state.get("epoch_next"))
        global_step = int(state.get("global_step", 0))
        print(f"[Resume ckpt] {resume_dir} -> start_epoch={start_epoch}, step={global_step}")      
    else:
        print("[Resume ckpt] No checkpoint found. Starting fresh.")
    

    # ------------------ Training Loop --------------------
    if start_epoch >= cfg.training.num_epochs:
        print(f"[TRAINING] Training complete/No-Op: start_epoch: {start_epoch} >= num_epochs: {cfg.training.num_epochs}.")
        logs = {"loss": None}
        return logs 

    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")  # datetime warning?
    
        for step, batch in enumerate(pbar): 
            with debug_section("train_step"):

                if isinstance(batch, dict):
                    x, y = batch["image"].to(device), batch["label"].to(device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, y = batch[0].to(device), batch[1].to(device)
                else:
                    raise TypeError(f"Unexpected batch types: {type(batch)}")

                t = torch.randint(low=0, high=cfg.schedule.timesteps, size=(x.size(0),), device=x.device, dtype=torch.long)
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise=noise) 

                with autocast(device_type=str(device), enabled=cfg.training.amp):
                    noise_pred = model(x_noisy, t) 
                    loss, _ = loss_fn(noise_pred, noise, t) 

                debug_log(f"Step {step} | Loss: {loss.item():.4f}", name="train_loop")

                assert torch.isfinite(loss), "Loss is not finite"
                scaler.scale(loss).backward()
                


                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                

                # grad logging:
                params = list(model.named_parameters()) 
                if (epoch + 1) % cfg.training.viz_interval == 0:              
                    viz.log("grad_flow", step=epoch, named_parameters=params)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema: ema.update()        # does not take model

                # Log to console and optionally W&B/tensorboard
                loss_value = float(loss.item())
                logs = {"loss": loss_value}

                if logger: logger.log_scalar("loss", loss.item(), step=epoch * len(dataloader) + step)
                
                global_step += 1

            if loss_scheduler: loss_scheduler.step()  

            


        # ------------ Evaluation & Logging ---------------
        if (epoch + 1) % cfg.training.viz_interval == 0:
            model.eval()      
            vis_img = sample_ddim_grid(cfg, model=model, shape=x_noisy.shape, step=epoch) 
            vis_misc = viz.run_all(step=epoch, model=model)     
            if logger:  
                if vis_img is not None:     # might swallow if sampler doesn't return tensor
                    try:
                        logger.log_image("image_samples", vis_img, step=epoch)
                    except Exception as e:
                        print(f"[viz] image_samples skipped: {e}")  

                if vis_misc is not None:
                    try:
                        logger.log_image("misc_viz", vis_misc, step=epoch)
                    except Exception as e:
                        print(f"[viz] misc_viz skipped: {e}")

        
        # ---------------  Checkpointing -------------------
        if (epoch+1) % cfg.training.ckpt_interval == 0:
            if cfg.checkpoint.backend in {"noop", None}:
                pass    # skip saving in tests
            else:
                final_dir = save_model(model, optimizer, loss_scheduler, ema, epoch=epoch, step=global_step, path=cfg.checkpoint.out_dir)    
                log_ckpt_artifact(final_dir, epoch=epoch, step=global_step, best=True)
    logger.finish()

    return logs





            




    













