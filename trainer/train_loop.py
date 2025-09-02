
import os, torch, time
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.debug import debug_log, debug_section

from utils.tensorstore_checkpointing import save_checkpoint, load_checkpoint
from utils.visualizer import visualize_everything
from trainer.logger import ExperimentLogger
from trainer.optim_utils import build_optimizer, build_scheduler
from trainer.ema_utils import EMA
from diffusion.schedule import get_diffusion_schedule
from trainer.losses import get_loss_fn
from tqdm import tqdm

# === NOTES:
"""
All these calls (like build_optimizer, EMA) are weird, not the same, and mabye should be crunched down.

I got a wandb init timeout error before, is my key not being accessed? Does wandb need internet? (exact error: wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec.)

"""


def train_loop(cfg, model, dataset):
    # --------- 1. Setup -----------
    device = torch.device(cfg.device)
    model.to(device)

    diffusion = get_diffusion_schedule(cfg.schedule.schedule, cfg.schedule.timesteps, cfg.schedule.beta_start, cfg.schedule.beta_end)           # does not take cfg, should unpack someday
    visualizer = visualize_everything(model, cfg=cfg)   # cfg issue, enabled at top level not under viz (fixed?)
    logger = ExperimentLogger(cfg)

    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)
    optimizer = build_optimizer(model.parameters(), cfg)                       
    schedule = build_scheduler(optimizer, cfg)    # should rename to 'build_loss_scheduler'?              
    ema = EMA

    loss_fn = get_loss_fn(cfg)

    if torch.cuda.is_available(): scaler = torch.amp.GradScaler(enabled=cfg.training.amp)
    else: scaler = None


    start_epoch = 0
    if cfg.resume_path:
        start_epoch = load_checkpoint(cfg.resume_path, model, optimizer, ema)   # is ema a scheduler or it's own? Also not how this is inputted.


    # ------------------ 2. Training Loop --------------------
    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")  # datetime warning?

        for step, batch in enumerate(pbar): 
            with debug_section("train_step"):
                x = batch["image"].to(device)
                t = diffusion.sample_timesteps(x.shape[0].to(device))
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise=noise) # q-sample is odd

                with autocast(enabled=cfg.training.amp):
                    noise_pred = model(x_noisy, t)
                    loss = loss_fn(noise_pred, noise, t)

                debug_log(f"Step {step} | Loss: {loss.item():.4f}", name="train_loop")

                scaler.scale(loss).backward()

                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema: ema.update(model)

                # Log to console and optionally W&B/tensorboard
                if logger: logger.log_metric("loss", loss.item(), step=epoch * len(dataloader) + step)
            
            if scheduler: scheduler.step() 


        # ------------ 3. Evaluation & Logging ---------------
        if (epoch + 1) % cfg.training.vis_interval == 0:
            model.eval()
            vis_imgs = visualizer.visualize(model, diffusion, device)
            if logger: logger.log_images("samples", vis_imgs, step=epoch)

        
        # --------------- 4. Checkpointing -------------------
        if (epoch+1) % cfg.training.ckpt_interval == 0:
            save_checkpoint(cfg.save_dir, model, optimizer, ema, epoch)

    logger.finish()





            




    













