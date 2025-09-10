
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils.debug import debug_log, debug_section    # not done yet

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
        start_epoch = load_model(model, optimizer, scheduler=None, path=cfg.resume_path)


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
                loss, _ = loss_fn(noise_pred, noise, t)

                debug_log(f"Step {step} | Loss: {loss.item():.4f}", name="train_loop")

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                #if ema: ema.update(model)       # should the ema state be saved in the model?

                # Log to console and optionally W&B/tensorboard
                loss_value = float(loss.item())
                logs = {"loss": loss_value}

                if logger: logger.log_scalar("loss", loss.item(), step=epoch * len(dataloader) + step)

        
        # ---------------- Evaluation & Logging ------------------------

        if (epoch+1) % cfg.training.vis_interval == 0:
            model.eval()
            vis_imgs = visualize_everything(model, cfg=cfg)
            if logger:
                if vis_imgs == None: pass
                logger.log_image("samples", vis_imgs, step=epoch)

        # -------------- Checkpointing -------------------

        if (epoch+1) % cfg.training.ckpt_interval == 0:
            if cfg.checkpoint.backend in {"noop", None}:        # this won't raise if catagory not present
                pass    # skip saving in tests
            else:
                save_model(model, optimizer, epoch=epoch, scheduler=None, step=step, path=cfg.checkpoint.out_dir)       # conisder adding EMA, mabye metadata.

    
    logger.finish()
    return logs


def find_latest_checkpoint(dirpath: str | Path):
    p = Path(dirpath)
    if not p.exists():
        return None
    
    resume_root = find_latest_checkpoint(cfg.resume_path)
    if resume_root is None:
        logger.info("No checkpoint found in %s; starting fresh.", cfg.resume_path)
        starting_epoch = 0
    else:
        state = load_state  # or just that zarr loader (basically make sure all these are loaded)
        trainer.load_state(state)
        starting_epoch = state["epoch_next"]    

        # in meta, store:
        #{"epoch_completed": N, "epoch_next": N+1, "global_step": S, ...}
        # makes sure the saver closes checkpointer.

        # theres a test too:

def test_resume_end_to_end(tmp_path):
    run_dir = tmp_path / "run1"
    ckpt_dir = run_dir / "ckpt"
    viz_dir = run_dir / "viz"
    ckpt_dir.mkdir(parents=True)
    viz_dir.mkdir()

    # First run: train 1 epoch and save
    cfg = make_cfg()
    cfg.checkpoint.out_dir = str(ckpt_dir)
    cfg.visualization.output_dir = str(viz_dir)
    train(cfg, epochs=1)  # should write meta.json with epoch_next=1

    # Second run: resume and do another epoch
    cfg2 = make_cfg()
    cfg2.resume_path = str(ckpt_dir)
    cfg2.checkpoint.out_dir = str(ckpt_dir)  # continue in same place
    start_epoch = resume_or_fresh(cfg2)      # should be 1
    assert start_epoch == 1
    train(cfg2, epochs=2, start_epoch=start_epoch)