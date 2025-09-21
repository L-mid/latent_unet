
import os, torch, time
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.debug import debug_log, debug_section 

from utils.checkpointing.tensorstore_checkpointing import save_checkpoint, load_checkpoint
from utils.visualizer import visualize_everything
from trainer.logger import build_logger
from trainer.optim_utils import build_optimizer, build_scheduler
from trainer.ema_utils import EMA
from trainer.losses import get_loss_fn
from diffusion.forward_process import ForwardProcess
from tqdm import tqdm

# === NOTES:
"""
All these calls (like build_optimizer, EMA) are weird, not the same, and mabye should be crunched down.
)

Weird noop logger mechanics.


There were bad warnings before. Something between 'build logger' and Experiment logger is now rid of them. 

"""


def train_loop(cfg, model, dataset):
    # --------- 1. Setup -----------
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

    start_epoch = 0
    if cfg.resume_path:
        start_epoch = load_checkpoint(model, optimizer, scheduler=loss_scheduler, path=cfg.resume_path)   


    # ------------------ 2. Training Loop --------------------
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

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema: ema.update(model)

                # Log to console and optionally W&B/tensorboard
                loss_value = float(loss.item())
                logs = {"loss": loss_value}

                if logger: logger.log_scalar("loss", loss.item(), step=epoch * len(dataloader) + step)
            
            if loss_scheduler: loss_scheduler.step()  # loss thing? can it step?


        # ------------ 3. Evaluation & Logging ---------------
        if (epoch + 1) % cfg.training.vis_interval == 0:
            model.eval()
            vis_imgs = visualize_everything(model, cfg=cfg) # disabled, not init properly.
            if logger: 
                if vis_imgs == None: pass
                else: logger.log_image("samples", vis_imgs, step=epoch)

        
        # --------------- 4. Checkpointing -------------------

        bytes_model = sum(p.numel()*p.element_size() for p in model.state_dict().values())
        bytes_ema   = bytes_model if ema else 0
        bytes_opt   = 2*bytes_model if optimizer else 0   # exp_avg + exp_avg_sq
        print((bytes_model+bytes_ema+bytes_opt)/1e9, "GB expected (fp32)")

        def nbytes(t): return t.numel() * t.element_size()

        def bytes_model(model):
            return sum(nbytes(p) for p in model.state_dict().values())

        def bytes_opt(optimizer):
            tot = 0
            for st in optimizer.state.values():
                for v in st.values():
                    if torch.is_tensor(v): tot += nbytes(v)
            return tot

        def bytes_ema(ema_model):  # if you keep a shadow copy
            return sum(nbytes(p) for p in ema_model.state_dict().values())


        import os, psutil, time
        proc = psutil.Process(os.getpid())

        def rss_mb(): return proc.memory_info().rss / (1024**2)

        print("RSS before:", rss_mb())
        t0 = time.perf_counter()
        save_checkpoint(model, optimizer=None, scheduler=loss_scheduler, epoch=epoch, step=step, path=cfg.checkpoint.out_dir)       # your function
        print("RSS after :", rss_mb(), "elapsed:", time.perf_counter()-t0, "s")


        if (epoch+1) % cfg.training.ckpt_interval == 0:
            raise RuntimeError
            if cfg.checkpoint.backend in {"noop", None}:
                pass    # skip saving in tests
            else:
                save_checkpoint(model, optimizer, scheduler=loss_scheduler, epoch=epoch, step=step, path=cfg.checkpoint.out_dir)    
                
    logger.finish()
    return logs





            




    













