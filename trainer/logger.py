
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Mapping

# === NOTES:
"""
DEPENDENT on tensorboard installation (if colab doesn't have keep in mind)

colab has spam issues with tensorboard, harmless but noting
"""

try: 
    import wandb
    from wandb.errors import CommError
    WANDB_AVAILIBLE = True
except Exception:
    wandb = None
    CommError = Exception 
    WANDB_AVAILIBLE = False

class BaseLogger:
    def __init__(self): pass
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_dict(self, metrics: Mapping[str, Any], step: int) -> None: ...
    def finish(self) -> None: ...

class NoopLogger(BaseLogger):
    def __init__(self): pass
    def log_scalar(self, tag: str, value: float, step: int): pass
    def log_dict(self, metrics, step): pass
    def finish(self): pass


def _make_tb_writer(log_dir=None):
    if os.getenv("ENABLE_TB", 0).lower() not in ("1", "true", "yes"):
        return None
    # Import ONLY when actually needed
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir) if log_dir else SummaryWriter()


class ExperimentLogger(BaseLogger):
    def __init__(self, cfg, output_dir="logs", use_wandb=True, debug_mode=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(output_dir) / f"{cfg.logging.project_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_dir=str(self.log_dir / "tensorboard")

        self.debug_mode = debug_mode 
        self.writer = _make_tb_writer(log_dir)

        self.use_wandb = use_wandb and WANDB_AVAILIBLE and cfg.logging.use_wandb
        self.wandb_run = None

        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=cfg.logging.project_name,
                    name=cfg.logging.run_name,
                    settings=wandb.Settings(init_timeout=10),
                    config=cfg,
                    dir=str(self.log_dir / "wandb"),
                    resume="allow"
                )
            except CommError:
                print("[LOGGER] W&B init failed: no internet or blocked API. Disabling W&B.")
                self.wandb_run = None
                self.use_wandb = False

        self._setup_python_logger()

    def _setup_python_logger(self):
        self.logger = logging.getLogger("trainer_logger")
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        fh = logging.FileHandler(self.log_dir / "log.txt")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if not self.debug_mode else logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_image(self, tag: str, image, step: int):
        self.writer.add_image(tag, image, step)     
        if self.use_wandb:
            wandb.log({tag: [wandb.Image(image, caption=tag)]}, step=step)

    def log_dict(self, metrics: Mapping[str, float], step: int):
        for k, v in metrics.items():
                self.log_scalar(k, float(v), step)

    def log_config(self, cfg):
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)
        if self.use_wandb:
            wandb.config.update(cfg)

    def print(self, msg: str, level: str = "info"):
        if level == "debug" and not self.debug_mode:
            return
        getattr(self.logger, level)(msg)

    def finish(self):
        self.writer.close()
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

    def log_model_graph(self, model, example_input):
        self.writer.add_graph(model, example_input)    


def build_logger(cfg) -> BaseLogger:
    use_tb   = getattr(cfg.logging, "enable_tb", False)
    use_wandb= getattr(cfg.logging, "enable_wandb", False)
    if not (use_tb or use_wandb):
        return NoopLogger()
    return ExperimentLogger(
        cfg=cfg,
        output_dir= cfg.logging.output_dir,
        use_wandb = use_wandb,
    )
# no tensorboard check