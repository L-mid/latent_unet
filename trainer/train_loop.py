
import os, torch, time
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from utils.debug import debug_log, debug_section

from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.visualizer import build_visualizer
from trainer.logger import build_logger
from trainer.optim_utils import build_optimizer, build_scheduler, build_ema
from diffusion.sampler_registry import build_diffusion
from trainer.losses import get_loss_fn
from tqdm import tqdm


def train_loop(cfg, model, dataset):