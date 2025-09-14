
import pytest
from trainer.logger import build_logger
from omegaconf import OmegaConf 
from tempfile import mkdtemp 

@pytest.mark.failure_injection
def test_wandb_init_500_falls_back_to_noop(fp):
    fp.once("logger.wandb.inil", exc=RuntimeError("HTTP 500")) 
    tmp_dir = mkdtemp()
    cfg = OmegaConf.create(
        {
        "logging": {
            "use_wandb": True,
            "use_tb": True,
            "output_dir": tmp_dir,
            "project_name": "project_name",
            "run_name": "run_name"
        }
    }
)
    logger = build_logger(cfg)
    # should not crash when logging even though W&B failed
    logger.log_dict({"loss": 1.23}, step=1)     # is it ideal for a logger to force the step?