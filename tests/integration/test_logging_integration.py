
import os, pytest
from trainer.logger import build_logger

pytestmark = pytest.mark.logging_integration # unknown?

from tempfile import mkdtemp
from pathlib import Path

def test_tb_wandb_smoke():
    tmp_path = mkdtemp()
    os.environ["ENABLE_TB"] = "1"
    os.environ["ENABLE_WANDB"] = "1"        # flip to 1 if you want W&B too
    os.environ["TB_LOGDIR"] = f"{tmp_path}/runs"

    class C: pass
    cfg = C(); cfg.logging = C()
    cfg.logging.use_tb = True; cfg.logging.tb_logdir = os.environ["TB_LOGDIR"] #that last part confuses me
    cfg.logging.use_wandb = True
    cfg.logging.output_dir = tmp_path
    cfg.logging.project_name = "project"
    cfg.logging.run_name = tmp_path

    logger = build_logger(cfg)
    logger.log_dict({"loss": 0.123}, step=0)
    logger.finish()


# never clears itself?