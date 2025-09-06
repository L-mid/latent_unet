
import os, pytest
from trainer.logger import build_logger

pytestmark = pytest.mark.logging_integration # unknown?

def test_tb_wandb_smoke(tmp_path):
    os.environ["ENABLE_TB"] = "1"
    os.environ["ENABLE_WANDB"] = "1"        # flip to 1 if you want W&B too
    os.environ["TB_LOGDIR"] = str(tmp_path / "runs")

    class C: pass
    cfg = C(); cfg.logging = C()
    cfg.logging.use_tb = True; cfg.logging.tb_logdir = os.environ["TB_LOGDIR"] #that last part confuses me
    cfg.logging.use_wandb = True

    logger = build_logger(cfg)
    logger.log_dict({"loss": 0.123}, step=0)
    logger.finish()


