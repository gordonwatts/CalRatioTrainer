from cal_ratio_trainer.config import load_config
from cal_ratio_trainer.training.runner_utils import training_runner_util


def test_default_run():
    "Make sure the default configuration runs correctly"

    config = load_config()
    training_runner_util(config)
