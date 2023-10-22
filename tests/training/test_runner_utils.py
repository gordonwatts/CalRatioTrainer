from pathlib import Path
from cal_ratio_trainer.config import TrainingConfig, load_config
from cal_ratio_trainer.training.runner_utils import training_runner_util


def test_small_run_felix_files():
    "Make sure the default configuration runs correctly, with minor mods"

    config = load_config(TrainingConfig)
    assert isinstance(config, TrainingConfig)
    config.main_training_file = "./tests/data/felix_main_training.pkl"
    config.cr_training_file = "./tests/data/felix_control_region.pkl"

    config.num_splits = 1
    config.epochs = 1

    training_runner_util(config, Path("./calratio_training"))
