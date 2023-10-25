from pathlib import Path
from cal_ratio_trainer.config import TrainingConfig, load_config
from cal_ratio_trainer.training.runner_utils import training_runner_util


def test_small_run_felix_files():
    """Make sure the default configuration runs correctly, with minor mods
    and on Felix's original files.
    """

    config = load_config(TrainingConfig)
    assert isinstance(config, TrainingConfig)
    config.main_training_file = "./tests/data/felix_main_training.pkl"
    config.cr_training_file = "./tests/data/felix_control_region.pkl"

    config.num_splits = 1
    config.epochs = 1

    training_runner_util(config, Path("./calratio_training"))


def test_small_run_cr_trainer_files():
    """Make sure the default configuration runs correctly, with minor mods
    and on files we write out with our `build` command.

    * For now, use the felix control region until our build does that too.
    """

    config = load_config(TrainingConfig)
    assert isinstance(config, TrainingConfig)
    config.main_training_file = "./tests/data/cr_trainer_main_training.pkl"
    config.cr_training_file = "./tests/data/felix_control_region.pkl"

    config.num_splits = 1
    config.epochs = 1

    training_runner_util(config, Path("./calratio_training"))
