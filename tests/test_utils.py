import argparse
from typing import Any, Dict

import pytest
from cal_ratio_trainer.config import TrainingConfig

from cal_ratio_trainer.utils import HistoryTracker, add_config_args, apply_config_args


def test_add_args_right_ones():
    """Test that the add_config_args function adds the correct arguments."""
    # Create an argument parser
    parser = argparse.ArgumentParser()
    add_config_args(TrainingConfig, parser)

    # Check that there are a few arguments in the parser that we expect.
    help = parser.format_help()
    assert "--model_name" in help
    assert "--dropout_array" in help

    # Anything with a complex type should not be in the argument parser
    assert "--filters_cnn_constit" not in help


def test_add_args_help_string():
    """Test that the add_config_args function adds the correct arguments."""
    # Create an argument parser
    parser = argparse.ArgumentParser()
    add_config_args(TrainingConfig, parser)

    assert "Name of the model" in parser.format_help()


def test_update_args():
    """Test add_config_args to see if it properly updates training config"""
    arguments: Dict[str, Any] = {"model_name": "forking"}
    t_orig = TrainingConfig(**arguments)

    # Create an argument parser
    parser = argparse.ArgumentParser()
    add_config_args(TrainingConfig, parser)

    # Now, parse dummy command line with the model_name argument
    args = parser.parse_args(["--model_name", "shirtballs"])

    # Next, do the update of t_orig.
    t_final = apply_config_args(TrainingConfig, t_orig, args)

    # Check the original did not change and the update did occur.
    assert t_orig.model_name == "forking"
    assert t_final.model_name == "shirtballs"


def test_history_basic_storage():
    h = HistoryTracker()

    h.adv_loss.append(20)

    assert h.adv_loss == [20]

    h.second_loss.append(33)
    h.second_loss.append(33)

    assert len(h) == 2

    assert str(h) == "HistoryTracker(epochs=2, items=['adv_loss', 'second_loss'])"


def test_history_file_io(tmp_path):
    h = HistoryTracker()

    h.adv_loss.append(20)
    h.save(tmp_path / "test")

    h1 = HistoryTracker(file=tmp_path / "test")
    assert len(h1) == 1
    assert h1.adv_loss == [20]


def test_history_find_smallest():
    h = HistoryTracker()

    h.loss.append(20)
    h.loss.append(30)
    h.loss.append(10)

    assert h.find_smallest("loss", 2) == [2, 0]


def test_history_find_smallest_none():
    h = HistoryTracker()

    h.loss.append(20)

    assert h.find_smallest("loss", 2) == [0]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_history_sum_n(n: int):
    h = HistoryTracker()

    for i in range(n):
        h.one.append(1)
        h.two.append(2)
        h.three.append(3)

    h.make_sum("sum", ["one", "two", "three"])
    assert h.sum == [6] * n


def test_history_get_values_for_epoch():
    h = HistoryTracker()

    h.loss.append(20)
    h.loss.append(30)
    h.loss.append(10)

    assert h.values_for(1) == {"loss": 30}
    assert h.values_for(2) == {"loss": 10}
