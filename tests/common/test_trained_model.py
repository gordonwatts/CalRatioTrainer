from pathlib import Path

from cal_ratio_trainer.common.trained_model import (
    load_model,
    load_trained_model_from_json,
    load_trained_model_from_training,
)


def test_load_old_json():
    """Test load a json file from a path"""
    data_path = Path("./tests/data/old_json_model.json")
    model = load_trained_model_from_json(data_path)
    assert model is not None
    assert model.model is not None


def test_load_model_epoch():
    """Test loading a model from a run/epoch number"""
    data_path = Path("./tests/data/train_results/default")
    model = load_trained_model_from_training(data_path, 1)
    assert model is not None
    assert model.model is not None


def test_load_model_string_training():
    """Test loading a model from a string of run/epoch"""
    model = load_model("default/1", base_path=Path("./tests/data/train_results"))
    assert model is not None
    assert model.model is not None


def test_load_model_string_json():
    """Test loading a model from a string of JSON location"""
    model = load_model("./tests/data/old_json_model.json")
    assert model is not None
    assert model.model is not None
