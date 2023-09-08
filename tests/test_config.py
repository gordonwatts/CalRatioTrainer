from cal_ratio_trainer.config import load_config
from pathlib import Path


def test_config_default():
    c = load_config()
    assert c.epochs is not None
    assert c.epochs > 0


def test_config_default_with_new():
    c = load_config(Path("tests/test_config.yaml"))

    # should be set by default
    assert c.epochs != 0
    assert c.epochs is not None
    # loaded from the new file
    assert c.nodes_track_lstm == 22


def test_str():
    c = load_config()
    assert "epochs" in str(c)
    assert "\n" in str(c)


def test_url():
    c = load_config(Path("tests/test_with_file_url.yaml"))
    assert c.main_training_file == "file:///home/gwatts/junk.pkl"
