from pathlib import Path
from cal_ratio_trainer.common.fileio import load_dataset


def test_load_simple(tmp_path: Path):
    ds = load_dataset("tests/data/cr_trainer_main_training.pkl", tmp_path)
    assert len(ds) > 0
