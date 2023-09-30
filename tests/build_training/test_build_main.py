from pathlib import Path
import pandas as pd
from cal_ratio_trainer.build.build_main import build_main_training
from cal_ratio_trainer.config import BuildMainTrainingConfig, training_input_file


def test_build_main_one_file(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=None
            )
        ],
        output_file=out_file,
        min_jet_pT=30,
        max_jet_pT=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 76


def test_build_main_one_file_length(tmp_path):
    out_file = tmp_path / "test_output.pkl"
    c = BuildMainTrainingConfig(
        input_files=[
            training_input_file(
                input_file=Path("tests/data/sig_311424_600_275.pkl"), num_events=2
            )
        ],
        output_file=out_file,
        min_jet_pT=30,
        max_jet_pT=400,
    )

    build_main_training(c)

    assert out_file.exists()
    df = pd.read_pickle(out_file)
    assert len(df) == 2
