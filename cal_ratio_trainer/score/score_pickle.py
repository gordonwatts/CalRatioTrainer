from pathlib import Path
import numpy as np

import pandas as pd

from cal_ratio_trainer.common.trained_model import TrainedModel, load_model_from_spec
from cal_ratio_trainer.config import ScorePickleConfig
from cal_ratio_trainer.reporting.evaluation_utils import load_test_data_from_df


def _score_pickle_file(model: TrainedModel, file_path: Path):
    """Run the ML scoring on the data in file_path.

    Args:
        file_path (Path): The path to the input pickle file.
    """
    # Load the data and format it for training. Note
    # that we assume signal - the label is never used.
    data = pd.read_pickle(file_path)
    assert isinstance(data, pd.DataFrame)
    training_data = load_test_data_from_df(data)

    # Run the prediction
    predict = model.predict(training_data)
    print(predict[:, 0])
    print(predict[:, 1])
    print(predict[:, 2])
    print(f"label 0: { np.sum(predict[:, 0] > 0.1 ) }")
    print(f"label 1: { np.sum(predict[:, 1] > 0.1 ) }")
    print(f"label 2: { np.sum(predict[:, 2] > 0.1 ) }")


def score_pkl_files(config: ScorePickleConfig):
    """Score a list of pickle files, writing out
    parquet files with the scored jets.

    Args:
        config (ScorePickleConfig): Config describing parameters for the job.
    """
    # Load the model for the training we have been given.
    assert config.training is not None

    # Load the training
    model = load_model_from_spec(config.training)

    # Run through each file
    assert config.input_files is not None
    for f in config.input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file {f} does not exist")

        _score_pickle_file(model, f)
