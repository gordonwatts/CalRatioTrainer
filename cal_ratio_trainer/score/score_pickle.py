from pathlib import Path
import numpy as np

import pandas as pd
import awkward as ak

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

    # Build the output and write it to parquet. For comparison reasons,
    # we need run, event, jet pt, eta, and phi, along with the three
    # prediction outputs. We'll create an awkward event.
    data_dict = {
        # "run_number": data["run_number"],
        # "event_number": data["event_number"],
        "jet_pt": data["jet_pt"],
        "jet_eta": data["jet_eta"],
        "jet_phi": data["jet_phi"],
        "label": data["label"],
        "jet_nn_qcd": predict[:, 0],
        "jet_nn_sig": predict[:, 1],
        "jet_nn_bib": predict[:, 2],
    }

    ak_data = ak.from_iter(data_dict)
    ak.to_parquet(ak_data, file_path.parent / f"{file_path.stem}-score.parquet")

    print(predict[:, 0])
    print(predict[:, 1])
    print(predict[:, 2])
    print(f"label 0: { np.sum(predict[:, 0] > 0.1 ) }")
    print(f"label 1: { np.sum(predict[:, 1] > 0.1 ) }")
    print(f"label 2: { np.sum(predict[:, 2] > 0.1 ) }")
    print(f"length: {len(predict[:, 0])}")


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
