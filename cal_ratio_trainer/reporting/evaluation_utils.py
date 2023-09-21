from pathlib import Path

import numpy as np
import pandas as pd

from cal_ratio_trainer.common.trained_model import TrainedModelData


def load_test_data(run_dir: Path) -> TrainedModelData:
    """Load up data and model for a particular epoch.

    Args:
        run_dir (Path): The directory where this epoch can be found
        epoch (int): Which epoch to load against

    Returns:
        TestInfo: Info containing the test data.
    """
    # Next, fetch the test data that needs to be used for all of this.
    # y_test,
    # Z_test,
    x_data = np.load(run_dir.parent / "x_to_test.npz")
    y_data = pd.read_pickle(run_dir.parent / "y_test.pkl")
    z_data = pd.read_pickle(run_dir.parent / "Z_test.pkl")
    weights = np.load(run_dir.parent / "weights_to_test.npz")

    return TrainedModelData(
        x=list(x_data.values()),
        y=y_data,
        z=z_data,
        weights=list(weights.values()),
    )
