from pathlib import Path

import numpy as np
import pandas as pd

import logging

from typing import List

from cal_ratio_trainer.common.trained_model import TrainedModelData
from cal_ratio_trainer.common.column_names import (
    col_llp_mass_names,
    col_cluster_names,
    col_track_names,
    col_mseg_names,
    col_jet_names,
)


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


def load_test_data_from_df(data: pd.DataFrame) -> TrainedModelData:
    """
    Loads test data from a DataFrame and prepares it for model evaluation.
    Args:
        data (pd.DataFrame): The DataFrame containing the test data.
    Returns:
        TrainedModelData: The prepared test data in the form of TrainedModelData object.
    """

    # TODO: Fix something upstream to not have to do this anymore
    data = data.replace(np.nan, 0.0)

    def pull_columns(df: pd.DataFrame, col_list: List[str]) -> np.ndarray:
        """
        Extracts the specified columns from a DataFrame and reshapes the resulting array.
        Args:
            df (pd.DataFrame): The DataFrame from which to extract columns.
            col_list (List[str]): The list of column names to extract.
        Returns:
            np.ndarray: The reshaped array containing the extracted columns.
        """
        df_train = df.loc[:, col_list]
        logging.debug(f"df_train: {df_train.shape}: {df_train.columns}")
        train = df_train.values

        # Find the largest value of the integer in column names ending with "_<integer>"
        # and reshape the inputs so they look like features x depth. If we don't find anything
        # like that, leave this as a simple array of features.
        max_index = max(
            [
                int(c_name.split("_")[-1])
                for c_name in col_list
                if len(c_name.split("_")) > 1 and c_name.split("_")[-1].isdigit()
            ],
            default=-1,
        )

        if max_index > 0:
            rows = int(len(col_list) / (max_index + 1))
            assert len(col_list) == rows * (
                max_index + 1
            ), f"Found {rows} rows, expected {max_index + 1}"

            return train.reshape(train.shape[0], max_index + 1, rows)
        else:
            return train

    cluster_inputs = pull_columns(data, col_cluster_names)
    logging.debug(f"cluster_inputs: {cluster_inputs.shape}")
    track_inputs = pull_columns(
        data, [t for t in col_track_names if "track_Pixel" not in t]
    )
    logging.debug(f"track_inputs: {track_inputs.shape}")
    mseg_inputs = pull_columns(data, col_mseg_names)
    logging.debug(f"mseg_inputs: {mseg_inputs.shape}")
    jet_inputs = data.loc[:, col_jet_names].values
    logging.debug(f"jet_inputs: {jet_inputs.shape}")

    return TrainedModelData(
        x=[
            cluster_inputs,
            track_inputs,
            mseg_inputs,
            jet_inputs,
        ],
        y=data["label"],
        z=data[col_llp_mass_names],
        weights=[data["mcEventWeight"].values],  # type: ignore
    )
