import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def create_directories(model_to_do: str, signal_filename: str) -> str:
    """Creates directories to store model plots + Keras files and returns directory
    name."""
    # Append time/date to directory name
    creation_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S/"))
    dir_name = model_to_do + signal_filename + "_" + creation_time

    # Create directories
    plot_path = Path("./plots") / dir_name
    plot_path.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Directory {plot_path} created!")

    keras_outputs_path = Path("./keras_outputs") / dir_name
    keras_outputs_path.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Directory {keras_outputs_path} created!")

    return dir_name


def load_dataset(filename: Path) -> pd.DataFrame:
    """Loads .pkl file, does some pre-processing and returns Pandas DataFrame"""
    # Load dataset
    df = pd.read_pickle(filename)
    # Replace infs with nan's
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replace nan's with 0
    # TODO: this should be moved to writing the file out so we don't have to do this
    # here.
    df = df.fillna(0)

    # Delete some 'virtual' variables only needed for pre-processing
    # TODO: Remove these guys before writing them out too
    del df["track_sign"]
    del df["clus_sign"]

    # Delete track_vertex vars in tracks
    # TODO: this should be removed ahead of time.
    vertex_delete = [col for col in df if col.startswith("nn_track_vertex_x")]
    vertex_delete += [col for col in df if col.startswith("nn_track_vertex_y")]
    vertex_delete += [col for col in df if col.startswith("nn_track_vertex_z")]
    for item in vertex_delete:
        del df[item]

    # Print sizes of inputs for signal, qcd, and bib
    logging.debug(df.head())
    logging.info("  Length of Signal is: " + str(df[df.label == 1].shape[0]))
    logging.info("  Length of QCD is: " + str(df[df.label == 0].shape[0]))
    logging.info("  Length of BIB is: " + str(df[df.label == 2].shape[0]))

    return df


def match_adversary_weights(df):
    """Match the sum of weights of data/QCD in input df'

    NOTE: The dataframe will be sorted with QCD and then Data,
    so it would be good to shuffle (using DataFrame.sample)
    or similar before using the data.

    :param df: input dataframe
    :param random_state: keep track of random seed
    :return: returns dataframe with matching sum of mcEvent Weight
    """
    # TODO: could we speed this up by modifying things in place rather than
    # splitting the two out into two different dataframes? Or just copying the one
    # column?
    qcd = df.loc[(df["label"] == 0)].copy()
    data = df.loc[(df["label"] == 2)].copy()

    qcd_weight_sum = np.sum(qcd["mcEventWeight"])
    data_weight_sum = np.sum(data["mcEventWeight"])

    # Reweight the QCD to match the data.
    qcd.loc[:, "mcEventWeight"] = qcd.loc[:, "mcEventWeight"] * (
        data_weight_sum / qcd_weight_sum
    )

    # Put them back together
    df = pd.concat([qcd, data])
    return df


# def low_or_high_pt_selection(
#     df: pd.DataFrame,
#     lowPt: bool,
#     highPt: bool,
#     random_state: Optional[Any] = None,
# ) -> pd.DataFrame:
#     """Only selects low mH (60, 125, 200) or high mH (400, 600, 1000) - DEPRECATED

#     TODO: This needs to be part of building the sample files, not part of the
#           loading.

#     :param df: input dataframe
#     :param lowPt: bool if including low M
#     :param highPt: bool if including high M
#     :param random_state: seed if we want to reproduce random shuffling
#     :return: dataframe with selected signal samples
#     """
#     if lowPt and not highPt:
#         signal = df.loc[
#             ((df["label"] == 1) & (df["llp_mH"] == 60))
#             | ((df["label"] == 1) & (df["llp_mH"] == 125))
#             | ((df["label"] == 1) & (df["llp_mH"] == 200))
#         ]
#         qcd = df.loc[
#             ((df["label"] == 0) & (df["llp_mH"] == 60))
#             | ((df["label"] == 0) & (df["llp_mH"] == 125))
#             | ((df["label"] == 0) & (df["llp_mH"] == 200))
#         ]
#         bib = df.loc[
#             ((df["label"] == 2) & (df["llp_mH"] == 60))
#             | ((df["label"] == 2) & (df["llp_mH"] == 125))
#             | ((df["label"] == 2) & (df["llp_mH"] == 200))
#         ]
#     elif highPt and not lowPt:
#         signal = df.loc[
#             ((df["label"] == 1) & (df["llp_mH"] == 400))
#             | ((df["label"] == 1) & (df["llp_mH"] == 600))
#             | ((df["label"] == 1) & (df["llp_mH"] == 1000))
#         ]
#         qcd = df.loc[
#             ((df["label"] == 0) & (df["llp_mH"] == 400))
#             | ((df["label"] == 0) & (df["llp_mH"] == 600))
#             | ((df["label"] == 0) & (df["llp_mH"] == 1000))
#         ]
#         bib = df.loc[
#             ((df["label"] == 2) & (df["llp_mH"] == 400))
#             | ((df["label"] == 2) & (df["llp_mH"] == 600))
#             | ((df["label"] == 2) & (df["llp_mH"] == 1000))
#         ]
#     else:
#         raise ValueError("Must select either lowPt or highPt, not both or neither!")

#     if (lowPt and not highPt) or (highPt and not lowPt):
#         signal, qcd = match_pandas_length(signal, qcd)
#         signal, bib = match_pandas_length(signal, bib)
#         df = pd.concat([qcd, signal, bib])
#         df = df.sample(frac=1.0, random_state=random_state)
#         return df
#     return df

T = TypeVar("T", pd.Series, pd.DataFrame)


def match_pandas_length(df_1: T, df_2: T) -> Tuple[T, T]:
    """utility function to make two dataframes the same length

    :param df_1:
    :param df_2:
    :return:
    """
    length_1 = df_1.shape[0]
    length_2 = df_2.shape[0]
    min_length = min(length_1, length_2)
    df_1 = df_1.iloc[0:min_length]
    df_2 = df_2.iloc[0:min_length]
    return df_1, df_2


def low_or_high_pt_selection_train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    weights: pd.Series,
    mc_weights: pd.Series,
    Z: pd.DataFrame,
    lowPt: bool,
    highPt: bool,
    random_state: Optional[Any] = None,
):
    """Only selects low pt (low M: 60, 125, 200) or high pt (high M: 400, 600, 1000)
    signal jets

    TODO: This needs to be part of building the sample files, not part of the
          loading (other than perhaps making them the same length).

    :param X: input variable dataframe
    :param Y: jet labels
    :param weights: flattened weights
    :param mc_weights: mc-only weights
    :param Z: signal mH, mS
    :param lowPt: bool if taking low M
    :param highPt: bool if taking high M
    :param random_state: if we shuffle, keep seed (not used)
    :return:
    """
    if lowPt and not highPt:
        signal_X = X.loc[
            ((Y == 1) & (Z["llp_mH"] == 60))
            | ((Y == 1) & (Z["llp_mH"] == 125))
            | ((Y == 1) & (Z["llp_mH"] == 200))
        ]
        signal_Y = Y.loc[
            ((Y == 1) & (Z["llp_mH"] == 60))
            | ((Y == 1) & (Z["llp_mH"] == 125))
            | ((Y == 1) & (Z["llp_mH"] == 200))
        ]
        signal_Z = Z.loc[
            ((Y == 1) & (Z["llp_mH"] == 60))
            | ((Y == 1) & (Z["llp_mH"] == 125))
            | ((Y == 1) & (Z["llp_mH"] == 200))
        ]
        signal_weights = weights.loc[
            ((Y == 1) & (Z["llp_mH"] == 60))
            | ((Y == 1) & (Z["llp_mH"] == 125))
            | ((Y == 1) & (Z["llp_mH"] == 200))
        ]
        signal_mc_weights = mc_weights.loc[
            ((Y == 1) & (Z["llp_mH"] == 60))
            | ((Y == 1) & (Z["llp_mH"] == 125))
            | ((Y == 1) & (Z["llp_mH"] == 200))
        ]

        qcd_X = X.loc[
            ((Y == 0) & (Z["llp_mH"] == 60))
            | ((Y == 0) & (Z["llp_mH"] == 125))
            | ((Y == 0) & (Z["llp_mH"] == 200))
        ]
        qcd_Y = Y.loc[
            ((Y == 0) & (Z["llp_mH"] == 60))
            | ((Y == 0) & (Z["llp_mH"] == 125))
            | ((Y == 0) & (Z["llp_mH"] == 200))
        ]
        qcd_Z = Z.loc[
            ((Y == 0) & (Z["llp_mH"] == 60))
            | ((Y == 0) & (Z["llp_mH"] == 125))
            | ((Y == 0) & (Z["llp_mH"] == 200))
        ]
        qcd_weights = weights.loc[
            ((Y == 0) & (Z["llp_mH"] == 60))
            | ((Y == 0) & (Z["llp_mH"] == 125))
            | ((Y == 0) & (Z["llp_mH"] == 200))
        ]
        qcd_mc_weights = mc_weights.loc[
            ((Y == 0) & (Z["llp_mH"] == 60))
            | ((Y == 0) & (Z["llp_mH"] == 125))
            | ((Y == 0) & (Z["llp_mH"] == 200))
        ]

        bib_X = X.loc[
            ((Y == 2) & (Z["llp_mH"] == 60))
            | ((Y == 2) & (Z["llp_mH"] == 125))
            | ((Y == 2) & (Z["llp_mH"] == 200))
        ]
        bib_Y = Y.loc[
            ((Y == 2) & (Z["llp_mH"] == 60))
            | ((Y == 2) & (Z["llp_mH"] == 125))
            | ((Y == 2) & (Z["llp_mH"] == 200))
        ]
        bib_Z = Z.loc[
            ((Y == 2) & (Z["llp_mH"] == 60))
            | ((Y == 2) & (Z["llp_mH"] == 125))
            | ((Y == 2) & (Z["llp_mH"] == 200))
        ]
        bib_weights = weights.loc[
            ((Y == 2) & (Z["llp_mH"] == 60))
            | ((Y == 2) & (Z["llp_mH"] == 125))
            | ((Y == 2) & (Z["llp_mH"] == 200))
        ]
        bib_mc_weights = mc_weights.loc[
            ((Y == 2) & (Z["llp_mH"] == 60))
            | ((Y == 2) & (Z["llp_mH"] == 125))
            | ((Y == 2) & (Z["llp_mH"] == 200))
        ]

    elif highPt and not lowPt:
        signal_X = X.loc[
            ((Y == 1) & (Z["llp_mH"] == 400))
            | ((Y == 1) & (Z["llp_mH"] == 600))
            | ((Y == 1) & (Z["llp_mH"] == 1000))
        ]
        signal_Y = Y.loc[
            ((Y == 1) & (Z["llp_mH"] == 400))
            | ((Y == 1) & (Z["llp_mH"] == 600))
            | ((Y == 1) & (Z["llp_mH"] == 1000))
        ]
        signal_Z = Z.loc[
            ((Y == 1) & (Z["llp_mH"] == 400))
            | ((Y == 1) & (Z["llp_mH"] == 600))
            | ((Y == 1) & (Z["llp_mH"] == 1000))
        ]
        signal_weights = weights.loc[
            ((Y == 1) & (Z["llp_mH"] == 400))
            | ((Y == 1) & (Z["llp_mH"] == 600))
            | ((Y == 1) & (Z["llp_mH"] == 1000))
        ]
        signal_mc_weights = mc_weights.loc[
            ((Y == 1) & (Z["llp_mH"] == 400))
            | ((Y == 1) & (Z["llp_mH"] == 600))
            | ((Y == 1) & (Z["llp_mH"] == 1000))
        ]

        qcd_X = X.loc[
            ((Y == 0) & (Z["llp_mH"] == 400))
            | ((Y == 0) & (Z["llp_mH"] == 600))
            | ((Y == 0) & (Z["llp_mH"] == 1000))
        ]
        qcd_Y = Y.loc[
            ((Y == 0) & (Z["llp_mH"] == 400))
            | ((Y == 0) & (Z["llp_mH"] == 600))
            | ((Y == 0) & (Z["llp_mH"] == 1000))
        ]
        qcd_Z = Z.loc[
            ((Y == 0) & (Z["llp_mH"] == 400))
            | ((Y == 0) & (Z["llp_mH"] == 600))
            | ((Y == 0) & (Z["llp_mH"] == 1000))
        ]
        qcd_weights = weights.loc[
            ((Y == 0) & (Z["llp_mH"] == 400))
            | ((Y == 0) & (Z["llp_mH"] == 600))
            | ((Y == 0) & (Z["llp_mH"] == 1000))
        ]
        qcd_mc_weights = mc_weights.loc[
            ((Y == 0) & (Z["llp_mH"] == 400))
            | ((Y == 0) & (Z["llp_mH"] == 600))
            | ((Y == 0) & (Z["llp_mH"] == 1000))
        ]

        bib_X = X.loc[
            ((Y == 2) & (Z["llp_mH"] == 400))
            | ((Y == 2) & (Z["llp_mH"] == 600))
            | ((Y == 2) & (Z["llp_mH"] == 1000))
        ]
        bib_Y = Y.loc[
            ((Y == 2) & (Z["llp_mH"] == 400))
            | ((Y == 2) & (Z["llp_mH"] == 600))
            | ((Y == 2) & (Z["llp_mH"] == 1000))
        ]
        bib_Z = Z.loc[
            ((Y == 2) & (Z["llp_mH"] == 400))
            | ((Y == 2) & (Z["llp_mH"] == 600))
            | ((Y == 2) & (Z["llp_mH"] == 1000))
        ]
        bib_weights = weights.loc[
            ((Y == 2) & (Z["llp_mH"] == 400))
            | ((Y == 2) & (Z["llp_mH"] == 600))
            | ((Y == 2) & (Z["llp_mH"] == 1000))
        ]
        bib_mc_weights = mc_weights.loc[
            ((Y == 2) & (Z["llp_mH"] == 400))
            | ((Y == 2) & (Z["llp_mH"] == 600))
            | ((Y == 2) & (Z["llp_mH"] == 1000))
        ]
    else:
        # Here we just train everything we have
        return X, Y, Z, weights, mc_weights

    signal_X, qcd_X = match_pandas_length(signal_X, qcd_X)
    signal_X, bib_X = match_pandas_length(signal_X, bib_X)
    signal_Y, qcd_Y = match_pandas_length(signal_Y, qcd_Y)
    signal_Y, bib_Y = match_pandas_length(signal_Y, bib_Y)
    signal_Z, qcd_Z = match_pandas_length(signal_Z, qcd_Z)
    signal_Z, bib_Z = match_pandas_length(signal_Z, bib_Z)
    signal_weights, qcd_weights = match_pandas_length(signal_weights, qcd_weights)
    signal_weights, bib_weights = match_pandas_length(signal_weights, bib_weights)
    signal_mc_weights, qcd_mc_weights = match_pandas_length(
        signal_mc_weights, qcd_mc_weights
    )
    signal_mc_weights, bib_mc_weights = match_pandas_length(
        signal_mc_weights, bib_mc_weights
    )

    X = pd.concat([qcd_X, signal_X, bib_X])  # type: ignore
    Y = pd.concat([qcd_Y, signal_Y, bib_Y])  # type: ignore
    Z = pd.concat([qcd_Z, signal_Z, bib_Z])  # type: ignore

    # TODO: all the random states should be somehow tracked so
    # we don't loose the ability to have some randomness.
    # in particular here, this next arg needs a random_state.
    X, Y, Z = shuffle(X, Y, Z)  # type: ignore

    weights = pd.concat([qcd_weights, signal_weights, bib_weights])  # type: ignore
    mc_weights = pd.concat(
        [qcd_mc_weights, signal_mc_weights, bib_mc_weights]
    )  # type: ignore

    return X, Y, Z, weights, mc_weights
