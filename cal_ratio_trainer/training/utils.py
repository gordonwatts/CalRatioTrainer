import logging
from datetime import datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd


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


def make_local(file_path: str, cache: Path) -> Path:
    """Uses the `fsspec` library to copy a non-local file locally in the `cache`.
    If the `file_path` is already a local file, then it isn't copied locally.

    Args:
        file_path (str): The URI of the file we want to be local.

    Returns:
        Path: Path on the local system to the data.
    """
    if file_path.startswith("file://"):
        return Path(file_path[7:])
    else:
        local_path = cache / Path(file_path).name
        if local_path.exists():
            return local_path

        # Ok - copy block by block.
        logging.warning(f"Copying file {file_path} locally to {cache}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_file_path = local_path.with_suffix(".tmp")
        with open(tmp_file_path, "wb") as f_out:
            with fsspec.open(file_path, "rb") as f_in:
                # Read `f_in` in chunks of 1 MB and write them to `f_out`
                while True:
                    data = f_in.read(50 * 1024**2)  # type: ignore
                    if not data:
                        break
                    f_out.write(data)
        tmp_file_path.rename(local_path)
        logging.warning(f"Done copying file {file_path}")
        return local_path


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
    logging.info("Length of Signal is: " + str(df[df.label == 1].shape[0]))
    logging.info("Length of QCD is: " + str(df[df.label == 0].shape[0]))
    logging.info("Length of BIB is: " + str(df[df.label == 2].shape[0]))

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
