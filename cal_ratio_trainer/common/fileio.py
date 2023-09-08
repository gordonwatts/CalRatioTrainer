import logging
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd


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


def load_dataset(file_url: str, cache: Path) -> pd.DataFrame:
    """
    Loads a .pkl file from the given URL, replaces infs with nans and nans with 0s,
    deletes some virtual variables only needed for pre-processing, and returns a Pandas
    DataFrame.

    Args:
        file_url (str): The URL of the .pkl file to load.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the loaded data.
    """
    # Load dataset
    filename = make_local(file_url, cache)
    df = pd.read_pickle(filename)
    # Replace infs with nan's
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replace nan's with 0
    # TODO: this should be moved to writing the file out so we don't have to do this
    # here.
    df = df.fillna(0)

    # Delete some 'virtual' variables only needed for pre-processing.
    # Make sure they exist before deleting them.
    # TODO: Remove these guys before writing them out too
    for t_name in ["track_sign", "clus_sign"]:
        if t_name in df.columns:
            del df[t_name]

    # Delete track_vertex vars in tracks
    # TODO: this should be removed ahead of time.
    vertex_delete = [col for col in df.columns if col.startswith("nn_track_vertex_x")]
    vertex_delete += [col for col in df.columns if col.startswith("nn_track_vertex_y")]
    vertex_delete += [col for col in df.columns if col.startswith("nn_track_vertex_z")]
    for item in vertex_delete:
        del df[item]

    # Print sizes of inputs for signal, qcd, and bib
    logging.debug(df.head())
    logging.info("  Length of Signal is: " + str(df[df.label == 1].shape[0]))
    logging.info("  Length of QCD is: " + str(df[df.label == 0].shape[0]))
    logging.info("  Length of BIB is: " + str(df[df.label == 2].shape[0]))

    return df
