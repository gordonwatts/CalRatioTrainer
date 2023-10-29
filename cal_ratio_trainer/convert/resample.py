from pathlib import Path
import pandas as pd

from cal_ratio_trainer.common.fileio import make_local


def resample_training_file(
    input_uri: str, output_file_path: Path, fraction: float, cache: Path
):
    """Resample a training file by a given fraction.

    Args:
        input_file_path (Path): The source file to read from.
        output_file_path (Path): The destination file to write to.
        fraction (float): The fraction to resample by. (0.0-1.0)
    """
    input_file_path = make_local(input_uri, cache)

    # Read input file
    input_df = pd.read_pickle(input_file_path)

    # Split by label
    labels = input_df["label"].unique()
    label_dfs = [input_df[input_df["label"] == label] for label in labels]

    # Now resample each one by the fraction
    resampled_dfs = [label_df.sample(frac=fraction) for label_df in label_dfs]

    # Finally, combine them, and then resample to randomize the order
    resampled_df = pd.concat(resampled_dfs).sample(frac=1)

    # Write out resulting pkl file. Make sure the filetype is `.pkl`
    resampled_df.to_pickle(output_file_path.with_suffix(".pkl"))
