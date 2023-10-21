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

    # Resample by fraction
    resampled_df = input_df.sample(frac=fraction)

    # Write out resulting pkl file. Make sure the filetype is `.pkl`
    resampled_df.to_pickle(output_file_path.with_suffix(".pkl"))
