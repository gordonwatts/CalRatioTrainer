import logging
from datetime import datetime
from pathlib import Path

import fsspec


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
                    data = f_in.read(50 * 1024**2)
                    if not data:
                        break
                    f_out.write(data)
        tmp_file_path.rename(local_path)
        logging.warning(f"Done copying file {file_path}")
        return local_path
