from datetime import datetime
import logging
from pathlib import Path


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
