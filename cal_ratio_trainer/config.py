import logging
from pathlib import Path
from typing import List, Optional
import fsspec
from pydantic import AnyUrl, BaseModel, Field
import yaml

# WARNING:
# This should include no other modules that eventually import TensorFlow
# This gets imported in just about every instance, so worth keeping clean.


class TrainingConfig(BaseModel):
    """Configuration to run a complete training, from source data files on!"""

    # Name of the model - for user reference only
    model_name: Optional[str] = Field(
        description="Name of the model - for user reference only"
    )

    # NAdam learning rate.
    lr_values: Optional[float] = Field(
        description="NAdam learning rate for both main network and adversary"
    )

    filters_cnn_constit: Optional[List[int]]
    frac_list: Optional[float]
    nodes_constit_lstm: Optional[int]
    reg_values: Optional[float]
    dropout_array: Optional[float]
    adversary_weight: Optional[float]
    layers_list: Optional[int]
    filters_cnn_track: Optional[List[int]]
    nodes_track_lstm: Optional[int]
    filters_cnn_MSeg: Optional[List[int]]
    nodes_MSeg_lstm: Optional[int]

    # Number of epochs for training
    epochs: Optional[int]
    # Number of mini-batches
    num_splits: Optional[int]
    # TODO: Why do we have both batch_size and num_splits?
    batch_size: Optional[int]

    hidden_layer_fraction: Optional[float]

    mH_parametrization: Optional[bool] = False
    mS_parametrization: Optional[bool] = False

    # Which sets of signal should we include in teh training?
    # TODO: Make this part of the signal file prep - we need something more
    # flexible about what data we include that a bunch of bools like this.
    include_low_mass: Optional[bool] = False
    include_high_mass: Optional[bool] = False

    # The path to the main training data file (signal, qcd, and bib)
    # Use "file://xxx" to specify a local file.
    main_training_file: Optional[AnyUrl] = None
    cr_training_file: Optional[AnyUrl] = None

    data_cache: Optional[Path] = None

    @property
    def main_file(self) -> Path:
        assert (
            self.data_cache is not None
        ), "Must have a valid data_cache in configuration parameters"
        return make_local(str(self.main_training_file), self.data_cache)

    @property
    def cr_file(self) -> Path:
        assert (
            self.data_cache is not None
        ), "Must have a valid data_cache in configuration parameters"
        return make_local(str(self.cr_training_file), self.data_cache)


def _load_config_from_file(p: Path) -> TrainingConfig:
    """Load a TrainingConfig from a file, without taking into account defaults."""
    with open(p, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def load_config(p: Optional[Path] = None) -> TrainingConfig:
    """Load a TrainingConfig from a file, taking into account defaults."""
    r = _load_config_from_file(Path(__file__).parent / "default_training_config.yaml")

    if p is not None:
        specified = _load_config_from_file(p)
        d = specified.dict()
        for k, v in d.items():
            if v is not None:
                setattr(r, k, v)

    return r


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
