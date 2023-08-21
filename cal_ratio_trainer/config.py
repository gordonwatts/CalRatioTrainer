from pathlib import Path
from typing import List, Optional
from pydantic import AnyUrl, BaseModel
import yaml

from cal_ratio_trainer.training.utils import make_local


class TrainingConfig(BaseModel):
    """Configuration to run a complete training, from source data files on!"""

    # Name of the model - for user reference only
    model_name: Optional[str]
    learning_rate: Optional[float]
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
    batch_size: Optional[int]
    epochs: Optional[int]
    lr_values: Optional[float]
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
            setattr(r, k, v)

    return r
