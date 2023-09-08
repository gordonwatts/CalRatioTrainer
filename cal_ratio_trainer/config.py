import io
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
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
    # Use "file:///xxx" to specify a local file with absolute path on linux.
    main_training_file: Optional[str] = None
    cr_training_file: Optional[str] = None

    def __str__(self) -> str:
        string_out = io.StringIO()
        for k, v in self.dict().items():
            print(f"{k}: {v}", file=string_out)
        return string_out.getvalue()


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


class plot_file(BaseModel):
    "Information about a plot file"

    # The url that we can use to read this file
    input_file: str

    # The name to use on the legend of the plot for data from this file.
    legend_name: str


class ReportingConfig(BaseModel):
    "Configuration to run a plotting/reporting job for a training input file"

    # List of the plots to make with all files. If the sources
    # can have different names, then this is a list. Otherwise it is
    # just a single item.
    common_plots: Optional[List[Union[str, List[str]]]] = None

    # The data labels and what they mean in the
    # adversary training data file.
    data_labels_adversary: Optional[Dict[int, str]] = None

    # The data labels and what they mean in the
    # main training data file.
    data_labels_main: Optional[Dict[int, str]] = None

    plot_every_column: Optional[bool] = Field(
        description="If this is true, it "
        "will generate a plot for every row in the column list table. This is "
        "very slow!"
    )

    output_report: Optional[Path] = Field(
        description="The path to the output report file. All plots will be "
        "written to the directory this file is in (so put it in a clean sub-dir!)."
    )

    input_files: Optional[List[plot_file]] = None


def _load_reporting_config_from_file(p: Path) -> ReportingConfig:
    """Load a TrainingConfig from a file, without taking into account defaults."""
    with open(p, "r") as f:
        config_dict = yaml.safe_load(f)
    return ReportingConfig(**config_dict)


def load_report_config(p: Optional[Path] = None) -> ReportingConfig:
    """Load a ReportingConfig from a file, taking into account defaults."""
    r = _load_reporting_config_from_file(
        Path(__file__).parent / "default_reporting_config.yaml"
    )

    if p is not None:
        specified = _load_reporting_config_from_file(p)
        d = specified.dict()
        for k, v in d.items():
            if v is not None:
                setattr(r, k, v)

    return r
