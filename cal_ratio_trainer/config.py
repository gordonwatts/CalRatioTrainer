import io
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

# WARNING:
# This should include no other modules that eventually import TensorFlow
# This gets imported in just about every instance, so worth keeping clean.
# IF you put a default in here, it should be `None`!!! Otherwise the logic
# will think this is properly set value. Put defaults in the `default_xxx.yaml`
# files.


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

    # Which sets of signal should we include in the training?
    # TODO: Make this part of the signal file prep - we need something more
    # flexible about what data we include that a bunch of bools like this.
    include_low_mass: Optional[bool] = True
    include_high_mass: Optional[bool] = True

    # The path to the main training data file (signal, qcd, and bib)
    # Use "file:///xxx" to specify a local file with absolute path on linux.
    main_training_file: Optional[str] = None
    cr_training_file: Optional[str] = None

    def __str__(self) -> str:
        string_out = io.StringIO()
        for k, v in self.dict().items():
            print(f"{k}: {v}", file=string_out)
        return string_out.getvalue()


def _load_config_from_file(config_type, p: Path):
    """Load a config from a file, without taking into account defaults."""
    with open(p, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_type(**config_dict)


def load_config(config_type: type, p: Optional[Path] = None):
    """Load a Config from a file, taking into account defaults."""
    r = _load_config_from_file(
        config_type, Path(__file__).parent /
        f"{config_default_file[config_type]}.yaml"
    )

    if p is not None:
        specified = _load_config_from_file(config_type, p)
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


class training_spec(BaseModel):
    "A specification for a training run"

    # The name of the training run
    name: str

    # The run
    run: int


class epoch_spec(BaseModel):
    "Specification for an epoch of a training run"

    # The name of the training run
    name: str

    # The run number
    run: int

    # The epoch of the run.
    epoch: int


class AnalyzeConfig(BaseModel):
    "Configuration for analysis"

    runs_to_analyze: Optional[List[training_spec]] = None

    output_report: Optional[Path] = Field(
        description="The path to the output report file. All plots will be "
        "written to the directory this file is in (so put it in a clean sub-dir!)."
    )


class ConvertTrainingConfig(BaseModel):
    "Configuration for training to JSON-CPP Conversion"

    run_to_convert: Optional[epoch_spec] = None

    output_json: Optional[Path] = Field(
        description="The path to the output model and weight json file."
    )


class DiVertFileType(str, Enum):
    "The type of file in a divertanalysis output file"
    sig = "sig"
    qcd = "qcd"
    bib = "bib"


class DiVertAnalysisInputFile(BaseModel):
    "Information about a divertanalysis input file"

    # The location of the input file
    input_file: Path = Field(description="The location of the input file.")

    # The name to use on the legend of the plot for data from this file.
    data_type: DiVertFileType = Field(
        description="The type of data in this file. One of 'sig', 'qcd', 'bib'."
    )

    # Output directory name, which is under the `output_path` of the master config.
    output_dir: Optional[str] = Field(
        description="The name of the output directory for this file. Stored under the "
        "`output_path` of the master config."
    )

    llp_mH: Optional[float] = Field(
        description="The mass of the heavy higgs like particle. Zero if not a signal"
        " file.",
        default=None,
    )

    llp_mS: Optional[float] = Field(
        description="The mass of the dark sector light LLP like particle. Zero if not a"
        " signal file.",
        default=None,
    )


class ConvertDiVertAnalysisConfig(BaseModel):
    "Configuration for converting a divertanalysis output file"

    input_files: Optional[List[DiVertAnalysisInputFile]] = Field(
        description="The list of input files to convert."
    )

    output_path: Optional[Path] = Field(
        description="The path to the directory where the converted pd.DataFrame pickle"
        "files will be written.",
    )

    signal_branches: Optional[List[str]] = Field(
        description="The list of branches to store in the output training file "
        "for signal."
    )
    bib_branches: Optional[List[str]] = Field(
        description="The list of branches to store in the output training file for bib."
    )
    qcd_branches: Optional[List[str]] = Field(
        description="The list of branches to store in the output training file for qcd."
    )
    rename_branches: Optional[Dict[str, str]] = Field(
        description="A dictionary of branches to rename. The key is the old name, the "
        "value is the new name."
    )

    min_jet_pt: Optional[float] = Field(
        description="The minimum jet pT to use for the training [GeV]."
    )

    max_jet_pt: Optional[float] = Field(
        description="The maximum jet pT to use for the training [GeV]."
    )

    llp_mH: Optional[float] = Field(
        description="The default mass of the heavy higgs like particle. Ignored if not a"
        " signal file. Applied only to files on the command line.",
        default=0.0,
    )

    llp_mS: Optional[float] = Field(
        description="The default mass of the dark sector light LLP like particle. Ignored"
        " if not a signal file. Applied only to files on the command line.",
        default=0.0,
    )


class ConvertxAODConfig(BaseModel):
    "Configuration for converting an xAOD file"

    input_files: Optional[List[Path]] = Field(
        description="The list of input files to convert."
    )

    output_path: Optional[Path] = Field(
        description="The path to the directory where the converted pd.DataFrame pickle"
        "files will be written.",
    )

    clean: Optional[bool] = Field(
        description="If true, will remove all compiled source code first.",
        default=False,
    )

    working_directory: str = Field(
        description="The working directory to use for the compilation. If not specified,"
        " will use ~/cr_trainer_DiVertAnalysis.",
        default="~/cr_trainer_DiVertAnalysis",
    )

    nevents: Optional[int] = Field(
        description="The number of events to convert. If None, all events will be "
        "converted. The "
        "total number is applied to all files that match any wildcards. Events kept "
        "are just the first events in the file. Defaults to None.",
        default=None,
    )

    skip_build: Optional[bool] = Field(
        description="If true, will skip the checkout and build step. This is useful if you have "
        "already built the code and just want to run the conversion.",
        default=False,
    )

    add_training: Optional[List[epoch_spec]] = Field(
        description="Add a training file to the build. This will be used to add an extra NN "
        "evaluation. "
        "  The format is "
        "`training_name/training_number/epoch`. Use the `analyze` sub-command to determine which"
        " epoch.",
        default=[],
    )


class training_input_file(BaseModel):
    "Specs for a single input file for training"

    # The location of the input file, can be a wild card.
    input_file: Path = Field(
        description="The location of the input file, wildcard is ok."
    )

    # The number of events to include in building the training file.
    # If None, all events will be used. If you ask for more than exist,
    # a warning will be applied. The total number is applied to all files
    # that match any wildcards. Events kept are randomly sampled from the
    # full set of events.
    num_events: Optional[int] = Field(
        description="The number of events to include in building the training file.",
        default=None,
    )

    # An expression that can be applied to the dataframe at the event level to
    # filter filter events. Use the bar column name. This will need to be a valid
    # pandas `query` call. For example, odd events only would be "eventNumber % 2 == 1"
    # A `None` here means no events will be filtered.
    event_filter: Optional[str] = Field(
        description="An expression that can be applied to the dataframe at the event "
        "level to filter filter events. Use the bar column name. This will need to be "
        "a valid pandas `query` expression. For example, odd events only would be "
        "'eventNumber % 2 == 1'. A `None` here means no events will be filtered.",
        default=None,
    )


class BuildMainTrainingConfig(BaseModel):
    "Configuration for building a main training file"

    input_files: Optional[List[training_input_file]] = Field(
        description="The list of input files to convert."
    )

    output_file: Optional[Path] = Field(
        description="The path to the output training file. Must not already exist."
    )

    min_jet_pt: Optional[float] = Field(
        description="The minimum pT to use for the training [GeV]. Applied to jets, "
        "tracks, etc."
    )
    max_jet_pt: Optional[float] = Field(
        description="The maximum jet pT to use for the training [GeV]."
    )
    remove_branches: Optional[List[str]] = Field(
        description="The list of branches to exclude from the training file"
        " (if present).",
        default=None,
    )


config_default_file = {
    TrainingConfig: "default_training_config",
    ReportingConfig: "default_reporting_config",
    AnalyzeConfig: "default_analyze_config",
    ConvertTrainingConfig: "default_convert_config",
    ConvertDiVertAnalysisConfig: "default_divert_config",
    BuildMainTrainingConfig: "default_build_main_training_config",
    ConvertxAODConfig: "default_xaod_config",
}
