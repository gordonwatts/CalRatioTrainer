import argparse
import logging
from pathlib import Path
from typing import List

from cal_ratio_trainer.config import (
    AnalyzeConfig,
    BuildMainTrainingConfig,
    ConvertDiVertAnalysisConfig,
    ConvertTrainingConfig,
    ConvertxAODConfig,
    DiVertAnalysisInputFile,
    DiVertFileType,
    ReportingConfig,
    ScorePickleConfig,
    TrainingConfig,
    epoch_spec,
    load_config,
    plot_file,
    training_input_file,
    training_spec,
)
from cal_ratio_trainer.utils import add_config_args, apply_config_args

cache = Path("./calratio_training")


# NOTE: Many imports are inside the methods. This is because these imports
# pull in TensorFlow and that is a fairly costly operation, and slows down the
# overall usage of the tool.


def do_train(args):
    # Get the config loaded.
    c_config = load_config(TrainingConfig, args.config)

    # Next, look at the arguments and see if anything should be changed.
    c = apply_config_args(TrainingConfig, c_config, args)

    # Figure out what to do next
    if args.print_settings:
        # Pretty Print the training configuration
        print(c)
        return
    else:
        # Now, run the training.
        from cal_ratio_trainer.training.runner_utils import training_runner_util

        training_runner_util(c, cache=cache, continue_from=args.continue_from)


def do_plot(args):
    from cal_ratio_trainer.reporting.training_file import make_report_plots

    r_config = load_config(ReportingConfig, args.config)
    r = apply_config_args(ReportingConfig, r_config, args)
    # Special handling of the input files argument:
    if len(args.input_files) > 0:
        input_files: List[plot_file] = []
        for i, f_name in enumerate(args.input_files):
            assert isinstance(f_name, str), f"Input file {f_name} is not a string."
            if "=" in f_name:
                name, f = f_name.split("=", 2)
            else:
                name, f = f"file_{i}", f_name
            input_files.append(plot_file(input_file=f, legend_name=name))
        r.input_files = input_files

    make_report_plots(
        cache,
        r,
    )


def do_analyze(args):
    from cal_ratio_trainer.reporting.analyze import analyze_training_runs

    a_config = load_config(AnalyzeConfig, args.config)
    a = apply_config_args(AnalyzeConfig, a_config, args)

    # The training runs to analyze are in args.training, if specified.
    if len(args.training) > 0:
        a.runs_to_analyze = []
        for t in args.training:
            name, run = t.split("/")
            a.runs_to_analyze.append(training_spec(name=name, run=int(run)))

    analyze_training_runs(cache, a)


def parse_epoch_spec(spec: str) -> epoch_spec:
    """Parse the training + epoch spec. The accepted format
    is "name/run/epoch".

    Args:
        spec (str): The string specification to be parsed

    Returns:
        epoch_spec: The parsed specification
    """
    name, run, epoch = spec.split("/")
    return epoch_spec(name=name, run=int(run), epoch=int(epoch))


def do_cpp_convert(args):
    """
    Converts a training file to a frugally-deep JSON file.

    Args:
        args: An object containing the command-line arguments.

    Returns:
        None
    """
    a_config = load_config(ConvertTrainingConfig, args.config)
    a = apply_config_args(ConvertTrainingConfig, a_config, args)

    # The training epochs are special. Analyze is specified.
    if len(args.training) > 0:
        a.run_to_convert = parse_epoch_spec(args.training)

    from cal_ratio_trainer.convert.convert_json import convert_file

    convert_file(a)


def do_divert_convert(args):
    a_config = load_config(ConvertDiVertAnalysisConfig, args.config)
    a = apply_config_args(ConvertDiVertAnalysisConfig, a_config, args)

    if len(args.input_files) > 0:
        a.input_files = [
            DiVertAnalysisInputFile(
                input_file=f,
                data_type=args.data_type,
                output_dir=None,
                llp_mH=a.llp_mH,
                llp_mS=a.llp_mS,
            )
            for f in args.input_files
        ]

    # Check the output path is a directory or does not exist.
    if a.output_path.exists() and not a.output_path.is_dir():
        raise RuntimeError(
            f"Output path {a.output_path} exists and is not a directory."
        )
    a.output_path.mkdir(parents=True, exist_ok=True)

    # And run the conversion.
    from cal_ratio_trainer.convert.convert_divert import convert_divert

    convert_divert(a)


def do_xaod_convert(args):
    a_config = load_config(ConvertxAODConfig, args.config)
    a = apply_config_args(ConvertxAODConfig, a_config, args)

    if len(args.input_files) > 0:
        a.input_files = args.input_files

    # Check the output path does not yet exist.
    if a.output_path.exists():
        raise RuntimeError(
            f"Output path {a.output_path} exists. Please remove before " f"running."
        )

    for t in args.add_trainings:
        a.add_training.append(parse_epoch_spec(t))

    # And run the conversion.
    from cal_ratio_trainer.convert.convert_xaod import convert_xaod

    convert_xaod(a)


def do_model_dump(args):
    """
    Dumps the model from a training file to a text file.

    Args:
        args: An object containing the command-line arguments.

    Returns:
        None
    """
    from cal_ratio_trainer.reporting.dump import dump_model

    dump_model(args.training)


def do_score_pkl(args):
    a_config = load_config(ScorePickleConfig, args.config)
    a = apply_config_args(ScorePickleConfig, a_config, args)

    if args.input_file is not None:
        if a.input_files is None:
            a.input_files = []
        a.input_files.append(Path(args.input_file))

    a.training = parse_epoch_spec(args.training)

    from cal_ratio_trainer.score.score_pickle import score_pkl_files

    score_pkl_files(a)


def do_build_main_training(args):
    a_config = load_config(BuildMainTrainingConfig, args.config)
    a = apply_config_args(BuildMainTrainingConfig, a_config, args)

    # Check the output path is a directory or does not exist.
    if a.output_file.exists() and not a.output_file.is_dir():
        raise RuntimeError(
            f"Output path {a.output_file} exists. Please remove before running."
        )

    # If there are any input files, replace the list from the config.
    if len(args.input_files) > 0:
        a.input_files = [
            training_input_file(input_file=f, num_events=None) for f in args.input_files
        ]

    # Make sure we have at least one input file, and the input files all exist.
    if a.input_files is None or len(a.input_files) == 0:
        raise RuntimeError("No input files specified.")

    from cal_ratio_trainer.build.build_main import build_main_training

    build_main_training(a)


def do_resample(args):
    from cal_ratio_trainer.convert.resample import resample_training_file

    resample_training_file(args.input_file, args.output_file, args.fraction, cache)


def main():
    # Create the master command line parser
    parser = argparse.ArgumentParser(
        description="Training and Utilities for the CalRatio RNN training"
    )
    subparsers = parser.add_subparsers(help="cr_trainer commands")
    parser.set_defaults(func=lambda _: parser.print_help())

    # Add a top level verbose flag that will count the number of
    # v's on the command line to turn on levels of verbosity.
    parser.add_argument("-v", "--verbose", action="count", default=0)

    # Sub-command train to actually run the training, using a config file to load.
    parser_train = subparsers.add_parser("train", help="Train the CalRatio RNN model")
    parser_train.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for training",
    )
    parser_train.add_argument(
        "--print-settings",
        action="store_true",
        default=False,
        help="Print the training configuration and exit",
    )
    parser_train.add_argument(
        "--continue",
        dest="continue_from",
        type=int,
        default=None,
        help="Continue training from the end of a previous training run. The argument "
        "is the training number to start from. Use -1 for the most recently completed.",
    )

    add_config_args(TrainingConfig, parser_train)
    parser_train.set_defaults(func=do_train)

    # Add the plot command which will plot the training input variables. The command
    # will accept multiple input files, and needs an output directory where everything
    # can be dumped.
    parser_plot = subparsers.add_parser(
        "plot",
        help="Plot the training input variables",
        epilog="Note: There are several forms the input_files argument can take:\n"
        "1. If you specify nothing, the 2019 adversary and (small) main training data "
        "files will be used.\n"
        "2. If you specify 'file://path/to/file', then that file will be used.\n"
        "3. If you specify 'http://path/to/dir', then that fill will be copied locally "
        "and used.\n"
        "4. If you specify multiple files, a single report with comparison plots is "
        "made.\n"
        "5. If you use the format <name>=<file> then the name will be used in the "
        "legend. Otherwise 'file_0' will be used.",
    )
    parser_plot.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for training",
    )
    parser_plot.add_argument(
        "input_files",
        type=str,
        nargs="*",
        default=[],
        help="Path to the input files to plot. Can be multiple files. If not files are "
        "used 2019 files are used by default.",
    )
    add_config_args(ReportingConfig, parser_plot)
    parser_plot.set_defaults(func=do_plot)

    # The `analyze` command will analyze a series of runs ("name/number") and find
    # the most likely "best" epochs that could be used for eventual inference in the
    # analysis.
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a series of training runs to find the best epochs",
    )
    parser_analyze.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for analysis",
    )
    parser_analyze.add_argument(
        "training",
        nargs="*",
        default=[],
        help="The training runs to analyze. Can be repeated multiple times. "
        "Form is <name>/<number>.",
    )
    add_config_args(AnalyzeConfig, parser_analyze)
    parser_analyze.set_defaults(func=do_analyze)

    # The `convert` command facilitates converting files from one form to another.
    # It currently does two things, depending on a "noun" that it gets:
    #  1. will take a training ("name/number/epoch") and convert it
    #     to an output JSON file that can be used in our C++ DiVertAnalysis code.
    #  2. Take a DiVertAnalysis file and convert it to a raw training file (which
    #     must still be built into a complete training data file).
    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert file formats",
    )
    # Create a sub-parser that will process two commands, "training" and
    # "divertanalysis".
    subparsers_convert = parser_convert.add_subparsers(
        help="Convert one file type to another"
    )
    parser_convert.set_defaults(func=lambda _: parser_convert.print_help())

    # First, the training conversion command.
    parser_convert = subparsers_convert.add_parser(
        "training",
        help="Convert a training/run/epoch to a JSON file to be used by fdeep "
        "in DiVertAnalysisR21",
    )
    parser_convert.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for analysis",
    )
    parser_convert.add_argument(
        "training",
        help="The training runs to analyze. Form is <training-name>/<number>/<epoch>.",
    )
    add_config_args(ConvertTrainingConfig, parser_convert)
    parser_convert.set_defaults(func=do_cpp_convert)

    # Next, the divertanalysis sub-command
    parser_divertanalysis_convert = subparsers_convert.add_parser(
        "divertanalysis",
        help="Convert a DiVertAnalysis file to a training file",
        description="Convert a DiVertAnalysis file to a training file. This"
        " will process all the files specified on the command line or in the yaml"
        " config file. Please watch warning and error messages carefully to make sure"
        " all the files expected are processed. You can run multiple instances at once"
        " - they should not step on each other.",
    )
    parser_divertanalysis_convert.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for analysis",
    )
    parser_divertanalysis_convert.add_argument(
        "input_files",
        nargs="*",
        default=[],
        help="The input files to convert. Can be repeated multiple times. Written to "
        "the output directory",
        type=Path,
    )
    parser_divertanalysis_convert.add_argument(
        "--data_type",
        type=DiVertFileType,
        default="sig",
        help="The Datatype of the file to be converted (sig, qcd, bib). 'sig' "
        "is default.",
    )
    add_config_args(ConvertDiVertAnalysisConfig, parser_divertanalysis_convert)
    parser_divertanalysis_convert.set_defaults(func=do_divert_convert)

    # And convert a xAOD file to a DiVertAnalysis ntuple
    parser_xaod_convert = subparsers_convert.add_parser(
        "xaod",
        help="Convert a xAOD file to a DiVertAnalysis ntuple",
        description="Convert a xAOD file to a DiVertAnalysis ntuple. This"
        " will process all the files specified on the command line or in the yaml"
        " config file. Please watch warning and error messages carefully to make sure"
        " all the files expected are processed.",
    )
    parser_xaod_convert.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for analysis",
    )
    parser_xaod_convert.add_argument(
        "input_files",
        nargs="*",
        default=[],
        help="The input files to convert. Can be repeated multiple times. Written to "
        "the output directory",
        type=Path,
    )
    parser_xaod_convert.add_argument(
        "--add_trainings",
        nargs="*",
        default=[],
        help="Additional trainings to use in the conversion. fForm is "
        "name/number/epoch. Can be repeated multiple times.",
        type=str,
    )
    add_config_args(ConvertxAODConfig, parser_xaod_convert)
    parser_xaod_convert.set_defaults(func=do_xaod_convert)

    # The `model-dump` command will take a training ("name/number/epoch") and dump the
    # model to stdout.
    parser_model_dump = subparsers.add_parser(
        "model-dump",
        help="Dump the model from a training run to stdout",
    )
    parser_model_dump.add_argument(
        "training",
        help="The training runs to analyze. Form is <training-name>/<number>/<epoch> "
        "or a path to a JSON file.",
    )
    parser_model_dump.set_defaults(func=do_model_dump)

    # Score a pkl training file.
    parser_score_training_file = subparsers.add_parser(
        "score",
        help="Score a particular input file",
        description="Sub-command that will help score a input file.",
    )

    # The Score command to "score" particular types of files.
    score_subparsers = parser_score_training_file.add_subparsers(
        help="Use a NN Model to score a input file"
    )

    # Score a pkl training file.
    parser_pickle_score = score_subparsers.add_parser(
        "pkl",
        help="Score a pkl training file",
        description="Score a pkl training file",
    )
    parser_pickle_score.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for this scoring",
    )
    parser_pickle_score.add_argument(
        "training",
        help="Teh training run to use for scoring. Form is <training-name>/<number>/<epoch>.",
    )
    parser_pickle_score.add_argument(
        "input_file",
        help="The input file to score (a URI we can use to get at the file)",
        type=str,
    )
    add_config_args(ScorePickleConfig, parser_pickle_score)
    parser_pickle_score.set_defaults(func=do_score_pkl)

    # The build command, to build the main training file.
    parser_build = subparsers.add_parser(
        "build",
        help="Build the main training file",
    )
    parser_build.add_argument(
        "input_files",
        nargs="*",
        default=[],
        help="The input files to build. Can be repeated multiple times. "
        "Completed training is written to the output file.",
        type=Path,
    )
    parser_build.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for analysis",
    )
    add_config_args(BuildMainTrainingConfig, parser_build)
    parser_build.set_defaults(func=do_build_main_training)

    # The `resample` command which takes an input and output training file
    # and a sampling fraction (0.0-1.0) and resamples the input file to the
    # output file.
    parser_resample = subparsers.add_parser(
        "resample",
        help="Resample a training (pkl) file by some fraction",
    )
    parser_resample.add_argument(
        "input_file",
        help="The input file to resample (a URI we can use to get at the file)",
        type=str,
    )
    parser_resample.add_argument(
        "output_file",
        help="The output file to write",
        type=Path,
    )
    parser_resample.add_argument(
        "fraction",
        help="The fraction of events to keep (value between 0.0 and 1.0)",
        type=float,
    )
    parser_resample.set_defaults(func=do_resample)

    # Parse the command line arguments
    args = parser.parse_args()

    # Turn on verbosity by setting the log level to be "info" or "debug"
    # in python's `logging` module. Shut down matplotlib, it makes
    # way too much noise.
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)
    if args.verbose > 0:
        root_logger = logging.getLogger()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.removeHandler(root_logger.handlers[0])
        root_logger.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Call the function to execute the command.
    args.func(args)


if __name__ == "__main__":
    main()
