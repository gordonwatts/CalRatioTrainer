import argparse
import logging
from pathlib import Path
from typing import List

from cal_ratio_trainer.config import (
    AnalyzeConfig,
    ConvertConfig,
    ReportingConfig,
    TrainingConfig,
    epoch_spec,
    load_config,
    plot_file,
    training_spec,
)
from cal_ratio_trainer.utils import add_config_args, apply_config_args

cache = Path("./calratio_training")


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
            assert isinstance(f_name, str)
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


def do_cpp_convert(args):
    """
    Converts a training file to a frugally-deep JSON file.

    Args:
        args: An object containing the command-line arguments.

    Returns:
        None
    """
    a_config = load_config(ConvertConfig, args.config)
    a = apply_config_args(ConvertConfig, a_config, args)

    # The training epochs are special. Analyze is specified.
    if len(args.training) > 0:
        name, run, epoch = args.training.split("/")
        a.run_to_convert = epoch_spec(name=name, run=int(run), epoch=int(epoch))

    from cal_ratio_trainer.convert.convert_json import convert_file

    convert_file(a)


def main():
    # Create the master command line parser
    parser = argparse.ArgumentParser(
        description="Training and Utilities for the CalRatio RNN training"
    )
    subparsers = parser.add_subparsers(help="sub-command help")
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

    # The `convert` command will take a training ("name/number/epoch") and convert it
    # to an output JSON file that can be used in our C++ DiVertAnalysis code.
    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert a training run to a JSON file for C++",
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
    add_config_args(ConvertConfig, parser_convert)
    parser_convert.set_defaults(func=do_cpp_convert)

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
