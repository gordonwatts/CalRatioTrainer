import argparse
import logging
from pathlib import Path

from cal_ratio_trainer.config import load_config
from cal_ratio_trainer.utils import add_config_args, apply_config_args

cache = Path("./calratio_training")


def do_train(args):
    # Get the config loaded.
    c_config = load_config(args.config)

    # Next, look at the arguments and see if anything should be changed.
    c = apply_config_args(c_config, args)

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
    from cal_ratio_trainer.reporting.training_file import plot_file, make_report_plots

    make_report_plots(
        [
            plot_file(input_file=pf, legend_name=f"file_{i}")
            for i, pf in enumerate(args.input_files)
        ],
        cache,
    )


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

    # Add the plot command which will plot the training input variables. The command
    # will accept multiple input files, and needs an output directory where everything
    # can be dumped.
    parser_plot = subparsers.add_parser(
        "plot", help="Plot the training input variables"
    )
    parser_plot.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Path to the directory where the plots will be saved",
    )
    parser_plot.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Path to the input files to plot. Can be multiple files.",
    )
    parser_plot.set_defaults(func=do_plot)

    # Add all the training configuration options
    add_config_args(parser_train)
    parser_train.set_defaults(func=do_train)

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
