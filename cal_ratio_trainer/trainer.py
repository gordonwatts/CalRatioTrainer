import argparse
from pathlib import Path

from cal_ratio_trainer.config import load_config
from cal_ratio_trainer.training.runner_utils import training_runner_util


def do_train(args):
    # Get the config loaded.
    c = load_config(args.config)

    training_runner_util(c)


def main():
    # Create the master command line parser
    parser = argparse.ArgumentParser(
        description="Training and Utilities for the CalRatio RNN training"
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    parser.set_defaults(func=lambda _: parser.print_help())

    # Sub-command train to actually run the training, using a config file to load.
    parser_train = subparsers.add_parser("train", help="Train the CalRatio RNN model")
    parser_train.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to the config file to use for training",
    )
    parser_train.set_defaults(func=do_train)

    # Parse the command line arguments
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()