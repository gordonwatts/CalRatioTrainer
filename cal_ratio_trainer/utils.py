import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union


def as_bare_type(t: type) -> Union[type, None]:
    """Return the bare type for certain types:

    str, float, int -> return those.
    Optional[T] -> return T, as long as T is one of the above.
    Anything else -> return None (too complex, we can't deal with it).

    Args:
        t (type): The type to convert with the above rules.

    Returns:
        Union[type, None]: The bare type, or None if can't be found.
    """
    if t == str or t == float or t == int:
        return t
    elif hasattr(t, "__origin__") and t.__origin__ == Union:  # type: ignore
        # This is an Optional[T], so we need to get T.
        # We can only deal with T if it is one of the above types.
        if len(t.__args__) != 2 and type(None) not in t.__args__:  # type: ignore
            return None
        for t2 in t.__args__:  # type: ignore
            if t2 == str or t2 == float or t2 == int or t2 == bool or t2 == Path:
                return t2
        return None
    else:
        return None


def _good_config_args(config_class: type) -> Generator[Tuple[str, type], None, None]:
    """Return the list of training params that are good for being set
    on the command line (e.g. their types are something we can easily deal
    with).

    Yields:
        Generator[str, None, None]: Generates all the names of the good arguments
    """
    dummy = config_class(**{}).dict()

    for prop in dummy.keys():
        p_type = as_bare_type(config_class.__annotations__[prop])
        if p_type is not None:
            yield prop, p_type


def add_config_args(config_class, args: argparse.ArgumentParser) -> None:
    """Add all the configuration arguments to the argument parser.

    This includes adding any help strings that python knows about
    as the argument help.

    Parameters
    ----------
    args : argparse.ArgumentParser
        The argument parser to add the arguments to.
    """
    # Look at the properties of the TrainingConfig object.
    for prop, p_type in _good_config_args(config_class):
        help_str = (
            config_class.__fields__[prop].field_info.description
            if prop in config_class.__fields__
            else None
        )
        if p_type != bool:
            args.add_argument(
                f"--{prop}",
                type=p_type,
                help=help_str,
            )
        else:
            args.add_argument(
                f"--{prop}",
                action="store_true",
                default=None,
                help=help_str,
            )


def apply_config_args(config_class: type, config, args):
    """Using args, anything that isn't set to None that matches as config
    option in `config_class` (e.g. TrainingConfig), update the config and return a
    new one.

    Note: The `config` will not be updated - a new one will be returned.

    Args:
        config (TrainingConfig): The base training config to start from
        args (_type_): The list of arguments

    Returns:
        (TrainingConfig): New training config with updated arguments
    """
    # Start by making a copy of config for updating.
    r = config_class(**config.dict())

    # Next, loop through all possible arguments from the training config.
    for prop, p_type in _good_config_args(config_class):
        # If the argument is set, then update the config.
        if getattr(args, prop) is not None:
            setattr(r, prop, getattr(args, prop))

    return r


def find_training_result(
    model_to_do: str, continue_from: Optional[int] = None, base_dir: Path = Path(".")
) -> Path:
    # If it doesn't exist, do our best.
    model_dir = base_dir / "training_results" / model_to_do
    if not model_dir.exists():
        if continue_from is not None:
            raise ValueError(
                f"Directory {model_dir} does not exist. Cannot continue from the "
                f"{continue_from} run it."
            )
        else:
            return model_dir / "00000"

    # The model dir exists. Lets see hwo well we can do here.
    # If we have been given a continue_from directory, make sure it exists
    # and then use that.
    if continue_from is not None:
        if continue_from < 0:
            # get a sorted list of all sub-directories, and then take the one
            # one back from the end.
            try:
                run_number = sorted(
                    int(md.name) for md in model_dir.iterdir() if md.name.isdigit()
                )[continue_from]
                return model_dir / f"{run_number:05d}"
            except IndexError:
                raise ValueError(f"No runs in {model_dir} to continue from.")
        else:
            run_dir = model_dir / f"{continue_from:05d}"
            if not run_dir.exists():
                raise ValueError(
                    f"Directory {run_dir} does not exist. Cannot continue from it."
                )
            return run_dir
    # Ok - no continue_from listed - so we need to make up a new directory.
    try:
        biggest_run_number = max(
            int(item.name)
            for item in model_dir.iterdir()
            if (item.is_dir() and item.name.isdigit())
        )
        biggest_run_number += 1
    except ValueError:
        biggest_run_number = 0

    return model_dir / f"{biggest_run_number:05d}"


class HistoryTracker:
    "Keep together all the arrays we want to track per-epoch"

    def __init__(self, file: Optional[Path] = None):
        self._cache = defaultdict(list)

        if file is not None:
            self.load(file)

    def __str__(self):
        return f"HistoryTracker(epochs={len(self)}, items={list(self._cache.keys())})"

    def __getattr__(self, name):
        "Return the list for the requested tracking name"
        return self._cache[name]

    def __len__(self):
        if len(self._cache) == 0:
            return 0

        else:
            # Return the max length of all the lists
            return max(len(item) for item in self._cache.values())

    def save(self, filename: Path):
        "Save the history to a file"

        if filename.suffix != ".json":
            filename = filename.with_suffix(".json")

        with filename.open("w") as f:
            json.dump(self._cache, f)

    def load(self, filename: Path):
        "Load history from a file"

        if filename.suffix != ".json":
            filename = filename.with_suffix(".json")

        with filename.open("r") as f:
            self._cache = json.load(f)

    def find_smallest(self, item: str, num: int) -> List[int]:
        """Returns the epoch of the smallest `num` values in the
        item history.

        Args:
            item (str): The item to look for smallest values
            num (int): How many of them to return

        Returns:
            List[int]: List of the indices of the smallest values
        """
        assert num > 0
        s_item = sorted(enumerate(self._cache[item]), key=lambda x: x[1])
        return [x[0] for x in s_item[:num]]

    def make_sum(self, result_item: str, items: List[str]):
        """Sum the items together and place them in `result_item`

        Args:
            result_item (str): The result should be stored as this item
            items (List[str]): The items to sum
        """
        assert len(items) > 0
        assert result_item not in self._cache

        # Make sure all the items are the same length
        for item in items:
            assert len(self._cache[item]) == len(self._cache[items[0]])

        # Now sum them
        self._cache[result_item] = [
            sum([self._cache[item][i] for item in items])
            for i in range(len(self._cache[items[0]]))
        ]

    def values_for(self, epoch: int) -> Dict[str, float]:
        """Get all the values for a given epoch as a dictionary.

        Args:
            epoch (int): The epoch to get the values for

        Returns:
            Dict[str, float]: The values for that epoch
        """
        return {k: v[epoch] for k, v in self._cache.items()}
