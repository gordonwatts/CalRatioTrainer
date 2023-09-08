import argparse
from typing import Generator, Tuple, Union


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
            if t2 == str or t2 == float or t2 == int or t2 == bool:
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
