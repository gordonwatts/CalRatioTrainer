from pathlib import Path

from cal_ratio_trainer.convert.convert_xaod import (
    ConvertxAODConfig,
    convert_xaod,
    delete_directory,
    dir_exists,
    execute_commands,
)


def default_config_xaod() -> ConvertxAODConfig:
    return ConvertxAODConfig(
        input_files=[Path("tests/xaod/small_file.root")],
        output_path=Path("tests/xaod/small_file.root"),
    )


def test_small_convert(tmp_path: Path):
    """Test the call to convert_xaod.

    NOTE: This is quite slow since it probably will download and compile the executable
    DiVertAnalysis!
    """
    convert_xaod(default_config_xaod())


def test_directory_test():
    assert dir_exists("~")
    assert not dir_exists("~/bogus/dude")


def test_delete_directory():
    dir = "~/bogus/dude"
    assert not dir_exists(dir)
    delete_directory(dir)

    # Create directory and then make sure delete removes it.
    execute_commands([f"mkdir -p {dir}"])
    assert dir_exists(dir)
    delete_directory(dir)
    assert not dir_exists(dir)
