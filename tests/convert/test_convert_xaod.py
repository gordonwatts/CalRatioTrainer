from pathlib import Path
import pytest

from cal_ratio_trainer.convert.convert_xaod import (
    ConvertxAODConfig,
    convert_xaod,
    delete_directory,
    dir_exists,
    execute_commands,
)


def default_config_xaod(tmp_path) -> ConvertxAODConfig:
    return ConvertxAODConfig(
        input_files=[Path("tests/data/small_xaod_EXOT15_file.root")],
        output_path=Path(f"{tmp_path}/output.root"),
    )


# # TODO: Generate a small xAOD file for testing.
# @pytest.mark.skip(reason="Missing a small xAOD file for running the testing on.")
# def test_small_convert(tmp_path: Path):
#     """Test the call to convert_xaod.

#     NOTE: This is quite slow since it probably will download and compile the executable
#     DiVertAnalysis!
#     """
#     c = default_config_xaod(tmp_path)
#     convert_xaod(c)

#     assert c.output_path is not None
#     assert c.output_path.exists()


# @pytest.mark.skip(reason="Only run on WSL2")
# def test_directory_test():
#     assert dir_exists("~")
#     assert not dir_exists("~/bogus/dude")


# @pytest.mark.skip(reason="Only run on WSL2")
# def test_delete_directory():
#     dir = "~/bogus/dude"
#     assert not dir_exists(dir)
#     delete_directory(dir)

#     # Create directory and then make sure delete removes it.
#     execute_commands([f"mkdir -p {dir}"])
#     assert dir_exists(dir)
#     delete_directory(dir)
#     assert not dir_exists(dir)


# @pytest.mark.skip(reason="Only run on WSL2")
# def test_execute_commands():
#     r = execute_commands(["echo 'hello world'"])
#     assert "hello world" in r


# @pytest.mark.skip(reason="Only run on WSL2")
# def test_execute_bad_command():
#     with pytest.raises(Exception):
#         execute_commands(["cp my_left_foot_is_not_here.root junk.root"])
