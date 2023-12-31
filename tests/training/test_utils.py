from typing import Tuple
import pytest
from cal_ratio_trainer.training.utils import create_directories


def test_create_directories(tmp_path):
    final_path = create_directories("me_model", tmp_path)

    assert "me_model" in str(final_path)
    assert "00000" in str(final_path)


def test_create_directories_sequence(tmp_path):
    create_directories("me_model", tmp_path)
    create_directories("me_model", tmp_path)
    final_path = create_directories("me_model", tmp_path)

    assert "00002" in str(final_path)


def test_create_directories_continue_from_no(tmp_path):
    create_directories("me_model", tmp_path)
    create_directories("me_model", tmp_path)

    with pytest.raises(ValueError) as e:
        create_directories("me_model", tmp_path, continue_from=10)

    assert "does not" in str(e)


def test_create_directories_bad_file(tmp_path):
    create_directories("me_model", tmp_path)
    create_directories("me_model", tmp_path)

    bogus_file = tmp_path / "training_results" / "me_model" / "bogus.txt"
    bogus_file.parent.mkdir(exist_ok=True, parents=True)
    bogus_file.touch()

    final_path = create_directories("me_model", tmp_path, continue_from=-1)

    assert "00001" in str(final_path)


def test_create_directories_bad_file_new_dir(tmp_path):
    create_directories("me_model", tmp_path)

    bogus_file = tmp_path / "training_results" / "me_model" / "bogus.txt"
    bogus_file.parent.mkdir(exist_ok=True, parents=True)
    bogus_file.touch()
    (bogus_file.parent / "weird_dir").mkdir()

    r = create_directories("me_model", tmp_path)

    assert "00001" in str(r)


def test_create_directories_continue_from_nothing_neg(tmp_path):
    "Good error message when nothing has been run"
    with pytest.raises(ValueError) as e:
        create_directories("me_model", tmp_path, continue_from=-1)

    assert "Cannot continue from" in str(e)


@pytest.mark.parametrize(
    "back_from", [(0, "00000"), (1, "00001"), (-1, "00001"), (-2, "00000")]
)
def test_create_directories_continue_from_last(tmp_path, back_from: Tuple[int, str]):
    "Test various ways of referring to previous runs"
    create_directories("me_model", tmp_path)
    create_directories("me_model", tmp_path)

    final_path = create_directories("me_model", tmp_path, continue_from=back_from[0])

    assert back_from[1] in str(final_path)
