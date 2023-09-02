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


def test_create_directories_continue_from_yes(tmp_path):
    create_directories("me_model", tmp_path)
    create_directories("me_model", tmp_path)

    final_path = create_directories("me_model", tmp_path, continue_from=0)

    assert "00000" in str(final_path)
