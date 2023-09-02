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
