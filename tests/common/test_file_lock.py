from cal_ratio_trainer.common.file_lock import FileLock


def test_file_lock_simple(tmp_path):
    with FileLock(tmp_path / "test_file.txt") as lock:
        assert lock.is_locked

    lock_file = tmp_path / "test_file.txt.lock"
    assert not lock_file.exists()


def test_two_try(tmp_path):
    interesting_file = tmp_path / "test_file.txt"
    with FileLock(interesting_file) as lock1:
        assert lock1.is_locked
        with FileLock(interesting_file) as lock2:
            assert not lock2.is_locked
