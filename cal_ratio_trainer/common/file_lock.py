import fcntl
from pathlib import Path


class FileLock:
    def __init__(self, file_path: Path):
        self.file_path = file_path.with_suffix(f"{file_path.suffix}.lock")
        self.file = None

    @property
    def is_locked(self):
        return self.file is not None

    def __enter__(self):
        "Lock the lock-file"
        self.file = open(self.file_path, "w")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file is not None:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None

            # Try to remove file
            try:
                self.file_path.unlink()
            except Exception:
                pass
