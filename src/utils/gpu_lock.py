"""GPU task serialization via flock.

Ensures only one GPU-intensive process runs at a time on this machine.
Other processes block and wait (queue behavior, not error).

Usage::

    if __name__ == "__main__":
        from src.utils.gpu_lock import GpuLock
        with GpuLock():
            main()
"""

import fcntl
import os
import sys
from pathlib import Path

LOCK_FILE = "/tmp/gpu-task.lock"


class GpuLock:
    """Exclusive GPU lock using flock. Auto-releases on process exit/crash/kill."""

    def __init__(self, lock_file: str = LOCK_FILE, verbose: bool = True):
        self.lock_file = lock_file
        self.verbose = verbose
        self._fd = None

    def __enter__(self):
        self._fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR)
        if self.verbose:
            # Try non-blocking first to report waiting
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"GPU lock acquired ({self.lock_file})")
                return self
            except OSError:
                print(
                    f"Another GPU task is running. Waiting for lock ({self.lock_file})...",
                    file=sys.stderr,
                )
        # Blocking acquire
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        if self.verbose:
            print(f"GPU lock acquired ({self.lock_file})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        return False

    @staticmethod
    def status() -> str:
        """Check if the GPU lock is currently held."""
        if not Path(LOCK_FILE).exists():
            return "No lock file exists. GPU is free."
        fd = os.open(LOCK_FILE, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return "GPU lock is free."
        except OSError:
            return "GPU lock is held by another process."
        finally:
            os.close(fd)
