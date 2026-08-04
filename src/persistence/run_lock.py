"""Simple cross-platform file-based run lock so an overlapping scheduler
trigger is skipped (logged) instead of starting a second concurrent run
against the same profile (plan section 27.3 / M23 "Scheduler overlap")."""
from __future__ import annotations

import os
import time
from pathlib import Path


class RunLockHeld(Exception):
    pass


class RunLock:
    """Exclusive lock backed by atomic file creation (O_CREAT|O_EXCL), which is
    portable across Windows and POSIX. A lock older than `stale_after_seconds`
    is treated as abandoned (e.g. a crashed prior process) and reclaimed."""

    def __init__(self, lock_path: Path, stale_after_seconds: int = 3600):
        self.lock_path = Path(lock_path)
        self.stale_after_seconds = stale_after_seconds
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._is_stale():
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise RunLockHeld(f"lock already held: {self.lock_path}")
        os.write(self._fd, str(os.getpid()).encode())
        os.close(self._fd)

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return True
        return age > self.stale_after_seconds

    def release(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
