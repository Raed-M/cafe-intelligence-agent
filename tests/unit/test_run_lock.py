import time

import pytest

from src.persistence.run_lock import RunLock, RunLockHeld


def test_acquire_and_release(tmp_path):
    lock = RunLock(tmp_path / "test.lock")
    lock.acquire()
    assert (tmp_path / "test.lock").exists()
    lock.release()
    assert not (tmp_path / "test.lock").exists()


def test_second_acquire_raises_while_held(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock1 = RunLock(lock_path)
    lock1.acquire()
    lock2 = RunLock(lock_path)
    with pytest.raises(RunLockHeld):
        lock2.acquire()
    lock1.release()


def test_context_manager_releases_on_exit(tmp_path):
    lock_path = tmp_path / "test.lock"
    with RunLock(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_stale_lock_is_reclaimed(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock1 = RunLock(lock_path, stale_after_seconds=0)
    lock1.acquire()
    time.sleep(0.05)
    lock2 = RunLock(lock_path, stale_after_seconds=0)
    lock2.acquire()  # should reclaim since stale_after_seconds=0
    assert lock_path.exists()
    lock2.release()
