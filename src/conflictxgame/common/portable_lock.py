"""Cross-platform advisory file locking.

The dyadic pipeline (:mod:`conflictxgame.solo.simulated_conversation`) uses :mod:`fcntl`, which is
POSIX-only. The game-theoretic pipeline uses this shim instead so that caching
works on Windows (via :mod:`msvcrt`) as well as on Linux/macOS (via
:mod:`fcntl`).

Example:
    >>> from conflictxgame.common.portable_lock import file_lock
    >>> with file_lock("cache.json.lock"):
    ...     pass  # read-modify-write the cache here
"""

import os
import time
from contextlib import contextmanager


def replace_with_retry(src: str, dst: str, attempts: int = 8) -> None:
    """Atomically replace ``dst`` with ``src``, retrying Dropbox/Windows locks.

    ``os.replace`` is the commit for Message Batch manifests and journals.
    On a Dropbox path Windows can raise ``PermissionError`` (WinError 5)
    while the sync client has the destination open; failing immediately
    abandons an in-flight Anthropic batch.  Brief backoff is enough for
    the lock to clear without changing the atomic-replace contract.
    """
    delay = 0.25
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 4.0)
    assert last_error is not None
    raise last_error


try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # Windows
    import msvcrt

    _HAVE_FCNTL = False


@contextmanager
def file_lock(lock_path: str):
    """Acquire an exclusive advisory lock on ``lock_path`` for the block's duration.

    The lock file is created if it does not exist. Locking is advisory: all
    cooperating processes must use this same context manager for mutual
    exclusion to hold.

    Args:
        lock_path: Path of the lock file (conventionally ``<target>.lock``).

    Yields:
        None. The lock is held while the ``with`` block executes and released
        afterwards, even if the block raises.
    """
    directory = os.path.dirname(lock_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        if _HAVE_FCNTL:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            # msvcrt locks a byte range; ensure the file has at least one byte.
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write("x")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
