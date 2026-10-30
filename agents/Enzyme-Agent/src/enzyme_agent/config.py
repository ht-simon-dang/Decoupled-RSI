from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

MAX_BYTES = 50_000_000_000  # Decimal GB; never silently interpret as 50 GiB.
HEADROOM = 200_000_000


def data_root() -> Path:
    return Path(os.environ.get("ENZYME_DATA_DIR", str(Path.home() / ".cache/decoupled-rsi/enzyme"))).expanduser().resolve()


def encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def usage(root=None) -> int:
    root = Path(root or data_root())
    total = 0
    # Count logical file sizes (conservative for sparse files); do not follow symlinks.
    if not root.exists():return 0
    pending=[root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total+=entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    pass
    return total


def check_space(extra=0):
    used = usage()
    if used + extra + HEADROOM > MAX_BYTES:
        raise OSError(f"Enzyme storage quota: {used} + {extra} bytes would exceed 50 GB with reserve")
    return used


@contextlib.contextmanager
def quota_lock():
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".quota.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with quota_lock():
        check_space(len(content))
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(Path(path), encode(value))

