from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    source_root = package_root()
    default = (
        source_root / ".antibody-agent"
        if (source_root / "pyproject.toml").is_file()
        else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "decoupled-rsi/antibody"
    )
    return Path(os.environ.get("ANTIBODY_DATA_DIR", default)).expanduser().resolve()


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(encode(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
