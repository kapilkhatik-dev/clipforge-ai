"""Small helpers for durable cache writes and inexpensive file identity checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_FINGERPRINT_CHUNK_SIZE = 1024 * 1024


def fingerprint_file(path: Path) -> str:
    """Hash file metadata plus its first and last MiB without rereading large media."""
    resolved = path.resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    with resolved.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_CHUNK_SIZE))
        if stat.st_size > _FINGERPRINT_CHUNK_SIZE:
            handle.seek(max(0, stat.st_size - _FINGERPRINT_CHUNK_SIZE))
            digest.update(handle.read(_FINGERPRINT_CHUNK_SIZE))
    return digest.hexdigest()


def fingerprint_payload(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
