"""Bounded helpers for loading tensor-only PyTorch checkpoints."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Any

import torch


MINIMUM_RESTRICTED_LOADER_TORCH = (2, 10, 0)
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def _torch_release_tuple() -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(torch.__version__))
    if match is None:
        raise RuntimeError(f"cannot validate PyTorch version {torch.__version__!r}")
    return tuple(int(component) for component in match.groups())


def require_restricted_loader_runtime() -> None:
    """Refuse versions with known ``weights_only`` loader vulnerabilities."""
    if _torch_release_tuple() < MINIMUM_RESTRICTED_LOADER_TORCH:
        required = ".".join(str(part) for part in MINIMUM_RESTRICTED_LOADER_TORCH)
        raise RuntimeError(
            f"checkpoint loading requires PyTorch {required} or newer"
        )


def load_bounded_weights_only(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    max_bytes: int = MAX_CHECKPOINT_BYTES,
) -> tuple[Any, Path, str]:
    """Load a local checkpoint through PyTorch's restricted loader.

    The size bound reduces accidental resource exhaustion. It is not a claim
    that arbitrary third-party checkpoint files are safe; callers must still
    validate the returned schema and should verify an expected digest.
    """
    require_restricted_loader_runtime()
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive native int")

    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    with checkpoint_path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"checkpoint is not a regular file: {checkpoint_path}"
            )
        if before.st_size <= 0:
            raise ValueError("checkpoint file is empty")
        if before.st_size > max_bytes:
            raise ValueError(
                f"checkpoint is {before.st_size} bytes; "
                f"the evaluator limit is {max_bytes} bytes"
            )

        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after_hash = os.fstat(handle.fileno())
        if _file_identity(before) != _file_identity(after_hash):
            raise RuntimeError("checkpoint changed while it was being hashed")

        handle.seek(0)
        checkpoint = torch.load(
            handle,
            map_location=map_location,
            weights_only=True,
        )
        after_load = os.fstat(handle.fileno())
        if _file_identity(after_hash) != _file_identity(after_load):
            raise RuntimeError("checkpoint changed while it was being loaded")

    return checkpoint, checkpoint_path, digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return file attributes that change for replacement or normal writes."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
