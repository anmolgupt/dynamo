# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory transport for embedding requests and response vectors.

Embedding requests use UTF-8 JSON in SHM. Embedding responses use contiguous
float32 bytes. In both directions, small versioned metadata stays on Dynamo's
normal request plane.
"""

from __future__ import annotations

import json
import logging
from multiprocessing import resource_tracker
import multiprocessing.shared_memory as shm
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel
from prometheus_client import Counter, Histogram

from dynamo.common.utils.env import env_bool

logger = logging.getLogger(__name__)

_LABELS = ("model", "path", "status")
_FALLBACK_LABELS = ("model", "path", "reason")
_DEFAULT_REQUEST_SHM_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_RESPONSE_SHM_MAX_BYTES = 256 * 1024 * 1024
_REQUEST_PAYLOAD_KIND_JSON = "json"

EMBEDDING_SHM_REQUESTS = Counter(
    "dynamo_worker_embedding_shm_requests_total",
    "Embedding SHM response attempts by status.",
    _LABELS,
)
EMBEDDING_SHM_FALLBACKS = Counter(
    "dynamo_worker_embedding_shm_fallback_total",
    "Embedding SHM response fallback reasons.",
    _FALLBACK_LABELS,
)
EMBEDDING_SHM_BYTES_WRITTEN = Counter(
    "dynamo_worker_embedding_shm_bytes_written_total",
    "Embedding response bytes written to shared memory.",
    ("model", "path"),
)
EMBEDDING_SHM_WRITE_SECONDS = Histogram(
    "dynamo_worker_embedding_shm_write_seconds",
    "Time spent writing embedding responses to shared memory.",
    ("model", "path"),
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
EMBEDDING_SHM_CLEANUP_FAILURES = Counter(
    "dynamo_worker_embedding_shm_cleanup_failures_total",
    "Embedding SHM cleanup failures observed by the writer.",
    ("model", "path"),
)


class EmbeddingRequestShmMetadata(BaseModel):
    """Wire metadata for a single embedding request SHM segment."""

    version: int = 1
    model: str
    path: str
    name: str
    size_bytes: int
    payload_kind: str = _REQUEST_PAYLOAD_KIND_JSON
    field: str


class EmbeddingResponseShmMetadata(BaseModel):
    """Wire metadata for a single embedding response SHM segment."""

    version: int = 1
    model: str
    path: str
    name: str
    size_bytes: int
    dtype: str = "float32"
    endianness: str = "native"
    shape: list[int]
    embedding_type: str = "float"
    encoding_format: str = "float"


def _validate_shm_name(name: str) -> None:
    """Reject unsafe or unrelated POSIX SHM names."""

    if (
        not name
        or "/" in name
        or ".." in name
        or "\x00" in name
        or not name.startswith(("dyn_embed_req_", "dyn_embed_resp_"))
    ):
        raise ValueError(f"invalid embedding SHM name {name!r}")


def request_shm_max_bytes() -> int:
    """Maximum request SHM payload size accepted by the Python worker."""

    raw = os.environ.get("DYN_EMBEDDING_SHM_REQUEST_MAX_BYTES")
    if raw is None:
        return _DEFAULT_REQUEST_SHM_MAX_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid DYN_EMBEDDING_SHM_REQUEST_MAX_BYTES=%r; using %d",
            raw,
            _DEFAULT_REQUEST_SHM_MAX_BYTES,
        )
        return _DEFAULT_REQUEST_SHM_MAX_BYTES


def response_shm_max_bytes() -> int:
    """Maximum response SHM payload size written by the Python worker."""

    raw = os.environ.get("DYN_EMBEDDING_SHM_RESPONSE_MAX_BYTES")
    if raw is None:
        return _DEFAULT_RESPONSE_SHM_MAX_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid DYN_EMBEDDING_SHM_RESPONSE_MAX_BYTES=%r; using %d",
            raw,
            _DEFAULT_RESPONSE_SHM_MAX_BYTES,
        )
        return _DEFAULT_RESPONSE_SHM_MAX_BYTES


def apply_embedding_request_shm(request: dict[str, Any]) -> None:
    """Read an internal request SHM payload and restore it onto the request dict."""

    raw_meta = request.get("embedding_request_shm")
    if raw_meta is None:
        return

    meta = EmbeddingRequestShmMetadata.model_validate(raw_meta)
    if meta.version != 1:
        raise ValueError(f"unsupported embedding request SHM version {meta.version}")
    if meta.payload_kind != _REQUEST_PAYLOAD_KIND_JSON:
        raise ValueError(
            f"unsupported embedding request SHM payload kind {meta.payload_kind!r}"
        )
    if meta.field not in ("input", "token_ids"):
        raise ValueError(f"unsupported embedding request SHM field {meta.field!r}")
    if meta.size_bytes <= 0:
        raise ValueError("embedding request SHM payload is empty")

    _validate_shm_name(meta.name)
    path = os.path.join("/dev/shm", meta.name)
    max_bytes = request_shm_max_bytes()
    if meta.size_bytes > max_bytes:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise ValueError(
            f"embedding request SHM payload too large: {meta.size_bytes} bytes, "
            f"max {max_bytes}"
        )
    fd: int | None = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        stat = os.fstat(fd)
        if stat.st_size < meta.size_bytes:
            raise ValueError(
                f"embedding request SHM segment {meta.name} size mismatch: "
                f"got {stat.st_size}, expected {meta.size_bytes}"
            )
        if stat.st_size > meta.size_bytes:
            raise ValueError(
                f"embedding request SHM segment {meta.name} is larger than metadata: "
                f"got {stat.st_size}, expected {meta.size_bytes}"
            )
        with os.fdopen(fd, "rb", closefd=True) as f:
            fd = None
            data = f.read(meta.size_bytes)
        if len(data) != meta.size_bytes:
            raise ValueError(
                f"embedding request SHM segment {meta.name} size mismatch: "
                f"got {len(data)}, expected {meta.size_bytes}"
            )
        request[meta.field] = json.loads(data.decode("utf-8"))
        request.pop("embedding_request_shm", None)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@dataclass
class EmbeddingShmResult:
    """Result of writing embedding vectors to shared memory."""

    metadata: EmbeddingResponseShmMetadata
    handle: shm.SharedMemory

    def close(self) -> None:
        """Close the writer handle after handing unlink ownership to the consumer."""

        self.handle.close()
        # SharedMemory always registers creators on Python 3.12. The Rust
        # frontend is the consumer and unlinks this segment, so keeping the
        # writer registered causes a false leak warning when the worker exits.
        resource_tracker.unregister(self.handle._name, "shared_memory")


def shm_response_enabled() -> bool:
    """Return whether embedding response SHM is enabled for this process."""

    return env_bool("DYN_EMBEDDING_SHM_RESPONSE")


def shm_min_bytes() -> int:
    """Minimum payload size for SHM fast path."""

    raw = os.environ.get("DYN_EMBEDDING_SHM_MIN_BYTES", "65536")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid DYN_EMBEDDING_SHM_MIN_BYTES=%r; using 65536", raw)
        return 65536


def record_shm_fallback(model: str, path: str, reason: str) -> None:
    """Record a fallback to the normal JSON float-array response path."""

    EMBEDDING_SHM_REQUESTS.labels(model=model, path=path, status="fallback").inc()
    EMBEDDING_SHM_FALLBACKS.labels(model=model, path=path, reason=reason).inc()


def pooling_output_to_tensor(data: Any) -> torch.Tensor:
    """Convert a vLLM pooling output payload to a flat CPU float32 tensor."""

    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", dtype=torch.float32).flatten()
    if isinstance(data, (list, tuple)):
        # torch.as_tensor handles both flat and nested numeric lists.
        return torch.as_tensor(data, dtype=torch.float32, device="cpu").flatten()
    raise TypeError(
        f"Unsupported PoolingOutput.data type {type(data).__name__}; "
        "expected torch.Tensor or list"
    )


def maybe_write_embedding_response_shm(
    rows: list[torch.Tensor],
    *,
    model: str,
    path: str,
    encoding_format: str | None,
    embedding_type: str = "float",
) -> EmbeddingShmResult | None:
    """Write embedding rows to SHM if the fast path is enabled and applicable.

    Returns ``None`` when callers should use the normal JSON response path.
    The caller owns the returned handle and should close it after yielding the
    metadata. The Rust frontend owns unlinking after it has read the segment.
    """

    if not shm_response_enabled():
        record_shm_fallback(model, path, "disabled")
        return None
    if encoding_format not in (None, "float"):
        record_shm_fallback(model, path, "unsupported_encoding_format")
        return None
    if embedding_type != "float":
        record_shm_fallback(model, path, "unsupported_embedding_type")
        return None
    if not rows:
        record_shm_fallback(model, path, "empty")
        return None

    dim = rows[0].numel()
    if dim == 0:
        record_shm_fallback(model, path, "empty")
        return None
    for row in rows:
        if row.numel() != dim:
            record_shm_fallback(model, path, "ragged_rows")
            return None

    stacked = torch.stack([row.contiguous() for row in rows]).contiguous()
    size_bytes = stacked.numel() * stacked.element_size()
    if size_bytes < shm_min_bytes():
        record_shm_fallback(model, path, "small_payload")
        return None

    if size_bytes > response_shm_max_bytes():
        record_shm_fallback(model, path, "large_payload")
        return None
    name = f"dyn_embed_resp_{os.getpid()}_{uuid.uuid4().hex[:16]}"
    start = time.perf_counter()
    segment: shm.SharedMemory | None = None
    view: memoryview | None = None
    try:
        segment = shm.SharedMemory(name=name, create=True, size=size_bytes)
        view = memoryview(stacked.numpy()).cast("B")
        segment.buf[:size_bytes] = view
        view.release()
        view = None
    except Exception:
        EMBEDDING_SHM_REQUESTS.labels(model=model, path=path, status="fallback").inc()
        EMBEDDING_SHM_FALLBACKS.labels(
            model=model, path=path, reason="write_failed"
        ).inc()
        logger.warning("Failed to write embedding response SHM", exc_info=True)
        if view is not None:
            view.release()
        if segment is not None:
            try:
                segment.close()
                segment.unlink()
            except Exception:
                EMBEDDING_SHM_CLEANUP_FAILURES.labels(model=model, path=path).inc()
        return None

    EMBEDDING_SHM_WRITE_SECONDS.labels(model=model, path=path).observe(
        time.perf_counter() - start
    )
    EMBEDDING_SHM_BYTES_WRITTEN.labels(model=model, path=path).inc(size_bytes)
    EMBEDDING_SHM_REQUESTS.labels(model=model, path=path, status="success").inc()

    return EmbeddingShmResult(
        metadata=EmbeddingResponseShmMetadata(
            model=model,
            path=path,
            name=name,
            size_bytes=size_bytes,
            shape=[len(rows), dim],
            embedding_type=embedding_type,
        ),
        handle=segment,
    )
