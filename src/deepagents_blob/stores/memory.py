"""In-memory blob store for tests and local development."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import IO

from deepagents_blob.stores.protocol import (
    DEFAULT_PRESIGN_TTL,
    BlobCapabilities,
    BlobInfo,
    BlobNotFoundError,
    BlobRef,
    BlobStore,
)


class InMemoryBlobStore(BlobStore):
    """Dict-backed store. Declares S3-like capabilities so tests exercise the
    same code paths as ``S3CompatibleStore`` (range reads included)."""

    capabilities = BlobCapabilities(
        presign_get=True,
        presign_put=True,
        range_get=True,
        multipart=False,
        native_list=True,
        server_side_lifecycle=False,
    )

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}  # key -> (data, modified_at)

    def put(self, key: str, data: bytes | IO[bytes], *, content_type: str | None = None) -> BlobRef:
        raw = data if isinstance(data, bytes) else data.read()
        self._objects[key] = (raw, datetime.now(timezone.utc).isoformat())
        return BlobRef(key=key, size=len(raw), etag=None)

    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        if key not in self._objects:
            raise BlobNotFoundError(key)
        raw = self._objects[key][0]
        if byte_range is not None:
            start, end = byte_range
            return raw[start : end + 1]
        return raw

    def exists(self, key: str) -> bool:
        return key in self._objects

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def list(self, prefix: str) -> Iterator[BlobInfo]:
        for key, (raw, ts) in sorted(self._objects.items()):
            if key.startswith(prefix):
                yield BlobInfo(key=key, size=len(raw), modified_at=ts)

    def presign_get(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        return f"memory://{key}?ttl={ttl}"

    def presign_put(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        return f"memory://{key}?ttl={ttl}&put=1"
