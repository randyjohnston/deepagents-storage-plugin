"""Provider-agnostic blob storage protocol.

Deliberately *not* S3-shaped: Box and Egnyte are folder/file-ID systems with
OAuth'd chunked upload sessions and no presigned URLs, so the protocol is a
minimal key-value contract plus capability flags. Callers (e.g.
``OffloadingBackend``) must branch on ``capabilities`` rather than assume
S3 semantics.

Sync methods are required; async twins default to ``asyncio.to_thread`` so
providers with native async clients (aioboto3, httpx) can override them.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import IO, Final

DEFAULT_PRESIGN_TTL: Final = 900
"""Default presigned-URL lifetime in seconds. Keep short; pointers in
LangGraph state should carry the *key*, never a presigned URL."""


class BlobStoreError(Exception):
    """Base error for blob store operations."""


class BlobNotFoundError(BlobStoreError):
    """Requested key does not exist in the store."""


@dataclass(frozen=True)
class BlobCapabilities:
    """What a provider can do. ``OffloadingBackend`` and any ingress API
    branch on these instead of try/except-driven feature detection."""

    presign_get: bool = False
    presign_put: bool = False
    range_get: bool = False
    """Supports byte-range reads (S3 ``Range`` header). When False, callers
    must GET the whole object even for windowed reads."""
    multipart: bool = False
    """Native multipart / chunked upload sessions for large objects."""
    native_list: bool = False
    """Efficient prefix listing (S3 ListObjectsV2). Folder-mapped providers
    emulate listing by walking folders, which is slower and rate-limited."""
    server_side_lifecycle: bool = False
    """Provider can expire objects via bucket lifecycle rules. When False,
    orphan GC must be done client-side (see OffloadingBackend docs)."""


@dataclass(frozen=True)
class BlobRef:
    """Result of a successful ``put``. This is what gets embedded (as fields
    of the pointer stub) into LangGraph state — small and serializable."""

    key: str
    size: int
    etag: str | None = None
    """Provider-native integrity token, when available. NOT guaranteed to be
    a content hash (S3 multipart etags aren't); the content sha256 used for
    addressing lives in the pointer stub, not here."""


@dataclass(frozen=True)
class BlobInfo:
    """A single entry from ``list``."""

    key: str
    size: int
    modified_at: str | None = None
    """ISO 8601, when the provider reports it."""


@dataclass(frozen=True)
class UploadTarget:
    """Instructions for a client-side direct upload (Option D ingress).

    For presign-capable stores this is a single PUT URL. For session-based
    providers (Box, Egnyte) it describes the provider handshake the client
    must perform; ``url`` is the session endpoint and ``headers`` carries
    auth. Providers that support neither return ``None`` from
    ``create_upload_target`` and the caller falls back to proxied streaming.
    """

    scheme: str  # "presigned_put" | "box_session" | "egnyte_chunked" | ...
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    expires_in: int = DEFAULT_PRESIGN_TTL


class BlobStore(abc.ABC):
    """Minimal key-value blob contract every provider adapter implements.

    Keys are opaque ``/``-separated strings chosen by the caller (the
    offloading layer uses content-addressed keys, see ``keys.py``). Adapters
    for folder-based providers own the key<->folder/file-id mapping
    internally; callers never see provider IDs.
    """

    capabilities: BlobCapabilities = BlobCapabilities()

    # -- required sync surface -------------------------------------------------

    @abc.abstractmethod
    def put(
        self,
        key: str,
        data: bytes | IO[bytes],
        *,
        content_type: str | None = None,
    ) -> BlobRef:
        """Store ``data`` at ``key``, overwriting any existing object.

        Must accept a file-like object so large payloads can be streamed
        (mirroring the Fleet file API's streamed multipart behavior) rather
        than buffered.
        """

    @abc.abstractmethod
    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        """Fetch object bytes.

        Args:
            key: Object key.
            byte_range: Optional inclusive ``(start, end)`` byte range.
                Only honored when ``capabilities.range_get``; adapters
                without range support MUST raise ``NotImplementedError``
                for a non-None range rather than silently returning the
                full object.

        Raises:
            BlobNotFoundError: If the key does not exist.
        """

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Cheap existence probe (HEAD). Used by the offloading layer to skip
        re-uploading content-addressed blobs that are already present."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object. Idempotent: deleting a missing key is a no-op
        (mirrors the Fleet sandbox DELETE semantics)."""

    @abc.abstractmethod
    def list(self, prefix: str) -> Iterator[BlobInfo]:
        """Iterate objects under ``prefix``. May be expensive on providers
        without ``native_list``; callers should prefer state-side metadata
        (the pointer stubs) and treat this as a reconciliation tool."""

    # -- optional sync surface -------------------------------------------------

    def presign_get(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        """Short-lived GET URL, or ``None`` if unsupported."""
        return None

    def presign_put(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        """Short-lived PUT URL, or ``None`` if unsupported."""
        return None

    def create_upload_target(self, key: str, size: int) -> UploadTarget | None:
        """Provider-appropriate direct-upload instructions for ingress
        (Option D), or ``None`` -> caller proxies the bytes itself."""
        url = self.presign_put(key)
        if url is not None:
            return UploadTarget(scheme="presigned_put", url=url)
        return None

    # -- async twins (override with native async clients where available) ------

    async def aput(
        self,
        key: str,
        data: bytes | IO[bytes],
        *,
        content_type: str | None = None,
    ) -> BlobRef:
        return await asyncio.to_thread(self.put, key, data, content_type=content_type)

    async def aget(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        return await asyncio.to_thread(self.get, key, byte_range=byte_range)

    async def aexists(self, key: str) -> bool:
        return await asyncio.to_thread(self.exists, key)

    async def adelete(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)
