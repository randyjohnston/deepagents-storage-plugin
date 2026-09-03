"""Egnyte adapter.

Egnyte is friendlier to a key-value facade than Box: its public API is
*path-based* (``/pubapi/v1/fs/...`` and ``/pubapi/v1/fs-content/...``), so
keys map 1:1 to paths under a root folder with no ID bookkeeping. Remaining
mismatches: OAuth2 bearer auth (no presign), chunked uploads via
``X-Egnyte-Chunk-Num`` headers for large files, folder listing instead of
prefix listing, and per-tenant QPS rate limits that make deep walks slow.

Stubbed with real endpoint shapes; wire up with ``httpx`` (async twin can
override the ``a*`` methods with an ``httpx.AsyncClient``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

from deepagents_blob.stores.protocol import (
    BlobCapabilities,
    BlobInfo,
    BlobNotFoundError,
    BlobRef,
    BlobStore,
)

_CHUNK_SIZE: int = 100 * 1024 * 1024
"""Egnyte chunked-upload chunk size ceiling; single-request uploads are fine
below roughly this size."""


class EgnyteBlobStore(BlobStore):
    """Blob store facade over Egnyte's fs / fs-content API."""

    capabilities = BlobCapabilities(
        presign_get=False,
        presign_put=False,
        range_get=False,  # fs-content GET has no documented range support
        multipart=True,  # chunked upload headers
        native_list=False,
        server_side_lifecycle=False,
    )

    def __init__(self, domain: str, token: str, root: str = "/Shared/agent-blobs", client: Any = None) -> None:
        """Args:
        domain: Tenant domain, e.g. ``"acme"`` for ``acme.egnyte.com``.
        token: OAuth2 bearer token.
        root: Folder under which all keys live.
        client: Optional pre-built ``httpx.Client`` (for pooling/testing).
        """
        self._base = f"https://{domain}.egnyte.com/pubapi/v1"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._root = root.rstrip("/")
        self._client = client  # TODO: default to httpx.Client(headers=self._headers)

    def _path(self, key: str) -> str:
        return f"{self._root}/{key}"

    # -- required surface ------------------------------------------------------

    def put(self, key: str, data: bytes | IO[bytes], *, content_type: str | None = None) -> BlobRef:
        # Small payloads: single request —
        #   POST {base}/fs-content/{path}  (body = bytes)
        # Egnyte auto-creates intermediate folders on fs-content POST.
        # Large payloads: chunked —
        #   POST {base}/fs-content-chunked/{path} with X-Egnyte-Chunk-Num: 1..N,
        #   final chunk carries X-Egnyte-Last-Chunk: true; server returns
        #   checksum per chunk to verify.
        # TODO: implement both branches; return BlobRef(key, size,
        #   etag=response checksum).
        raise NotImplementedError("TODO: Egnyte fs-content upload")

    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        if byte_range is not None:
            raise NotImplementedError("Egnyte adapter does not support range reads")
        # TODO: GET {base}/fs-content/{path}; 404 -> BlobNotFoundError(key)
        raise NotImplementedError("TODO: Egnyte fs-content download")

    def exists(self, key: str) -> bool:
        # TODO: GET {base}/fs/{path} (metadata); 200 -> True, 404 -> False
        raise NotImplementedError("TODO: Egnyte fs metadata probe")

    def delete(self, key: str) -> None:
        # TODO: DELETE {base}/fs/{path}; treat 404 as success (idempotent)
        raise NotImplementedError("TODO: Egnyte delete")

    def list(self, prefix: str) -> Iterator[BlobInfo]:
        # TODO: GET {base}/fs/{folder}?list_content=true (paged via count/offset),
        # recurse into subfolders. Mind tenant QPS limits — prefer state-side
        # pointer metadata over walking this for routine ls calls.
        raise NotImplementedError("TODO: Egnyte folder walk")

    # Ingress: no presign and no public downscoped-token upload flow -> return
    # None from create_upload_target (inherited default already does this when
    # presign_put is None), meaning ingress proxies bytes through your API.
