"""Pluggable large-file offloading for deepagents (Option B).

Public surface:

- ``OffloadingBackend`` — decorator around any ``BackendProtocol``.
- ``BlobStore`` protocol + ``BlobCapabilities`` — provider seam.
- Stores: ``S3CompatibleStore`` (AWS / Backblaze B2 / MinIO / R2 / GCS
  interop), ``BoxBlobStore``, ``EgnyteBlobStore``, ``InMemoryBlobStore``.
"""

from deepagents_blob.offloading import (
    DEFAULT_THRESHOLD,
    EXTERNAL_ENCODING,
    OffloadingBackend,
    parse_pointer,
)
from deepagents_blob.stores.protocol import (
    BlobCapabilities,
    BlobInfo,
    BlobNotFoundError,
    BlobRef,
    BlobStore,
    BlobStoreError,
    UploadTarget,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "EXTERNAL_ENCODING",
    "BlobCapabilities",
    "BlobInfo",
    "BlobNotFoundError",
    "BlobRef",
    "BlobStore",
    "BlobStoreError",
    "OffloadingBackend",
    "UploadTarget",
    "parse_pointer",
]
