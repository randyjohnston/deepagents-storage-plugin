"""S3-compatible blob store.

One implementation covers AWS S3, Backblaze B2 (S3 API), MinIO, Cloudflare R2,
and GCS in interoperability mode — everything hides behind ``endpoint_url``.

Requires ``boto3``. For a native-async variant, subclass and override the
``a*`` twins with ``aioboto3``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, TYPE_CHECKING, Any

from deepagents_blob.stores.protocol import (
    DEFAULT_PRESIGN_TTL,
    BlobCapabilities,
    BlobInfo,
    BlobNotFoundError,
    BlobRef,
    BlobStore,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client  # optional stubs; plain Any at runtime


class S3CompatibleStore(BlobStore):
    """Blob store backed by any S3-compatible endpoint.

    Examples:
        >>> # AWS
        >>> S3CompatibleStore(bucket="agent-blobs", region_name="us-east-1")
        >>> # Backblaze B2
        >>> S3CompatibleStore(
        ...     bucket="agent-blobs",
        ...     endpoint_url="https://s3.us-west-004.backblazeb2.com",
        ... )
        >>> # MinIO / self-hosted
        >>> S3CompatibleStore(bucket="agent-blobs", endpoint_url="http://minio:9000")
    """

    capabilities = BlobCapabilities(
        presign_get=True,
        presign_put=True,
        range_get=True,
        multipart=True,
        native_list=True,
        server_side_lifecycle=True,
    )

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        key_prefix: str = "",
        client: "S3Client | None" = None,
        client_kwargs: dict[str, Any] | None = None,
        sse: str | None = "AES256",
    ) -> None:
        """Create the store.

        Args:
            bucket: Target bucket name.
            endpoint_url: Non-AWS endpoint (B2/MinIO/R2/GCS-interop). ``None``
                targets AWS.
            region_name: AWS region, when applicable.
            key_prefix: Prepended to every key — use for tenant isolation,
                e.g. ``f"{tenant_id}/"``. Presign/IAM policies can then be
                scoped per prefix.
            client: Pre-built boto3 client (wins over the kwargs below).
            client_kwargs: Extra ``boto3.client("s3", ...)`` kwargs
                (credentials, signature version — B2 wants ``s3v4``).
            sse: Server-side encryption algorithm passed on ``put``
                (``"AES256"`` or ``"aws:kms"``); ``None`` disables.
        """
        if client is None:
            import boto3  # noqa: PLC0415 - optional dependency, import at use

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region_name,
                **(client_kwargs or {}),
            )
        self._client = client
        self._bucket = bucket
        self._prefix = key_prefix
        self._sse = sse

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # -- required surface ------------------------------------------------------

    def put(self, key: str, data: bytes | IO[bytes], *, content_type: str | None = None) -> BlobRef:
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        if self._sse:
            extra["ServerSideEncryption"] = self._sse

        if isinstance(data, bytes):
            resp = self._client.put_object(Bucket=self._bucket, Key=self._k(key), Body=data, **extra)
            return BlobRef(key=key, size=len(data), etag=resp.get("ETag"))

        # File-like: let boto3 manage multipart chunking for large streams.
        self._client.upload_fileobj(data, self._bucket, self._k(key), ExtraArgs=extra or None)
        head = self._client.head_object(Bucket=self._bucket, Key=self._k(key))
        return BlobRef(key=key, size=head["ContentLength"], etag=head.get("ETag"))

    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": self._k(key)}
        if byte_range is not None:
            kwargs["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        try:
            return self._client.get_object(**kwargs)["Body"].read()
        except self._client.exceptions.NoSuchKey as exc:
            raise BlobNotFoundError(key) from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._k(key))
        except self._client.exceptions.ClientError:
            return False
        return True

    def delete(self, key: str) -> None:
        # S3 DeleteObject is already idempotent (204 for missing keys).
        self._client.delete_object(Bucket=self._bucket, Key=self._k(key))

    def list(self, prefix: str) -> Iterator[BlobInfo]:
        paginator = self._client.get_paginator("list_objects_v2")
        strip = len(self._prefix)
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._k(prefix)):
            for obj in page.get("Contents", []):
                yield BlobInfo(
                    key=obj["Key"][strip:],
                    size=obj["Size"],
                    modified_at=obj["LastModified"].isoformat() if obj.get("LastModified") else None,
                )

    # -- presign ---------------------------------------------------------------

    def presign_get(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": self._k(key)}, ExpiresIn=ttl
        )

    def presign_put(self, key: str, ttl: int = DEFAULT_PRESIGN_TTL) -> str | None:
        return self._client.generate_presigned_url(
            "put_object", Params={"Bucket": self._bucket, "Key": self._k(key)}, ExpiresIn=ttl
        )
