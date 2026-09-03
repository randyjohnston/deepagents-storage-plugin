"""Box adapter: key-value facade over a folder/file-ID system.

Box is *not* object storage: content lives under numeric folder/file IDs,
uploads over ~20MB should use chunked upload sessions, there are no presigned
URLs (auth is OAuth2 / CCG / JWT service accounts), and listing means walking
folders. This adapter owns the key<->ID mapping so callers never see Box IDs.

Key mapping: a key like ``sha256/ab/abcd...ef`` becomes the path
``<root>/sha256/ab/`` (folders, created on demand) containing a file named
``abcd...ef``. Folder IDs are memoized; for multi-process deployments swap
``_folder_cache`` for a shared cache, since ``folders.create_folder`` on an
existing name 409s and requires a lookup retry.

Requires ``box-sdk-gen``::

    from box_sdk_gen import BoxCCGAuth, BoxClient, CCGConfig
    auth = BoxCCGAuth(CCGConfig(client_id=..., client_secret=..., enterprise_id=...))
    store = BoxBlobStore(BoxClient(auth=auth))
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import IO, Any

from deepagents_blob.stores.protocol import (
    BlobCapabilities,
    BlobInfo,
    BlobNotFoundError,
    BlobRef,
    BlobStore,
)

_CHUNKED_UPLOAD_MIN: int = 20 * 1024 * 1024
"""Box requires chunked upload sessions for files >= 20MB; below that the
simple ``uploads.upload_file`` endpoint is fine."""

_PAGE_LIMIT: int = 1000
_ITEM_FIELDS = ["type", "id", "name", "size", "modified_at", "etag"]


def _type_of(item: Any) -> str:
    t = getattr(item, "type", None)
    return getattr(t, "value", t) or ""


class BoxBlobStore(BlobStore):
    """Blob store facade over Box.

    Capability notes vs. S3:
    - no presign: ingress must proxy bytes or drive a Box upload session
      client-side with a downscoped token.
    - range_get: Box honors the ``Range`` header on content downloads, so
      windowed reads are supported.
    - no server_side_lifecycle: GC must be client-side (see
      ``OffloadingBackend`` docs on orphan sweeps).
    """

    capabilities = BlobCapabilities(
        presign_get=False,
        presign_put=False,
        range_get=True,
        multipart=True,  # chunked upload sessions
        native_list=False,
        server_side_lifecycle=False,
    )

    def __init__(self, client: Any, root_folder_id: str = "0") -> None:
        """Args:
        client: An authenticated ``box_sdk_gen.BoxClient`` (CCG or JWT
            service account recommended for headless use).
        root_folder_id: Folder under which all keys live ("0" = root).
        """
        self._client = client
        self._root = root_folder_id
        self._folder_cache: dict[str, str] = {"": root_folder_id}  # path -> folder_id
        self._file_cache: dict[str, str] = {}  # key -> file_id

    # -- key <-> ID mapping ----------------------------------------------------

    def _iter_items(self, folder_id: str) -> Iterator[Any]:
        marker: str | None = None
        while True:
            page = self._client.folders.get_folder_items(
                folder_id, fields=_ITEM_FIELDS, usemarker=True, marker=marker, limit=_PAGE_LIMIT
            )
            yield from page.entries or []
            marker = getattr(getattr(page, "next_marker", None), "value", None) or getattr(page, "next_marker", None)
            if not marker:
                return

    def _find_child(self, folder_id: str, name: str, kind: str) -> Any | None:
        for item in self._iter_items(folder_id):
            if item.name == name and _type_of(item) == kind:
                return item
        return None

    @staticmethod
    def _conflict_id(exc: Exception) -> str | None:
        """Pull the existing item's ID out of a 409 response, when present."""
        body = getattr(getattr(exc, "response_info", None), "body", None) or {}
        conflicts = body.get("context_info", {}).get("conflicts")
        if isinstance(conflicts, list) and conflicts:
            conflicts = conflicts[0]
        if isinstance(conflicts, dict):
            return conflicts.get("id")
        return None

    def _walk_folder(self, folder_path: str, *, create: bool) -> str | None:
        """Resolve the folder ID for ``a/b/c``, optionally creating segments.
        Returns ``None`` when absent and ``create`` is False."""
        from box_sdk_gen import BoxAPIError  # noqa: PLC0415 - optional dependency
        from box_sdk_gen.managers.folders import CreateFolderParent  # noqa: PLC0415

        if folder_path in self._folder_cache:
            return self._folder_cache[folder_path]
        parent_id = self._root
        built = ""
        for seg in folder_path.split("/") if folder_path else []:
            built = f"{built}/{seg}" if built else seg
            cached = self._folder_cache.get(built)
            if cached is not None:
                parent_id = cached
                continue
            found = self._find_child(parent_id, seg, "folder")
            if found is not None:
                folder_id = found.id
            elif not create:
                return None
            else:
                try:
                    folder_id = self._client.folders.create_folder(seg, CreateFolderParent(id=parent_id)).id
                except BoxAPIError as exc:
                    if getattr(exc.response_info, "status_code", None) != 409:
                        raise
                    folder_id = self._conflict_id(exc)
                    if folder_id is None:  # racing writer; the folder exists now
                        refound = self._find_child(parent_id, seg, "folder")
                        if refound is None:
                            raise
                        folder_id = refound.id
            self._folder_cache[built] = folder_id
            parent_id = folder_id
        return parent_id

    def _ensure_folder(self, folder_path: str) -> str:
        folder_id = self._walk_folder(folder_path, create=True)
        assert folder_id is not None
        return folder_id

    def _resolve_file_id(self, key: str) -> str:
        if key in self._file_cache:
            return self._file_cache[key]
        folder_path, _, name = key.rpartition("/")
        folder_id = self._walk_folder(folder_path, create=False)
        if folder_id is None:
            raise BlobNotFoundError(key)
        found = self._find_child(folder_id, name, "file")
        if found is None:
            raise BlobNotFoundError(key)
        self._file_cache[key] = found.id
        return found.id

    # -- required surface ------------------------------------------------------

    def put(self, key: str, data: bytes | IO[bytes], *, content_type: str | None = None) -> BlobRef:
        from box_sdk_gen import BoxAPIError  # noqa: PLC0415
        from box_sdk_gen.managers.uploads import (  # noqa: PLC0415
            UploadFileAttributes,
            UploadFileAttributesParentField,
            UploadFileVersionAttributes,
        )

        folder_path, _, name = key.rpartition("/")
        folder_id = self._ensure_folder(folder_path)
        raw: bytes = data if isinstance(data, bytes) else data.read()
        stream = BytesIO(raw)

        if len(raw) >= _CHUNKED_UPLOAD_MIN:
            entry = self._client.chunked_uploads.upload_big_file(stream, name, len(raw), folder_id)
        else:
            attrs = UploadFileAttributes(name=name, parent=UploadFileAttributesParentField(id=folder_id))
            try:
                entry = self._client.uploads.upload_file(attrs, stream).entries[0]
            except BoxAPIError as exc:
                if getattr(exc.response_info, "status_code", None) != 409:
                    raise
                file_id = self._conflict_id(exc) or self._resolve_file_id(key)
                stream.seek(0)
                entry = self._client.uploads.upload_file_version(
                    file_id, UploadFileVersionAttributes(name=name), stream
                ).entries[0]
        self._file_cache[key] = entry.id
        return BlobRef(key=key, size=len(raw), etag=getattr(entry, "etag", None))

    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        file_id = self._resolve_file_id(key)
        range_header = f"bytes={byte_range[0]}-{byte_range[1]}" if byte_range is not None else None
        stream = self._client.downloads.download_file(file_id, range=range_header)
        if stream is None:  # pragma: no cover - SDK returns None only for 202 retry-later
            raise BlobNotFoundError(key)
        return stream.read()

    def exists(self, key: str) -> bool:
        try:
            self._resolve_file_id(key)
        except BlobNotFoundError:
            return False
        return True

    def delete(self, key: str) -> None:
        from box_sdk_gen import BoxAPIError  # noqa: PLC0415

        try:
            file_id = self._resolve_file_id(key)
        except BlobNotFoundError:
            return  # idempotent
        try:
            self._client.files.delete_file_by_id(file_id)
        except BoxAPIError as exc:
            if getattr(exc.response_info, "status_code", None) != 404:
                raise
        self._file_cache.pop(key, None)

    def list(self, prefix: str) -> Iterator[BlobInfo]:
        """Recursive folder walk (paged, rate-limited — treat as a
        reconciliation tool, not a hot path)."""
        yield from self._list_under("", self._root, prefix)

    def _list_under(self, path: str, folder_id: str, prefix: str) -> Iterator[BlobInfo]:
        for item in self._iter_items(folder_id):
            child = f"{path}{item.name}"
            kind = _type_of(item)
            if kind == "folder":
                subtree = f"{child}/"
                if subtree.startswith(prefix) or prefix.startswith(subtree):
                    self._folder_cache.setdefault(child, item.id)
                    yield from self._list_under(subtree, item.id, prefix)
            elif kind == "file" and child.startswith(prefix):
                yield BlobInfo(key=child, size=getattr(item, "size", 0) or 0, modified_at=getattr(item, "modified_at", None))
