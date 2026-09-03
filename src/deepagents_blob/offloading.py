"""Transparent large-file offloading for any deepagents backend.

``OffloadingBackend`` wraps an inner ``BackendProtocol`` (typically
``StateBackend``) and a ``BlobStore``. Content above ``threshold`` bytes goes
to the blob store; the inner backend receives only a small *pointer stub*, so
LangGraph checkpoints and trace payloads stay small. Because it sits at the
backend seam, every existing consumer is fixed for free: the model's file
tools, ``_overflow_clip``'s ``/large_tool_results/`` offload, and the
summarization middleware's history/media offload all flow through it.

Pointer stubs
-------------
A stub is a normal ``FileData`` whose ``encoding`` is ``"external"`` and whose
``content`` is a compact JSON pointer::

    {"__deepagents_external__": 1, "key": "sha256/ab/abcd...", "size": 1048576,
     "sha256": "abcd...", "text": true, "content_type": null}

Anything reading the ``files`` channel directly (custom nodes, UIs) must
tolerate this shape — check ``encoding == EXTERNAL_ENCODING`` before treating
``content`` as file text.

Keys & garbage collection
-------------------------
Keys are content-addressed (``sha256/<h[:2]>/<hash>``): forked checkpoints
referencing the same blob are naturally safe, identical uploads dedupe, and
``exists()`` lets us skip re-uploads. The trade is that precise deletion is
impossible (another checkpoint may still point at the blob), so GC is a
bucket lifecycle/TTL policy aligned with thread TTLs — or, on providers
without server-side lifecycle (Box, Egnyte), a periodic sweep that lists the
prefix and deletes blobs older than the retention window. ``delete()`` here
therefore only removes the stub from the inner backend.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from typing import Final

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    create_file_data,
    perform_string_replacement,
    slice_read_response,
)

from deepagents_blob.stores.protocol import BlobNotFoundError, BlobStore

EXTERNAL_ENCODING: Final = "external"
"""``FileData.encoding`` value marking a pointer stub. Extends the upstream
``"utf-8" | "base64"`` contract — see module docstring."""

_MARKER: Final = "__deepagents_external__"

DEFAULT_THRESHOLD: Final = 64 * 1024
"""Offload cutoff in bytes. Below this, state overhead is cheaper than a
blob round-trip; raise it for providers without range reads (Box/Egnyte)
where every read fetches the whole object."""


def default_key_fn(sha256: str) -> str:
    """Content-addressed key layout: ``sha256/ab/abcd...``. The two-char fan-out
    keeps prefix listings and per-prefix lifecycle rules manageable."""
    return f"sha256/{sha256[:2]}/{sha256}"


def make_pointer(*, key: str, size: int, sha256: str, is_text: bool, content_type: str | None = None) -> str:
    """Serialize a pointer stub's ``content`` JSON."""
    return json.dumps(
        {_MARKER: 1, "key": key, "size": size, "sha256": sha256, "text": is_text, "content_type": content_type},
        separators=(",", ":"),
    )


def parse_pointer(file_data: FileData) -> dict | None:
    """Return the pointer dict when ``file_data`` is a stub, else ``None``.

    Detection is content-based (marker key in the JSON), not encoding-based:
    ``encoding == EXTERNAL_ENCODING`` is only guaranteed when the inner
    backend exposes a FileData-level write (StateBackend); the generic
    fallback stores the pointer JSON as ordinary ``utf-8`` content.
    """
    content = file_data.get("content", "")
    if _MARKER not in content[:64]:
        return None
    try:
        ptr = json.loads(content)
    except ValueError:
        return None
    return ptr if isinstance(ptr, dict) and _MARKER in ptr else None


class _ByteCache:
    """Tiny LRU keyed by sha256, bounded by total bytes. Softens read
    amplification when the model re-reads a hot offloaded file (pointer
    dereference = full GET on providers without range support)."""

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._used = 0
        self._data: OrderedDict[str, bytes] = OrderedDict()

    def get(self, sha256: str) -> bytes | None:
        if sha256 in self._data:
            self._data.move_to_end(sha256)
            return self._data[sha256]
        return None

    def put(self, sha256: str, raw: bytes) -> None:
        if len(raw) > self._max:
            return
        self._data[sha256] = raw
        self._used += len(raw)
        while self._used > self._max:
            _, evicted = self._data.popitem(last=False)
            self._used -= len(evicted)


class OffloadingBackend(BackendProtocol):
    """Decorator backend: large content -> ``store``, stubs -> ``inner``.

    Examples:
        >>> from deepagents import create_deep_agent
        >>> from deepagents.backends import StateBackend
        >>> backend = OffloadingBackend(StateBackend(), S3CompatibleStore(bucket="agent-blobs"))
        >>> agent = create_deep_agent(backend=backend, ...)
    """

    def __init__(
        self,
        inner: BackendProtocol,
        store: BlobStore,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        key_fn: Callable[[str], str] = default_key_fn,
        cache_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.inner = inner
        self.store = store
        self.threshold = threshold
        self._key_fn = key_fn
        self._cache = _ByteCache(cache_bytes)

    # -- offload / materialize core -------------------------------------------

    def _offload(self, raw: bytes, *, is_text: bool) -> FileData:
        """Put ``raw`` in the blob store and return the stub FileData."""
        digest = hashlib.sha256(raw).hexdigest()
        key = self._key_fn(digest)
        if not self.store.exists(key):  # content-addressed: skip duplicate PUTs
            self.store.put(key, raw)
        self._cache.put(digest, raw)
        pointer = make_pointer(key=key, size=len(raw), sha256=digest, is_text=is_text)
        stub = create_file_data(pointer)  # sets created/modified timestamps
        stub["encoding"] = EXTERNAL_ENCODING
        return stub

    def _fetch(self, ptr: dict) -> bytes:
        raw = self._cache.get(ptr["sha256"])
        if raw is None:
            raw = self.store.get(ptr["key"])
            self._cache.put(ptr["sha256"], raw)
        return raw

    def _materialize(self, ptr: dict) -> str:
        """Pointer -> the string form the rest of deepagents expects
        (utf-8 text, or base64 for binary — matching FileData conventions)."""
        raw = self._fetch(ptr)
        if ptr.get("text", True):
            return raw.decode("utf-8")
        import base64  # noqa: PLC0415

        return base64.b64encode(raw).decode("ascii")

    def _read_stub(self, file_path: str) -> dict | None:
        """Fetch ``file_path`` from the inner backend and return its pointer
        if it's a stub. Uses a 1-line inner read: stubs are single-line JSON,
        so this stays cheap even against remote inner backends."""
        result = self.inner.read(file_path, offset=0, limit=1)
        if result.error or result.file_data is None:
            return None
        return parse_pointer(result.file_data)

    # -- BackendProtocol: reads -----------------------------------------------

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        ptr = self._read_stub(file_path)
        if ptr is None:
            # Missing file or not offloaded — delegate with the caller's window.
            return self.inner.read(file_path, offset=offset, limit=limit)
        try:
            content = self._materialize(ptr)
        except BlobNotFoundError:
            return ReadResult(error=f"File '{file_path}': external blob {ptr['key']} not found (expired or deleted)")
        encoding = "utf-8" if ptr.get("text", True) else "base64"
        file_data = FileData(content=content, encoding=encoding)
        if encoding == "base64":
            return ReadResult(file_data=file_data)
        return slice_read_response(file_data, offset, limit)

    def ls(self, path: str) -> LsResult:
        # Delegated as-is. Note: for stubs the inner backend reports the
        # pointer's byte length as `size`, not the true object size.
        # TODO: rewrite entry sizes from pointer metadata (needs a batched
        # inner read of stub heads; cheap on StateBackend, chatty elsewhere).
        return self.inner.ls(path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None) -> GrepResult:
        result = self.inner.grep(pattern, path=path, glob=glob, max_count=max_count)
        if result.matches:
            # Drop hits inside pointer JSON (e.g. searching for "sha256").
            result.matches = [m for m in result.matches if _MARKER not in m.get("text", "")]
        # TODO (opt-in, `grep_external=True`): additionally glob for stubs,
        # materialize text ones, and grep them via
        # deepagents.backends.utils.grep_matches_from_files. Off by default —
        # it downloads every offloaded file under `path`.
        return result

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.inner.glob(pattern, path)

    # -- BackendProtocol: writes ----------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        raw = content.encode("utf-8")
        if len(raw) <= self.threshold:
            return self.inner.write(file_path, content)
        stub = self._offload(raw, is_text=True)
        return self._write_stub(file_path, stub)

    def _write_stub(self, file_path: str, stub: FileData) -> WriteResult:
        """Persist a stub through the inner backend.

        Inner backends write plain strings, which would store the pointer with
        ``encoding="utf-8"``. For ``StateBackend`` we instead push the stub
        FileData directly into the ``files`` channel so ``encoding="external"``
        survives; other inner backends fall back to writing the pointer JSON
        as file content (still functional: ``parse_pointer`` on read checks
        content, so add a content-based fallback if you take that path).

        TODO: upstream a ``write_file_data`` hook on BackendProtocol to avoid
        reaching into StateBackend internals here.
        """
        send = getattr(self.inner, "_send_files_update", None)
        if callable(send):
            send({file_path: dict(stub)})
            return WriteResult(path=file_path)
        return self.inner.write(file_path, stub["content"])

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        ptr = self._read_stub(file_path)
        if ptr is None:
            return self.inner.edit(file_path, old_string, new_string, replace_all)
        if not ptr.get("text", True):
            return EditResult(error=f"Error: '{file_path}' is an offloaded binary file and cannot be edited")
        try:
            content = self._materialize(ptr)
        except BlobNotFoundError:
            return EditResult(error=f"Error: external blob for '{file_path}' not found (expired or deleted)")
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        if isinstance(result, str):
            return EditResult(error=result)
        new_content, occurrences = result
        write_result = self.write(file_path, new_content)  # re-offloads if still large
        if write_result.error:
            return EditResult(error=write_result.error)
        return EditResult(path=file_path, occurrences=int(occurrences))

    def delete(self, file_path: str) -> DeleteResult:
        # Only the stub is removed; the blob stays (content-addressed, other
        # checkpoints may reference it) and is reclaimed by lifecycle GC.
        return self.inner.delete(file_path)

    # -- BackendProtocol: byte transfer ---------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        passthrough: list[tuple[str, bytes]] = []
        for path, data in files:
            if len(data) <= self.threshold:
                passthrough.append((path, data))
                continue
            is_text = _decodes_utf8(data)
            try:
                stub = self._offload(data, is_text=is_text)
                self._write_stub(path, stub)
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:  # noqa: BLE001 - normalize per-file, batch continues
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        if passthrough:
            responses.extend(self.inner.upload_files(passthrough))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        # Detect stubs via a FileData-level read BEFORE delegating: inner
        # backends serialize bytes by encoding, and a stub's "external"
        # encoding hits StateBackend's base64 branch, mangling the pointer.
        responses: dict[str, FileDownloadResponse] = {}
        passthrough: list[str] = []
        for path in paths:
            ptr = self._read_stub(path)
            if ptr is None:
                passthrough.append(path)
                continue
            try:
                responses[path] = FileDownloadResponse(path=path, content=self._fetch(ptr))
            except BlobNotFoundError:
                responses[path] = FileDownloadResponse(path=path, content=None, error="file_not_found")
        for inner_resp in self.inner.download_files(passthrough) if passthrough else []:
            if inner_resp.content is not None:
                # Inner backends without FileData-level writes store pointer
                # JSON as ordinary utf-8 content — catch those here.
                ptr = _pointer_from_bytes(inner_resp.content)
                if ptr is not None:
                    try:
                        inner_resp = FileDownloadResponse(path=inner_resp.path, content=self._fetch(ptr))
                    except BlobNotFoundError:
                        inner_resp = FileDownloadResponse(path=inner_resp.path, content=None, error="file_not_found")
            responses[inner_resp.path] = inner_resp
        return [responses[p] for p in paths]

    # -- async twins ----------------------------------------------------------
    # The BackendProtocol defaults (asyncio.to_thread over the sync methods)
    # are correct here. TODO: override aread/awrite/aupload_files to use
    # store.aget/aput for native-async stores (aioboto3) once wired.


def _decodes_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _pointer_from_bytes(data: bytes) -> dict | None:
    """Pointer detection for the bytes path (download_files)."""
    if _MARKER.encode() not in data[:64]:
        return None
    try:
        ptr = json.loads(data)
    except ValueError:
        return None
    return ptr if isinstance(ptr, dict) and _MARKER in ptr else None
