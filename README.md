# deepagents-blob

Pluggable large-file offloading for [deepagents](https://docs.langchain.com/oss/python/deepagents) /
headless LangSmith Fleet. `OffloadingBackend` wraps any deepagents backend
(typically `StateBackend`): file content above a size threshold goes to a
pluggable blob store, and LangGraph state keeps only a ~230-byte
content-addressed pointer stub — so checkpoints and traces stay small while the
agent's `write_file` / `read_file` / `edit_file` tools keep working unchanged.

## Provider seam

`BlobStore` (`deepagents_blob/stores/protocol.py`) is a minimal key/value
contract plus `BlobCapabilities` flags — deliberately *not* S3-shaped, so
folder/file-ID providers (Box, Egnyte, SharePoint) fit as first-class adapters.

Included stores:

| Store | Covers |
|---|---|
| `S3CompatibleStore` | AWS S3, GCS (interop mode), Backblaze B2, MinIO, Cloudflare R2 — anything behind `endpoint_url` |
| `BoxBlobStore` | Box (folder-mapped, chunked upload sessions) |
| `EgnyteBlobStore` | Egnyte |
| `InMemoryBlobStore` | tests / local dev |

Azure Blob or SharePoint = one new class implementing `BlobStore`.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# no credentials needed:
.venv/bin/python tests/test_offloading_smoke.py   # backend unit behavior
.venv/bin/python tests/test_agent_e2e.py          # full deepagents graph, scripted model

# live agent runs (configure via .env — see .env.example):
.venv/bin/python examples/run_agent.py   # one-shot demo
.venv/bin/python examples/chat.py        # interactive CLI: ingest URLs, retrieve to ~/Downloads
```

Configuration lives in `.env` (auto-loaded): a model key (`ANTHROPIC_API_KEY`
or `OPENAI_API_KEY`), and `BLOB_STORE=s3|box|memory` plus that provider's
credentials (`S3_BUCKET`/`S3_ENDPOINT_URL`, or `BOX_CLIENT_ID`/
`BOX_CLIENT_SECRET`/`BOX_ENTERPRISE_ID` for a CCG app authorized in the
Admin Console).

## Wiring

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemState
from deepagents_blob import OffloadingBackend
from deepagents_blob.stores import S3CompatibleStore

backend = OffloadingBackend(StateBackend(), S3CompatibleStore(bucket="agent-blobs"))
agent = create_deep_agent(backend=backend, state_schema=FilesystemState)
```

> **`state_schema=FilesystemState` is required.** deepagents (0.7.13) registers
> the `files` state channel only when it sees `StateBackend`/`CompositeBackend`
> directly (`_uses_state_backend` in `deepagents/middleware/filesystem.py`);
> a decorator backend must register it explicitly or every file write is
> silently dropped ("wrote to unknown channel files").

See `examples/wiring.py` for composition with `CompositeBackend` prefix
routing and provider-specific threshold notes (Box/Egnyte have no range reads,
so use a higher threshold).

## Design notes

- **Pointer stubs**: a stub is a normal `FileData` whose content is one line of
  JSON (`{"__deepagents_external__":1, "key":..., "sha256":..., ...}`).
  Anything reading the `files` channel directly must tolerate this shape —
  use `parse_pointer()`.
- **Content-addressed keys** (`sha256/ab/abcd…`): forked checkpoints share
  blobs safely, identical uploads dedupe, re-uploads are skipped via
  `exists()`. Trade-off: precise deletion is impossible, so GC is a bucket
  lifecycle/TTL policy (or a periodic prefix sweep on providers without
  server-side lifecycle). `delete()` removes only the stub.
- **Read amplification** is softened by a small in-process LRU keyed by sha256.
