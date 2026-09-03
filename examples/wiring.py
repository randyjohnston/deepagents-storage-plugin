"""Wiring examples: OffloadingBackend with create_deep_agent.

Always pass ``state_schema=FilesystemState`` alongside a decorator backend:
deepagents' ``_uses_state_backend`` only recognizes StateBackend /
CompositeBackend, so the ``files`` channel is not registered automatically
for a wrapper (verified against deepagents 0.7.13).
"""

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemState

from deepagents_blob import OffloadingBackend
from deepagents_blob.stores import S3CompatibleStore


def s3_agent():
    """Default state files, with anything > 64KB offloaded to S3/B2/MinIO."""
    store = S3CompatibleStore(
        bucket="agent-blobs",
        # endpoint_url="https://s3.us-west-004.backblazeb2.com",  # Backblaze B2
        # endpoint_url="http://minio:9000",                        # MinIO
        key_prefix="tenant-acme/",  # scope IAM/lifecycle rules per tenant
    )
    backend = OffloadingBackend(StateBackend(), store, threshold=64 * 1024)
    return create_deep_agent(backend=backend, state_schema=FilesystemState)


def composite_agent(langgraph_store):
    """Offloading composes with prefix routing: /memories/ stays in the
    LangGraph BaseStore, everything else is state + blob offload."""
    blob = S3CompatibleStore(bucket="agent-blobs")
    backend = CompositeBackend(
        default=OffloadingBackend(StateBackend(), blob),
        routes={"/memories/": StoreBackend()},
    )
    return create_deep_agent(backend=backend, store=langgraph_store, state_schema=FilesystemState)


def box_agent():
    """Box supports range reads (Range header on downloads), but key->file-ID
    resolution costs extra API calls per access — raise the threshold so only
    genuinely large payloads pay that cost."""
    from box_sdk_gen import BoxCCGAuth, BoxClient, CCGConfig

    from deepagents_blob.stores import BoxBlobStore

    auth = BoxCCGAuth(CCGConfig(client_id=..., client_secret=..., enterprise_id=...))
    store = BoxBlobStore(BoxClient(auth=auth), root_folder_id="0")
    backend = OffloadingBackend(StateBackend(), store, threshold=512 * 1024)
    return create_deep_agent(backend=backend, state_schema=FilesystemState)
