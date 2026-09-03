"""Running example: a deep agent whose large files transparently offload.

The agent gets a tool that returns a ~500KB dataset. It saves the data with
its normal ``write_file`` tool — ``OffloadingBackend`` intercepts the write,
ships the bytes to the blob store, and leaves a ~230-byte pointer stub in
LangGraph state. ``read_file``/``edit_file`` dereference transparently.

Usage::

    export ANTHROPIC_API_KEY=sk-ant-...   # or: export OPENAI_API_KEY=sk-...
    python examples/run_agent.py                 # in-memory blob store

    # Any S3-compatible provider (AWS, GCS interop, Backblaze B2, MinIO, R2):
    export S3_BUCKET=agent-blobs
    export S3_ENDPOINT_URL=http://localhost:9000   # omit for AWS
    python examples/run_agent.py

    # Override the model explicitly (any init_chat_model string):
    MODEL=openai:gpt-5.1 python examples/run_agent.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemState

from deepagents_blob import OffloadingBackend, parse_pointer
from deepagents_blob.stores import BoxBlobStore, InMemoryBlobStore, S3CompatibleStore


def make_store():
    """Provider is pluggable. BLOB_STORE=s3|box|memory picks explicitly;
    unset, it auto-selects from whichever credentials are configured."""
    choice = os.environ.get("BLOB_STORE", "").lower()
    if not choice:
        choice = "s3" if os.environ.get("S3_BUCKET") else "box" if os.environ.get("BOX_CLIENT_ID") else "memory"
    if choice == "s3":
        if not os.environ.get("S3_BUCKET"):
            sys.exit("BLOB_STORE=s3 but S3_BUCKET is not set — add it to .env.")
        return S3CompatibleStore(bucket=os.environ["S3_BUCKET"], endpoint_url=os.environ.get("S3_ENDPOINT_URL"))
    if choice == "box":
        missing = [v for v in ("BOX_CLIENT_ID", "BOX_CLIENT_SECRET", "BOX_ENTERPRISE_ID") if not os.environ.get(v)]
        if missing:
            sys.exit(f"BLOB_STORE=box but {', '.join(missing)} not set — add to .env (or unset BLOB_STORE to auto-select).")
        from box_sdk_gen import BoxCCGAuth, BoxClient, CCGConfig

        auth = BoxCCGAuth(
            CCGConfig(
                client_id=os.environ["BOX_CLIENT_ID"],
                client_secret=os.environ["BOX_CLIENT_SECRET"],
                enterprise_id=os.environ["BOX_ENTERPRISE_ID"],
            )
        )
        return BoxBlobStore(BoxClient(auth=auth))
    if choice != "memory":
        print(f"Unknown BLOB_STORE={choice!r}, using in-memory store.")
    return InMemoryBlobStore()


def fetch_sales_data() -> str:
    """Download the complete raw sales dataset as CSV text. Large."""
    rows = [
        f"2026-{m:02d}-{d:02d},store-{d % 5},{(m * d * 7919) % 997}"
        for m in range(1, 13)
        for d in range(1, 29)
        for _ in range(30)
    ]
    return "date,store,revenue\n" + "\n".join(rows)


def pick_threshold() -> int:
    """Offload cutoff from BLOB_THRESHOLD: plain bytes or K/KB/M/MB suffix
    (e.g. "512KB", "1M", "65536"). Defaults to 512KB."""
    raw = os.environ.get("BLOB_THRESHOLD", "512KB").strip().upper().replace(" ", "")
    for suffix, factor in (("KB", 1024), ("MB", 1024 * 1024), ("K", 1024), ("M", 1024 * 1024), ("B", 1)):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * factor)
    return int(raw)


def pick_model() -> str | None:
    """Model is provider-agnostic: any LangChain init_chat_model string."""
    if os.environ.get("MODEL"):
        return os.environ["MODEL"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic:claude-sonnet-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-5.1"
    return None


def main() -> int:
    model = pick_model()
    if model is None:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or MODEL=provider:name) to run this example.")
        print("The no-key integration test is tests/test_agent_e2e.py.")
        return 1

    store = make_store()
    backend = OffloadingBackend(StateBackend(), store, threshold=pick_threshold())

    agent = create_deep_agent(
        model=model,
        tools=[fetch_sales_data],
        backend=backend,
        # Required for any decorator backend: registers the `files` channel
        # (deepagents' _uses_state_backend only recognizes StateBackend /
        # CompositeBackend, not wrappers around them).
        state_schema=FilesystemState,
        system_prompt=(
            "You are a data analyst. When asked about the sales data, first call "
            "fetch_sales_data and save the raw output to /data/sales.csv with "
            "write_file, then answer from the file."
        ),
    )

    print(f"model: {model} | blob store: {type(store).__name__}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Pull the sales data, save it, and tell me the date range and row count."}]}
    )

    print("\n--- agent answer " + "-" * 50)
    print(result["messages"][-1].text)

    print("\n--- what actually got stored " + "-" * 39)
    for path, file_data in sorted(result.get("files", {}).items()):
        ptr = parse_pointer(file_data)
        if ptr:
            print(f"{path}: {len(file_data['content'])}B stub in state -> {ptr['size']}B blob at {ptr['key']}")
        else:
            print(f"{path}: {len(file_data['content'])}B inline in state")
    if isinstance(store, InMemoryBlobStore):
        print(f"blob store now holds {len(store._objects)} object(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
