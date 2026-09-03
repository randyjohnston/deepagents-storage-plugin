"""Running example: a deep agent whose large files transparently offload.

The agent gets a tool that builds a ~240KB dataset and writes it through the
backend. ``OffloadingBackend`` intercepts the write, ships the bytes to the
blob store, and leaves a ~230-byte pointer stub in LangGraph state. The model
then reads small windows of that file; ``read_file``/``edit_file`` dereference
the stub transparently.

The dataset never enters the conversation. Do not ask the model to copy a
large tool result into ``write_file`` itself: deepagents already evicts any
oversized tool result to ``/large_tool_results/<tool_call_id>``, so the model
would have to page the whole file back in and re-emit it as one tool argument
— tens of thousands of output tokens, many minutes, and usually a truncated
write. Tools that produce bulk data must write it through the backend.

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
import time
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


def build_sales_csv() -> str:
    """A ~240KB CSV: one row per store per day, 10080 data rows."""
    rows = [
        f"2026-{m:02d}-{d:02d},store-{s:02d},{(m * d * s * 7919) % 997}"
        for m in range(1, 13)
        for d in range(1, 29)
        for s in range(1, 31)
    ]
    return "date,store,revenue\n" + "\n".join(rows) + "\n"


def make_fetch_tool(backend: OffloadingBackend):
    """The tool writes through the backend, so the bytes go state -> blob store
    without ever passing through the model's context."""

    def fetch_sales_data(file_path: str) -> str:
        """Download the raw sales dataset and save it in the agent filesystem
        at file_path. Returns a one-line summary, not the data itself."""
        csv = build_sales_csv()
        result = backend.write(file_path, csv)
        if result.error:
            return f"Save failed: {result.error}"
        row_count = csv.count("\n") - 1  # minus the header
        return (
            f"Saved {len(csv.encode()):,} bytes to {file_path}: "
            f"a header line plus {row_count:,} data rows, columns date,store,revenue. "
            "Read small windows with read_file; do not read the whole file."
        )

    return fetch_sales_data


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


def run_and_trace(agent, prompt: str) -> dict:
    """Stream the run so each step is visible. ``agent.invoke`` prints nothing
    until the whole run ends, which makes a slow run look like a hang."""
    seen = 0
    state: dict = {}
    started = time.monotonic()
    for state in agent.stream({"messages": [{"role": "user", "content": prompt}]}, stream_mode="values"):
        for msg in state["messages"][seen:]:
            stamp = f"[{time.monotonic() - started:6.1f}s]"
            if msg.type == "ai":
                for call in getattr(msg, "tool_calls", []):
                    args = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in call["args"].items()}
                    print(f"{stamp} call  {call['name']} {args}")
                if msg.text:
                    print(f"{stamp} say   {msg.text[:200]}")
            elif msg.type == "tool":
                print(f"{stamp} ->    {str(msg.content)[:120].replace(chr(10), ' ')}")
        seen = len(state["messages"])
    return state


def main() -> int:
    model = pick_model()
    if model is None:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or MODEL=provider:name) to run this example.")
        print("The no-key integration test is tests/test_agent_e2e.py.")
        return 1

    store = make_store()
    threshold = pick_threshold()
    backend = OffloadingBackend(StateBackend(), store, threshold=threshold)

    agent = create_deep_agent(
        model=model,
        tools=[make_fetch_tool(backend)],
        backend=backend,
        # Required for any decorator backend: registers the `files` channel
        # (deepagents' _uses_state_backend only recognizes StateBackend /
        # CompositeBackend, not wrappers around them).
        state_schema=FilesystemState,
        system_prompt=(
            "You are a data analyst working on a large CSV.\n"
            "- Call fetch_sales_data with file_path='/data/sales.csv'. It saves the "
            "data and reports the row count; the data itself never reaches you.\n"
            "- Check the date range with two small read_file calls: the first rows, "
            "then the last rows using an offset near the end. Use limit=5 or less.\n"
            "- Never read the whole file, and never pass the data to write_file.\n"
            "- Write your findings to /data/summary.md with write_file, then answer."
        ),
    )

    print(f"model: {model} | blob store: {type(store).__name__} | offload threshold: {threshold:,}B\n")
    result = run_and_trace(
        agent, "Pull the sales data, save it, and tell me the date range and row count."
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
