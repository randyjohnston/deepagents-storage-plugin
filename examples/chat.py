"""Interactive CLI: chat with a deep agent that ingests URLs into blob
storage and retrieves files back into ~/Downloads on request.

Bytes never pass through the model: ``fetch_url`` streams URL content straight
into the backend (large payloads become blob-store objects with a pointer stub
in state), and ``save_to_downloads`` streams them back out to local disk.

Usage::

    export ANTHROPIC_API_KEY=... (or OPENAI_API_KEY / MODEL=provider:name)
    python examples/chat.py                       # in-memory blob store
    S3_BUCKET=agent-blobs python examples/chat.py # or BOX_CLIENT_ID etc.

Commands: /files (show agent filesystem), /quit.
"""

import os
import sqlite3
import sys
import urllib.request
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemState

from deepagents_blob import OffloadingBackend, parse_pointer
from run_agent import make_store, pick_model, pick_threshold  # examples/ sibling module

DOWNLOADS_DIR = Path(os.environ.get("DEEPAGENTS_DOWNLOADS_DIR", Path.home() / "Downloads"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DB = Path(os.environ.get("CHAT_STATE_DB", PROJECT_ROOT / ".chat_state.sqlite"))
THREAD_FILE = STATE_DB.with_suffix(".thread")


def current_thread_id(*, new: bool = False) -> str:
    """Sticky thread id so sessions resume across restarts; /new rotates it."""
    if not new and THREAD_FILE.exists():
        return THREAD_FILE.read_text().strip()
    thread_id = uuid.uuid4().hex
    THREAD_FILE.write_text(thread_id)
    return thread_id

SYSTEM_PROMPT = """You are a file librarian with a persistent filesystem.

- Your filesystem (paths like /ingest/...) is agent storage backed by a blob
  store — it is NOT the user's computer. Files only reach the user's actual
  Downloads folder via the save_to_downloads tool.
- When the user gives you a URL to ingest, call fetch_url with a tidy
  destination path like /ingest/<filename>. The bytes go straight to
  storage; you only see a summary. Do not read binary files afterwards, and
  read only small windows of large text files (read_file with a limit).
- After ingesting, tell the user it is stored, and that they can ask you to
  save it to their Downloads folder to get a local copy.
- When the user asks to get a file back / export / download it, find it
  (ls, glob) if needed and call save_to_downloads.
- NEVER claim a file is in the user's Downloads folder unless a
  save_to_downloads result in this conversation says so — quote the exact
  path from that result. fetch_url alone puts nothing on their computer.
- Keep answers short; report sizes and paths.
"""


def build_tools(backend: OffloadingBackend):
    def fetch_url(url: str, file_path: str) -> str:
        """Download a URL and store it in the agent filesystem at file_path.

        The bytes stream straight to storage and never enter the conversation.
        Returns a one-line summary (size, content type).
        """
        req = urllib.request.Request(url, headers={"User-Agent": "deepagents-blob-cli/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - user-supplied URL, CLI context
            data = resp.read()
            ctype = resp.headers.get_content_type()
        result = backend.upload_files([(file_path, data)])[0]
        if result.error:
            return f"Upload failed: {result.error}"
        return (
            f"Stored in agent blob storage as {file_path} ({len(data):,} bytes, {ctype}). "
            "This is NOT on the user's computer — nothing was written to their Downloads folder. "
            "Offer save_to_downloads if they want a local copy."
        )

    def save_to_downloads(file_path: str, filename: str | None = None) -> str:
        """Copy a file from the agent filesystem into the user's Downloads
        folder (dereferencing blob storage). Returns the local path written."""
        resp = backend.download_files([file_path])[0]
        if resp.error or resp.content is None:
            return f"Could not retrieve {file_path}: {resp.error or 'no content'}"
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        name = Path(filename or Path(file_path).name).name  # strip any directories
        target = DOWNLOADS_DIR / name
        counter = 1
        while target.exists():
            target = DOWNLOADS_DIR / f"{Path(name).stem}-{counter}{Path(name).suffix}"
            counter += 1
        target.write_bytes(resp.content)
        return f"Wrote {len(resp.content):,} bytes to {target} on the user's computer. Report this exact path to the user."

    return [fetch_url, save_to_downloads]


def build_agent(model, store):
    backend = OffloadingBackend(StateBackend(), store, threshold=pick_threshold())
    return create_deep_agent(
        model=model,
        tools=build_tools(backend),
        backend=backend,
        state_schema=FilesystemState,
        checkpointer=SqliteSaver(sqlite3.connect(STATE_DB, check_same_thread=False)),
        system_prompt=SYSTEM_PROMPT,
    )


def show_files(agent, config) -> None:
    files = agent.get_state(config).values.get("files", {})
    if not files:
        print("  (no files yet)")
        return
    for path, file_data in sorted(files.items()):
        ptr = parse_pointer(file_data)
        if ptr:
            kind = "text" if ptr.get("text", True) else "binary"
            print(f"  {path}  {ptr['size']:,}B {kind}, offloaded -> {ptr['key']}")
        else:
            print(f"  {path}  {len(file_data['content']):,}B inline in state")


def run_turn(agent, config, user_input: str) -> None:
    seen = 0
    for state in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"
    ):
        messages = state["messages"]
        for msg in messages[seen:]:
            if msg.type == "ai":
                for call in getattr(msg, "tool_calls", []):
                    brief = {k: (v[:60] + "…" if isinstance(v, str) and len(v) > 60 else v) for k, v in call["args"].items()}
                    print(f"  [tool] {call['name']} {brief}")
                if msg.text:
                    print(f"agent> {msg.text}")
            elif msg.type == "tool":
                text = str(msg.content).replace("\n", " ")
                print(f"  [-->] {text[:150]}")
        seen = len(messages)


def main() -> int:
    model = pick_model()
    if model is None:
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or MODEL=provider:name) first.")
        return 1
    store = make_store()
    agent = build_agent(model, store)
    config = {"configurable": {"thread_id": current_thread_id()}}

    print(f"deepagents-blob chat — model: {model} | store: {type(store).__name__} | offload threshold: {pick_threshold():,}B")
    print("Session persists across restarts. URLs to ingest, ask for files back; /files, /new (fresh session), /quit.")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in {"/q", "/quit", "/exit"}:
            break
        if user_input == "/files":
            show_files(agent, config)
            continue
        if user_input == "/new":
            config = {"configurable": {"thread_id": current_thread_id(new=True)}}
            print("started a fresh session (old files stay in the blob store; their listing is gone)")
            continue
        try:
            run_turn(agent, config, user_input)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            print(f"  [error] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
